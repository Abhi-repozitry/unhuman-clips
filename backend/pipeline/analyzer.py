"""Video analysis module — multi-stage LLM reel plan generation.

Architecture:
  Rich Timeline
       ↓
  Semantic Block Builder (Python) + Importance Scoring (Python)
       ↓
  LLM #1  Structure Planner
       ↓
  LLM #2  Clip Planner
       ↓
  LLM #3  Narration Writer
       ↓
  LLM #4  Critic (optional revision)
       ↓
  Python Validator (finalize_edit)
       ↓
  Final ReelPlan
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.config import (
    CLIP_COUNT_MAX,
    CLIP_COUNT_MIN,
    CLIP_DURATION_SOFT_MAX,
    CLIP_DURATION_SOFT_MIN,
    HOOK_SECONDS,
    INSIGHT_SECONDS_MAX,

    MAX_OUTPUT_DURATION,
    MAX_OUTPUT_TOKENS,
    MIN_OUTPUT_DURATION,
    MODEL_FALLBACK_MAP,
    OPENCODE_API_KEY,
    OPENCODE_BASE_URL,
    OPENCODE_MODEL,
    OPENCODE_MODEL_FALLBACK,
)
from backend.models import LLMInteraction, ReelPlan, RichTimeline
from backend.pipeline.plan_validator import finalize_edit
from backend.pipeline.sanitize import sanitize_text
from backend.providers.llm import call_llm_sync

__all__ = ["select_reel_plan", "select_clips"]

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic blocks + importance scoring (Python owns the numbers)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticBlock:
    block_id: int
    start: float
    end: float
    text: str
    speech_energy: float
    volume_db: float | None
    ocr: list[str]
    silence_before: bool
    black_frame: bool
    freeze: bool
    importance: float
    peak_offset: float  # seconds from block start to peak energy moment
    segment_ids: list[int] = field(default_factory=list)
    has_question: bool = False
    has_exclamation: bool = False
    has_emphasis: bool = False
    word_density: float = 0.0
    # Gen-Z engagement signals
    has_vulgarity: bool = False
    has_flirting: bool = False
    has_drama: bool = False
    has_bold_statement: bool = False
    has_high_stakes: bool = False
    has_spectacle: bool = False
    has_elimination: bool = False
    has_comedy: bool = False
    has_tea: bool = False
    has_reaction: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def engagement_score(self) -> float:
        """Composite engagement signal score (0-10). Higher = more viral potential."""
        return float(self.has_vulgarity + self.has_flirting + self.has_drama
                     + self.has_bold_statement + self.has_high_stakes
                     + self.has_spectacle + self.has_elimination
                     + self.has_comedy + self.has_tea + self.has_reaction)

    def summary_line(self) -> str:
        energy_bar = "█" * int(self.speech_energy * 10) + "░" * (10 - int(self.speech_energy * 10))
        ocr_str = f" | OCR: {'; '.join(self.ocr[:3])}" if self.ocr else ""
        flags = []
        if self.silence_before:
            flags.append("SILENCE_BEFORE")
        if self.black_frame:
            flags.append("BLACK")
        if self.freeze:
            flags.append("FREEZE")
        if self.has_question:
            flags.append("Q")
        if self.has_exclamation:
            flags.append("!")
        if self.has_emphasis:
            flags.append("CAPS")
        if self.has_vulgarity:
            flags.append("VULGAR")
        if self.has_flirting:
            flags.append("FLIRT")
        if self.has_drama:
            flags.append("DRAMA")
        if self.has_bold_statement:
            flags.append("BOLD")
        if self.has_high_stakes:
            flags.append("STAKES")
        if self.has_spectacle:
            flags.append("WOW")
        if self.has_elimination:
            flags.append("ELIM")
        if self.has_comedy:
            flags.append("FUNNY")
        if self.has_tea:
            flags.append("TEA")
        if self.has_reaction:
            flags.append("REACT")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        vol = f" vol={self.volume_db:.1f}dB" if self.volume_db is not None else ""
        wps = f" wps={self.word_density:.1f}" if self.word_density > 0 else ""
        return (
            f"Block {self.block_id} [{self.start:.1f}-{self.end:.1f}s] "
            f"imp={self.importance:.0f} energy={energy_bar}({self.speech_energy:.2f})"
            f"{vol} peak=+{self.peak_offset:.1f}s{wps}{flag_str}: "
            f"{self.text[:220]}{ocr_str}"
        )


def _compute_importance(
    energy: float,
    volume_db: float | None,
    has_ocr: bool,
    silence_before: bool,
    black: bool,
    freeze: bool,
) -> float:
    """Deterministic importance 0–100. Python calculates; LLM only ranks editorially."""
    score = 0.0
    # Speech energy (0–40)
    score += min(40.0, energy * 40.0)
    # Volume relative boost (0–15) — louder than -25 dB helps
    if volume_db is not None:
        # typical speech ~-20 to -10; map roughly
        vol_norm = max(0.0, min(1.0, (volume_db + 40) / 30.0))
        score += vol_norm * 15.0
    # OCR = strong key-moment signal (0–15)
    if has_ocr:
        score += 15.0
    # Natural cut point (0–10)
    if silence_before:
        score += 10.0
    # Penalties
    if black:
        score -= 25.0
    if freeze:
        score -= 20.0
    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Gen-Z engagement signal detection (keyword-based)
# ─────────────────────────────────────────────────────────────────────────────

_VULGARITY_KEYWORDS = {
    # Sexual/physical attraction
    "sexy", "hot", "fine", "thick", "bad", "body", "ass", "tits", "babe",
    "attractive", "physical", "looks", "appearance", "face", "smile",
    # Dating/relationship crude
    "hookup", "situationship", "red flag", "green flag", "ick", "std",
    "body count", "onlyfans", "slut", "whore", "player", "fuckboy",
    # Crude humor
    "ugly", "mid", "fugly", "butterface", "roasted", "destroyed",
}

_FLIRTING_KEYWORDS = {
    # Romantic interest
    "date", "dating", "match", "couple", "together", "relationship",
    "boyfriend", "girlfriend", "crush", "flirt", "flirting", "rizz",
    "charm", "charming", "smooth", "game", "pull", "slide",
    # Physical affection
    "kiss", "kissing", "hug", "hold", "touch", "cuddle", "snuggle",
    "chemistry", "connection", "vibe", "energy", "attraction",
    # Dating show specific
    "pop", "balloon", "match", "no match", "compatible", "single",
    "looking for", "type", "preference", "standard",
}

_DRAMA_KEYWORDS = {
    # Conflict
    "hate", "argue", "fight", "disagree", "problem", "issue",
    "mad", "angry", "furious", "annoyed", "pissed", "done",
    "leave", "walk away", "get out", "shut up", "stupid", "dumb",
    # Rejection
    "no", "nah", "not interested", "not feeling", "not my type",
    "pop", "popped", "rejected", "denied", "turned down",
    # Confrontation
    "really", "seriously", "are you serious", "no way", "excuse me",
    "how dare", "unbelievable", "ridiculous", "absurd",
}

_BOLD_STATEMENT_KEYWORDS = {
    # Strong opinions / hot takes
    "only", "never", "always", "must", "require", "demand",
    "settling", "lowering", "compromise", "worth", "deserve",
    "standard", "requirement", "condition", "rule",
    # Extreme declarations
    "best", "worst", "greatest", "ugliest", "prettiest",
    "perfect", "ideal", "dream", "nightmare", "dealbreaker",
    # Confidence / arrogance
    "king", "queen", "prize", "catch", "upgrade", "downgrade",
    "out of your league", "too good", "above you", "beneath me",
    # Unhinged takes
    "controversial", "unpopular opinion", "hot take", "hear me out",
    "i would rather", "i'd rather", "i refuse", "i will never",
}

# Challenge/stunt/competition engagement signals
_HIGH_STAKES_KEYWORDS = {
    "money", "cash", "prize", "win", "winner", "winn", "dollars", "thousand", "million",
    "last to leave", "eliminated", "elimination", "final", "finals",
    "challenge", "compete", "competition", "contest", "stakes",
    "whoever", "takes all", "winner takes", "loser", "loses",
    "bet", "wager", "risk", "reward", "incentive",
}
_SPECTACLE_KEYWORDS = {
    "insane", "crazy", "unbelievable", "impossible", "never before",
    "record", "biggest", "largest", "most expensive", "extreme",
    "first time", "nobody has", "never seen", "mind blown", "jaw drop",
    "holy", "oh my god", "no way", "what the", "this is",
}
_ELIMINATION_KEYWORDS = {
    "eliminated", "you're out", "you are out", "voted off", "kicked out",
    "last to leave", "going home", "game over", "disqualified",
    "removed", "banished", "exiled", "sent home", "packing",
    "done", "finished", "over", "out of here", "bye",
}

# Comedy/skit engagement signals
_COMEDY_KEYWORDS = {
    "funny", "hilarious", "laugh", "joke", "prank", "comedy", "skit",
    "relatable", "literally me", "that's so me", "me irl", "when you",
    "plot twist", "twist", "unexpected", "nobody expected", "wait for it",
    "dead", "i'm dead", "dying", "can't breathe", "lmao", "rofl",
}

# Podcast/commentary/tea engagement signals
_TEA_KEYWORDS = {
    "receipts", "exposed", "exposing", "the truth", "let me tell you",
    "allegedly", "rumor", "gossip", "spill", "tea", "shade", "throw shade",
    "called out", "called out", "nobody is talking", "the real story",
    "behind the scenes", "secret", "hidden", "confessed", "admitted",
    "controversy", "scandal", "caught", "busted", "lied", "lying",
}

# Reaction/review engagement signals
_REACTION_KEYWORDS = {
    "terrible", "amazing", "overrated", "underrated", "garbage", "masterpiece",
    "worst", "best", "destroyed", "obliterated", "cooked", "bodied",
    "cringe", "based", "take", "hot take", "unpopular", "controversial",
    "roast", "roasted", "ratio", "L", "W", "massive W", "biggest L",
    "hate", "love", "obsessed", "addicted", "fan", "stanning",
}


def _detect_engagement_signals(text: str) -> dict[str, bool]:
    """Detect engagement signals from transcript text.

    Works across dating, challenge, comedy, podcast, and reaction content.
    Returns dict with 10 boolean signal keys.
    """
    if not text:
        return {k: False for k in (
            "has_vulgarity", "has_flirting", "has_drama", "has_bold_statement",
            "has_high_stakes", "has_spectacle", "has_elimination",
            "has_comedy", "has_tea", "has_reaction",
        )}

    text_lower = text.lower()
    words = set(text_lower.split())

    def _match_any(keywords: set[str]) -> bool:
        for kw in keywords:
            if " " in kw:
                if kw in text_lower:
                    return True
            elif kw in words:
                return True
        return False

    return {
        "has_vulgarity": _match_any(_VULGARITY_KEYWORDS),
        "has_flirting": _match_any(_FLIRTING_KEYWORDS),
        "has_drama": _match_any(_DRAMA_KEYWORDS),
        "has_bold_statement": _match_any(_BOLD_STATEMENT_KEYWORDS),
        "has_high_stakes": _match_any(_HIGH_STAKES_KEYWORDS),
        "has_spectacle": _match_any(_SPECTACLE_KEYWORDS),
        "has_elimination": _match_any(_ELIMINATION_KEYWORDS),
        "has_comedy": _match_any(_COMEDY_KEYWORDS),
        "has_tea": _match_any(_TEA_KEYWORDS),
        "has_reaction": _match_any(_REACTION_KEYWORDS),
    }


def _build_semantic_blocks(
    rich_timeline: RichTimeline | None,
    transcript: list[dict],
    max_block_seconds: float = 28.0,
    min_block_seconds: float = 4.0,
) -> list[SemanticBlock]:
    """
    Collapse fine-grained segments into coherent semantic blocks.
    Target ~120–180 blocks for a long video instead of 500+ tiny segments.
    """
    if rich_timeline and rich_timeline.segments:
        segs = rich_timeline.segments
        items = []
        for seg in segs:
            items.append({
                "id": seg.segment_id,
                "start": seg.start,
                "end": seg.end,
                "text": (seg.speech or "").strip(),
                "energy": float(getattr(seg, "speech_energy", 0.0) or 0.0),
                "volume_db": getattr(seg.metrics, "volume_db", None) if hasattr(seg, "metrics") else None,
                "ocr": list(seg.ocr) if getattr(seg, "ocr", None) else [],
                "silence_before": bool(getattr(seg, "silence_before", False)),
                "black": bool(getattr(seg.metrics, "black_frame", False)) if hasattr(seg, "metrics") else False,
                "freeze": bool(getattr(seg.metrics, "freeze_detected", False)) if hasattr(seg, "metrics") else False,
                "has_question": bool(getattr(seg, "has_question", False)),
                "has_exclamation": bool(getattr(seg, "has_exclamation", False)),
                "has_emphasis": bool(getattr(seg, "has_emphasis", False)),
                "word_density": float(getattr(seg, "word_density", 0.0) or 0.0),
            })
        # Detect engagement signals from text after building items
        for item in items:
            signals = _detect_engagement_signals(item["text"])
            item.update(signals)
    else:
        items = []
        for i, entry in enumerate(transcript):
            items.append({
                "id": i,
                "start": float(entry.get("start", 0.0)),
                "end": float(entry.get("end", 0.0)),
                "text": (entry.get("text") or "").strip(),
                "energy": 0.5,
                "volume_db": None,
                "ocr": [],
                "silence_before": False,
                "black": False,
                "freeze": False,
                "has_question": False,
                "has_exclamation": False,
                "has_emphasis": False,
                "word_density": 0.0,
            })
        # Detect engagement signals from text
        for item in items:
            signals = _detect_engagement_signals(item["text"])
            item.update(signals)

    if not items:
        return []

    blocks: list[SemanticBlock] = []
    cur: list[dict] = [items[0]]

    def _flush(group: list[dict]) -> None:
        if not group:
            return
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(g["text"] for g in group if g["text"]).strip()
        energies = [g["energy"] for g in group]
        avg_energy = sum(energies) / len(energies)
        # peak offset = midpoint of highest-energy segment relative to block start
        peak_seg = max(group, key=lambda g: g["energy"])
        peak_offset = max(0.0, (peak_seg["start"] + peak_seg["end"]) / 2.0 - start)
        vols = [g["volume_db"] for g in group if g["volume_db"] is not None]
        volume_db = sum(vols) / len(vols) if vols else None
        ocr: list[str] = []
        for g in group:
            for t in g["ocr"]:
                if t and t not in ocr:
                    ocr.append(t)
        silence_before = group[0]["silence_before"]
        black = any(g["black"] for g in group)
        freeze = any(g["freeze"] for g in group)
        has_question = any(g.get("has_question", False) for g in group)
        has_exclamation = any(g.get("has_exclamation", False) for g in group)
        has_emphasis = any(g.get("has_emphasis", False) for g in group)
        has_vulgarity = any(g.get("has_vulgarity", False) for g in group)
        has_flirting = any(g.get("has_flirting", False) for g in group)
        has_drama = any(g.get("has_drama", False) for g in group)
        has_bold_statement = any(g.get("has_bold_statement", False) for g in group)
        has_high_stakes = any(g.get("has_high_stakes", False) for g in group)
        has_spectacle = any(g.get("has_spectacle", False) for g in group)
        has_elimination = any(g.get("has_elimination", False) for g in group)
        has_comedy = any(g.get("has_comedy", False) for g in group)
        has_tea = any(g.get("has_tea", False) for g in group)
        has_reaction = any(g.get("has_reaction", False) for g in group)
        word_densities = [g["word_density"] for g in group if g.get("word_density", 0) > 0]
        avg_word_density = sum(word_densities) / len(word_densities) if word_densities else 0.0
        importance = _compute_importance(
            avg_energy, volume_db, bool(ocr), silence_before, black, freeze
        )
        # Boost importance for high-engagement blocks (up to +30, 3 per signal)
        engagement_count = (has_vulgarity + has_flirting + has_drama + has_bold_statement
                           + has_high_stakes + has_spectacle + has_elimination
                           + has_comedy + has_tea + has_reaction)
        engagement_boost = min(30.0, engagement_count * 3.0)
        importance = max(0.0, min(100.0, importance + engagement_boost))
        blocks.append(SemanticBlock(
            block_id=len(blocks),
            start=start,
            end=end,
            text=text,
            speech_energy=avg_energy,
            volume_db=volume_db,
            ocr=ocr[:5],
            silence_before=silence_before,
            black_frame=black,
            freeze=freeze,
            importance=importance,
            peak_offset=peak_offset,
            segment_ids=[g["id"] for g in group],
            has_question=has_question,
            has_exclamation=has_exclamation,
            has_emphasis=has_emphasis,
            word_density=avg_word_density,
            has_vulgarity=has_vulgarity,
            has_flirting=has_flirting,
            has_drama=has_drama,
            has_bold_statement=has_bold_statement,
            has_high_stakes=has_high_stakes,
            has_spectacle=has_spectacle,
            has_elimination=has_elimination,
            has_comedy=has_comedy,
            has_tea=has_tea,
            has_reaction=has_reaction,
        ))

    for item in items[1:]:
        cur_dur = cur[-1]["end"] - cur[0]["start"]
        gap = item["start"] - cur[-1]["end"]
        # Split on silence boundary or when block gets long
        should_split = (
            item["silence_before"]
            or gap > 0.8
            or cur_dur >= max_block_seconds
        )
        if should_split and (cur[-1]["end"] - cur[0]["start"]) >= min_block_seconds:
            _flush(cur)
            cur = [item]
        else:
            cur.append(item)
    _flush(cur)

    logger.info(
        f"Semantic blocks: {len(blocks)} blocks from {len(items)} segments "
        f"(avg {sum(b.duration for b in blocks)/max(len(blocks),1):.1f}s)"
    )
    return blocks


def _format_blocks_for_llm(blocks: list[SemanticBlock], top_n: int | None = None) -> str:
    """Compact block list for LLM. Optionally emphasize top importance blocks."""
    if not blocks:
        return ""
    lines = [b.summary_line() for b in blocks]
    if top_n and len(blocks) > top_n:
        ranked = sorted(blocks, key=lambda b: b.importance, reverse=True)[:top_n]
        top_ids = {b.block_id for b in ranked}
        header = f"TOP-{top_n} importance block_ids: {sorted(top_ids)}\n"
        return header + "\n".join(lines)
    return "\n".join(lines)


def _usable_duration(blocks: list[SemanticBlock], start: float, end: float) -> float:
    """Deterministic usable seconds inside [start, end] (excludes black/freeze/low-importance)."""
    total = 0.0
    for b in blocks:
        if b.end <= start or b.start >= end:
            continue
        if b.black_frame or b.freeze or b.importance < 25:
            continue
        overlap_start = max(b.start, start)
        overlap_end = min(b.end, end)
        total += max(0.0, overlap_end - overlap_start)
    return total


def _score_clip_as_hook(
    clip: dict,
    blocks: list[SemanticBlock],
) -> float:
    """Score a clip's suitability as a hook (0-100). Higher = better hook.

    Factors:
    - Blocks with questions or exclamations in the clip range
    - SILENCE_BEFORE on the first block (clean entry point)
    - High importance blocks
    - Shorter duration (fast hooks)
    - Position not at the very start (curiosity gap > intro)
    """
    clip_start = clip.get("source_start", 0.0)
    clip_end = clip.get("source_end", 0.0)
    clip_duration = clip_end - clip_start

    score = 0.0
    blocks_in_clip = [
        b for b in blocks
        if b.end > clip_start and b.start < clip_end
    ]
    if not blocks_in_clip:
        return 0.0

    # Question/exclamation density (curiosity signals)
    q_count = sum(1 for b in blocks_in_clip if b.has_question)
    e_count = sum(1 for b in blocks_in_clip if b.has_exclamation)
    score += min(30.0, (q_count * 15.0) + (e_count * 10.0))

    # Clean entry (silence before first block in clip)
    first_block = min(blocks_in_clip, key=lambda b: b.start)
    if first_block.silence_before:
        score += 15.0

    # Average importance of blocks in clip
    avg_imp = sum(b.importance for b in blocks_in_clip) / len(blocks_in_clip)
    score += avg_imp * 0.25  # 0-25 points

    # Shorter clips make better hooks (faster payoff)
    if clip_duration <= 5.0:
        score += 15.0
    elif clip_duration <= 10.0:
        score += 8.0
    elif clip_duration <= 15.0:
        score += 3.0

    # Not at the very start (curiosity gap)
    if clip_start > 3.0:
        score += 10.0

    return min(100.0, score)


def _rank_hook_candidates(
    groups: list[dict],
    blocks: list[SemanticBlock],
) -> int:
    """Re-evaluate hook clip choices. Returns number of hook swaps made.

    For each group, if another clip scores higher as a hook than the
    currently marked hook clip, swap the hook flag. Max 1 swap per group.
    """
    swaps = 0
    for group in groups:
        clips = group.get("source_clips", [])
        if len(clips) < 2:
            continue

        hook_idx = None
        for i, c in enumerate(clips):
            if c.get("is_hook_clip", False):
                hook_idx = i
                break
        if hook_idx is None:
            continue

        current_hook = clips[hook_idx]
        current_score = _score_clip_as_hook(current_hook, blocks)

        best_idx = hook_idx
        best_score = current_score
        for i, c in enumerate(clips):
            if i == hook_idx:
                continue
            s = _score_clip_as_hook(c, blocks)
            if s > best_score:
                best_score = s
                best_idx = i

        if best_idx != hook_idx:
            clips[hook_idx]["is_hook_clip"] = False
            clips[best_idx]["is_hook_clip"] = True
            swaps += 1
            logger.info(
                f"Hook swap in group {group.get('group_index', '?')}: "
                f"block {current_hook.get('source_start', 0):.1f}s (score={current_score:.0f}) "
                f"-> block {clips[best_idx].get('source_start', 0):.1f}s (score={best_score:.0f})"
            )

    return swaps


# ─────────────────────────────────────────────────────────────────────────────
# LLM plumbing
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(
    messages: list[dict[str, Any]],
    progress_cb: Callable[[str, float], None] | None = None,
    reporter: Any = None,
    interactions: list[LLMInteraction] | None = None,
    stage_name: str = "reel_plan",
    max_tokens: int = MAX_OUTPUT_TOKENS,
    model: str | None = None,
) -> str:
    if not OPENCODE_API_KEY:
        raise RuntimeError(
            "OPENCODE_API_KEY is not set. Skipping LLM analysis and using local fallback."
        )

    primary_model = model if model else OPENCODE_MODEL
    models_to_try = [primary_model]
    # Use cross-fallback map: each model falls back to the other
    fallback = MODEL_FALLBACK_MAP.get(primary_model, OPENCODE_MODEL_FALLBACK)
    if fallback and fallback != primary_model:
        models_to_try.append(fallback)

    last_error = None
    for model in models_to_try:
        try:
            logger.info(f"Calling LLM ({stage_name}) model={model}")
            raw_content = call_llm_sync(
                messages=messages,
                model=model,
                api_key=OPENCODE_API_KEY,
                base_url=OPENCODE_BASE_URL,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=480.0,
                reporter=reporter,
                interactions=interactions,
                stage_name=stage_name,
            )
            if reporter and interactions is not None:
                reporter.set_stage_data_key(
                    "llm_interactions", [i.model_dump() for i in interactions]
                )
            try:
                from backend.config import WORKING_DIR
                debug_path = WORKING_DIR / f"llm_debug_{stage_name}_{int(time.time())}.txt"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(raw_content)
            except Exception as log_e:
                logger.warning(f"Failed to write LLM debug log: {log_e}")
            stripped = raw_content.strip()
            if not stripped:
                raise RuntimeError(
                    f"LLM returned empty response for stage '{stage_name}'. "
                    f"Raw length: {len(raw_content)}. Check debug log: {debug_path}"
                )
            return stripped
        except Exception as e:
            logger.warning(f"LLM call failed ({stage_name}) model={model}: {e}")
            last_error = e

    raise RuntimeError(f"All OpenCode models failed ({stage_name}). Last error: {last_error}") from last_error


def _try_repair_truncated_json(text: str) -> str:
    if not text:
        return ""
    repaired = re.sub(r',\s*([}\]])', r'\1', text.strip())
    unescaped_quotes = len(re.findall(r'(?<!\\)"', repaired))
    if unescaped_quotes % 2 != 0:
        repaired += '"'
    open_braces = repaired.count("{")
    close_braces = repaired.count("}")
    open_brackets = repaired.count("[")
    close_brackets = repaired.count("]")
    repaired += "}" * max(0, open_braces - close_braces)
    repaired += "]" * max(0, open_brackets - close_brackets)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass
    try:
        for start_pos in [repaired.find("{"), repaired.find("[")]:
            if start_pos < 0:
                continue
            depth = 0
            in_string = False
            escape = False
            for i in range(start_pos, len(repaired)):
                ch = repaired[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch in ('{', '['):
                    depth += 1
                elif ch in ('}', ']'):
                    depth -= 1
                    if depth == 0:
                        candidate = repaired[start_pos:i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            continue
    except (json.JSONDecodeError, IndexError):
        pass
    return ""


def _extract_json_object(text: str) -> str:
    t = text.strip()
    if not t:
        logger.error("Empty text passed to _extract_json_object")
        raise ValueError("No JSON object found in LLM response (empty input).")
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", t, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    if t.startswith("```json"):
        t = t[len("```json"):].strip()
    if t.startswith("```"):
        t = t[len("```"):].strip()
    if t.endswith("```"):
        t = t[:-len("```")].strip()
    # Strip <thinking> blocks from reasoning models (greedy match to handle nested tags)
    t = re.sub(r"<thinking>[\s\S]*?</thinking>\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<think>[\s\S]*?</think>\s*", "", t, flags=re.IGNORECASE)
    # Also strip <think> blocks that may not be properly closed
    t = re.sub(r"<think>[\s\S]*$", "", t, flags=re.IGNORECASE)
    t = t.strip()
    # Try to find the LAST valid JSON object (reasoning models prepend thinking text)
    last_brace = t.rfind("}")
    last_bracket = t.rfind("]")
    end_pos = max(last_brace, last_bracket)
    if end_pos >= 0:
        # Walk backwards from end to find matching open brace/bracket
        # When walking backwards: } increments depth, { decrements it
        close_ch = "}" if last_brace > last_bracket else "]"
        open_ch = "{" if close_ch == "}" else "["
        depth = 0
        in_string = False
        escape = False
        for i in range(end_pos, -1, -1):
            ch = t[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == close_ch:
                depth += 1
            elif ch == open_ch:
                depth -= 1
                if depth == 0:
                    candidate = t[i:end_pos + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        logger.debug("Backward JSON extraction failed, trying forward scan")
    # Fallback: scan forward for first complete JSON object
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        logger.error(f"No JSON object found in LLM response (length={len(text)}). Preview: {text[:300]}")
        raise ValueError("No JSON object found in LLM response.")
    return m.group(0).strip()


def _parse_json_response(raw: str) -> dict:
    raw_json = _extract_json_object(raw)
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        repaired = _try_repair_truncated_json(raw_json)
        if repaired:
            logger.info("Repaired malformed JSON from LLM")
            return json.loads(repaired)
        raise


def _compute_group_count_ceiling(source_duration_seconds: float) -> int:
    """Hard ceiling only. Content decides the actual count.

    Duration-based ceilings:
    - <6 min    → 1 group (single Short)
    - 6-10 min  → 2 groups max
    - 11-25 min → 6 groups max
    - 26-35 min → 8 groups max
    - >35 min   → 10 groups max
    """
    minutes = source_duration_seconds / 60.0
    if minutes < 6:
        return 1
    if minutes <= 10:
        return 2
    if minutes <= 25:
        return 6
    if minutes <= 35:
        return 8
    return 10


def _compute_group_count_floor(source_duration_seconds: float) -> int:
    """Minimum groups the LLM must produce.

    Duration-based floors:
    - <6 min    → 1 group
    - 6-10 min  → 1 group
    - 11-25 min → 3 groups
    - 26-35 min → 4 groups
    - >35 min   → 5 groups
    """
    minutes = source_duration_seconds / 60.0
    if minutes < 6:
        return 1
    if minutes <= 10:
        return 1
    if minutes <= 25:
        return 3
    if minutes <= 35:
        return 4
    return 5


# ─────────────────────────────────────────────────────────────────────────────
# Stage prompts — each LLM has one job
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_structure_planner(
    video_title: str,
    video_description: str,
    blocks_text: str,
    source_duration: float,
    min_groups: int,
    max_groups: int,
    block_usable_hints: str,
) -> str:
    return f"""You are a viral Shorts editor. Your ONLY job: find the moments that make people STOP SCROLLING.
Do NOT select clips. Do NOT write narration.
You are NOT dividing a video into chapters. You are extracting dopamine hits.

Think like a viewer, not a creator. A viewer decides in 1-2 seconds whether to stay.
The FIRST thing visible on screen must be visually striking, emotionally charged, or curiosity-inducing.
If the first 4-6 seconds are someone talking to camera with no context — it's not a hook. Merge it.

SOURCE
Title: {video_title}
Description: {video_description[:4000]}
Duration: {source_duration:.1f}s
Max groups: {max_groups} (CEILING — produce {min_groups}-{max_groups} only if each is genuinely addictive)

SEMANTIC BLOCKS (pre-scored by Python — imp=importance 0-100):
{blocks_text}

PRECOMPUTED USABLE SECONDS BY REGION:
{block_usable_hints}

═══════════════════════════════════════════════════════
STEP 1: HOOK AUDIT — before grouping, scan every block
═══════════════════════════════════════════════════════

For each candidate unit, ask ONLY about the first 4-6 seconds:
- Is there a visually striking moment? (shock, spectacle, unusual visual)
- Is there an emotional spike? (anger, excitement, fear, surprise)
- Is there a curiosity gap? (unexplained action, mid-sentence start, unexpected visual)
- Is there an ENGAGEMENT SIGNAL? ([DRAMA] [BOLD] [FLIRT] [VULGAR] [STAKES] [WOW] [ELIM] [FUNNY] [TEA] [REACT] flags in the blocks)
- Does it start with DEAD AIR, INTRO, or SETUP? → This unit needs a different entry point.

If the best entry point is NOT at the start of the section, shift the unit boundary to start at the hook moment. Content before the hook is MERGED with whatever comes before it.

═══════════════════════════════════════════════════════
STEP 2: ARC VALIDATION — every unit must pass ALL tests
═══════════════════════════════════════════════════════

A unit is KEEPABLE only if it has ALL four:

1. HOOK (first 4-6s): A moment that makes someone stop scrolling
   - NOT: "Hey guys welcome back", introductions, slow fade-ins, setup without context
   - YES: mid-action start, crowd already reacting, person already panicking, visual spectacle, unexpected statement
   - The hook must be long enough to establish curiosity but short enough to not feel slow

2. ESCALATION (middle): Tension that RISES, not flat
   - Each block should feel like stakes are increasing
   - If 3+ blocks feel the same intensity → compress them into fewer clips
   - Dead air, repeated explanations, waiting = must be cut

3. PEAK (final 30%): The moment everyone came for
   - Winner declared, record broken, prank revealed, reaction explosion
   - This is what the thumbnail would show

4. PAYOFF (last 5-10s): Resolution that satisfies
   - NOT: cut mid-sentence, fade to black, host saying "that's it"
   - YES: clear result, winner celebration, aftermath reaction, emotional landing

═══════════════════════════════════════════════════════
STEP 3: SCORING — rate each candidate unit
═══════════════════════════════════════════════════════

Rate each dimension 0-10 with these ANCHORS:

HOOK POTENTIAL (first 4-6 seconds):
  0 = starts with intro/greeting/setup
  5 = starts with talking but interesting topic
  8 = starts mid-action or with striking visual
  10 = starts with crowd gasping / person screaming / visual spectacle

TENSION CURVE:
  0 = flat intensity throughout
  5 = some variation but no clear rise
  8 = steady escalation with clear peak moment
  10 = rollercoaster — multiple peaks building to massive payoff

PAYOFF SATISFACTION:
  0 = no clear result, ends abruptly
  5 = result shown but underwhelming
  8 = clear satisfying result with reaction
  10 = moment that makes you share the video

UNIQUE MOMENT:
  0 = could be any video, generic content
  5 = interesting but seen before
  8 = genuinely unexpected or never-seen-before
  10 = "I've never seen anything like this" reaction

Units averaging ≥6 across these 4 dimensions are KEEPABLE.
Units averaging <6 should be merged with adjacent content.

═══════════════════════════════════════════════════════
STEP 3B: ENGAGEMENT BOOST — Gen-Z viral triggers
═══════════════════════════════════════════════════════

Look at the BLOCK FLAGS in the semantic blocks. These are engagement signals detected by Python:

- [VULGAR] = sexual/crude/provocative language → HIGH engagement (comments, shares)
- [FLIRT] = romantic/attraction/chemistry language → HIGH engagement (will-they-won't-they suspense)
- [DRAMA] = conflict/rejection/confrontation → HIGHEST engagement (debate, takes, opinions)
- [BOLD] = strong opinions/hot takes/unhinged declarations → HIGH engagement (screenshot & share)
- [STAKES] = money, prizes, winner-takes-all, high-risk challenges → HIGH engagement (suspense)
- [WOW] = insane spectacle, never-before-seen, record-breaking → HIGH engagement (wow factor)
- [ELIM] = elimination, voted off, last to leave → HIGH engagement (tension)
- [FUNNY] = comedy, prank, relatable, plot twist → HIGH engagement (shares, rewatch)
- [TEA] = gossip, exposed, receipts, scandal → HIGH engagement (comments, debate)
- [REACT] = strong reaction, roast, hot take, L/W → HIGH engagement (comments, quote tweets)

Content type auto-detection from title + blocks:
- Dating/reality → prioritize [DRAMA] [BOLD] [FLIRT] [VULGAR]
- Challenge/stunt → prioritize [STAKES] [WOW] [ELIM] [DRAMA]
- Comedy/skit → prioritize [FUNNY] [TEA] [REACT]
- Podcast/commentary → prioritize [TEA] [DRAMA] [BOLD] [REACT]
- Reaction/review → prioritize [REACT] [DRAMA] [FUNNY]

PRIORITIZATION for viral Shorts (by content type):

DATING/REALITY:
1. DRAMA → rejection moments, "oh snap" callsouts → COMMENTS
2. BOLD → unhinged takes, controversial opinions → SHARES
3. FLIRT → chemistry, "will they match?" → WATCH TIME
4. VULGAR → crude humor, shock value → REWATCHES

CHALLENGE/STUNT:
1. STAKES → money, prizes, "winner takes all" → SUSPENSE
2. WOW → insane spectacle, record-breaking → WOW FACTOR
3. ELIM → elimination, last to leave → TENSION
4. DRAMA → betrayal, arguments → COMMENTS

COMEDY/SKIT:
1. FUNNY → prank, relatable, plot twist → SHARES
2. TEA → gossip, exposed, scandal → COMMENTS
3. REACT → strong reaction, roast → DEBATE

PODCAST/COMMENTARY:
1. TEA → gossip, receipts, scandal → COMMENTS
2. DRAMA → conflict, confrontation → DEBATE
3. BOLD → hot takes, controversial → SHARES
4. REACT → strong reaction → ENGAGEMENT

When choosing between two otherwise equal units, PRIORITIZE the one with more engagement flags.
A unit with 2+ engagement flags is almost always better than a unit with 0 flags.

═══════════════════════════════════════════════════════
STEP 4: MERGE DECISIONS
═══════════════════════════════════════════════════════

RULES:
1. Classify video_type first.
2. Every unit MUST have a complete arc: hook → escalation → peak → payoff.
3. If a section has setup but no payoff → MERGE with adjacent content that provides the payoff.
4. If two adjacent sections share the same climax (e.g., same challenge) → MERGE into one unit.
5. If a section is just "introduction" or "talking head setup" → MERGE, do not keep.
6. Never split a challenge/contest if the climax immediately follows.
7. Never combine two unrelated climaxes into one group — they are separate stories.
8. The max groups ({max_groups}) is a HARD limit. Fewer strong groups > more weak ones.
9. Produce {min_groups}-{max_groups} groups IF content supports it. ALWAYS produce at least {min_groups}.
10. Every group MUST end at or after the payoff moment. If a challenge ends at 5:00, extend past 5:00.

═══════════════════════════════════════════════════════
THINK INTERNALLY about each unit's hook → arc → payoff before outputting.
═══════════════════════════════════════════════════════

OUTPUT — STRICT JSON ONLY
{{
  "structure_analysis": {{
    "video_type": "challenge|comedy|listicle|podcast|tutorial|vlog|review|documentary|continuous_story|other",
    "identified_units": [
      {{
        "name": "unit name — describe the hook moment",
        "approx_start": 0.0,
        "approx_end": 90.0,
        "usable_seconds": 72,
        "hook_block_id": 0,
        "hook_description": "what the viewer sees in the first 4-6 seconds",
        "peak_block_id": 5,
        "peak_description": "the biggest moment — what the video is about",
        "kept": true
      }}
    ],
    "final_group_count": 1,
    "reasoning": "Why this many groups. For each: what is the hook, what is the escalation, what is the payoff. Why a viewer would watch the entire Short."
  }}
}}"""


def _prompt_clip_planner(
    video_title: str,
    structure: dict,
    blocks_text: str,
    source_duration: float,
    dur_min: int,
    dur_max: int,
    top_blocks_hint: str = "",
) -> str:
    units = structure.get("identified_units", [])
    kept = [u for u in units if u.get("kept", True)]
    units_json = json.dumps(kept, ensure_ascii=False, indent=2)

    top_section = ""
    if top_blocks_hint:
        top_section = f"\nTOP BLOCKS BY IMPORTANCE (use these to anchor your strongest clips):\n{top_blocks_hint}\n"

    return f"""You are a senior YouTube Shorts clip editor.
Your ONLY job: select the strongest source clips for each group defined by the structure plan.
Do NOT write narration text. Do NOT change the number of groups.

You choose clips for maximum ENGAGEMENT, not transcript continuity.
Every second must earn its place. If nothing meaningful happens, cut it.
A 5-second clip with [DRAMA]+[BOLD] beats a 15-second clip with just [DRAMA].

STRUCTURE PLAN (already decided):
{json.dumps(structure.get("structure_analysis", structure), ensure_ascii=False, indent=2)}

KEPT UNITS:
{units_json}

SEMANTIC BLOCKS (imp = importance 0-100, peak = best moment offset):
{blocks_text}
{top_section}
═══════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════

RULE 1 — NO SOURCE OVERLAP
Every source timestamp may belong to ONE reel only.
If reel A uses 20.0-35.0s, NO other reel may touch 20.0-35.0s.
Exception: explicit recap clips (must state "RECAP" in reason).
Violating this creates duplicate Shorts. This is the #1 rule.

RULE 2 — ENGAGEMENT DENSITY
A clip's value = engagement flags ÷ duration.
- 5s clip with [DRAMA]+[BOLD] = 2 flags ÷ 5s = 0.40 density ← BEST
- 15s clip with [DRAMA] = 1 flag ÷ 15s = 0.067 density ← weak
Prefer SHORT clips that pack multiple engagement signals.
If a clip has no engagement flags AND is longer than 10s, it's probably filler.

RULE 3 — EMOTIONAL JOURNEY
Every reel must take the viewer on a ride:
- HOOK (first 4-6s): Shock, curiosity, or spectacle that stops the scroll
- RISE (middle): Tension building, stakes increasing, one signal per clip
- PEAK (final 30%): The biggest moment — the reason the video exists
- LAND (last 5-10s): Satisfying resolution that makes them want more

RULE 4 — ANCHOR CLIP
Every reel has ONE anchor clip — the single most engaging moment.
This is the clip that would be the thumbnail, the screenshot, the "you HAVE to see this" moment.
Build the reel AROUND the anchor. Hook leads to it, escalation builds to it, payoff follows it.

═══════════════════════════════════════════════════════
CLIP SELECTION — by content type
═══════════════════════════════════════════════════════

Detect content type from title + blocks, then pick clips in this priority:

DATING/REALITY:
1. [DRAMA] — rejection, confrontation, "oh snap" → COMMENTS
2. [BOLD] — unhinged takes, controversial opinions → SHARES
3. [FLIRT] — chemistry, "will they match?" → WATCH TIME
4. [VULGAR] — crude humor, shock value → REWATCHES
5. Payoff — match reveal, final moment (MANDATORY last clip)

CHALLENGE/STUNT:
1. [STAKES] — money, prizes, "winner takes all" → SUSPENSE
2. [WOW] — insane spectacle, record-breaking → WOW FACTOR
3. [ELIM] — elimination, last to leave → TENSION
4. [DRAMA] — betrayal, arguments → COMMENTS
5. Payoff — winner declared (MANDATORY last clip)

COMEDY/SKIT:
1. [FUNNY] — prank, relatable, plot twist → SHARES
2. [TEA] — gossip, exposed, scandal → COMMENTS
3. [REACT] — strong reaction, roast → DEBATE
4. Payoff — punchline, twist ending (MANDATORY last clip)

PODCAST/COMMENTARY:
1. [TEA] — gossip, receipts, scandal → COMMENTS
2. [DRAMA] — conflict, confrontation → DEBATE
3. [BOLD] — hot takes → SHARES
4. Payoff — final verdict (MANDATORY last clip)

REACTION/REVIEW:
1. [REACT] — strong reaction, roast, L/W → COMMENTS
2. [DRAMA] — conflict → DEBATE
3. [FUNNY] — comedy → SHARES
4. Payoff — final rating (MANDATORY last clip)

GENERAL (fallback):
1. Curiosity gap — "What happens next?"
2. Emotional peak — reactions, shock
3. Visual action — motion, spectacle

═══════════════════════════════════════════════════════
HOOK CLIPS — the make-or-break moment
═══════════════════════════════════════════════════════

The hook clip is 4-6 seconds. It MUST:
- Start mid-action or with a striking visual/statement
- Create instant curiosity ("wait, what?")
- Have at least ONE engagement flag

Best hook sources by content type:
- [DRAMA]: someone getting rejected, called out, confronted
- [BOLD]: wild declaration, "I only...", unhinged take
- [STAKES]: "winner takes $500K", "last to leave wins"
- [WOW]: "this has never been done", insane spectacle
- [ELIM]: "you're out", tense elimination
- [FUNNY]: prank reveal, relatable moment, plot twist
- [TEA]: gossip, exposed, receipts
- [REACT]: strong reaction, "this is terrible/amazing"
- [FLIRT]: unexpected chemistry, smooth line

NEVER hook with: introductions, greetings, "Hey guys", slow builds, context setup.

═══════════════════════════════════════════════════════
COMMENT BAIT — clips that drive engagement
═══════════════════════════════════════════════════════

Select at least 1 clip per reel specifically designed to generate COMMENTS:
- Controversial take → people argue in comments
- "No way he said that" → people quote and reply
- Rejection moment → people take sides
- Bold opinion → people agree/disagree

Select at least 1 clip per reel designed to get SHARES:
- Screenshot-worthy moment
- "You have to see this" visual
- Relatable moment ("this is so me")
- Unexpected twist

═══════════════════════════════════════════════════════
AVOID — common mistakes that kill engagement
═══════════════════════════════════════════════════════

NEVER select a clip that:
- Has 0 engagement flags AND is longer than 10s (filler)
- Requires context from a previous reel to understand
- Starts with "Hey guys" or "Welcome back" (no hook)
- Has flat energy throughout (no peak moment)
- Ends mid-sentence or mid-action (no payoff)
- Is just talking heads with no visual/emotional change
- Repeats information from another clip in the same reel
- Shows explanation BEFORE the visual action (always SHOW THEN EXPLAIN)

═══════════════════════════════════════════════════════
FILLER REMOVAL — cut ruthlessly
═══════════════════════════════════════════════════════

Aggressively remove:
- Repeated explanations saying the same thing
- Countdowns that go nowhere
- Slow walking, waiting, dead air
- Reset moments ("Okay let's try again")
- Any 5+ second stretch where nothing visually or emotionally happens
Trim clips to the minimum that delivers the moment. Shorter is almost always better.

═══════════════════════════════════════════════════════
CLIP MIX — pacing rules
═══════════════════════════════════════════════════════

- SHORT 3-5s: punchy moments, reactions, single statements
- MEDIUM 6-15s: conversations, interactions, mini-scenes
- LONG 16-30s: full story arcs, complex moments, build-ups

Rules:
- No back-to-back LONG clips (viewer fatigue)
- No 3+ SHORT clips in a row (feels choppy)
- Strongest moment in the final 30% (anchor clip position)
- Final clip must be MEDIUM or LONG (never end on SHORT)
- Never < 3.0s per clip (too fast to register)
- Mix types for rhythm: SHORT-MEDIUM-SHORT-MEDIUM-LONG etc.

═══════════════════════════════════════════════════════
DURATION RULES (HARD)
═══════════════════════════════════════════════════════

- Each group estimated_duration = sum of (source_end - source_start) of its clips.
- Must be between {dur_min} and {dur_max} seconds.
- Narration is overlaid later — it does NOT add duration.
- Prefer high-imp blocks. Avoid BLACK / FREEZE unless intentional.
- Prefer SILENCE_BEFORE as cut points.
- If duration is too short, expand existing clips (wider time ranges) rather than adding filler clips.

═══════════════════════════════════════════════════════
OUTPUT — STRICT JSON ONLY
═══════════════════════════════════════════════════════

Every reel_groups entry MUST include:
- engagement_signals: list of which flags this reel contains (e.g. ["DRAMA", "BOLD"])
- anchor_clip_index: which clip is the anchor (0-indexed)
- emotional_journey: one-line description of the hook→rise→peak→land arc

{{
  "reel_groups": [
    {{
      "group_index": 0,
      "group_reasoning": "Why this arc works: hook moment, key engagement signals, anchor clip, emotional payoff.",
      "estimated_duration_seconds": 120.0,
      "engagement_signals": ["DRAMA", "BOLD"],
      "anchor_clip_index": 2,
      "emotional_journey": "Hook: rejection moment → Rise: escalating takes → Peak: bold declaration → Land: aftermath reaction",
      "reel_summary": {{
        "title": "≤60 chars — what makes this Short compelling",
        "short_description": "≤150 chars",
        "source_understanding": "what this covers",
        "narrative_angle": "unique emotional framing",
        "key_moment": "strongest moment — what makes someone stop scrolling"
      }},
      "source_clips": [
        {{"source_start": 12.0, "source_end": 17.0, "reason": "HOOK: [DRAMA] rejection moment — instant curiosity", "is_hook_clip": true}},
        {{"source_start": 20.0, "source_end": 28.0, "reason": "MEDIUM: [BOLD] escalating take — builds tension", "is_hook_clip": false}},
        {{"source_start": 35.0, "source_end": 45.0, "reason": "LONG: [DRAMA]+[BOLD] anchor clip — biggest moment", "is_hook_clip": false}},
        {{"source_start": 50.0, "source_end": 57.0, "reason": "MEDIUM: [REACT] aftermath reaction — satisfying land", "is_hook_clip": false}}
      ]
    }}
  ]
}}"""


def _prompt_narration_writer(
    video_title: str,
    groups_without_narration: list[dict],
) -> str:
    groups_json = json.dumps(groups_without_narration, ensure_ascii=False, indent=2)
    return f"""You are a narration writer for vertical reels.
Your ONLY job: write narration_events for each group.
Do NOT change clips, timestamps, or group count.

Video title: {video_title}

GROUPS (clips already locked):
{groups_json}

═══════════════════════════════════════════════════════
EVENT STRUCTURE — max 3 events per group
═══════════════════════════════════════════════════════

1. HOOK (required): reel_start=0.0, reel_end=4.0-6.0
   - 5-8 words, specific curiosity gap
   - Must match the hook clip's first 4-6 seconds
   - Should make viewer think "wait, what?"

   GOOD HOOKS (specific, curiosity-driven):
   - "First kiss panic over sweat patches?"
   - "Guy pops because name isn't moanable?"
   - "Birthday girl gets to pop 5 guys?"
   - "Final girl rejected for shallow reasons?"
   - "Guy vows to pop any woman under 10?"

   BANNED HOOKS (generic, clickbait):
   - "Watch what happens", "You won't believe", "This is insane"
   - "Wait for it", "No way this is real", "I can't believe it"
   - "This is crazy", "Look at this", "Oh my god", "Bro what"

2. COMMENTARY 1 (middle): 4-6 words, place at the ESCALATION moment
   - Don't use a fixed percentage — read the clips and place it where tension rises
   - Typically during the 2nd or 3rd clip (the build-up)
   - Must REACT to what's on screen, not describe it

3. COMMENTARY 2 (end): 4-6 words, place at the PEAK or PAYOFF moment
   - Don't use a fixed percentage — read the clips and place it where the biggest moment is
   - Typically during the anchor clip or just before the final clip
   - Must REACT to the biggest moment in the reel

═══════════════════════════════════════════════════════
PERSONAS — match to content type
═══════════════════════════════════════════════════════

Detect the content type from the group's clips and title, then use matching personas.
USE EDGY, VULGAR LANGUAGE — this is Gen-Z content, not a family show.
Bad words are ENCOURAGED when they land the joke harder.

DATING/REALITY:
- roast: "Guys are dumbass for popping themselves, lol."
- brutally_honest: "She's outta his league and he knows it, bullshit excuse."
- sarcastic: "Oh yeah, great personality bro, that'll get you far."
- deadpan: "Dude really said that with a straight face. Embarrassing."

CHALLENGE/STUNT:
- narrator: "Last to leave wins $500K, this is gonna get ugly."
- hype_man: "This shit is INSANE, nobody's done this before!"
- underdog: "Everyone counted him out, now watch this bastard win."
- dramatic: "One wrong move and it's game over, no pressure."

COMEDY/SKIT:
- deadpan: "This is painfully relatable and I hate it."
- absurdist: "Why is this actually my life? Fuck."
- self_deprecating: "I feel personally attacked and I don't like it."
- hype_man: "The plot twist nobody asked for but everybody needed."

PODCAST/COMMENTARY:
- investigator: "Let's talk about the bullshit nobody's mentioning."
- gossiper: "So apparently the tea is spilling and it's messy."
- analyst: "Here's why this actually matters, no bullshit."
- roast: "The audacity is actually impressive, what a dumbass."

REACTION/REVIEW:
- roasted: "This is trash and I'm obsessed with it."
- hype_man: "This is actually fire, fight me if you disagree."
- deadpan: "10/10, no notes, fuck the haters."
- brutally_honest: "Let's be real, this is mid as hell."

═══════════════════════════════════════════════════════
WRITING RULES
═══════════════════════════════════════════════════════

- Punchy style: short, sharp, impactful. Think one-liner, not sentence.
- Commentary must REACT to what is on screen, not DESCRIBE it.
- USE SWEAR WORDS WHEN THEY FIT: fuck, shit, ass, dumbass, bullshit, sucks, hell, damn
- Make it sound like your funniest friend roasting the video alongside you
- Match the energy: roast for rejection, hype for wins, deadpan for cringe
- The more savage and unexpected the commentary, the better
- Comments should make people screenshot and share "look what the AI said"

═══════════════════════════════════════════════════════
TIMING RULES
═══════════════════════════════════════════════════════

- ≥0.8s gap between events
- Never cover key_moment (the anchor clip's peak moment)
- Last 3-5s free of narration (let the payoff breathe)
- Hook narration must fit within the hook clip's duration (4-6s)
- Commentary must land during a clip that has the matching engagement signal

═══════════════════════════════════════════════════════
CHARACTER RULES
═══════════════════════════════════════════════════════

- Allowed: letters numbers . , ! ? ' - — " : ;
- BANNED: / \ | @ # $ % ^ & * ( ) [ ] {{ }} < > ~ `

═══════════════════════════════════════════════════════
OUTPUT — STRICT JSON ONLY
═══════════════════════════════════════════════════════

{{
  "reel_groups": [
    {{
      "group_index": 0,
      "narration_events": [
        {{"event_type": "hook", "reel_start": 0.0, "reel_end": 5.0, "text": "First kiss panic over sweat patches?", "persona": null, "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 22.0, "reel_end": 23.5, "text": "Dude's sweating more than the kiss.", "persona": "deadpan", "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 55.0, "reel_end": 56.5, "text": "No wonder he's still a virgin, dumbass.", "persona": "roast", "voice_id": null}}
      ]
    }}
  ]
}}"""


def _detect_content_type(title: str, blocks: list[SemanticBlock]) -> str:
    """Detect content type from video title and semantic blocks."""
    text = (title or "").lower()
    block_text = " ".join(
        b.text.lower() for b in blocks if b.text
    )
    combined = text + " " + block_text

    # Dating / reality
    dating_kw = ["dating", "kiss", "love", "rejected", "proposal", "married", "boyfriend", "girlfriend",
                  "first date", "blind date", "couple", "single", "crush", "flirting", "pickup"]
    if any(k in combined for k in dating_kw):
        return "dating"

    # Challenge / stunt
    challenge_kw = ["challenge", "last to", "stunt", "dare", "competition", "winner", "prize",
                    "eliminated", "round", "fastest", "strongest", "endurance"]
    if any(k in combined for k in challenge_kw):
        return "challenge"

    # Comedy / skit
    comedy_kw = ["comedy", "funny", "skit", "prank", "joke", "hilarious", "laugh", "meme"]
    if any(k in combined for k in comedy_kw):
        return "comedy"

    # Podcast / commentary
    podcast_kw = ["podcast", "interview", "talk show", "discussion", "debate", "opinion", "explained"]
    if any(k in combined for k in podcast_kw):
        return "podcast"

    # Reaction / review
    reaction_kw = ["reaction", "react", "review", "rating", "responding"]
    if any(k in combined for k in reaction_kw):
        return "reaction"

    # Default: detect from block characteristics
    has_dialogue = any(b.get("dialogue", "").strip() for b in blocks)
    has_narration = any(b.get("narration_text", "").strip() for b in blocks)
    if has_dialogue and not has_narration:
        return "podcast"
    return "dating"  # safe default for most YouTube Shorts


def _prompt_critic(plan: dict, dur_min: int, dur_max: int, content_type: str) -> str:
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)[:60000]
    return f"""You are a senior YouTube Shorts editor with 10M+ subscriber experience.
Do NOT invent new groups. Do NOT rewrite everything.
Critique the draft reel plan and apply concrete fixes.

Draft:
{plan_json}

Content type: {content_type}

═══════════════════════════════════════════════════════
PRIORITY ORDER — fix most impactful issues first
═══════════════════════════════════════════════════════

1. HOOK CLIPS (highest impact)
   - Must create curiosity, not introduce
   - Hook clips must be 4-6 seconds
   - Show moment FIRST, then explain (show-first rule)
   - If hook clip is just talking head → swap for visual-first clip

2. EMOTIONAL ARC (retention)
   - Strongest moment must be in FINAL 30% (peak)
   - Don't front-load the best content
   - Build tension: hook → rise → peak → land
   - If peak is in first half → swap clips to move peak later

3. ENGAGEMENT SIGNALS (virality)
   - Each group should have 2-3 engagement signals (drama, vulgarity, flirting, bold statements, etc.)
   - Commentary should amplify these moments, not ignore them
   - If a group has 0-1 engagement signals → it's boring, swap clips or add commentary that creates engagement

4. PACING (retention)
   - Vary clip lengths: mix short (1-3s) + medium (4-6s) + long (7-10s)
   - No back-to-back same-length clips
   - Dead air / filler clips → remove or speed up
   - Groups that don't standalone without another reel → merge or add context

5. COMMENTARY QUALITY (shareability)
   - Commentary must be 4-6 words max (punchy)
   - Must REACT to what's on screen, not DESCRIBE it
   - Use edgy/savage language when appropriate: dumbass, bullshit, sucks, fuck, etc.
   - Commentary must land at actual engagement moments (not fixed percentages)
   - No banned words: like, basically, honestly, so, well, you know
   - No banned characters: / \ | @ # $ % ^ & * ( ) [ ] {{ }} < > ~ `

6. DURATION
   - Every group must be {dur_min}-{dur_max}s
   - If too short → expand clips, don't pad with filler
   - If too long → remove weakest clip, not the peak

7. REDUNDANCY
   - Zero timestamp overlap across reels (every source timestamp in ONE reel only)
   - If two reels cover similar content → merge or remove the weaker one

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════

Return the FULL revised plan as STRICT JSON with the same schema.
For each group, include ALL required fields:
- source_clips (with source_start, source_end, clip_index)
- hook_block_id, hook_description
- peak_block_id, peak_description
- engagement_signals (updated if commentary changed)
- anchor_clip_index
- emotional_journey
- narration_events (hook + commentary, with reel_start, reel_end, text, persona)
- explanations

Only change what needs fixing. Keep strong parts intact.
Explain EACH fix in the explanations array — what was wrong and what you changed."""


# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage orchestration
# ─────────────────────────────────────────────────────────────────────────────

def select_reel_plan(
    transcript: list[dict],
    video_title: str,
    video_description: str,
    progress_cb: Callable[[str, float], None] | None = None,
    reporter: Any = None,
    interactions: list[LLMInteraction] | None = None,
    rich_timeline: RichTimeline | None = None,
    model: str | None = None,
    checkpoint: Any = None,
    resume_from: str | None = None,
) -> ReelPlan:
    if progress_cb:
        progress_cb("Building semantic blocks...", 5)

    if not transcript:
        raise RuntimeError("Transcript is empty; cannot build reel plan.")

    source_duration = float(transcript[-1]["end"]) if transcript else 0.0
    max_groups = _compute_group_count_ceiling(source_duration)
    min_groups = _compute_group_count_floor(source_duration)

    # Duration targets — must land between 90-150s for output compliance
    if source_duration < 90:
        reel_dur_min = max(45, int(source_duration * 0.8))
        reel_dur_max = min(int(source_duration * 0.95), 90)
    elif source_duration < 150:
        reel_dur_min = 90
        reel_dur_max = min(int(source_duration * 0.95), 150)
    else:
        reel_dur_min = 90
        reel_dur_max = 150
    if reel_dur_max - reel_dur_min < 30:
        reel_dur_max = reel_dur_min + 30
    reel_dur_max = min(reel_dur_max, int(MAX_OUTPUT_DURATION))

    # ── Python: semantic blocks + importance ──
    blocks = _build_semantic_blocks(rich_timeline, transcript)
    blocks_text = _format_blocks_for_llm(blocks)

    # Precomputed usable hints for structure planner (no LLM math)
    regions = [
        ("early", 0.0, source_duration * 0.25),
        ("mid", source_duration * 0.25, source_duration * 0.75),
        ("late", source_duration * 0.75, source_duration),
    ]
    usable_hints = "\n".join(
        f"  {name}: {start:.0f}-{end:.0f}s → usable≈{_usable_duration(blocks, start, end):.0f}s"
        for name, start, end in regions
    )
    # Also coarse whole-video usable
    whole_usable = _usable_duration(blocks, 0.0, source_duration)
    usable_hints = f"  whole_video usable≈{whole_usable:.0f}s\n" + usable_hints

    description = (video_description or "")[:10000]
    logger.info(
        f"MULTI-STAGE PLAN source={source_duration:.1f}s groups={min_groups}-{max_groups} "
        f"blocks={len(blocks)} duration_target={reel_dur_min}-{reel_dur_max}s"
    )

    # ── LLM #1 Structure Planner ──
    sa = None
    if checkpoint and resume_from and resume_from != "structure_planner":
        # Loading from checkpoint - skip structure planner
        ckpt_data = checkpoint.load_stage("analyze_structure")
        if ckpt_data and "structure_analysis" in ckpt_data:
            sa = ckpt_data["structure_analysis"]
            logger.info("Resuming from checkpoint: structure planner already complete")

    if sa is None:
        if progress_cb:
            progress_cb("Structure planning...", 20)
        raw1 = _call_llm(
            [
                {"role": "system", "content": "Respond with ONLY valid JSON."},
                {"role": "user", "content": _prompt_structure_planner(
                    video_title, description, blocks_text, source_duration, min_groups, max_groups, usable_hints
                )},
            ],
            progress_cb, reporter, interactions, stage_name="structure_planner",
            max_tokens=65536, model=model,
        )
        structure = _parse_json_response(raw1)
        sa = structure.get("structure_analysis", structure)
        logger.info(
            f"STRUCTURE: type={sa.get('video_type')} groups={sa.get('final_group_count')} "
            f"reason={sa.get('reasoning')}"
        )
        # Save intermediate checkpoint
        if checkpoint:
            checkpoint.save_stage("analyze_structure", {"structure_analysis": sa})

    # ── LLM #2 Clip Planner ──
    groups = None
    if checkpoint and resume_from and resume_from not in ("structure_planner", "clip_planner"):
        # Loading from checkpoint - skip clip planner
        ckpt_data = checkpoint.load_stage("analyze_clips")
        if ckpt_data and "groups" in ckpt_data:
            groups = ckpt_data["groups"]
            logger.info("Resuming from checkpoint: clip planner already complete")

    if groups is None:
        if progress_cb:
            progress_cb("Selecting clips...", 45)

        # Build top-blocks hint per kept unit for the clip planner
        top_blocks_lines: list[str] = []
        kept_units = [u for u in sa.get("identified_units", []) if u.get("kept", True)]
        for unit in kept_units:
            u_start = unit.get("approx_start", 0.0)
            u_end = unit.get("approx_end", source_duration)
            unit_blocks = [
                b for b in blocks
                if b.end > u_start and b.start < u_end and not b.black_frame and not b.freeze
            ]
            unit_blocks.sort(key=lambda b: b.importance, reverse=True)
            top5 = unit_blocks[:5]
            if top5:
                block_strs = ", ".join(
                    f"Block {b.block_id} (imp={b.importance:.0f}, peak=+{b.peak_offset:.1f}s)"
                    for b in top5
                )
                top_blocks_lines.append(
                    f"  {unit.get('name', 'unit')} [{u_start:.0f}-{u_end:.0f}s]: {block_strs}"
                )
        top_blocks_hint = "\n".join(top_blocks_lines)

        raw2 = _call_llm(
            [
                {"role": "system", "content": "Respond with ONLY valid JSON."},
                {"role": "user", "content": _prompt_clip_planner(
                    video_title, structure, blocks_text, source_duration, reel_dur_min, reel_dur_max,
                    top_blocks_hint=top_blocks_hint,
                )},
            ],
            progress_cb, reporter, interactions, stage_name="clip_planner",
            max_tokens=65536, model=model,
        )
        clips_plan = _parse_json_response(raw2)
        groups = clips_plan.get("reel_groups", [])
        if not groups:
            raise RuntimeError("Clip planner returned no reel_groups")

        # Strip any narration the clip planner may have hallucinated
        for g in groups:
            g.pop("narration_events", None)

        # ── Python: rank hook candidates ──
        hook_swaps = _rank_hook_candidates(groups, blocks)
        if hook_swaps > 0:
            logger.info(f"Hook ranking: swapped {hook_swaps} hook clip(s)")

        # Save intermediate checkpoint
        if checkpoint:
            checkpoint.save_stage("analyze_clips", {"groups": groups})

    # ── LLM #3 Narration Writer ──
    if progress_cb:
        progress_cb("Writing narration...", 65)
    raw3 = _call_llm(
        [
            {"role": "system", "content": "Respond with ONLY valid JSON. Do NOT include any reasoning, thinking, or commentary. Output ONLY the JSON object."},
            {"role": "user", "content": _prompt_narration_writer(video_title, groups)},
        ],
        progress_cb, reporter, interactions, stage_name="narration_writer",
        max_tokens=65536, model=model,
    )
    narr_plan = _parse_json_response(raw3)
    narr_by_idx = {
        g.get("group_index", i): g.get("narration_events", [])
        for i, g in enumerate(narr_plan.get("reel_groups", []))
    }
    for i, g in enumerate(groups):
        idx = g.get("group_index", i)
        g["narration_events"] = narr_by_idx.get(idx, [])

    # Assemble draft
    draft = {
        "structure_analysis": sa,
        "ranked_segments": clips_plan.get("ranked_segments", []),
        "reel_groups": groups,
        "explanations": clips_plan.get("explanations", []),
    }

    # ── LLM #4 Critic (one revision) ──
    from backend.config import FAST_MODE
    content_type = _detect_content_type(video_title, blocks)
    if not FAST_MODE:
        if progress_cb:
            progress_cb("Critic pass...", 80)
        try:
            raw4 = _call_llm(
                [
                    {"role": "system", "content": "Respond with ONLY valid JSON."},
                    {"role": "user", "content": _prompt_critic(draft, reel_dur_min, reel_dur_max, content_type)},
                ],
                progress_cb, reporter, interactions, stage_name="critic",
                max_tokens=65536, model=model,
            )
            revised = _parse_json_response(raw4)
            revised_groups = revised.get("reel_groups", [])
            if isinstance(revised_groups, list) and revised_groups:
                # Validate Critic didn't change group count
                original_count = len(groups)
                revised_count = len(revised_groups)
                if revised_count != original_count:
                    logger.warning(
                        f"Critic changed group count {original_count} -> {revised_count}, "
                        f"keeping original draft"
                    )
                else:
                    draft = revised
                    if "structure_analysis" not in draft:
                        draft["structure_analysis"] = sa
                    logger.info("Critic applied revisions")
            else:
                logger.info("Critic returned empty groups — keeping draft")
        except Exception as e:
            logger.warning(f"Critic pass failed, keeping draft: {e}")
    else:
        if progress_cb:
            progress_cb("Skipping critic (FAST_MODE)...", 80)
        logger.info("FAST_MODE: skipping critic pass")

    # Validate shape
    groups = draft.get("reel_groups", [])
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("Final plan missing reel_groups")
    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            raise RuntimeError(f"Group {i} must be an object")
        if not group.get("source_clips"):
            raise RuntimeError(f"Group {i} missing source_clips")
        if "narration_events" not in group or not isinstance(group["narration_events"], list):
            group["narration_events"] = []

    logger.info(f"LLM returned {len(groups)} groups (ceiling {max_groups})")

    # ── Python validator owns final numbers ──
    if progress_cb:
        progress_cb("Validating plan...", 90)
    reel_plan = finalize_edit(draft, source_duration, min_groups=min_groups)

    if progress_cb:
        progress_cb(f"Built reel plan with {len(reel_plan.reel_groups)} group(s)", 100)

    total_clips = sum(len(g.source_clips) for g in reel_plan.reel_groups)
    total_narr = sum(len(g.narration_events) for g in reel_plan.reel_groups)
    avg_dur = sum(g.estimated_duration_seconds for g in reel_plan.reel_groups) / max(
        len(reel_plan.reel_groups), 1
    )
    logger.info(
        f"REEL PLAN STATS: {len(reel_plan.reel_groups)} groups, {total_clips} clips, "
        f"{total_narr} narrations, avg duration {avg_dur:.1f}s"
    )

    if reporter and interactions is not None:
        reporter.set_stage_data_key(
            "llm_interactions", [i.model_dump() for i in interactions]
        )

    return reel_plan


# ─────────────────────────────────────────────────────────────────────────────
# Legacy flat clip selection (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def _format_full_transcript(transcript: list) -> str:
    if not transcript:
        return ""
    lines = []
    for i, entry in enumerate(transcript):
        start = entry.get("start", 0.0)
        end = entry.get("end", 0.0)
        text = entry.get("text", "").strip()
        if text:
            lines.append(f"Seg {i} [{start:.1f}-{end:.1f}s]: {text}")
    return "\n".join(lines)


def _normalize_clip_range(transcript: list, start_seg: int, end_seg: int) -> tuple[int, int]:
    max_idx = len(transcript) - 1
    start_seg = min(max(start_seg, 0), max_idx)
    end_seg = min(max(end_seg, start_seg), max_idx)
    while transcript[end_seg]["end"] - transcript[start_seg]["start"] < CLIP_DURATION_SOFT_MIN:
        if end_seg < max_idx:
            end_seg += 1
        elif start_seg > 0:
            start_seg -= 1
        else:
            break
    while transcript[end_seg]["end"] - transcript[start_seg]["start"] > CLIP_DURATION_SOFT_MAX and end_seg > start_seg:
        remove_after = transcript[end_seg - 1]["end"] - transcript[start_seg]["start"]
        remove_before = transcript[end_seg]["end"] - transcript[start_seg + 1]["start"]
        if abs(remove_after - CLIP_DURATION_SOFT_MAX) <= abs(remove_before - CLIP_DURATION_SOFT_MAX):
            end_seg -= 1
        else:
            start_seg += 1
    return start_seg, end_seg


def select_clips(
    transcript: list[dict],
    video_title: str,
    video_description: str,
    progress_cb: Callable[[str, float], None] | None = None,
) -> list[dict]:
    """Legacy flat clip selection (kept for compatibility)."""
    if progress_cb:
        progress_cb("Preparing transcript for analysis...", 10)
    if not transcript:
        raise RuntimeError("Transcript is empty; cannot select clips.")

    transcript_text = _format_full_transcript(transcript)
    description = (video_description or "")[:500]
    if progress_cb:
        progress_cb("Sending transcript to LLM for clip selection...", 30)

    prompt = f"""You are a JSON-only output machine. Output ONLY valid JSON.

Video title: {video_title}
Description: {description[:500]}

Transcript:
{transcript_text}

Select {CLIP_COUNT_MIN}-{CLIP_COUNT_MAX} moments, each ~{CLIP_DURATION_SOFT_MIN:.0f}-{CLIP_DURATION_SOFT_MAX:.0f}s.
Hard cap {MAX_OUTPUT_DURATION}s total including hooks/insights.

[
  {{"start_segment": 0, "end_segment": 2, "topic": "...", "why_it_hooks": "...", "payoff": "...", "reason": "..."}}
]"""

    raw_content = _call_llm(
        [
            {"role": "system", "content": "Respond with ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        progress_cb,
        stage_name="legacy_clips",
    )

    def _extract_json_array(text: str) -> str:
        t = text.strip()
        if t.startswith("```json"):
            t = t[len("```json"):].strip()
        if t.startswith("```"):
            t = t[len("```"):].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
        m = re.search(r"\[[\s\S]*\]", t)
        if not m:
            raise ValueError("No JSON array found")
        return m.group(0).strip()

    if progress_cb:
        progress_cb("Parsing clip selections...", 80)
    try:
        clip_data = json.loads(_extract_json_array(raw_content))
    except Exception:
        raw_retry = _call_llm(
            [{"role": "user", "content": prompt + "\n\nCRITICAL: ONLY the JSON array."}],
            progress_cb,
            stage_name="legacy_clips_retry",
        )
        clip_data = json.loads(_extract_json_array(raw_retry))

    if not isinstance(clip_data, list):
        raise RuntimeError("Expected JSON array")

    clips = []
    for item in clip_data:
        if not isinstance(item, dict):
            continue
        start_seg, end_seg = item.get("start_segment"), item.get("end_segment")
        if not isinstance(start_seg, int) or not isinstance(end_seg, int):
            continue
        start_seg, end_seg = _normalize_clip_range(transcript, start_seg, end_seg)
        actual_start = transcript[start_seg]["start"]
        actual_end = transcript[end_seg]["end"]
        duration = actual_end - actual_start
        planned = duration + HOOK_SECONDS + INSIGHT_SECONDS_MAX
        if sum((c["end"] - c["start"]) + HOOK_SECONDS + INSIGHT_SECONDS_MAX for c in clips) + planned > MAX_OUTPUT_DURATION:
            continue
        clips.append({
            "start": actual_start,
            "end": actual_end,
            "start_segment": start_seg,
            "end_segment": end_seg,
            "topic": item.get("topic", ""),
            "why_it_hooks": item.get("why_it_hooks", ""),
            "payoff": item.get("payoff", ""),
            "reason": item.get("reason", ""),
        })

    if progress_cb:
        progress_cb(f"Selected {len(clips)} clips", 100)
    return clips