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
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_DURATION,
    MAX_OUTPUT_TOKENS,
    MIN_OUTPUT_DURATION,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    NVIDIA_MODEL_FALLBACK,
    REASONING_EFFORT,
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

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

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
        has_question = any(g["has_question"] for g in group)
        has_exclamation = any(g["has_exclamation"] for g in group)
        has_emphasis = any(g["has_emphasis"] for g in group)
        word_densities = [g["word_density"] for g in group if g["word_density"] > 0]
        avg_word_density = sum(word_densities) / len(word_densities) if word_densities else 0.0
        importance = _compute_importance(
            avg_energy, volume_db, bool(ocr), silence_before, black, freeze
        )
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
    reasoning_effort: str = REASONING_EFFORT,
) -> str:
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Skipping LLM analysis and using local fallback."
        )

    models_to_try = [NVIDIA_MODEL]
    if NVIDIA_MODEL_FALLBACK and NVIDIA_MODEL_FALLBACK != NVIDIA_MODEL:
        models_to_try.append(NVIDIA_MODEL_FALLBACK)

    last_error = None
    for model in models_to_try:
        try:
            logger.info(f"Calling LLM ({stage_name}) model={model}")
            raw_content = call_llm_sync(
                messages=messages,
                model=model,
                api_key=NVIDIA_API_KEY,
                base_url=NVIDIA_BASE_URL,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=480.0,
                reporter=reporter,
                interactions=interactions,
                stage_name=stage_name,
                reasoning_effort=reasoning_effort,
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
            return raw_content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed ({stage_name}) model={model}: {e}")
            last_error = e

    raise RuntimeError(f"All NVIDIA models failed ({stage_name}). Last error: {last_error}") from last_error


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
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", t, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    if t.startswith("```json"):
        t = t[len("```json"):].strip()
    if t.startswith("```"):
        t = t[len("```"):].strip()
    if t.endswith("```"):
        t = t[:-len("```")].strip()
    # Strip <thinking> blocks from reasoning models
    t = re.sub(r"<thinking>[\s\S]*?</thinking>\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<think>[\s\S]*?</think>\s*", "", t, flags=re.IGNORECASE)
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
    return f"""You are an editorial structure planner for YouTube Shorts.
Your ONLY job: identify the strongest standalone story units worth turning into individual Shorts.
Do NOT select clips. Do NOT write narration.

You think like a professional editor, not a chapter detector.
Chapters mirror the source. Editorial units tell complete stories.

SOURCE
Title: {video_title}
Description: {video_description[:4000]}
Duration: {source_duration:.1f}s
Max groups allowed: {max_groups} (this is a CEILING, not a goal — produce {min_groups}-{max_groups} strong standalone stories)

SEMANTIC BLOCKS (pre-scored by Python — imp=importance 0-100):
{blocks_text}

PRECOMPUTED USABLE SECONDS BY REGION:
{block_usable_hints}

EDITORIAL SCORING — before grouping, score each candidate section on:
- Emotional intensity (0-10): crowd gasps, player reactions, stakes
- Novelty (0-10): unexpected outcome, never-seen-before moment
- Stakes (0-10): what is at risk for the participants
- Visual action (0-10): motion, spectacle, physical comedy
- Surprise (0-10): twist, reversal, unexpected winner
- Payoff clarity (0-10): is there a clear result/reveal

Only sections scoring ≥5 average deserve standalone reels.

GROUPING RULES
1. Classify video_type.
2. For each candidate unit, reason about its internal arc:
   - Setup: what is established or promised
   - Conflict: what tension or challenge builds
   - Peak: the climax, reveal, or payoff
   - Resolution: the actual outcome, winner reveal, result, or aftermath (REQUIRED — every unit must end by showing what happened)
3. A unit is ONLY a group if it has a complete arc (setup → peak → resolution).
4. If a section has setup but no payoff, merge it with adjacent content that provides the payoff.
5. Never split a challenge/contest if the climax immediately follows — keep them as one unit.
6. Never combine two unrelated climaxes into one group — they are separate stories.
7. Would someone watch this as a standalone Short? If not, merge it.
8. Avoid groups that are just "introduction" or "setup" with no payoff.
9. The max groups ({max_groups}) is a hard limit. Fewer strong groups always beats more weak ones.
10. Produce {min_groups}-{max_groups} groups if content supports it. ALWAYS produce at least {min_groups} group(s).
11. Every group MUST end at or after the moment the viewer sees the outcome. If a challenge ends at 5:00, the group must extend past 5:00 to capture the result.

THINK INTERNALLY about each unit's arc before outputting. Only output the final boundaries.

OUTPUT — STRICT JSON ONLY
{{
  "structure_analysis": {{
    "video_type": "challenge|listicle|podcast|tutorial|vlog|review|documentary|continuous_story|other",
    "identified_units": [
      {{
        "name": "unit name",
        "approx_start": 0.0,
        "approx_end": 90.0,
        "usable_seconds": 72,
        "kept": true
      }}
    ],
    "final_group_count": 1,
    "reasoning": "Why this many groups. What makes each unit a complete standalone story."
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

You choose clips for maximum impact, not transcript continuity.
Every second must earn its place. If nothing meaningful happens, cut it.

STRUCTURE PLAN (already decided):
{json.dumps(structure.get("structure_analysis", structure), ensure_ascii=False, indent=2)}

KEPT UNITS:
{units_json}

SEMANTIC BLOCKS (imp = importance 0-100, peak = best moment offset):
{blocks_text}
{top_section}
CRITICAL RULE — NO SOURCE OVERLAP
Every source timestamp may belong to ONE reel only.
If reel A uses 20.0-35.0s, NO other reel may touch 20.0-35.0s.
Exception: explicit recap clips (must state "RECAP" in reason).
Violating this creates duplicate Shorts. This is the #1 rule.

CLIP SELECTION PRIORITIES (ranked)
1. Payoff — result, winner reveal, outcome, final moment (MANDATORY for last clip)
2. Curiosity gap — "What happens next?" / "No way..." / unexpected visuals
3. Emotional peak — crowd gasps, player reactions, shock, celebration
4. Visual action — motion, spectacle, physical moments
5. Stakes — what is at risk, what could go wrong
6. Surprise — twist, reversal, unexpected result
NEVER choose a clip just because it comes next in the transcript.
Dialogue continuity does NOT matter. Impact does.

CLIP STORY ARC — every reel must contain within its clips:
- Hook: the curiosity trigger (first clip)
- Escalation: tension building (middle clips)
- Peak: the climax or reveal (later clip)
- Resolution: the actual outcome/result/winner reveal (REQUIRED final clip — show what happens at the end)
The Resolution clip is NOT optional. Every reel MUST end by showing what happened.
Find the exact moment where the winner is declared, the record is broken, the prize is awarded, or the challenge concludes. This is the payoff the viewer stays for.

HOOK CLIPS — chosen by curiosity, NOT earliest timestamp.
Search the entire unit for:
- "What happens if..." moments
- Unexpected visuals or instant action
- Crowd reactions before you see why
- Oh-my-God expressions
- Visual spectacle that makes you stop scrolling
Avoid: introductions, greetings, "Hey guys", slow builds, setup without payoff.
is_hook_clip=true only on the hook clip.

FILLER REMOVAL — aggressively cut:
- Repeated explanations saying the same thing
- Countdowns that go nowhere
- Slow walking, waiting, dead air
- Reset moments ("Okay let's try again")
- Repeated commentary
- Any 5+ second stretch where nothing visually or emotionally happens
Trim clips to the minimum that delivers the moment. Shorter is almost always better.

CLIP MIX
- SHORT ≤6s, MEDIUM 7-15s, LONG 16-20s — mix them for pacing.
- No back-to-back LONG clips. No 3+ SHORT in a row.
- Strongest moment must be in the final 30% of the reel.
- Final clip must be MEDIUM — payoff moments need enough time for the reveal.
- Never < 3.0s per clip.
- Prefer SHOW THEN EXPLAIN: visual action first, commentary after.

IF TWO ADJACENT CLIPS COMMUNICATE THE SAME INFORMATION
Keep only the stronger one. Redundancy kills pacing.

EVERY REEL STANDS ALONE
No clip should require a previous reel for context.
A viewer seeing this Short cold must understand what is happening.

DURATION RULES (HARD)
- Each group estimated_duration = sum of (source_end - source_start) of its clips.
- Must be between {dur_min} and {dur_max} seconds.
- Narration is overlaid later — it does NOT add duration.
- Prefer high-imp blocks. Avoid BLACK / FREEZE unless intentional.
- Prefer SILENCE_BEFORE as cut points.
- If duration is too short, expand existing clips (wider time ranges) rather than adding filler clips.

OUTPUT — STRICT JSON ONLY
{{
  "reel_groups": [
    {{
      "group_index": 0,
      "group_reasoning": "Short:N Medium:N Long:N, total duration, why this arc works as a standalone story.",
      "estimated_duration_seconds": 120.0,
      "reel_summary": {{
        "title": "≤60 chars — what makes this Short compelling",
        "short_description": "≤150 chars",
        "source_understanding": "what this covers",
        "narrative_angle": "unique emotional framing",
        "key_moment": "strongest moment — what makes someone stop scrolling"
      }},
      "source_clips": [
        {{"source_start": 12.0, "source_end": 15.0, "reason": "HOOK: curiosity gap — ...", "is_hook_clip": true}},
        {{"source_start": 20.0, "source_end": 32.0, "reason": "MEDIUM: escalation — ...", "is_hook_clip": false}}
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

STYLE: Write like a witty friend reacting live — not a narrator reading a script.
Be punchy, specific, and unexpected. Every line should earn its spot.
Reference what's actually on screen. Use the creator's energy, not generic hype.
Humor beats inspiration. Specificity beats vagueness.

HOOK (reel_start=0.0, reel_end=3.0-5.0, 6-10 words)
- Drop the viewer into the most intriguing moment
- BANNED: "Watch what happens", "You won't believe", "This is insane", "Wait for it"
- GOOD: specific curiosity that demands resolution
- Example hooks: "This is the worst idea I've ever had." / "He has no idea I'm here." / "3 AM. Empty stadium. One ball."

COMMENTARIES (up to 2 per group, 8-14 words each)
- Commentary 1: place at ~35-45% of estimated_duration
- Commentary 2: place at ~70-80% of estimated_duration
- Must have persona. Every commentary must feel like a real person talking.

PERSONAS (pick the one that fits the moment):
- roast: playful jab at what's happening. Not mean, just sharp. "Of course he brought a backup plan. And a backup for the backup."
- brutally_honest: say what everyone's thinking but won't admit. "Let's be real, this could go horribly wrong."
- friendly: warm, excited, rooting for them. "Okay, this is actually adorable."
- sarcastic: dry wit, understated reactions. "Oh sure, because THAT always works out."
- hype: genuine excitement, but specific — not just "LET'S GOOO". "The crowd just lost it. All of them. At once."
- deadpan: flat delivery, maximum impact. "He missed. In front of everyone. On camera."

BANNED filler for commentaries: "This is crazy", "No way", "Insane", "Literally dying", "I can't even", "Best thing ever"
BANNED vague commentary: "That was amazing", "What a moment", "So cool"

RULES
- ≥0.8s gap between events. Never cover key_moment. Last 3-5s free of narration.
- Allowed chars: letters numbers . , ! ? ' - — " : ;

OUTPUT — STRICT JSON ONLY
{{
  "reel_groups": [
    {{
      "group_index": 0,
      "narration_events": [
        {{"event_type": "hook", "reel_start": 0.0, "reel_end": 5.0, "text": "...", "persona": null, "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 35.0, "reel_end": 38.0, "text": "...", "persona": "roast", "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 70.0, "reel_end": 73.0, "text": "...", "persona": "hype", "voice_id": null}}
      ]
    }}
  ]
}}"""


def _prompt_critic(plan: dict, dur_min: int, dur_max: int) -> str:
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)[:60000]
    return f"""You are a senior YouTube Shorts editor.
Do NOT invent new groups. Do NOT rewrite everything.
Critique the draft reel plan and apply concrete fixes.

Draft:
{plan_json}

Check for:
- boring openings (weak hook clips that don't create curiosity)
- weak pacing (too many similar lengths, no rhythm)
- redundant clips across groups (EVERY source timestamp may belong to ONE reel only — zero overlap)
- confusing story / missing payoff in any group
- dead air / low-value clips (repeated explanations, waiting, filler)
- missed emotional peaks (strongest moment buried in middle instead of final 30%)
- duration outside {dur_min}-{dur_max}s
- groups that don't stand alone (require context from another reel)
- hook clips that are introductions instead of curiosity triggers
- clips showing then explaining instead of explaining then showing (prefer show-first)

Return the FULL revised plan as STRICT JSON with the same schema:
structure_analysis (if present), ranked_segments (optional), reel_groups (with source_clips + narration_events), explanations.
Only change what needs fixing. Keep strong parts intact."""


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
) -> ReelPlan:
    if progress_cb:
        progress_cb("Building semantic blocks...", 5)

    if not transcript:
        raise RuntimeError("Transcript is empty; cannot build reel plan.")

    source_duration = float(transcript[-1]["end"]) if transcript else 0.0
    max_groups = _compute_group_count_ceiling(source_duration)
    min_groups = _compute_group_count_floor(source_duration)

    # Duration targets — must land between MIN-MAX_OUTPUT_DURATION for output compliance
    if source_duration < MIN_OUTPUT_DURATION:
        reel_dur_min = max(45, int(source_duration * 0.8))
        reel_dur_max = min(int(source_duration * 0.95), MIN_OUTPUT_DURATION)
    elif source_duration < MAX_OUTPUT_DURATION:
        reel_dur_min = MIN_OUTPUT_DURATION
        reel_dur_max = min(int(source_duration * 0.95), MAX_OUTPUT_DURATION)
    else:
        reel_dur_min = MIN_OUTPUT_DURATION
        reel_dur_max = MAX_OUTPUT_DURATION
    if reel_dur_max - reel_dur_min < 20:
        reel_dur_max = reel_dur_min + 20
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
        max_tokens=32768,
    )
    structure = _parse_json_response(raw1)
    sa = structure.get("structure_analysis", structure)
    logger.info(
        f"STRUCTURE: type={sa.get('video_type')} groups={sa.get('final_group_count')} "
        f"reason={sa.get('reasoning')}"
    )

    # ── LLM #2 Clip Planner ──
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
        max_tokens=65536,
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

    # ── LLM #3 Narration Writer ──
    if progress_cb:
        progress_cb("Writing narration...", 65)
    raw3 = _call_llm(
        [
            {"role": "system", "content": "Respond with ONLY valid JSON. Do NOT include any reasoning, thinking, or commentary. Output ONLY the JSON object."},
            {"role": "user", "content": _prompt_narration_writer(video_title, groups)},
        ],
        progress_cb, reporter, interactions, stage_name="narration_writer",
        max_tokens=32768,
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
    if not FAST_MODE:
        if progress_cb:
            progress_cb("Critic pass...", 80)
        try:
            raw4 = _call_llm(
                [
                    {"role": "system", "content": "Respond with ONLY valid JSON."},
                    {"role": "user", "content": _prompt_critic(draft, reel_dur_min, reel_dur_max)},
                ],
                progress_cb, reporter, interactions, stage_name="critic",
                max_tokens=32768,
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