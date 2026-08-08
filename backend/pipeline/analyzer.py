"""Video analysis module — story-driven, audience-retention-first reel planning.

Architecture (PLAN_MODE=executor, default):

  Rich Timeline / Transcript
       ↓
  Content Type Detection  (Python — hardcoded, committed early)
       ↓
  Audience Worth Gate     (Python — "Is this worth posting?")
       ↓
  Semantic Block Builder  (Python — importance + retention scoring)
       ↓
  LLM #1  Story Planner   (genre-specific prompt, regions/beats only — no timestamps)
       ↓
  LLM #1.5 Beat Relevance Ranker (block IDs per beat intent — no timestamps)
       ↓
  Python Executor         (windows, slots, budgets, clip order — all deterministic)
       ↓
  Story Flow Assembly     (groups ordered chronologically through source)
       ↓
  LLM #2  Narration Writer (text only — Python owns all numbers)
       ↓
  Python Validator        (finalize_edit)
       ↓
  Final ReelPlan

Legacy PLAN_MODE=llm path (select_reel_plan A/B branch) uses the old
Structure Planner → Clip Planner → Narration Writer → Critic chain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.pipeline.plan_schema import StoryPlan

from backend.config import (
    CLIP_DURATION_SOFT_MAX,
    CLIP_DURATION_SOFT_MIN,
    ENTITY_MIN_SEGMENT_SECONDS,
    ENTITY_MAX_SEGMENTS_MULTIPLIER,
    MAX_OUTPUT_DURATION,
    MAX_OUTPUT_TOKENS,
    MIN_OUTPUT_DURATION,
    MIN_USABLE_BLOCK_FRACTION,
    MULTIMODAL_ENABLED,
    OPENCODE_API_KEY,
    OPENCODE_BASE_URL,
    OPENCODE_MODEL,
    REASONING_EFFORT,
)
from backend.models import (
    ContentIdentity,
    EntitySegment,
    HookMode,
    LLMInteraction,
    MultimodalSignals,
    ReelPlan,
    RichTimeline,
    SourceMetadata,
)
from backend.pipeline.plan_validator import finalize_edit

__all__ = ["select_reel_plan"]

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer phase registry — single source of truth for the structured,
# frontend-facing breakdown of the ANALYZING stage. `id` values here MUST
# match the `stage_name=` passed to `_call_llm` (for kind="llm" phases) or the
# id used in the manual `reporter.update_analyzer_phase(...)` calls scattered
# through this module (for kind="python" phases). See progress.ProgressReporter.
# ─────────────────────────────────────────────────────────────────────────────

ANALYZER_PHASE_REGISTRY: dict[str, dict[str, str]] = {
    "semantic_blocks": {"label": "Semantic Blocks", "kind": "python"},
    "identifier": {"label": "Content Identifier", "kind": "llm"},
    "multimodal_ocr": {"label": "Multimodal & OCR", "kind": "python"},
    "content_classification": {"label": "Content Classification", "kind": "python"},
    # Story-planner branch — exactly one of these three runs per job.
    "entity_group_planner": {"label": "Story Planner · Entity", "kind": "llm"},
    "genre_story_planner": {"label": "Story Planner · Genre", "kind": "llm"},
    "structure_planner": {"label": "Story Planner · Structure", "kind": "llm"},
    # Clip selection — executor mode ranks beats to moments; legacy mode plans clips directly.
    "moment_beat_matcher": {"label": "Clip Selection", "kind": "llm"},
    "clip_planner": {"label": "Clip Selection", "kind": "llm"},
    "plan_execution": {"label": "Plan Execution", "kind": "python"},
    "completeness_critic": {"label": "Completeness QA", "kind": "llm"},
    # Narration writer — executor mode vs legacy mode use different prompts.
    "script_narration_writer": {"label": "Narration Writer", "kind": "llm"},
    "narration_writer": {"label": "Narration Writer", "kind": "llm"},
    "narration_placement": {"label": "Narration Placement", "kind": "python"},
    "critic": {"label": "Critic Pass", "kind": "llm"},
    "validation_finalize": {"label": "Validation & Finalize", "kind": "python"},
}


def _phase(*ids: str) -> list[dict[str, str]]:
    """Look up one or more phase ids in the registry as ordered plan entries."""
    return [{"id": i, **ANALYZER_PHASE_REGISTRY[i]} for i in ids]


def _build_analyzer_phase_plan(
    *,
    entity_grouped: bool,
    plan_mode: str,
    fast_mode: bool = False,
) -> list[dict[str, str]]:
    """Return the exact ordered list of phases that WILL run for this job.

    Pure and deterministic — no I/O, no reporter — so it's unit-testable on
    its own. This is the branch decision made explicit: which planner runs
    (entity vs genre vs legacy structure), and whether the critic pass is
    skipped (FAST_MODE, legacy path only).

    `multimodal_ocr` is always listed — whether it actually runs (vs is
    skipped because MULTIMODAL_ENABLED is off or OCR_MODE="skip") is a
    runtime status, not a plan-shape decision, and is reported via
    reporter.update_analyzer_phase("multimodal_ocr", "skipped") at the point
    that decision is made.
    """
    plan = _phase("semantic_blocks", "identifier", "multimodal_ocr", "content_classification")

    if plan_mode == "executor":
        planner_id = "entity_group_planner" if entity_grouped else "genre_story_planner"
        plan += _phase(planner_id, "moment_beat_matcher", "plan_execution")
        plan += _phase("script_narration_writer", "narration_placement")
        plan += _phase("validation_finalize")
        # Completeness critic is a read-only quality gate that runs AFTER
        # finalize_edit in executor mode — not before, unlike the legacy
        # path's critic (which revises the draft before validation). Skipped
        # entirely (not just fast) under FAST_MODE — see _completeness_critic.
        if not fast_mode:
            plan += _phase("completeness_critic")
    else:
        plan += _phase("structure_planner", "clip_planner", "narration_writer")
        if not fast_mode:
            plan += _phase("critic")
        plan += _phase("validation_finalize")

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Debug artifact dump — consolidated pipeline stage outputs per job
# ─────────────────────────────────────────────────────────────────────────────

def _write_debug_artifact(
    job_id: str | None,
    *,
    content_identity: Any = None,
    entity_segments: list | None = None,
    planner_branch: str = "unknown",
    story_plan: Any = None,
    relevance: dict | None = None,
    pre_qa_groups: list | None = None,
    post_qa_groups: list | None = None,
    final_groups: list | None = None,
    completeness_verdicts: dict | None = None,
    source_duration: float = 0.0,
    max_groups: int = 0,
) -> None:
    """Write a single consolidated debug artifact for a pipeline run.

    Output: backend/storage/working/debug_artifact_{job_id}.json
    Contains every pipeline stage output in one file for 5-minute diagnosis.
    """
    if not job_id:
        return
    try:
        from backend.config import WORKING_DIR
        artifact_dir = os.path.join(WORKING_DIR, job_id)
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, f"debug_artifact_{job_id}.json")

        def _safe_dump(obj: Any) -> Any:
            if obj is None:
                return None
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if isinstance(obj, list):
                return [_safe_dump(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _safe_dump(v) for k, v in obj.items()}
            return obj

        artifact = {
            "job_id": job_id,
            "source_duration": source_duration,
            "max_groups": max_groups,
            "identifier": _safe_dump(content_identity),
            "entity_segments": [
                {
                    "id": s.entity_segment_id,
                    "name": s.entity_name,
                    "start": s.start,
                    "end": s.end,
                    "evidence": s.evidence,
                    "speaker_ids": s.speaker_ids,
                    "block_count": len(s.block_ids),
                }
                for s in (entity_segments or [])
            ],
            "planner_branch": planner_branch,
            "story_plan": _safe_dump(story_plan),
            "relevance": _safe_dump(relevance),
            "pre_qa_groups": _safe_dump(pre_qa_groups),
            "post_qa_groups": _safe_dump(post_qa_groups),
            "final_groups": _safe_dump(final_groups),
            "completeness_verdicts": completeness_verdicts or {},
        }
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Debug artifact written: {artifact_path}")
    except Exception as e:
        logger.warning(f"Failed to write debug artifact: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Clip duration classification (single source of truth — LLM prompt + validators)
# ─────────────────────────────────────────────────────────────────────────────

CLIP_SHORT_MAX_SECONDS = 6.0
CLIP_MEDIUM_MAX_SECONDS = 15.0
CLIP_LONG_MAX_SECONDS = 25.0


def classify_clip_duration(duration: float) -> str:
    """Classify a clip as SHORT, MEDIUM, or LONG based on duration."""
    if duration <= CLIP_SHORT_MAX_SECONDS:
        return "SHORT"
    if duration <= CLIP_MEDIUM_MAX_SECONDS:
        return "MEDIUM"
    return "LONG"


# ─────────────────────────────────────────────────────────────────────────────
# Content type detection — Python-hardcoded, committed early in the pipeline.
# This classification drives genre-specific story planning, clip scoring, and
# narration style.  Detection is data-driven via _GENRE_REGISTRY: to add a new
# genre, add one entry there and a matching narration style block — no new
# prompt function needed.
# ─────────────────────────────────────────────────────────────────────────────

# Supported genre identifiers (single source of truth)
GENRE_GAME_CHALLENGE   = "game_challenge"    # MrBeast-style: prizes, elimination, stunts
GENRE_DATING_REALITY   = "dating_reality"    # Dating shows, matchmaking, chemistry/rejection
GENRE_ROAST_REACTION   = "roast_reaction"    # Roast, reaction, rating, commentary
GENRE_PODCAST          = "podcast_interview" # Long-form conversation, insight-driven
GENRE_VLOG_PERSONAL    = "vlog_personal"     # Personal vlogs, meet-ups, celebrity encounters, travel
GENRE_EXPERIMENT       = "experiment"        # Science/social experiments, "what happens if..."
GENRE_SPORTS_FITNESS   = "sports_fitness"    # Sports highlights, gym, training, competition
GENRE_COMEDY_SKETCH    = "comedy_sketch"     # Scripted comedy, skits, pranks
GENRE_TUTORIAL         = "tutorial"          # How-to, educational, guide
GENRE_MUSIC_PERF       = "music_performance" # Music videos, live performances, covers
GENRE_NEWS_DOC         = "news_documentary"  # News, documentary, investigation
GENRE_GENERAL          = "general"           # Fallback — no strong signal

# Identifier corroboration is deliberately small: Python's keyword and block
# signals still determine the genre, but an agreeing Identifier can lift a
# borderline, otherwise-supported genre over the commitment threshold.
IDENTIFIER_GENRE_BONUS = 5.0

# ── Genre Registry ──────────────────────────────────────────────────────────
# Each entry defines: keywords, which block flag boosts the score, and the
# multipliers applied to keyword hits (kw_weight) and flag density (flag_weight).
# Detection scores each genre as: kw_score * kw_weight + flag_density * flag_weight.
# Highest score wins; if below min_threshold the fallback is GENRE_GENERAL.

_GENRE_REGISTRY: dict[str, dict] = {
    GENRE_GAME_CHALLENGE: {
        "keywords": {
            "challenge", "last to", "last one", "eliminated", "wins", "winner", "prize",
            "cash", "$", "dollars", "survive", "whoever", "battle", "tournament", "competition",
            "mr beast", "mrbeast", "beast", "stunt", "dares", "dared", "bet",
        },
        "flag_field": "has_stakes",
        "kw_weight": 2.0,
        "flag_weight": 30.0,
    },
    GENRE_DATING_REALITY: {
        "keywords": {
            "date", "dating", "match", "matched", "crush", "flirt", "chemistry",
            "boyfriend", "girlfriend", "single", "relationship", "love", "attraction",
            "reject", "rejected", "ask out", "kiss", "couple", "together",
        },
        "flag_field": "has_dating",
        "kw_weight": 2.0,
        "flag_weight": 30.0,
    },
    GENRE_ROAST_REACTION: {
        "keywords": {
            "roast", "roasted", "react", "reaction", "rating", "rate", "review",
            "exposed", "called out", "brutal", "savage", "clowned", "embarrassed",
            "reading comments", "responding",
        },
        "flag_field": "has_roast",
        "kw_weight": 2.0,
        "flag_weight": 25.0,
    },
    GENRE_PODCAST: {
        "keywords": {
            "interview", "podcast", "episode", "guest", "host", "conversation",
            "advice", "lessons", "mindset", "success", "failure", "philosophy",
        },
        "flag_field": None,
        "kw_weight": 1.5,
        "flag_weight": 0.0,
    },
    GENRE_VLOG_PERSONAL: {
        "keywords": {
            "vlog", "day in my life", "grwm", "get ready", "finally", "met",
            "first time", "surprise", "birthday", "road trip", "travel", "moving",
            "story time", "storytime", "my experience", "i tried", "celebrity",
            "meeting", "picture with", "photo with", "autograph",
        },
        "flag_field": None,
        "kw_weight": 2.0,
        "flag_weight": 0.0,
    },
    GENRE_EXPERIMENT: {
        "keywords": {
            "experiment", "what happens", "what if", "test", "testing", "hypothesis",
            "science", "social experiment", "prank", "hidden camera", "will it",
            "can you", "is it possible", "we tried", "results",
        },
        "flag_field": "has_stakes",
        "kw_weight": 2.0,
        "flag_weight": 15.0,
    },
    GENRE_SPORTS_FITNESS: {
        # flag_field=None because has_stakes is shared with game_challenge;
        # sports is disambiguated by keywords alone (nfl, ufc, boxing, etc.)
        "keywords": {
            "goal", "training", "workout", "gym", "fitness", "athlete",
            "championship", "final", "score", "tackle", "dunk", "knockout",
            "world record", "personal best", "marathon", "football", "basketball",
            "soccer", "cricket", "boxing", "mma", "ufc", "nba", "nfl",
        },
        "flag_field": None,
        "kw_weight": 2.5,
        "flag_weight": 0.0,
    },
    GENRE_COMEDY_SKETCH: {
        "keywords": {
            "comedy", "sketch", "skit", "funny", "hilarious", "parody", "spoof",
            "impression", "stand up", "standup", "comedian", "joke", "punchline",
            "acting", "roleplay",
        },
        "flag_field": None,
        "kw_weight": 2.0,
        "flag_weight": 0.0,
    },
    GENRE_TUTORIAL: {
        "keywords": {
            "tutorial", "how to", "how-to", "guide", "step by step", "learn",
            "tips", "tricks", "diy", "recipe", "cooking", "setup", "install",
            "beginner", "advanced", "masterclass", "explained",
        },
        "flag_field": None,
        "kw_weight": 2.0,
        "flag_weight": 0.0,
    },
    GENRE_MUSIC_PERF: {
        "keywords": {
            "music", "song", "cover", "live performance", "concert", "singing",
            "rapper", "beat", "album", "track", "freestyle", "remix", "acoustic",
            "band", "guitar", "piano", "drums", "vocal",
        },
        "flag_field": None,
        "kw_weight": 2.0,
        "flag_weight": 0.0,
    },
    GENRE_NEWS_DOC: {
        "keywords": {
            "news", "breaking", "documentary", "investigation", "report",
            "scandal", "court", "trial", "arrest", "politics", "election",
            "war", "conflict", "crisis", "update", "announcement",
        },
        "flag_field": "has_stakes",
        "kw_weight": 1.8,
        "flag_weight": 10.0,
    },
}


def detect_content_type(
    video_title: str,
    video_description: str,
    blocks: list["SemanticBlock"],
    identifier_genre: str | None = None,
) -> str:
    """Detect genre from title, description, and block-level engagement signals.

    Returns one of the GENRE_* constants. This is a Python-hardcoded decision
    that commits early — every downstream stage (prompts, scoring, flow) adapts
    to this result.

    Data-driven: iterates _GENRE_REGISTRY, scores each genre by keyword hits
    in title/description (weighted) plus block flag density (weighted).
    Highest score wins; if below 3.0 falls back to GENRE_GENERAL.
    """
    combined_text = " ".join([
        (video_title or "").lower(),
        (video_description or "")[:500000].lower(),
    ])
    words = set(combined_text.split())
    n_blocks = max(1, len(blocks))

    def _kw_score(kw_set: set[str]) -> float:
        """Count keyword hits, weighting multi-word phrases higher."""
        score = 0.0
        for kw in kw_set:
            if " " in kw:
                if kw in combined_text:
                    score += 2.0
            else:
                if kw in words:
                    score += 1.0
        return score

    def _flag_density(flag_field: str | None) -> float:
        """Fraction of blocks with the given flag (0.0-1.0)."""
        if not flag_field or not blocks:
            return 0.0
        return sum(1 for b in blocks if getattr(b, flag_field, False)) / n_blocks

    scores: dict[str, float] = {GENRE_GENERAL: 0.1}
    for genre, cfg in _GENRE_REGISTRY.items():
        kw = _kw_score(cfg["keywords"]) * cfg["kw_weight"]
        fd = _flag_density(cfg.get("flag_field")) * cfg["flag_weight"]
        scores[genre] = kw + fd

    # ── Keyword-corroboration: penalise flag-only wins ──────────────────────
    # A genre that scores purely via flag density (kw=0) is ambiguous — the
    # same flag (e.g. has_stakes) fires in both a MrBeast challenge AND a
    # World Cup final. Halve the score of any genre whose raw keyword score is
    # zero so that a genre with real keyword evidence can beat it.
    raw_kw: dict[str, float] = {}
    for genre, cfg in _GENRE_REGISTRY.items():
        raw_kw[genre] = _kw_score(cfg["keywords"]) * cfg["kw_weight"]

    adjusted_scores: dict[str, float] = {}
    for genre, total in scores.items():
        kw = raw_kw.get(genre, 0.0)
        if kw == 0.0 and total > 0.1:
            adjusted_scores[genre] = total * 0.3  # flag-only, no keyword evidence → heavy penalty
        else:
            adjusted_scores[genre] = total

    # Apply Identifier genre bonus BEFORE selection so it can lift a
    # borderline genre into first place (matching the docstring intent).
    if identifier_genre and identifier_genre != GENRE_GENERAL:
        adjusted_scores[identifier_genre] = adjusted_scores.get(identifier_genre, 0.0) + IDENTIFIER_GENRE_BONUS

    best_genre = max(adjusted_scores, key=lambda g: adjusted_scores[g])
    best_score = adjusted_scores[best_genre]

    # Only commit to a specific genre if the signal is meaningful (>= 3.0)
    if best_score < 3.0:
        best_genre = GENRE_GENERAL

    logger.info(
        f"CONTENT TYPE: detected '{best_genre}' "
        f"(scores: {', '.join(f'{g}={s:.1f}' for g, s in sorted(adjusted_scores.items(), key=lambda x: -x[1])[:5])})"
    )
    return best_genre




# ─────────────────────────────────────────────────────────────────────────────
# Audience Worth Gate — asks "Is this content worth an audience watching?"
# The result is a soft warning (logged + attached to plan metadata). It never
# blocks processing — the creator decides whether to post.
# ─────────────────────────────────────────────────────────────────────────────

def assess_audience_worth(
    blocks: list["SemanticBlock"],
    content_type: str,
    source_duration: float,
) -> tuple[bool, str, dict]:
    """Score content for audience-worthiness. Returns (is_worth, reason, breakdown).

    Criteria:
    1. Engagement signal density — blocks with ROAST/STAKES/DATING/VULGAR flags
    2. Peak energy concentration — are high-importance moments spread or clustered
       in a non-useful way (all at start, or all missing)
    3. Story completeness — is there a discernible setup AND payoff pattern
    4. Content volume — enough material to make a compelling reel
    5. Genre-specific thresholds — challenge content needs STAKES moments;
       dating content needs DATING moments; etc.

    Score is 0-100. Below 35 = likely not worth posting.
    """
    if not blocks:
        return False, "No content blocks found in video", {}

    score = 0.0
    reasons: list[str] = []

    n = len(blocks)

    # 1. Engagement signal density (0-30 pts)
    flag_count = sum(
        1 for b in blocks
        if b.has_stakes or b.has_dating or b.has_roast or b.has_vulgarity
    )
    flag_density = flag_count / max(1, n)
    engagement_score = min(30.0, flag_density * 100.0)
    score += engagement_score
    if engagement_score < 10:
        reasons.append("low engagement signal density (few hooks/stakes/roast moments)")

    # 2. Peak energy distribution (0-20 pts)
    # Good content: high-importance blocks spread throughout, not all at one end
    importance_vals = [b.importance for b in blocks]
    avg_imp = sum(importance_vals) / max(1, n)
    high_imp_blocks = [b for b in blocks if b.importance >= 50]
    high_imp_density = len(high_imp_blocks) / max(1, n)

    # Check if high-importance blocks span the video (good distribution)
    if high_imp_blocks:
        first_peak_frac = high_imp_blocks[0].start / max(1.0, source_duration)
        last_peak_frac = high_imp_blocks[-1].end / max(1.0, source_duration)
        peak_span = last_peak_frac - first_peak_frac
        distribution_score = min(10.0, peak_span * 20.0)  # 0-10
    else:
        distribution_score = 0.0

    energy_score = min(20.0, avg_imp * 0.15 + distribution_score + high_imp_density * 10.0)
    score += energy_score
    if energy_score < 8:
        reasons.append("weak energy peaks throughout video")

    # 3. Story completeness (0-25 pts)
    # Does the video have discernible setup AND payoff signals?
    early_blocks = [b for b in blocks if b.start < source_duration * 0.3]
    late_blocks = [b for b in blocks if b.start > source_duration * 0.6]
    has_setup = any(b.importance > 30 for b in early_blocks)
    has_payoff = any(b.importance > 40 for b in late_blocks)
    has_questions = any(b.has_question for b in blocks)
    has_exclamations = any(b.has_exclamation for b in blocks)

    story_score = 0.0
    if has_setup:
        story_score += 8.0
    if has_payoff:
        story_score += 12.0
    if has_questions:
        story_score += 3.0
    if has_exclamations:
        story_score += 2.0
    score += story_score
    if not has_payoff:
        reasons.append("no clear payoff/climax in the latter part of the video")

    # 4. Content volume (0-15 pts)
    usable_blocks = [b for b in blocks if b.importance >= 25 and not b.black_frame and not b.freeze]
    usable_duration = sum(b.duration for b in usable_blocks)
    volume_score = min(15.0, (usable_duration / max(1.0, source_duration)) * 30.0)
    score += volume_score
    if volume_score < 5:
        reasons.append("very little usable/watchable content")

    # 5. Genre-specific bonus (0-10 pts)
    genre_score = 0.0
    genre_cfg = _GENRE_REGISTRY.get(content_type)
    if genre_cfg and genre_cfg.get("flag_field"):
        flag_field = genre_cfg["flag_field"]
        flagged_blocks = [b for b in blocks if getattr(b, flag_field, False)]
        if len(flagged_blocks) >= 3:
            genre_score = 10.0
        elif len(flagged_blocks) >= 1:
            genre_score = 5.0
        else:
            reasons.append(f"{content_type} content detected but few matching engagement moments found")
    elif content_type != GENRE_GENERAL:
        if flag_count >= 2:
            genre_score = 5.0
        if avg_imp >= 40:
            genre_score += 3.0
        if high_imp_density >= 0.2:
            genre_score += 2.0
    else:
        if flag_count >= 2:
            genre_score = 5.0
        if avg_imp >= 40:
            genre_score += 5.0

    score += genre_score

    breakdown = {
        "engagement": {"score": round(engagement_score, 1), "max": 30},
        "energy": {"score": round(energy_score, 1), "max": 20},
        "story": {"score": round(story_score, 1), "max": 25},
        "volume": {"score": round(volume_score, 1), "max": 15},
        "genre": {"score": round(genre_score, 1), "max": 10},
    }

    # Final verdict
    is_worth = score >= 35.0
    score_str = f"{score:.0f}/100"

    if is_worth:
        verdict = f"[PASS] AUDIENCE WORTH: {score_str} - content has good audience potential"
        if reasons:
            verdict += f" (minor concerns: {'; '.join(reasons)})"
    else:
        verdict = (
            f"[WARN] AUDIENCE WORTH: {score_str} - content may not hold audience attention. "
            f"Issues: {'; '.join(reasons) if reasons else 'overall low engagement signals'}"
        )

    logger.info(f"AUDIENCE WORTH GATE: {verdict}")
    return is_worth, verdict, breakdown


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
    has_vulgarity: bool = False
    has_dating: bool = False
    has_roast: bool = False
    has_stakes: bool = False
    # Retention-specific signals (computed by enhanced importance scorer)
    has_curiosity_gap: bool = False   # ends mid-sentence or with a question
    has_pattern_interrupt: bool = False  # sudden energy spike after sustained low energy
    has_social_proof: bool = False    # crowd/reaction language
    has_viral_trigger: bool = False   # emotional confrontation + stakes
    entity_name: str | None = None
    entity_segment_id: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def summary_line(self) -> str:
        energy_bar = "█" * int(self.speech_energy * 10) + "░" * (10 - int(self.speech_energy * 10))
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
        if self.has_dating:
            flags.append("DATING")
        if self.has_roast:
            flags.append("ROAST")
        if self.has_stakes:
            flags.append("STAKES")
        if self.has_curiosity_gap:
            flags.append("CUR_GAP")
        if self.has_pattern_interrupt:
            flags.append("INTERRUPT")
        if self.has_social_proof:
            flags.append("SOCIAL")
        if self.has_viral_trigger:
            flags.append("VIRAL")
        if self.entity_name:
            flags.append(f"ENTITY={self.entity_name}")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        vol = f" vol={self.volume_db:.1f}dB" if self.volume_db is not None else ""
        wps = f" wps={self.word_density:.1f}" if self.word_density > 0 else ""
        return (
            f"Block {self.block_id} [{self.start:.1f}-{self.end:.1f}s] "
            f"imp={self.importance:.0f} energy={energy_bar}({self.speech_energy:.2f})"
            f"{vol} peak=+{self.peak_offset:.1f}s{wps}{flag_str}: "
            f"{self.text[:1000]}"
        )


_ENTITY_NAME_STOP_WORDS = {
    "a", "an", "and", "challenge", "contestant", "episode", "follow", "for",
    "finally", "he", "i", "like", "my", "round", "she", "subscribe", "the",
    "then", "they", "this", "we", "winner", "youtube",
    # Whisper filler words — capitalized at sentence starts, not real entities
    "oh", "you", "yes", "no", "well", "whoa", "who", "what", "how", "why",
    "so", "right", "okay", "ok", "hey", "hi", "hello", "wow", "ah", "um",
    "yeah", "nah", "sure", "really", "just", "now", "here", "there",
    "that", "this", "those", "these", "will", "we're", "you're", "they're",
    "he's", "she's", "it's", "i'm", "can't", "don't", "won't", "let's",
    "going", "come", "look", "got", "get", "go", "see", "know", "think",
    # Additional Whisper artifacts — common words misidentified as entity names
    "ai", "to", "was", "you'll", "we'll", "they'll", "he'll", "she'll",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "last", "next", "back", "new", "old", "big", "little",
    "every", "each", "all", "some", "many", "much", "very",
    "still", "even", "also", "too", "only", "just", "already",
    "thing", "things", "way", "time", "day", "man", "guy",
    "okay", "alright", "oh", "ah", "wow", "ooh", "ugh",
    "yeah", "yep", "nope", "nah", "uh", "hm", "hmm",
}
_TITLE_CASE_NAME = re.compile(r"\b[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*){0,2}\b")


def _clean_entity_name(value: str) -> str | None:
    """Keep conservative person-like labels from Identifier, OCR, or transcript text."""
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n.,:;!?\"'`()[]{}")
    words = value.split()
    if (
        not 1 <= len(words) <= 3
        or value.casefold() in _ENTITY_NAME_STOP_WORDS
        or any(word.casefold() in _ENTITY_NAME_STOP_WORDS for word in words)
    ):
        return None
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", word) for word in words):
        return None
    return value


def _entity_name_in_text(name: str, text: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE))


def _best_entity_name(candidates: list[tuple[str, int]]) -> str | None:
    """Resolve evidence with deterministic strength, frequency, then spelling order."""
    scores: dict[str, tuple[int, int, str]] = {}
    for raw_name, strength in candidates:
        name = _clean_entity_name(raw_name)
        if not name:
            continue
        key = name.casefold()
        old_strength, old_count, old_name = scores.get(key, (0, 0, name))
        scores[key] = (max(old_strength, strength), old_count + 1, old_name)
    if not scores:
        return None
    return max(scores.values(), key=lambda item: (item[0], item[1], item[2].casefold()))[2]


def _segment_entities(
    blocks: list[SemanticBlock],
    rich_timeline: RichTimeline | None,
    multimodal_signals: MultimodalSignals | None,
    content_identity: ContentIdentity | None,
    source_duration: float,
) -> tuple[list[EntitySegment], bool]:
    """Fuse speaker, OCR, Identifier, transcript, and cuts into entity windows.

    Speaker changes and named evidence are preferred over scene changes. Scene cuts
    remain usable boundaries for unknown multi-entity videos, but a confirmed
    single-narrative Identifier prevents cut-only grouping regressions.
    """
    if not blocks or source_duration <= 0:
        return [], bool(content_identity and content_identity.structure == "multi_entity")

    timeline_by_id = {
        segment.segment_id: segment
        for segment in (rich_timeline.segments if rich_timeline else [])
    }
    known_names = [
        name for raw_name in (content_identity.entity_names if content_identity else [])
        if (name := _clean_entity_name(raw_name))
    ]
    signals = multimodal_signals or MultimodalSignals()
    block_speakers: dict[int, list[str]] = {}
    block_candidates: dict[int, list[tuple[str, int, str]]] = {}

    for block in blocks:
        speaker_ids = sorted({
            segment.speaker_id.strip()
            for segment_id in block.segment_ids
            if (segment := timeline_by_id.get(segment_id)) and segment.speaker_id and segment.speaker_id.strip()
        })
        block_speakers[block.block_id] = speaker_ids
        candidates: list[tuple[str, int, str]] = []
        for signal in signals.on_screen_text:
            if signal.text and block.start <= signal.timestamp < block.end:
                candidates.append((signal.text, 30, "ocr"))
        for name in known_names:
            if _entity_name_in_text(name, block.text):
                candidates.append((name, 20, "identifier_transcript"))
        for match in _TITLE_CASE_NAME.finditer(block.text):
            # Whisper usually capitalizes sentence starts; a lone first token is
            # not reliable enough to treat as a person without Identifier support.
            if match.start() == 0 and " " not in match.group(0):
                continue
            if (name := _clean_entity_name(match.group(0))):
                candidates.append((name, 10, "transcript_name"))
        block_candidates[block.block_id] = candidates

    # A speaker ID decides ownership; a nearby name only makes its label human-readable.
    speaker_labels: dict[str, str] = {}
    for speaker_id in sorted({speaker for ids in block_speakers.values() for speaker in ids}):
        speaker_candidates = [
            (name, strength)
            for block in blocks if speaker_id in block_speakers[block.block_id]
            for name, strength, _ in block_candidates[block.block_id]
        ]
        speaker_labels[speaker_id] = _best_entity_name(speaker_candidates) or speaker_id

    block_names: dict[int, str | None] = {}
    block_evidence: dict[int, list[str]] = {}
    for block in blocks:
        speakers = block_speakers[block.block_id]
        candidates = block_candidates[block.block_id]
        if speakers:
            block_names[block.block_id] = speaker_labels[speakers[0]]
            block_evidence[block.block_id] = ["speaker_id"]
        else:
            block_names[block.block_id] = _best_entity_name([(name, strength) for name, strength, _ in candidates])
            block_evidence[block.block_id] = sorted({kind for _, _, kind in candidates})

    # ── Name-entrance-first boundary detection ──
    # Name-entrance events (OCR lower-third hits) are the authoritative
    # segment boundaries.  Speaker-change/scene-cut signals are secondary:
    # they can subdivide within name-defined regions, but never override
    # a name-entrance boundary.
    #
    # Strength hierarchy:
    #   10 — name entrance (OCR signal) — always kept
    #    5 — transcript name change (different person identified in speech)
    #    4 — speaker change (diarization boundary)
    #    1 — scene cut (visual discontinuity)
    #
    name_entrance_times: set[float] = set()
    boundaries: dict[float, int] = {0.0: 99, round(source_duration, 3): 99}

    for signal in signals.on_screen_text:
        if signal.text and signal.scene_cut_at and 0.0 < signal.scene_cut_at < source_duration:
            ts = round(signal.scene_cut_at, 3)
            boundaries[ts] = max(boundaries.get(ts, 0), 10)
            name_entrance_times.add(ts)

    ordered_blocks = sorted(blocks, key=lambda block: (block.start, block.end, block.block_id))
    previous_speakers: list[str] = []
    previous_name: str | None = None
    for block in ordered_blocks:
        speakers = block_speakers[block.block_id]
        name = block_names[block.block_id]
        ts = round(block.start, 3)
        if block.start > 0.0 and speakers and previous_speakers and speakers != previous_speakers:
            boundaries[ts] = max(boundaries.get(ts, 0), 4)
        elif block.start > 0.0 and name and previous_name and name.casefold() != previous_name.casefold():
            # Transcript-detected name change — secondary to OCR name-entrance
            boundaries[ts] = max(boundaries.get(ts, 0), 5)
        if speakers:
            previous_speakers = speakers
        if name:
            previous_name = name

    for cut_at in signals.scene_cut_at:
        ts = round(cut_at, 3)
        if 0.0 < ts < source_duration and ts not in name_entrance_times:
            boundaries[ts] = max(boundaries.get(ts, 0), 1)

    candidates = sorted(
        ((timestamp, strength) for timestamp, strength in boundaries.items() if 0.0 < timestamp < source_duration),
        key=lambda item: (-item[1], item[0]),
    )
    kept_boundaries = [0.0, round(source_duration, 3)]
    for timestamp, _ in candidates:
        if all(abs(timestamp - boundary) >= ENTITY_MIN_SEGMENT_SECONDS for boundary in kept_boundaries):
            kept_boundaries.append(timestamp)
    kept_boundaries.sort()

    entity_segments: list[EntitySegment] = []
    for index in range(1, len(kept_boundaries)):
        start, end = kept_boundaries[index - 1], kept_boundaries[index]
        segment_blocks = [
            block for block in ordered_blocks
            if block.end > start and block.start < end
        ]
        name_candidates: list[tuple[str, int]] = []
        speaker_ids: set[str] = set()
        evidence: set[str] = {"scene_cut"} if any(start <= cut <= end for cut in signals.scene_cut_at) else set()
        for block in segment_blocks:
            speaker_ids.update(block_speakers[block.block_id])
            evidence.update(block_evidence[block.block_id])
            if block_names[block.block_id]:
                strength = 40 if block_speakers[block.block_id] else 30
                name_candidates.append((block_names[block.block_id], strength))
        entity_name = _best_entity_name(name_candidates)
        segment_id = f"entity-{index}"
        for block in segment_blocks:
            block.entity_segment_id = segment_id
            block.entity_name = block_names[block.block_id] or entity_name
        entity_segments.append(EntitySegment(
            entity_segment_id=segment_id,
            entity_name=entity_name,
            start=round(start, 3),
            end=round(end, 3),
            block_ids=[block.block_id for block in segment_blocks],
            speaker_ids=sorted(speaker_ids),
            evidence=sorted(evidence),
        ))

    qualifying_segments = [
        segment for segment in entity_segments
        if segment.end - segment.start >= ENTITY_MIN_SEGMENT_SECONDS and segment.block_ids
    ]
    entity_grouped = bool(content_identity and content_identity.structure == "multi_entity")
    if entity_grouped and content_identity and len(content_identity.entity_names) >= 2:
        # Identifier confidently identified multiple entities — always group
        pass
    elif content_identity and content_identity.structure == "single_narrative":
        # Identifier says single narrative — require sustained non-scene-cut evidence
        # (≥2 consecutive segments with speaker_id or name evidence) to override
        non_scene_streak = 0
        max_streak = 0
        for seg in qualifying_segments:
            if seg.evidence != ["scene_cut"]:
                non_scene_streak += 1
                max_streak = max(max_streak, non_scene_streak)
            else:
                non_scene_streak = 0
        if max_streak >= 2:
            entity_grouped = True
    elif len(qualifying_segments) >= 3:
        # No Identifier or ambiguous — use segment count heuristic, BUT only
        # if segments have distinct named entities (from OCR/scene names).
        # Podcasts/interviews produce many speaker-change segments without
        # distinct names — they should use the genre planner, not entity mode.
        named_segments = [s for s in qualifying_segments if s.entity_name]
        distinct_names = {s.entity_name for s in named_segments}
        if len(distinct_names) >= 3:
            entity_grouped = True
    logger.info(
        "ENTITY SEGMENTATION: %d segments (%d qualifying), grouped=%s",
        len(entity_segments), len(qualifying_segments), entity_grouped,
    )
    return entity_segments, entity_grouped


def _score_entity_segment(
    segment: EntitySegment,
    blocks: list,
) -> float:
    """Score an entity segment for reel-worthiness.

    Higher score = better standalone reel subject. Considers duration,
    evidence quality, named identity, and content richness.
    """
    score = 0.0

    duration = segment.end - segment.start
    score += duration * 2.0

    evidence_set = set(segment.evidence)
    if "speaker_id" in evidence_set:
        score += 30.0
    if "ocr" in evidence_set or "identifier_transcript" in evidence_set:
        score += 20.0
    if "transcript_name" in evidence_set:
        score += 10.0
    if evidence_set == {"scene_cut"}:
        score -= 15.0

    if segment.entity_name:
        score += 25.0

    segment_blocks = [b for b in blocks if b.block_id in segment.block_ids]
    if segment_blocks:
        avg_importance = sum(getattr(b, "importance", 0) for b in segment_blocks) / len(segment_blocks)
        score += avg_importance * 10.0
        word_density = sum(getattr(b, "word_density", 0) for b in segment_blocks) / len(segment_blocks)
        score += word_density * 20.0

    return score


def _select_top_entity_segments(
    segments: list[EntitySegment],
    max_count: int,
    blocks: list,
) -> list[EntitySegment]:
    """Select the top N entity segments by reel-worthiness score.

    Unlike the old merge-based approach, this preserves each selected
    segment's narrative integrity — no hybrid multi-entity units.
    """
    if len(segments) <= max_count:
        return segments

    scored = [
        (seg, _score_entity_segment(seg, blocks))
        for seg in segments
    ]
    scored.sort(key=lambda x: -x[1])

    selected = [s for s, _ in scored[:max_count]]
    selected.sort(key=lambda s: s.start)

    removed = len(segments) - len(selected)
    logger.info(
        f"Entity segment selection: {len(segments)} → {len(selected)} segments "
        f"(removed {removed} lowest-scored, kept top {max_count} by content quality)"
    )

    segs = selected
    for i, seg in enumerate(segs):
        new_id = f"entity-{i + 1}"
        if seg.entity_segment_id != new_id:
            for block in blocks:
                if block.entity_segment_id == seg.entity_segment_id:
                    block.entity_segment_id = new_id
            seg.entity_segment_id = new_id

    return segs


def _premerge_entity_segments(
    segments: list[EntitySegment],
    blocks: list,
) -> list[EntitySegment]:
    """Merge segments sharing the same primary speaker_id.

    Handles two patterns:
    1. Consecutive same-speaker segments (direct merge)
    2. Non-adjacent same-speaker segments interleaved with other speakers
       (e.g., host→contestant→host→contestant → merge all contestant segments)

    Only merges when the shared speaker is the PRIMARY (first-listed) speaker
    in both segments.
    """
    if len(segments) <= 1:
        return segments

    segs = list(segments)

    # Pass 1: merge consecutive same-speaker segments
    merged_any = True
    while merged_any:
        merged_any = False
        i = 0
        while i < len(segs) - 1:
            a, b = segs[i], segs[i + 1]
            a_primary = a.speaker_ids[0] if a.speaker_ids else None
            b_primary = b.speaker_ids[0] if b.speaker_ids else None
            if a_primary and b_primary and a_primary == b_primary:
                merged_ids = list(dict.fromkeys(a.block_ids + b.block_ids))
                merged_speakers = sorted(set(a.speaker_ids) | set(b.speaker_ids))
                merged_evidence = sorted(set(a.evidence) | set(b.evidence))
                merged_name = a.entity_name or b.entity_name
                merged_id = a.entity_segment_id
                for block in blocks:
                    if block.entity_segment_id == b.entity_segment_id:
                        block.entity_segment_id = merged_id
                merged = EntitySegment(
                    entity_segment_id=merged_id,
                    entity_name=merged_name,
                    start=a.start,
                    end=b.end,
                    block_ids=merged_ids,
                    speaker_ids=merged_speakers,
                    evidence=merged_evidence,
                )
                segs[i] = merged
                segs.pop(i + 1)
                merged_any = True
            else:
                i += 1

    # Pass 2: merge non-adjacent same-speaker segments
    # (host→contestant→host→contestant pattern)
    # The merged segment's start/end is computed as the union of its own
    # block time ranges, NOT min(seg1.start) to max(seg2.end) which would
    # span across other speakers' time and cause cross-contamination.
    merged_any_pass2 = True
    while merged_any_pass2:
        merged_any_pass2 = False
        speaker_positions: dict[str, list[int]] = {}
        for i, seg in enumerate(segs):
            primary = seg.speaker_ids[0] if seg.speaker_ids else None
            if primary:
                speaker_positions.setdefault(primary, []).append(i)

        for speaker, positions in speaker_positions.items():
            if len(positions) < 2:
                continue
            first, last = positions[0], positions[-1]
            between = [i for i in range(first + 1, last) if i not in positions]
            between_speakers = set()
            for i in between:
                primary = segs[i].speaker_ids[0] if segs[i].speaker_ids else None
                if primary:
                    between_speakers.add(primary)
            if speaker not in between_speakers:
                merge_segs = [segs[i] for i in positions]
                merged_ids = []
                merged_speakers_set = set()
                merged_evidence_set = set()
                merged_name = None
                for s in merge_segs:
                    merged_ids.extend(s.block_ids)
                    merged_speakers_set |= set(s.speaker_ids)
                    merged_evidence_set |= set(s.evidence)
                    if s.entity_name and not merged_name:
                        merged_name = s.entity_name
                merged_ids = list(dict.fromkeys(merged_ids))
                merged_id = merge_segs[0].entity_segment_id
                for block in blocks:
                    if block.entity_segment_id in {s.entity_segment_id for s in merge_segs[1:]}:
                        block.entity_segment_id = merged_id
                # Compute effective range: union of merged segments' own bounds.
                # For non-adjacent merges, store sub-ranges so the executor can
                # filter blocks to actual content ranges (not the full span).
                merged_start = min(s.start for s in merge_segs)
                merged_end = max(s.end for s in merge_segs)
                # Check if segments are non-adjacent (gaps between them)
                has_gaps = any(
                    merge_segs[i + 1].start > merge_segs[i].end + 0.1
                    for i in range(len(merge_segs) - 1)
                )
                effective_ranges = (
                    [(s.start, s.end) for s in merge_segs]
                    if has_gaps
                    else []
                )
                merged = EntitySegment(
                    entity_segment_id=merged_id,
                    entity_name=merged_name,
                    start=merged_start,
                    end=merged_end,
                    block_ids=merged_ids,
                    speaker_ids=sorted(merged_speakers_set),
                    evidence=sorted(merged_evidence_set),
                    effective_ranges=effective_ranges,
                )
                segs[positions[0]] = merged
                for i in sorted(positions[1:], reverse=True):
                    segs.pop(i)
                merged_any_pass2 = True
                break  # restart with recalculated positions

    # Reassign sequential IDs
    for i, seg in enumerate(segs):
        new_id = f"entity-{i + 1}"
        if seg.entity_segment_id != new_id:
            for block in blocks:
                if block.entity_segment_id == seg.entity_segment_id:
                    block.entity_segment_id = new_id
            seg.entity_segment_id = new_id

    if len(segs) < len(segments):
        logger.info(f"Entity pre-merge: merged {len(segments)} → {len(segs)} segments by speaker_id")
    return segs


def _compute_importance(
    energy: float,
    volume_db: float | None,
    silence_before: bool,
    black: bool,
    freeze: bool,
    has_question: bool = False,
    has_exclamation: bool = False,
    has_emphasis: bool = False,
    # Retention-weighted signals (new)
    has_curiosity_gap: bool = False,
    has_pattern_interrupt: bool = False,
    has_social_proof: bool = False,
    has_viral_trigger: bool = False,
) -> float:
    """Deterministic importance 0-100. Python calculates; LLM only ranks editorially.

    Enhanced with audience retention signals:
    - curiosity_gap: +15  (block ends mid-sentence/question — keeps watching)
    - pattern_interrupt: +12  (sudden energy spike — stops scrolling)
    - social_proof: +10  (crowd/reaction language — social validation)
    - viral_trigger: +20  (emotional stakes + confrontation — most shareable)
    """
    score = 0.0
    # Speech energy (0-40)
    score += min(40.0, energy * 40.0)
    # Volume relative boost (0-15) - louder than -25 dB helps
    if volume_db is not None:
        # typical speech ~-20 to -10; map roughly
        vol_norm = max(0.0, min(1.0, (volume_db + 40) / 30.0))
        score += vol_norm * 15.0
    # Natural cut point (0-10)
    if silence_before:
        score += 10.0
    # Engagement signals (0-20)
    engagement = (8.0 if has_question else 0.0) + (7.0 if has_exclamation else 0.0) + (5.0 if has_emphasis else 0.0)
    score += min(20.0, engagement)
    # Penalties
    if black:
        score -= 25.0
    if freeze:
        score -= 20.0

    # ── Audience retention bonuses ──
    retention_bonus = 0.0
    if has_curiosity_gap:
        retention_bonus += 15.0
    if has_pattern_interrupt:
        retention_bonus += 12.0
    if has_social_proof:
        retention_bonus += 10.0
    if has_viral_trigger:
        retention_bonus += 20.0
    # Cap retention bonus so it can't dominate (max +30 in practice)
    score += min(30.0, retention_bonus)

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# Engagement signal detection
# ─────────────────────────────────────────────────────────────────────────────

# Small keyword sets for 4 engagement signals — kept short on purpose.
_VULGARITY_KEYWORDS = {"sexy", "hot", "fine", "thick", "ass", "body count", "ugly", "mid", "onlyfans", "savage"}
_DATING_KEYWORDS = {"date", "dating", "crush", "flirt", "flirting", "kiss", "couple", "boyfriend", "girlfriend", "chemistry", "single", "match"}
_ROAST_KEYWORDS = {"rejected", "no way", "shut up", "roast", "roasted", "exposed", "called out", "ridiculous", "unbelievable", "dumb"}
_STAKES_KEYWORDS = {"money", "prize", "win", "winner", "eliminated", "last one", "challenge", "record", "insane", "never before"}

# Retention signal keyword sets (new)
_CURIOSITY_GAP_ENDINGS = {"but", "and then", "so", "because", "which means", "wait", "actually"}
_SOCIAL_PROOF_KEYWORDS = {"everyone", "crowd", "they went crazy", "no one expected", "whole room", "all of them", "they couldn't believe"}
_VIRAL_TRIGGER_KEYWORDS = {"confrontation", "fight", "argument", "drama", "caught", "exposed", "shocking", "betrayal", "breakdown"}


def _detect_engagement_signals(text: str) -> dict[str, bool]:
    """Detect vulgarity / dating / roast / stakes signals from block text."""
    if not text:
        return {
            "has_vulgarity": False, "has_dating": False,
            "has_roast": False, "has_stakes": False,
        }
    text_lower = text.lower()
    words = set(text_lower.split())

    def _hit(keywords: set[str]) -> bool:
        return any((kw in text_lower) if " " in kw else (kw in words) for kw in keywords)

    return {
        "has_vulgarity": _hit(_VULGARITY_KEYWORDS),
        "has_dating":    _hit(_DATING_KEYWORDS),
        "has_roast":     _hit(_ROAST_KEYWORDS),
        "has_stakes":    _hit(_STAKES_KEYWORDS),
    }


def _detect_retention_signals(text: str, energy: float, prev_energy: float | None) -> dict[str, bool]:
    """Detect audience retention signals from block text and energy context.

    These signals inform the retention-weighted importance scoring.
    """
    if not text:
        return {
            "has_curiosity_gap": False,
            "has_pattern_interrupt": False,
            "has_social_proof": False,
            "has_viral_trigger": False,
        }
    text_lower = text.lower().strip()
    words = set(text_lower.split())

    def _hit(keywords: set[str]) -> bool:
        return any((kw in text_lower) if " " in kw else (kw in words) for kw in keywords)

    # Curiosity gap: block ends with ellipsis, "but", "and then", mid-thought, or question
    has_curiosity_gap = (
        text.endswith("?")
        or text.endswith("...")
        or any(text_lower.endswith(e) for e in _CURIOSITY_GAP_ENDINGS)
        or (len(text.split()) < 6 and not text.endswith("."))  # very short, incomplete thought
    )

    # Pattern interrupt: sudden energy spike (this block much louder than previous)
    has_pattern_interrupt = False
    if prev_energy is not None and prev_energy < 0.3 and energy > 0.65:
        has_pattern_interrupt = True

    # Social proof: crowd/group reaction language
    has_social_proof = _hit(_SOCIAL_PROOF_KEYWORDS)

    # Viral trigger: emotional confrontation + any engagement flag
    has_viral_trigger = _hit(_VIRAL_TRIGGER_KEYWORDS) and (
        _hit(_STAKES_KEYWORDS) or _hit(_ROAST_KEYWORDS) or _hit(_DATING_KEYWORDS)
    )

    return {
        "has_curiosity_gap": has_curiosity_gap,
        "has_pattern_interrupt": has_pattern_interrupt,
        "has_social_proof": has_social_proof,
        "has_viral_trigger": has_viral_trigger,
    }


def _build_semantic_blocks(
    rich_timeline: RichTimeline | None,
    transcript: list[dict],
    max_block_seconds: float = 28.0,
    min_block_seconds: float = 4.0,
) -> list[SemanticBlock]:
    """
    Collapse fine-grained segments into coherent semantic blocks.
    Target ~120-180 blocks for a long video instead of 500+ tiny segments.
    Enhanced with retention signal detection.
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
                "silence_before": bool(getattr(seg, "silence_before", False)),
                "black": bool(getattr(seg.metrics, "black_frame", False)) if hasattr(seg, "metrics") else False,
                "freeze": bool(getattr(seg.metrics, "freeze_detected", False)) if hasattr(seg, "metrics") else False,
                "has_question": bool(getattr(seg, "has_question", False)),
                "has_exclamation": bool(getattr(seg, "has_exclamation", False)),
                "has_emphasis": bool(getattr(seg, "has_emphasis", False)),
                "word_density": float(getattr(seg, "word_density", 0.0) or 0.0),
            })
        for item in items:
            item.update(_detect_engagement_signals(item["text"]))
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
                "silence_before": False,
                "black": False,
                "freeze": False,
                "has_question": False,
                "has_exclamation": False,
                "has_emphasis": False,
                "word_density": 0.0,
            })
        for item in items:
            item.update(_detect_engagement_signals(item["text"]))

    if not items:
        return []

    # Attach retention signals (requires prev_energy context)
    prev_energy: float | None = None
    for item in items:
        retention = _detect_retention_signals(item["text"], item["energy"], prev_energy)
        item.update(retention)
        prev_energy = item["energy"]

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
        silence_before = group[0]["silence_before"]
        black = any(g["black"] for g in group)
        freeze = any(g["freeze"] for g in group)
        has_question = any(g["has_question"] for g in group)
        has_exclamation = any(g["has_exclamation"] for g in group)
        has_emphasis = any(g["has_emphasis"] for g in group)
        has_vulgarity = any(g.get("has_vulgarity", False) for g in group)
        has_dating = any(g.get("has_dating", False) for g in group)
        has_roast = any(g.get("has_roast", False) for g in group)
        has_stakes = any(g.get("has_stakes", False) for g in group)
        # Retention signals (any block in group triggers flag)
        has_curiosity_gap = any(g.get("has_curiosity_gap", False) for g in group)
        has_pattern_interrupt = any(g.get("has_pattern_interrupt", False) for g in group)
        has_social_proof = any(g.get("has_social_proof", False) for g in group)
        has_viral_trigger = any(g.get("has_viral_trigger", False) for g in group)
        word_densities = [g["word_density"] for g in group if g["word_density"] > 0]
        avg_word_density = sum(word_densities) / len(word_densities) if word_densities else 0.0
        importance = _compute_importance(
            avg_energy, volume_db, silence_before, black, freeze,
            has_question, has_exclamation, has_emphasis,
            has_curiosity_gap, has_pattern_interrupt, has_social_proof, has_viral_trigger,
        )
        # Post-importance adjustment: penalize verbose commentary with no engagement flags.
        # High word density (>2.0 wps) means lots of talking — if none of it has
        # engagement flags (question/exclamation) or retention flags (stakes/viral/etc),
        # it's likely filler commentary that shouldn't score as high-importance content.
        has_any_flag = any((
            has_question, has_exclamation, has_emphasis,
            has_vulgarity, has_dating, has_roast, has_stakes,
            has_curiosity_gap, has_pattern_interrupt, has_social_proof, has_viral_trigger,
        ))
        if not has_any_flag and avg_word_density > 2.0:
            importance = max(0.0, importance - 15.0)
        elif not has_any_flag and avg_word_density > 1.5:
            importance = max(0.0, importance - 8.0)
        blocks.append(SemanticBlock(
            block_id=len(blocks),
            start=start,
            end=end,
            text=text,
            speech_energy=avg_energy,
            volume_db=volume_db,
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
            has_dating=has_dating,
            has_roast=has_roast,
            has_stakes=has_stakes,
            has_curiosity_gap=has_curiosity_gap,
            has_pattern_interrupt=has_pattern_interrupt,
            has_social_proof=has_social_proof,
            has_viral_trigger=has_viral_trigger,
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
    if top_n and len(blocks) > top_n:
        ranked = sorted(blocks, key=lambda b: b.importance, reverse=True)[:top_n]
        top_ids = {b.block_id for b in ranked}
        header = f"TOP-{top_n} importance block_ids: {sorted(top_ids)}\n"
        selected = [b for b in blocks if b.block_id in top_ids]
        return header + "\n".join(b.summary_line() for b in selected)
    return "\n".join(b.summary_line() for b in blocks)


def _format_full_transcript(transcript: list[dict]) -> str:
    """Format transcript segments for legacy callers and diagnostics."""
    lines = []
    for index, entry in enumerate(transcript):
        text = str(entry.get("text", "")).strip()
        if text:
            lines.append(
                f"Seg {index} [{float(entry.get('start', 0.0)):.1f}-{float(entry.get('end', 0.0)):.1f}s]: {text}"
            )
    return "\n".join(lines)


def _normalize_clip_range(transcript: list[dict], start_seg: int, end_seg: int) -> tuple[int, int]:
    """Clamp a legacy transcript range to the configured soft clip duration."""
    if not transcript:
        return 0, 0
    max_index = len(transcript) - 1
    start_seg = min(max(start_seg, 0), max_index)
    end_seg = min(max(end_seg, start_seg), max_index)
    while transcript[end_seg]["end"] - transcript[start_seg]["start"] < CLIP_DURATION_SOFT_MIN:
        if end_seg < max_index:
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


def _prompt_identifier(
    video_title: str,
    video_description: str,
    source_metadata: SourceMetadata | None,
    blocks_text: str,
) -> str:
    """Build the semantic-only source identification prompt."""
    channel_name = source_metadata.channel_name if source_metadata else None
    channel_description = source_metadata.channel_description if source_metadata else None
    return f"""You identify the editorial structure of a YouTube source before it is cut into Shorts.
You make semantic judgments only. Do not choose timestamps, durations, clips, group counts, or rewrite content.

SOURCE METADATA
Channel name: {channel_name or "unavailable"}
Channel description: {(channel_description or "unavailable")[:20000]}
Video title: {video_title}
Video description: {(video_description or "")[:20000]}

HIGH-IMPORTANCE TRANSCRIPT BLOCKS
{blocks_text}

Determine who made the video, what format it is, whether it is one continuous story or a sequence of separate people/acts, and the best cutting approach.
Use "multi_entity" only when distinct contestants, performers, acts, or people have separate editorial moments. Do not infer named people without evidence.
Use exactly one detected_genre from: {", ".join((*_GENRE_REGISTRY, GENRE_GENERAL))}.
planning_notes must be 2-4 concise sentences of editorial strategy, never numbers or timestamps.

OUTPUT -- STRICT JSON ONLY
{{
  "creator_name": "confirmed creator name or best guess, or null",
  "content_format": "plain-language format description",
  "detected_genre": "one allowed genre",
  "structure": "single_narrative|multi_entity",
  "entity_names": ["named person or act"],
  "planning_notes": "2-4 sentence strategy"
}}"""


def _parse_content_identity(
    raw: str,
    source_metadata: SourceMetadata | None,
) -> ContentIdentity | None:
    """Validate Identifier output and prefer confirmed yt-dlp creator data."""
    try:
        data = _parse_json_response(raw)
        if not isinstance(data, dict):
            return None
        identity = ContentIdentity.model_validate(data)
    except Exception as e:
        logger.warning(f"Identifier returned invalid content identity: {e}")
        return None

    valid_genres = {*_GENRE_REGISTRY, GENRE_GENERAL}
    if identity.detected_genre not in valid_genres:
        logger.warning("Identifier returned unknown genre '%s'", identity.detected_genre)
        return None

    identity.content_format = identity.content_format.strip()
    identity.planning_notes = identity.planning_notes.strip()
    if not identity.content_format or not identity.planning_notes:
        logger.warning("Identifier omitted content format or planning notes")
        return None

    seen_names: set[str] = set()
    identity.entity_names = [
        name
        for raw_name in identity.entity_names
        if (name := raw_name.strip())
        and name.casefold() not in seen_names
        and not seen_names.add(name.casefold())
    ]
    if identity.structure == "single_narrative":
        identity.entity_names = []
    if source_metadata and source_metadata.channel_name:
        identity.creator_name = source_metadata.channel_name
    return identity


def _identify_content(
    video_title: str,
    video_description: str,
    source_metadata: SourceMetadata | None,
    blocks: list[SemanticBlock],
    progress_cb: Callable[[str, float], None] | None,
    reporter: Any,
    interactions: list[LLMInteraction] | None,
) -> ContentIdentity | None:
    """Run the advisory Identifier stage. Failure always falls back to Python-only planning."""
    try:
        raw = _call_llm(
            [
                {"role": "system", "content": "Respond with ONLY valid JSON."},
                {
                    "role": "user",
                    "content": _prompt_identifier(
                        video_title,
                        video_description,
                        source_metadata,
                        _format_blocks_for_llm(blocks, top_n=30),
                    ),
                },
            ],
            progress_cb,
            reporter,
            interactions,
            stage_name="identifier",
            max_tokens=2048,
        )
        identity = _parse_content_identity(raw, source_metadata)
        if identity:
            logger.info(
                "IDENTIFIER: creator=%r format=%r structure=%s genre=%s",
                identity.creator_name,
                identity.content_format,
                identity.structure,
                identity.detected_genre,
            )
        return identity
    except Exception as e:
        logger.warning(f"Identifier failed — continuing with Python-only planning: {e}")
        return None


async def _run_identifier(
    video_title: str,
    video_description: str,
    source_metadata: SourceMetadata | None,
    source_path: str,
    blocks: list[SemanticBlock],
    progress_cb: Callable[[str, float], None] | None,
    reporter: Any,
    interactions: list[LLMInteraction] | None,
) -> ContentIdentity | None:
    """Run LLM identifier enrichment only."""
    try:
        return await asyncio.to_thread(
            _identify_content,
            video_title,
            video_description,
            source_metadata,
            blocks,
            progress_cb,
            reporter,
            interactions,
        )
    except Exception as e:
        logger.warning("Identifier task failed — continuing with Python-only planning: %s", e)
        return None


async def _run_multimodal(
    source_path: str,
    interactions: list[LLMInteraction] | None,
) -> MultimodalSignals:
    """Run multimodal (scene detection + OCR) enrichment only."""
    from backend.pipeline.multimodal import enrich_multimodal_signals

    try:
        return await enrich_multimodal_signals(source_path, interactions)
    except Exception as e:
        logger.warning("Multimodal enrichment failed — continuing without visual signals: %s", e)
        return MultimodalSignals()


def _run_preplanning_enrichment(
    video_title: str,
    video_description: str,
    source_metadata: SourceMetadata | None,
    source_path: str | None,
    blocks: list[SemanticBlock],
    progress_cb: Callable[[str, float], None] | None,
    reporter: Any,
    interactions: list[LLMInteraction] | None,
    cached_multimodal_signals: MultimodalSignals | None,
) -> tuple[ContentIdentity | None, MultimodalSignals | None]:
    """Run identifier and multimodal enrichment sequentially when not checkpointed."""
    content_identity = _identify_content(
        video_title,
        video_description,
        source_metadata,
        blocks,
        progress_cb,
        reporter,
        interactions,
    )
    from backend.config import OCR_MODE as _OCR_MODE
    if not MULTIMODAL_ENABLED or not source_path or _OCR_MODE == "skip":
        if progress_cb:
            progress_cb("Skipping multimodal enrichment (disabled)...", 10)
        if reporter:
            reporter.update_analyzer_phase("multimodal_ocr", "skipped", detail={"reason": "disabled"})
        return content_identity, cached_multimodal_signals
    # Invalidate stale cache: if cached signals are empty but OCR is now enabled,
    # re-run multimodal enrichment instead of returning the empty cache.
    if cached_multimodal_signals is not None:
        from backend.config import OCR_MODE
        has_data = bool(
            cached_multimodal_signals.scene_cut_at
            or cached_multimodal_signals.on_screen_text
        )
        if has_data:
            if progress_cb:
                progress_cb("Using cached multimodal signals...", 10)
            if reporter:
                reporter.update_analyzer_phase("multimodal_ocr", "skipped", detail={"reason": "cached"})
            return content_identity, cached_multimodal_signals
    if progress_cb:
        progress_cb("Running multimodal enrichment (scene detection + OCR)...", 10)
    if reporter:
        reporter.update_analyzer_phase("multimodal_ocr", "running")
    # Safe asyncio execution: if an event loop is already running (e.g. from
    # FastAPI or asyncio.to_thread), use loop.run_until_complete instead of
    # asyncio.run() which would crash with "RuntimeError: event loop already running".
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    try:
        if loop is not None:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, _run_multimodal(source_path, interactions)).result()
        else:
            result = asyncio.run(_run_multimodal(source_path, interactions))
    except Exception as e:
        if reporter:
            reporter.update_analyzer_phase("multimodal_ocr", "error", error=str(e)[:300])
        raise
    if reporter:
        reporter.update_analyzer_phase(
            "multimodal_ocr", "done", progress=100,
            detail={
                "scene_cuts": len(result.scene_cut_at or []),
                "ocr_text_events": len(result.on_screen_text or []),
            },
        )
    return content_identity, result


def _usable_duration(blocks: list[SemanticBlock], start: float, end: float) -> float:
    """Deterministic usable seconds inside [start, end] (excludes black/freeze/low-importance).

    The importance>=25 gate is relaxed when it would exclude nearly all blocks
    in the window (e.g. VAD produced no speech energy), so planning hints don't
    collapse to zero on degraded inputs.
    """
    in_window = [b for b in blocks if b.end > start and b.start < end]
    if not in_window:
        return 0.0
    usable_pool = [b for b in in_window if not (b.black_frame or b.freeze)]
    strict = [b for b in usable_pool if b.importance >= 25]
    if len(usable_pool) and len(strict) / len(usable_pool) < MIN_USABLE_BLOCK_FRACTION:
        pool = usable_pool
    else:
        pool = strict
    total = 0.0
    for b in pool:
        if b.end <= start or b.start >= end:
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
    - Position not at the very start (curiosity gap)
    - Retention signals: curiosity_gap, viral_trigger, pattern_interrupt
    - Content substance: penalize clips with very few words (fluff)
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

    # Content substance penalty: clips with very few total words are likely
    # fluff (exclamations, reactions) not meaningful intros.
    total_words = sum(len(b.text.split()) for b in blocks_in_clip)
    words_per_second = total_words / max(clip_duration, 0.1)
    if words_per_second < 1.0:
        # Very low word density = fluff (shouting, reactions, filler)
        score -= 15.0
    elif words_per_second < 1.5:
        score -= 5.0

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

    # Engagement signal bonus — vulgar/dating/roast/stakes moments hook hardest
    if any(b.has_vulgarity or b.has_dating or b.has_roast or b.has_stakes for b in blocks_in_clip):
        score += 15.0

    # Retention signal bonuses (new)
    if any(b.has_curiosity_gap for b in blocks_in_clip):
        score += 12.0  # curiosity gap = viewer must keep watching
    if any(b.has_viral_trigger for b in blocks_in_clip):
        score += 15.0  # viral trigger = most shareable
    if any(b.has_pattern_interrupt for b in blocks_in_clip):
        score += 8.0   # pattern interrupt = stops scrolling

    return max(0.0, min(100.0, score))


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
# Story flow enforcement — groups are ordered so the playlist progresses
# chronologically through the source video (start to end), giving viewers
# a clear narrative arc rather than random jumps.
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_story_flow(groups: list[dict]) -> list[dict]:
    """Hardcode story flow pillars across and within all reel groups.

    Pillars:
      1. Hook / Start: 4-6s opening clip setting up the premise
      2. Mid / Journey: Escalation clips strictly ordered chronologically by source time
      3. End / Payoff: Final climax reveal

    Guarantees timestamps never jump backwards or mix up clips within or across groups.
    """
    if not groups:
        return groups

    for g in groups:
        clips = g.get("source_clips", [])
        if len(clips) > 1:
            # Enforce strict chronological order for non-hook clips so journey & payoff never mix up
            has_hook = clips[0].get("is_hook_clip", False)
            hook = clips[0] if has_hook else None
            rest = clips[1:] if has_hook else clips
            rest.sort(key=lambda c: c.get("source_start", 0.0))
            g["source_clips"] = ([hook] + rest) if hook else rest

    def _group_earliest_source(group: dict) -> float:
        clips = group.get("source_clips", [])
        if not clips:
            return float("inf")
        return min(c.get("source_start", float("inf")) for c in clips)

    sorted_groups = sorted(groups, key=_group_earliest_source)

    # Renumber group_index to match new order
    for i, g in enumerate(sorted_groups):
        g["group_index"] = i

    original_order = [g.get("reel_summary", {}).get("title", f"group_{i}") for i, g in enumerate(groups)]
    new_order = [g.get("reel_summary", {}).get("title", f"group_{i}") for i, g in enumerate(sorted_groups)]
    if original_order != new_order:
        logger.info(f"STORY FLOW: reordered groups {original_order} → {new_order}")

    return sorted_groups


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
    from backend.config import AI_PROVIDER

    if not OPENCODE_API_KEY:
        raise RuntimeError(
            "OPENCODE_API_KEY not set. Cannot run LLM analysis."
        )
    model = OPENCODE_MODEL

    logger.info(f"Calling LLM ({stage_name}) provider={AI_PROVIDER} model={model}")
    if reporter:
        reporter.update_analyzer_phase(stage_name, "running")
    from backend.providers.llm import call_llm
    try:
        raw_content = call_llm(
            messages=messages,
            model=model,
            api_key=OPENCODE_API_KEY or "",
            base_url=OPENCODE_BASE_URL,
            temperature=0.1,
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

        if not _response_has_json(raw_content):
            preview = raw_content[:300] if raw_content else "<empty>"
            logger.warning(
                f"LLM returned no JSON ({stage_name}) provider={AI_PROVIDER} model={model} "
                f"len={len(raw_content)} preview={preview}"
            )
            raise ValueError(f"No JSON object in LLM response (len={len(raw_content)})")
    except Exception as e:
        if reporter:
            reporter.update_analyzer_phase(stage_name, "error", error=str(e)[:300])
        raise

    if reporter:
        reporter.update_analyzer_phase(stage_name, "done", progress=100)
    return raw_content.strip()


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


def _response_has_json(text: str) -> bool:
    """Check if *text* contains at least one JSON object or array."""
    t = text.strip()
    if not t:
        return False
    t = re.sub(r"```[\s\S]*?```", "", t)
    t = re.sub(r"<thinking>[\s\S]*?</thinking>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<think>[\s\S]*?</think>", "", t, flags=re.IGNORECASE)
    t = t.strip()
    return "{" in t or "[" in t


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
    - 6-20 min  → 7 groups max
    - 21-30 min → 9 groups max
    - 31-47 min → 12 groups max
    - >47 min   → 15 groups max
    """
    minutes = source_duration_seconds / 60.0
    if minutes < 6:
        return 1
    if minutes <= 20:
        return 7
    if minutes <= 30:
        return 9
    if minutes <= 47:
        return 12
    return 15


def _compute_group_count_floor(source_duration_seconds: float) -> int:
    """Minimum groups the LLM must produce.

    Duration-based floors:
    - <6 min    → 1 group
    - 6-20 min  → 4 groups
    - 21-30 min → 5 groups
    - 31-47 min → 6 groups
    - >47 min   → 7 groups
    """
    minutes = source_duration_seconds / 60.0
    if minutes < 6:
        return 1
    if minutes <= 20:
        return 4
    if minutes <= 30:
        return 5
    if minutes <= 47:
        return 6
    return 7


# ─────────────────────────────────────────────────────────────────────────────
# Unified story planner prompt — one prompt, genre rules injected.
#
# The old approach had 5 separate full prompt functions per genre (350+ lines).
# The new approach: ONE prompt with a genre-specific RULES section injected.
# The story arc (hook → escalation → payoff) is universal. What changes per
# genre is: what makes a good HOOK, what makes a good PAYOFF, what to AVOID,
# and the TONE. Those are short rule blocks in _GENRE_PLANNING_RULES.
# ─────────────────────────────────────────────────────────────────────────────

_GENRE_PLANNING_RULES: dict[str, str] = {
    GENRE_GAME_CHALLENGE: """CONTENT-SPECIFIC RULES (game/challenge):
- Start MUST be a STAKES or INTERRUPT moment. The prize reveal, the wildest near-miss, or the dramatic start.
  NEVER an intro/greeting. If someone says the prize amount or the rules, that's your opening.
  SKIP past any "welcome back", "today we're doing", or channel plugs — find the first REAL action moment.
- Escalation: build through near-misses, competitor reactions, mounting pressure. Each beat should feel harder than the last.
- Payoff MUST be the actual winner reveal, elimination announcement, or stunt result. The payoff is the chronologically latest strong moment — not a reaction from earlier.
- NEVER combine two separate game rounds into one unit unless they share a single winner reveal.
- Prioritize blocks with STAKES flags for both start AND payoff selections.
- If a challenge has multiple elimination rounds, each MAJOR elimination can be its own unit.
- Each unit MUST contain blocks with SUBSTANCE (sentences, descriptions, commentary) — not just short exclamations like "Come on!" or "Oh!".
  Check the block text: if most blocks are under 5 words, the unit needs different content.""",

    GENRE_DATING_REALITY: """CONTENT-SPECIFIC RULES (dating/reality):
- Start MUST be the most awkward, charged, or surprising moment. Someone says something shocking, a reaction shot before context, a flirtatious callout, a blunt rejection.
  NEVER an introduction or someone saying their name/age.
- Escalation: build through flirtation, tension, or confrontation. If someone gets roasted, escalate through the roast.
- Payoff MUST be the strongest emotional reaction: rejection, acceptance, embarrassed laugh, definitive punchline.
- DATING and ROAST flags mark the best start AND payoff candidates.
- HOT clips (VULGAR/DATING flags) should anchor the opening position.
- If the same couple/interaction spans multiple scenes, that's ONE unit. Different interactions = different units.""",

    GENRE_ROAST_REACTION: """CONTENT-SPECIFIC RULES (roast/reaction):
- Start MUST be the moment that demands a reaction: shocking claim, brutal opener, or a reaction shot before you see what caused it.
- Escalation: increasingly savage comments, mounting reactions, or growing embarrassment. Each clip more intense than the last.
- Payoff MUST be the hardest punchline, the most brutal callout, or the most shocked reaction. The mic-drop moment, not the setup.
- ROAST and VULGAR flags = best candidates for start and payoff.
- NEVER end on a setup. The viewer must see the REACTION or COMEBACK.
- Rating/review content: start = the most controversial rating; payoff = creator's response or most extreme rating reveal.""",

    GENRE_PODCAST: """CONTENT-SPECIFIC RULES (podcast/interview):
- Start MUST be the most provocative or surprising claim. Something that demands context. "I lost everything" or "That's when I realized no one cared."
  NEVER small talk, introductions, or topic transitions.
- Escalation: the story's context, mounting details, the tension of not yet knowing the resolution.
- Payoff: the insight, the surprising reveal, the "I never knew that" or "oh wow" moment.
- Units should be TOPICALLY unified: one unit = one major story or insight thread.
- Prioritize high word-density blocks (fast-talking, emphatic speech) for openings.
- Questions (Q flag) are strong opening candidates — they create the curiosity gap.""",

    GENRE_VLOG_PERSONAL: """CONTENT-SPECIFIC RULES (vlog/personal story):
- Start MUST be the PEAK EMOTIONAL MOMENT or the most visually dramatic beat — the big reveal, the meeting, the surprise, the "finally" moment.
  NEVER the travel footage, walking around, or "hey guys welcome back." The opening should make someone think "I need to see how this happened."
- Escalation: build through the journey/anticipation leading to the peak. Show the effort, the travel, the near-misses.
- Payoff: the resolution of the personal story — the photo taken, the gift received, the surprise reaction, the emotional climax.
- Vlog content lives and dies on PERSONAL MOMENTS. Prioritize blocks with exclamations, questions, and high energy.
- If there are multiple distinct experiences (e.g. visiting two places), each can be a separate unit.
- The viewer must feel the creator's emotion — cut anything that feels like filler or generic B-roll narration.""",

    GENRE_EXPERIMENT: """CONTENT-SPECIFIC RULES (experiment/what-if):
- Start MUST be the hypothesis, the question, or the most dramatic result teaser. "What happens if you..." or a flash of the final result.
- Escalation: show the setup, the process, the building tension. Each step should raise the question "will it work?"
- Payoff MUST be the actual result — the answer to the question. The moment of truth, the reveal, the success or failure.
- Prioritize STAKES flags — experiments with risk or stakes are more compelling.
- NEVER end before the result. An incomplete experiment is not a unit.
- Social experiments: treat the subjects' reactions as escalation; the conclusion/reveal is the payoff.""",

    GENRE_SPORTS_FITNESS: """CONTENT-SPECIFIC RULES (sports/fitness):
- Start MUST be the most spectacular play, the peak action moment, or the stakes (championship, record attempt).
  NEVER warm-up footage, pre-game interviews, or walking to the field.
- Escalation: build through increasingly intense plays, near-misses, score changes, or training progression.
- Payoff: the winning moment, the record broken, the final score, or the transformation result (for fitness).
- STAKES flags mark decisive moments — use them for payoffs.
- Crowd/social proof blocks make excellent escalation beats.
- Sports content needs FAST PACING. Prefer shorter clips with higher energy density.""",

    GENRE_COMEDY_SKETCH: """CONTENT-SPECIFIC RULES (comedy/sketch):
- Start MUST be the setup of the joke — the premise that creates the comedic expectation. Or a flash-forward to the punchline without context.
- Escalation: the absurdity builds, the misunderstanding deepens, the situation spirals.
- Payoff MUST be the punchline, the big reveal, or the climactic comedic moment. The laugh should be at the end.
- Timing is everything. Don't cut a beat too early or too late — the comedic beat must land.
- If a sketch has multiple punchlines, the BIGGEST one is the payoff. Smaller ones are escalation.
- NEVER explain the joke in escalation — let the visual/dialogue do it.""",

    GENRE_TUTORIAL: """CONTENT-SPECIFIC RULES (tutorial/how-to):
- Start MUST be the finished result or the specific problem being solved. Show WHY someone should watch.
  NEVER "hey guys today we're going to..." The opening is the payoff shown first.
- Escalation: the most interesting or surprising steps. Skip boring setup — show the transformation points.
- Payoff: the completed result, the before/after, the final product. The "wow it actually worked" moment.
- Prioritize steps where something visibly changes — cutting, transforming, assembling.
- Skip intro chatter, sponsor segments, and long explanations. Keep only the doing.""",

    GENRE_MUSIC_PERF: """CONTENT-SPECIFIC RULES (music/performance):
- Start MUST be the most powerful vocal moment, the best riff, or the crowd's peak reaction. Drop the viewer into the climax, not the intro.
- Escalation: the build-up, the verse that leads to the chorus, the crowd's growing energy.
- Payoff: the final chorus, the high note, the crowd eruption, or the performance's emotional peak.
- Energy and rhythm matter most. Prefer blocks with high speech_energy and exclamation flags.
- NEVER start with tuning, setup, or "testing 1-2-3." Start mid-performance.""",

    GENRE_NEWS_DOC: """CONTENT-SPECIFIC RULES (news/documentary):
- Start MUST be the most shocking fact, the key revelation, or the central question. Name the event or the stakes plainly.
- Escalation: build through evidence, testimony, developing events. Each beat adds new information.
- Payoff: the conclusion, the verdict, the consequence, or the unanswered question that haunts.
- STAKES flags mark the most consequential moments — use them.
- Tone: calm urgency. Let the facts carry the weight, not hype.
- NEVER editorialize. State what happened, show what happened.""",
}

# Fallback rules for any genre not in the dictionary (including GENRE_GENERAL)
_GENRE_PLANNING_RULES_DEFAULT = """CONTENT-SPECIFIC RULES (general):
- Start: the moment that creates the strongest curiosity — a question, a dramatic statement, or a visual that demands context. NEVER a greeting, intro, or channel plug.
- Escalation: tension building toward the payoff. Each beat more intense or revealing than the last.
- Payoff: the answer to the curiosity, the climax, the outcome. Whatever makes the viewer think "that was worth watching."
- Read the blocks carefully. Let the CONTENT dictate the arc, not a formula. A cooking video's payoff is the dish; a confession's payoff is the truth; a prank's payoff is the reaction.
- Engagement flags (VULGAR, DATING, ROAST, STAKES) mark high-interest moments — weight them heavily for start and payoff."""


def _prompt_planner_for_genre(
    content_type: str,
    video_title: str,
    video_description: str,
    blocks_text: str,
    source_duration: float,
    min_groups: int,
    max_groups: int,
    usable_hints: str,
    content_identity: ContentIdentity | None = None,
    reel_dur_min: int = 80,
    reel_dur_max: int = 100,
    hook_mode: HookMode = "skip",
) -> str:
    """Build the unified story planner prompt with genre-specific rules injected.

    ONE prompt handles all 12+ genres. The genre-specific rules section tells
    the LLM what makes a good start/escalation/payoff for THIS content type.
    """
    genre_rules = _GENRE_PLANNING_RULES.get(content_type, _GENRE_PLANNING_RULES_DEFAULT)
    identity_context = ""
    if content_identity:
        entity_names = ", ".join(content_identity.entity_names) or "none named"
        identity_context = f"""
IDENTIFIER CONTEXT (semantic guidance only; do not infer timestamps or durations)
Creator: {content_identity.creator_name or "unconfirmed"}
Content format: {content_identity.content_format}
Structure: {content_identity.structure}
Named entities/acts: {entity_names}
Planning notes: {content_identity.planning_notes}
"""

    arc_rules = """5. For each unit's arc: exactly one "start" (position "start"), zero to four "escalation", exactly one "payoff" (position "end"). Max 6 beats total.
   - "start" = introduces the challenge, person, or context — this IS the opening beat. Make the intent specific and curiosity-driving (e.g. "Chef Marco's ridiculous 60-second plating challenge" not just "introduce the scene").
   - "escalation" = builds tension toward the payoff
   - "payoff" = the resolution or reveal"""
    arc_example = """        {{"beat": "start", "position": "start", "flags": [], "intent": "specific curiosity-driving opening moment"}},
        {{"beat": "escalation", "position": "any", "flags": [], "intent": "..."}},
        {{"beat": "payoff", "position": "end", "flags": [], "intent": "..."}}"""

    return f"""You are an editorial structure planner for YouTube Shorts.
Your ONLY job: identify the strongest standalone story units worth turning into individual Shorts.
You NEVER pick timestamps, clip ranges, or narration. Another system selects the exact clips.
You choose WHERE each story lives (a region of the video) and WHAT beats it needs to work.

SOURCE
Title: {video_title}
Description: {video_description[:500000]}
Duration: {source_duration:.1f}s
Content type detected: {content_type}
TARGET REEL DURATION: {reel_dur_min}-{reel_dur_max}s per reel. Plan units that contain enough content for this duration range.
MIN CONTENT DURATION: {reel_dur_min}s — each unit must have at least this much usable content.
MAX CONTENT DURATION: {reel_dur_max}s — each unit should not exceed this much content.
Produce {min_groups}-{max_groups} units. Fewer strong stories always beats more weak ones.
{identity_context}

CRITICAL FULL VIDEO MANDATE:
- You MUST analyze the ENTIRE video from 0.0s to {source_duration:.1f}s.
- The climax, winner reveal, and final resolution of any story ALWAYS lie near the end of the video.
- Never stop at the first half of the video. The story unit MUST encompass the full setup-to-climax journey.

CHALLENGE NUMBERING:
- For competition/challenge content: number each unit by the challenge order in the video (e.g. "Challenge 1", "Challenge 2", etc.)
- Use the block timestamps to identify where each challenge begins and ends
- Each unit should cover one complete challenge from start to finish

SEMANTIC BLOCKS (pre-scored by Python -- imp=importance 0-100, flags mark engagement):
{blocks_text}

PRECOMPUTED USABLE SECONDS BY REGION:
{usable_hints}

UNIVERSAL STORY ARC RULES
1. A unit is ONLY worth a reel if it has a complete arc: a start that introduces the scene,
   escalation that builds tension, and a payoff that resolves it.
2. Choose region from where the unit's story lives: "early" = 0-25%, "mid" = 25-75%, "late" = 75-100%.
   Multiple units may share a region.
3. HARD RULE: the unit with priority=1 MUST be "early". Its reel is the first one the viewer sees.
4. priority=1 means must-include, 2 nice-to-have, 3 bonus.
{arc_rules}
6. flags tells the clip picker which engagement signals mark each beat's moment: VULGAR, DATING, ROAST, STAKES.
7. intent: one short phrase describing what the beat must deliver. The start beat's intent is especially important — it must open the reel with a specific, intriguing moment.
8. Blocks with CUR_GAP (curiosity gap), INTERRUPT (pattern interrupt), SOCIAL (social proof), or VIRAL (viral trigger) flags are high-retention moments -- prioritize them.

{genre_rules}

OUTPUT -- STRICT JSON ONLY (no timestamps anywhere, no reasoning text outside JSON)
{{
  "video_type": "{content_type}",
  "units": [
    {{
      "unit_id": 0,
      "name": "short unit name",
      "priority": 1,
      "region": "early|mid|late",
      "arc": [
{arc_example}
      ]
    }}
  ]
}}"""


_GENRE_ENTITY_HINTS: dict[str, str] = {
    "comedy_sketch": "Fast pacing. Short clips (3-6s). Each entity reel should feel like a rapid-fire showcase. Payoffs can be quick.",
    "roast_reaction": "Rapid cuts between personalities. SHORT payoffs natural. Entity reels should feel punchy.",
    "sports_fitness": "Energy-driven. Escalation clips build tension. Entity reels showcase individual performances.",
    "game_challenge": "Competition energy. Each entity is a contestant. Escalation = building stakes. Payoff = result.",
    "podcast_conversational": "Topic-driven, not person-driven. Group by topic flow, not by speaker. Each reel covers one discussion thread.",
    "tutorial_educational": "Step-by-step. Each entity reel covers one technique or concept. Payoff = completed result.",
    "documentary_narrative": "Story-driven. Entity reels follow narrative arcs. Payoff = resolution or emotional peak.",
}


def _entity_genre_section(content_type: str) -> str:
    """Return genre-specific guidance for entity planner if content_type is known."""
    hint = _GENRE_ENTITY_HINTS.get(content_type, "")
    if hint:
        return f"CONTENT TYPE: {content_type}\nGENRE GUIDANCE: {hint}\n"
    return ""


def _prompt_planner_entity(
    video_title: str,
    video_description: str,
    entity_segments: list,
    blocks: list,
    source_duration: float,
    content_identity: ContentIdentity | None = None,
    reel_dur_min: int = 80,
    reel_dur_max: int = 100,
    content_type: str = "",
    min_groups: int = 3,
    max_groups: int = 6,
) -> str:
    """Build the entity-mode planner prompt.

    Python pre-seeds one candidate per entity segment. The LLM only returns
    merge groups of related segments. No timestamps or durations are exposed
    as LLM decisions.
    """
    # Build segment descriptions with speaker IDs, usable content time, and top blocks
    block_lookup = {b.block_id: b for b in blocks}
    segment_lines = []
    for seg in entity_segments:
        entity_name = seg.entity_name or "unnamed"
        duration = seg.end - seg.start
        block_count = len(seg.block_ids)
        # Compute usable content time: sum of block durations within this segment
        usable = sum(
            block_lookup[bid].end - block_lookup[bid].start
            for bid in seg.block_ids if bid in block_lookup
        )
        evidence_str = ", ".join(seg.evidence) if seg.evidence else "none"
        speakers_str = ", ".join(seg.speaker_ids) if seg.speaker_ids else "none"
        seg_blocks = [block_lookup[bid] for bid in seg.block_ids if bid in block_lookup]
        seg_blocks.sort(key=lambda b: b.importance, reverse=True)
        top_blocks = seg_blocks[:3]
        block_details = []
        for tb in top_blocks:
            flags = ", ".join(
                f for f, has in (
                    ("STAKES", tb.has_stakes), ("VULGAR", tb.has_vulgarity),
                    ("ROAST", tb.has_roast), ("CUR_GAP", tb.has_curiosity_gap),
                    ("VIRAL", tb.has_viral_trigger), ("INTERRUPT", tb.has_pattern_interrupt),
                ) if has
            ) or "no flags"
            text = (tb.text or "").strip().replace("\n", " ")
            if len(text) > 100:
                text = text[:100].rstrip() + "..."
            wps = tb.word_density if tb.word_density > 0 else 0.0
            block_details.append(
                f"    B{tb.block_id} [{tb.start:.0f}-{tb.end:.0f}s] imp={tb.importance:.0f} wps={wps:.1f} [{flags}]: \"{text}\""
            )
        details_str = "\n".join(block_details) if block_details else "    (no blocks)"
        segment_lines.append(
            f"  {seg.entity_segment_id}: \"{entity_name}\" | "
            f"{seg.start:.1f}-{seg.end:.1f}s ({duration:.1f}s, ~{usable:.0f}s usable) | "
            f"{block_count} blocks | speakers: {speakers_str} | evidence: {evidence_str}\n"
            f"    TOP BLOCKS (highest importance first):\n{details_str}"
        )
    segments_text = "\n".join(segment_lines)

    # Build speaker-id summary: which segments share a speaker
    speaker_to_segments: dict[str, list[str]] = {}
    for seg in entity_segments:
        for spk in seg.speaker_ids:
            speaker_to_segments.setdefault(spk, []).append(seg.entity_segment_id)
    # Only show speakers that appear in 2+ segments (potential same-person merges)
    shared_speakers = {spk: segs for spk, segs in speaker_to_segments.items() if len(segs) >= 2}
    speaker_section = ""
    if shared_speakers:
        speaker_lines = []
        for spk, seg_ids in sorted(shared_speakers.items()):
            speaker_lines.append(f"  {spk}: {', '.join(seg_ids)}")
        speaker_section = f"""
SHARED SPEAKERS (segments with the same speaker ID likely feature the same person):
{chr(10).join(speaker_lines)}
"""

    identity_section = ""
    if content_identity:
        entity_names = ", ".join(content_identity.entity_names) or "none named"
        identity_section = f"""
IDENTIFIER CONTEXT
Structure: {content_identity.structure}
Named entities: {entity_names}
Planning notes: {content_identity.planning_notes}
"""

    return f"""You are a reel grouping assistant for multi-entity challenge content.
Your ONLY job: group the pre-identified entity segments into standalone Short reels.
You NEVER pick timestamps, clip ranges, or narration. Another system selects the exact clips.
You decide WHICH segments belong together in one reel.

SOURCE
Title: {video_title}
Description: {video_description[:500000]}
Duration: {source_duration:.1f}s
TARGET: {min_groups}-{max_groups} reels, each 90-100s long. Each reel must be ONE isolated challenge.
{identity_section}
{_entity_genre_section(content_type)}
ENTITY SEGMENTS (pre-identified by Python — DO NOT infer new segments):
Each segment is a short window of one person, contestant, or act.
{segments_text}
{speaker_section}
RULES
1. Create {min_groups}-{max_groups} reel groups (based on content).
2. Each segment appears in exactly one reel group.
3. Group segments that are part of the SAME challenge/contest into one reel.
4. DIFFERENT challenges/contests MUST be in SEPARATE groups (never mix).
5. Each group must have enough content for 90+ seconds of clips.
6. Segments sharing the same speaker ID or entity name are the same person — merge them.
7. Consider timing: segments close in time are likely the same challenge.
8. No timestamps or durations — Python owns all numbers.
9. Each reel should have: introduction, tension building, and payoff/resolution.
10. Use the TOP BLOCKS to judge content quality: segments with STAKES/VIRAL/INTERRUPT flags are high-value.
11. Combine SHORT + MEDIUM + LONG clips within each group to fill the reel.

OUTPUT — STRICT JSON ONLY (no timestamps, no reasoning text outside JSON)
{{
  "video_type": "entity",
  "merge_groups": [
    {{"segment_ids": ["entity-1", "entity-2"]}},
    {{"segment_ids": ["entity-3"]}},
    {{"segment_ids": ["entity-4", "entity-5", "entity-6"]}}
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Stage prompts — legacy LLM path (PLAN_MODE=llm)
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
Description: {video_description[:500000]}
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
   - Resolution: the actual outcome, winner reveal, final punchline, or strongest reaction (REQUIRED)
3. A unit is ONLY a group if it has a complete arc (setup → peak → resolution).
4. If a section has setup but no payoff, merge it with adjacent content that provides the payoff.
5. Never split a challenge/contest if the climax immediately follows — keep them as one unit.
6. Never combine two unrelated climaxes into one group — they are separate stories.
7. Would someone watch this as a standalone Short? If not, merge it.
8. Avoid groups that are just "introduction" or "setup" with no payoff.
9. The max groups ({max_groups}) is a hard limit. Fewer strong groups always beats more weak ones.
10. Produce {min_groups}-{max_groups} groups if content supports it. ALWAYS produce at least {min_groups} group(s).
11. Every group MUST capture its natural payoff.

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
    sa = structure.get("structure_analysis", structure)
    units = sa.get("identified_units", structure.get("identified_units", []))
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
Block flags VULGAR / DATING / ROAST / STAKES / CUR_GAP / INTERRUPT / VIRAL mark high-engagement moments — weight them heavily.

CRITICAL RULE — NO SOURCE OVERLAP
Every source timestamp may belong to ONE reel only.
If reel A uses 20.0-35.0s, NO other reel may touch 20.0-35.0s.

CLIP SELECTION PRIORITIES (ranked)
1. Payoff / Punchline — result, winner reveal, outcome, or final reaction (MANDATORY for last clip)
2. Curiosity gap — "What happens next?" / "No way..." / unexpected visuals (look for CUR_GAP flag)
3. Emotional peak — crowd gasps, player reactions, shock, celebration
4. Visual action — motion, spectacle, physical moments
5. Stakes — what is at risk, what could go wrong
6. Surprise — twist, reversal, unexpected result

CLIP STORY ARC — every reel must contain within its clips:
- Hook: the curiosity trigger (first clip) — look for VIRAL, CUR_GAP, INTERRUPT flags
- Escalation: tension building (middle clips)
- Peak: the climax or reveal (later clip)
- Resolution: the outcome — must be the LATEST-OCCURRING timestamp clip in the group

DURATION RULES (HARD)
- Each group estimated_duration = sum of (source_end - source_start) of its clips.
- Must be between {dur_min} and {dur_max} seconds.
- Never <3.0s per clip.

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
- roast: playful jab at what's happening. Not mean, just sharp.
- brutally_honest: say what everyone's thinking but won't admit.
- friendly: warm, excited, rooting for them.
- sarcastic: dry wit, understated reactions.
- hype: genuine excitement, but specific — not just "LET'S GOOO".
- deadpan: flat delivery, maximum impact.

BANNED filler for commentaries: "This is crazy", "No way", "Insane", "Literally dying", "I can't even"
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
- confusing story / missing payoff in any group (ensure the final clip is the LATEST timestamp clip showing the outcome/punchline)
- dead air / low-value clips (repeated explanations, waiting, filler)
- missed emotional peaks (strongest moment buried in middle instead of final 30%)
- duration outside {dur_min}-{dur_max}s
- groups that don't stand alone (require context from another reel)
- hook clips that are introductions instead of curiosity triggers
- payoff clips that are not the latest-occurring source timestamp in the group (fix this)

Return the FULL revised plan as STRICT JSON with the same schema:
structure_analysis (if present), ranked_segments (optional), reel_groups (with source_clips + narration_events), explanations.
Only change what needs fixing. Keep strong parts intact."""


# ─────────────────────────────────────────────────────────────────────────────
# Completeness critic — post-execution LLM quality gate
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_completeness_critic(groups: list[dict]) -> str:
    """Build the completeness critic prompt.

    The critic is a narrow read-only LLM call: it receives locked clips,
    arc labels, and final narration only.  It returns a per-group verdict
    with no ability to alter clips, numbers, groups, or narration.
    """
    compact = []
    for g in groups:
        clips_summary = []
        for c in g.get("source_clips", []):
            clips_summary.append({
                "beat": c.get("beat") or c.get("_beat", "unknown"),
                "source_start": c.get("source_start"),
                "source_end": c.get("source_end"),
                "text_hint": (c.get("text") or c.get("reason") or "")[:80],
            })
        narr = [
            {"event_type": e.get("event_type"), "text": (e.get("text") or "")[:120]}
            for e in g.get("narration_events", [])
            if isinstance(e, dict)
        ]
        compact.append({
            "group_index": g.get("group_index"),
            "arc": g.get("group_reasoning", "")[:120],
            "clips": clips_summary,
            "narration": narr,
            "estimated_duration_seconds": g.get("estimated_duration_seconds"),
        })
    payload = json.dumps(compact, ensure_ascii=False, indent=2)
    return f"""You are a completeness critic for YouTube Shorts reels.
Review each group and decide if it is self-contained and editorially complete.

A group is INCOMPLETE if:
- The clip sequence does not form a coherent story arc (hook → escalation → payoff).
- The payoff is missing or is not the latest-occurring source-time clip.
- The group relies on context from another group to make sense.
- There is no narration and the clips alone do not convey a story.
- Key emotional or informational beats are absent.

Do NOT suggest changes to clips, timestamps, or narration.
Do NOT invent new groups.

Groups:
{payload}

Return STRICT JSON:
{{
  "groups": [
    {{"group_index": <int>, "complete": <bool>, "reason": "<short explanation>"}}
  ]
}}"""


def _completeness_critic(
    groups: list[dict],
    progress_cb: Any | None = None,
    reporter: Any = None,
    interactions: Any | None = None,
) -> list[dict[str, Any]]:
    """Run the completeness critic on finalized groups.

    Returns the original groups unchanged on any failure (LLM error,
    malformed output, timeout).  On success, attaches a
    ``completeness_critic`` key to each group with the verdict.

    Batches when >30 groups to avoid input truncation.
    """
    from backend.config import FAST_MODE
    if FAST_MODE:
        logger.info("FAST_MODE: skipping completeness critic")
        return groups

    if progress_cb:
        progress_cb("Completeness critic...", 92)

    BATCH_SIZE = 30
    all_verdicts: dict[int, dict] = {}

    try:
        for batch_start in range(0, len(groups), BATCH_SIZE):
            batch = groups[batch_start:batch_start + BATCH_SIZE]
            raw = _call_llm(
                [
                    {"role": "system", "content": "Respond with ONLY valid JSON."},
                    {"role": "user", "content": _prompt_completeness_critic(batch)},
                ],
                progress_cb, reporter, interactions,
                stage_name="completeness_critic",
                max_tokens=8192,
            )
            result = _parse_json_response(raw)
            verdicts = result.get("groups", [])
            if isinstance(verdicts, list):
                for v in verdicts:
                    if isinstance(v, dict) and "group_index" in v:
                        all_verdicts[v["group_index"]] = v

        if not all_verdicts:
            logger.warning("Completeness critic returned no verdicts — keeping draft")
            return groups

        flagged: list[int] = []
        unevaluated: list[int] = []
        for g in groups:
            idx = g.get("group_index")
            verdict = all_verdicts.get(idx)
            if verdict:
                g["completeness_critic"] = {
                    "complete": bool(verdict.get("complete", True)),
                    "reason": str(verdict.get("reason", ""))[:500],
                }
                if not verdict.get("complete", True):
                    flagged.append(idx)
            else:
                unevaluated.append(idx)
                g["completeness_critic"] = {
                    "complete": None,
                    "reason": "Not evaluated by critic (batch limit)",
                }

        if flagged:
            logger.info(f"Completeness critic flagged {len(flagged)} group(s): {flagged}")
        if unevaluated:
            logger.warning(f"Completeness critic: {len(unevaluated)} group(s) not evaluated: {unevaluated[:10]}...")
        else:
            logger.info("Completeness critic: all groups evaluated")

    except Exception as e:
        logger.warning(f"Completeness critic failed, keeping draft: {e}")

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Narration writer style — genre-aware
# ─────────────────────────────────────────────────────────────────────────────

def _writer_style_section(video_type: str) -> str:
    """Style guidance for the narration writer, adapted to the content type.

    Data-driven: looks up GENRE_* constant or legacy video_type string in
    _GENRE_WRITER_STYLES. Falls back to a versatile general style.
    """
    t = video_type.strip().lower()

    # Map legacy video_type strings to GENRE_* constants
    _LEGACY_MAP = {
        "challenge": GENRE_GAME_CHALLENGE, "game": GENRE_GAME_CHALLENGE, "stunt": GENRE_GAME_CHALLENGE,
        "dating": GENRE_DATING_REALITY, "reality": GENRE_DATING_REALITY,
        "roast": GENRE_ROAST_REACTION, "reaction": GENRE_ROAST_REACTION, "review": GENRE_ROAST_REACTION, "rating": GENRE_ROAST_REACTION,
        "podcast": GENRE_PODCAST, "interview": GENRE_PODCAST, "conversation": GENRE_PODCAST, "debate": GENRE_PODCAST,
        "tutorial": GENRE_TUTORIAL, "guide": GENRE_TUTORIAL, "howto": GENRE_TUTORIAL, "educational": GENRE_TUTORIAL, "how_to": GENRE_TUTORIAL,
        "news": GENRE_NEWS_DOC, "documentary": GENRE_NEWS_DOC, "investigation": GENRE_NEWS_DOC, "current_affairs": GENRE_NEWS_DOC,
        "vlog": GENRE_VLOG_PERSONAL, "travel": GENRE_VLOG_PERSONAL, "personal": GENRE_VLOG_PERSONAL,
        "experiment": GENRE_EXPERIMENT, "science": GENRE_EXPERIMENT, "prank": GENRE_EXPERIMENT,
        "comedy": GENRE_COMEDY_SKETCH, "sketch": GENRE_COMEDY_SKETCH, "skit": GENRE_COMEDY_SKETCH, "funny": GENRE_COMEDY_SKETCH,
        "sports": GENRE_SPORTS_FITNESS, "fitness": GENRE_SPORTS_FITNESS, "gym": GENRE_SPORTS_FITNESS, "training": GENRE_SPORTS_FITNESS,
        "music": GENRE_MUSIC_PERF, "performance": GENRE_MUSIC_PERF, "concert": GENRE_MUSIC_PERF, "cover": GENRE_MUSIC_PERF,
    }
    genre = _LEGACY_MAP.get(t, t)  # try legacy map, else use as-is

    return _GENRE_WRITER_STYLES.get(genre, _GENRE_WRITER_STYLES[GENRE_GENERAL])


_GENRE_WRITER_STYLES: dict[str, str] = {
    GENRE_GAME_CHALLENGE: (
        "STYLE: Hype commentator -- energetic, prize-obsessed, always building toward the reveal.\n"
        "HOOK (6-10 words): Open on the stakes or the wildest moment. Name the prize.\n"
        "  Examples: '$10,000 to whoever survives the longest.' / 'He is one step from the million.'\n"
        "START (6-10 words): When no hook is present — plain scene identification. Name who/what/where.\n"
        "  Examples: 'Four contestants. One challenge. Ten thousand dollars.'\n"
        "COMMENTARY: reference challenge state (who leads, who struggles), build pre-payoff tension.\n"
        "PERSONAS: hype (near-miss excitement), deadpan (flat reaction to chaos), brutally_honest (who loses), sarcastic (strategy callout).\n"
        "BANNED: 'That was amazing', 'No way', 'Insane', 'Let us go', 'Watch what happens', 'You won't believe'\n"
    ),
    GENRE_DATING_REALITY: (
        "STYLE: Reality TV narrator -- OBSESSED with the drama. Read the room. Notice micro-expressions.\n"
        "HOOK (6-10 words): Open on chemistry, awkwardness, or the most charged moment.\n"
        "  Examples: 'She already knows she does not like him.' / 'Most painfully shy intro ever.'\n"
        "START (6-10 words): When no hook is present — plain identification of the people and setting.\n"
        "  Examples: 'Two strangers. One dinner. Zero chemistry.'\n"
        "COMMENTARY: read the subtext, tease the verdict without showing it.\n"
        "PERSONAS: roast (playful cringe callout), friendly (rooting for chemistry), deadpan (obvious tension), sarcastic (knows the ending).\n"
        "BANNED: 'That was amazing', 'No way', 'Insane', 'Literally dying', 'Watch what happens'\n"
    ),
    GENRE_ROAST_REACTION: (
        "STYLE: Sharp observer -- dry, never surprised by how brutal it gets. Clocks the best burns.\n"
        "HOOK (6-10 words): Most brutal or surprising moment -- the moment BEFORE context.\n"
        "  Examples: 'He just called her entire personality boring.' / 'A 3 out of 10. In front of everyone.'\n"
        "START (6-10 words): When no hook is present — plain scene setup.\n"
        "  Examples: 'A comedy roast. The panel looks nervous.'\n"
        "COMMENTARY: mid-roast observation, setup for the comeback.\n"
        "PERSONAS: deadpan (maximum understatement), roast (plays along), brutally_honest (quiet part loud), sarcastic (dry chaos commentary).\n"
        "BANNED: 'That was amazing', 'No way', 'Insane', 'Literally dying', 'Watch what happens'\n"
    ),
    GENRE_PODCAST: (
        "STYLE: Sharp listener sharing the conversation's best parts. Quote the substance, not the banter.\n"
        "HOOK (6-10 words): Drop into the most revealing claim or story beat.\n"
        "START (6-10 words): When no hook is present — identify the speaker and topic.\n"
        "  Examples: 'The host brings up the one thing nobody expected.'\n"
        "COMMENTARY: reference the specific insight, anecdote, or tension.\n"
        "PERSONAS: curious (detail-focused), skeptical (pokes weak claims), moved (reacts to story), dry (understated humor).\n"
        "BANNED: 'That was amazing', 'No way', 'Insane', 'So interesting', 'Watch what happens'\n"
    ),
    GENRE_VLOG_PERSONAL: (
        "STYLE: Best friend energy -- warm, personal, emotionally invested in the creator's journey.\n"
        "HOOK (6-10 words): The peak emotional moment or the big reveal. Make viewer feel envy/excitement.\n"
        "  Examples: 'Three years of trying. And then this happened.' / 'He had no idea I was there.'\n"
        "START (6-10 words): When no hook is present — plain introduction of the creator's activity.\n"
        "  Examples: 'Today we are exploring an abandoned mall downtown.'\n"
        "COMMENTARY: reference the personal stakes, the journey, the emotion of THIS specific moment.\n"
        "PERSONAS: friendly (warm, invested), hype (genuine excitement at payoff), brutally_honest (acknowledges difficulty), sarcastic (self-aware creator energy).\n"
        "BANNED: 'Watch what happens', 'You won't believe', 'Wait for it', 'So cool', 'What a moment'\n"
    ),
    GENRE_EXPERIMENT: (
        "STYLE: Curious scientist energy -- genuinely fascinated by the result, builds anticipation.\n"
        "HOOK (6-10 words): State the question or flash the most dramatic result.\n"
        "  Examples: 'Nobody thought this would actually work.' / 'What happens when you freeze fire?'\n"
        "START (6-10 words): When no hook is present — state the experiment setup plainly.\n"
        "  Examples: 'We tested five different adhesives on wet concrete.'\n"
        "COMMENTARY: reference the process, the uncertainty, the mounting tension before the result.\n"
        "PERSONAS: curious (genuinely fascinated), skeptical (did not expect that), hype (result excitement), deadpan (understates the chaos).\n"
        "BANNED: 'Watch what happens', 'You won't believe', 'So cool', 'Insane'\n"
    ),
    GENRE_SPORTS_FITNESS: (
        "STYLE: Sports commentator -- fast, action-focused, stakes-aware. Every word matches the energy.\n"
        "HOOK (6-10 words): The peak action or the stakes. Name the score/record/moment.\n"
        "  Examples: 'Down by two. Thirty seconds left.' / 'He has been training for this his entire life.'\n"
        "START (6-10 words): When no hook is present — identify the athlete and the event.\n"
        "  Examples: 'Semifinal. Second set. Tiebreak point.'\n"
        "COMMENTARY: reference the play, the stakes, the momentum shift.\n"
        "PERSONAS: hype (action excitement), deadpan (flat reaction to spectacular play), brutally_honest (calls the mistake), sarcastic (dry commentary on obvious outcome).\n"
        "BANNED: 'That was amazing', 'No way', 'Insane', 'Let us go', 'Watch what happens'\n"
    ),
    GENRE_COMEDY_SKETCH: (
        "STYLE: Setup artist -- you set up the punchline without stepping on it. Timing is everything.\n"
        "HOOK (6-10 words): The premise or a flash of the punchline without context.\n"
        "  Examples: 'He did not realize the camera was still on.' / 'This is why you do not ask twice.'\n"
        "START (6-10 words): When no hook is present — set the scene plainly.\n"
        "  Examples: 'A guy walks into a bar with a parrot on his shoulder.'\n"
        "COMMENTARY: build the absurdity, never explain the joke.\n"
        "PERSONAS: deadpan (flat delivery for maximum impact), sarcastic (dry observation), friendly (laughing along), roast (playful).\n"
        "BANNED: 'This is so funny', 'I am dying', 'No way', 'Insane', 'Watch what happens'\n"
    ),
    GENRE_TUTORIAL: (
        "STYLE: Sharp, friendly expert -- clear, direct, specific. Name steps and results exactly.\n"
        "HOOK (6-10 words): State the outcome or the problem being solved.\n"
        "START (6-10 words): When no hook is present — state what is being taught.\n"
        "  Examples: 'How to fix a wobbly chair in under two minutes.'\n"
        "COMMENTARY: reference a specific step, mistake, or result.\n"
        "PERSONAS: expert (confident, precise), patient (explains clearly), honest (flags pitfalls), amused (dry humor).\n"
        "BANNED: 'That was amazing', 'No way', 'Insane', 'So cool', 'Watch what happens'\n"
    ),
    GENRE_MUSIC_PERF: (
        "STYLE: Music critic who is genuinely moved -- specific about what makes THIS moment great.\n"
        "HOOK (6-10 words): The peak vocal/instrumental moment or the crowd eruption.\n"
        "  Examples: 'The entire room went silent for this note.' / 'Three chords. That is all he needed.'\n"
        "START (6-10 words): When no hook is present — identify the performer and song.\n"
        "  Examples: 'A covers singer. An empty bar. One song that changes everything.'\n"
        "COMMENTARY: reference the specific musical moment, the energy shift, the crowd.\n"
        "PERSONAS: moved (genuine emotion), hype (energy match), deadpan (understates brilliance), curious (notes the technique).\n"
        "BANNED: 'That was amazing', 'No way', 'Insane', 'So good', 'Watch what happens'\n"
    ),
    GENRE_NEWS_DOC: (
        "STYLE: Calm, observational urgency -- understands the weight. States facts plainly.\n"
        "HOOK (6-10 words): Name the event or question the unit answers.\n"
        "START (6-10 words): When no hook is present — state the subject and context.\n"
        "  Examples: 'A small town. A missing person. Three years of questions.'\n"
        "COMMENTARY: reference the specific development, detail, or consequence.\n"
        "PERSONAS: grounded (states what matters), concerned (notes stakes), analytical (explains why), restrained (understates for impact).\n"
        "BANNED: 'That was amazing', 'No way', 'Insane', 'So crazy', 'Watch what happens'\n"
    ),
    GENRE_GENERAL: (
        "STYLE: Witty friend reacting live -- punchy, specific, unexpected. Reference what is on screen.\n"
        "HOOK (6-10 words): Drop the viewer into the most intriguing moment.\n"
        "  Examples: 'This is the worst idea I have ever had.' / 'He has no idea I am here.'\n"
        "START (6-10 words): When no hook is present — plain scene/person identification.\n"
        "  Examples: 'One person. A room full of strangers. A microphone.'\n"
        "COMMENTARY: reference a specific beat of THIS group's arc.\n"
        "PERSONAS: roast (playful jab), brutally_honest (says what everyone thinks), friendly (warm), sarcastic (dry), hype (specific excitement), deadpan (flat, maximum impact).\n"
        "BANNED: 'This is crazy', 'No way', 'Insane', 'Literally dying', 'I can not even', 'Watch what happens', 'You won't believe'\n"
    ),
}


def _writer_opening_event_type(hook_mode: str) -> str:
    """Return the opening event type for the narrator based on hook_mode."""
    return "hook" if hook_mode == "required" else "start"


def _prompt_writer(
    video_title: str,
    groups: list[dict],
    structure: dict,
    blocks_hint: str,
    video_type: str = "other",
    hook_mode: str = "skip",
    reel_dur_min: int = 90,
    reel_dur_max: int = 100,
) -> str:
    groups_json = json.dumps(groups, ensure_ascii=False, indent=2)
    structure_json = json.dumps(structure, ensure_ascii=False, indent=2)
    hint_section = f"\nTOP BLOCKS PER UNIT (Python-scored — anchor your lines to these):\n{blocks_hint}" if blocks_hint else ""

    # Build simplified reel timelines so the writer knows when each beat plays
    timeline_lines = []
    for g in groups:
        clips = g.get("source_clips", [])
        if not clips:
            continue
        clips_sorted = sorted(clips, key=lambda c: c.get("source_start", 0))
        reel_start = clips_sorted[0].get("source_start", 0)
        parts = []
        for c in clips_sorted:
            beat = c.get("beat") or c.get("_beat", "content")
            rel_start = round(c.get("source_start", 0) - reel_start, 1)
            rel_end = round(c.get("source_end", 0) - reel_start, 1)
            parts.append(f"{rel_start}-{rel_end}s:{beat}")
        timeline_lines.append(
            f"  Group {g.get('group_index', '?')}: {' | '.join(parts)}"
        )
    timeline_section = ""
    if timeline_lines:
        timeline_section = f"\nREEL TIMELINES (when each beat plays relative to reel start):\n" + "\n".join(timeline_lines) + "\n"

    return f"""You are a narration writer and reel titler for vertical Shorts.
Your ONLY job: for each group, write the reel_summary and the narration LINE TEXT.
Do NOT choose timestamps, durations, or percentages — Python places every event.
Do NOT change clips, groups, or any number.

Video title: {video_title}
Content type: {video_type}
Hook mode: {hook_mode} ({'generate a curiosity-triggering hook opening' if hook_mode == 'required' else 'use a plain start opening, no hook'})
TARGET REEL DURATION: {reel_dur_min}-{reel_dur_max}s per reel. Write narration that fits this duration range — not too short, not too long.

VIDEO TYPE / STRUCTURE:
{structure_json}

GROUPS (clips already locked — reasons describe what is on screen):
{groups_json}
{hint_section}
{timeline_section}
{_writer_style_section(video_type)}
REEL_SUMMARY
- title: <=60 chars — what makes this Short compelling
- short_description: <=150 chars
- source_understanding: what this unit covers
- narrative_angle: unique emotional framing
- key_moment: the strongest moment — what makes someone stop scrolling (use the payoff beat intent)

OUTPUT — STRICT JSON ONLY (no reel_start, no reel_end, no seconds)
{{
  "reel_groups": [
    {{
      "group_index": 0,
      "group_name": "copy the unit name exactly from the GROUPS input above",
      "reel_summary": {{
        "title": "...",
        "short_description": "...",
        "source_understanding": "...",
        "narrative_angle": "...",
        "key_moment": "..."
      }},
      "narration_events": [
        {{"event_type": "{_writer_opening_event_type(hook_mode)}", "text": "...", "persona": null}},
        {{"event_type": "commentary", "text": "...", "persona": "neutral"}},
        {{"event_type": "commentary", "text": "...", "persona": "neutral"}}
      ]
    }}
  ]
}}

CRITICAL: You MUST return exactly the same number of groups as provided in the GROUPS input.
Each reel_group MUST include "group_name" matching the unit name from the input.
Do NOT reorder groups — return them in the same order as the input.

RULES for event types:
- "hook": Opens the reel with a curiosity trigger or emotional hook (only when hook_mode is "required").
- "start": Introduces the scene/person when no hook is present (plain identification, no dramatic framing). Used when hook_mode is "skip".
- "commentary": Mid-reel observations. Persona is one of: neutral, hype, deadpan, roast, sarcastic, brutally_honest, friendly, curious, skeptical, moved, expert, patient, honest, amused, grounded, concerned, analytical, restrained.
- Only ONE opening event per reel (either hook OR start, never both).
- Always include exactly 2 commentary events per reel."""


# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage orchestration — public entry point
# ─────────────────────────────────────────────────────────────────────────────

def select_reel_plan(
    transcript: list[dict],
    video_title: str,
    video_description: str,
    progress_cb: Callable[[str, float], None] | None = None,
    reporter: Any = None,
    interactions: list[LLMInteraction] | None = None,
    rich_timeline: RichTimeline | None = None,
    source_path: str | None = None,
    source_metadata: SourceMetadata | None = None,
    hook_mode: HookMode = "skip",
    multimodal_signals: MultimodalSignals | None = None,
) -> ReelPlan:
    if progress_cb:
        progress_cb("Building semantic blocks...", 5)
    if reporter:
        # Announce the fixed prefix immediately — the branch-specific tail
        # (story planner / clip selection / narration writer / ...) is
        # appended once entity_grouped and PLAN_MODE are known, below.
        reporter.set_analyzer_phase_plan(
            _phase("semantic_blocks", "identifier", "multimodal_ocr", "content_classification")
        )
        reporter.update_analyzer_phase("semantic_blocks", "running")

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
    if reporter:
        reporter.update_analyzer_phase(
            "semantic_blocks", "done", progress=100,
            detail={"block_count": len(blocks)},
        )

    # ── LLM #0 Identifier — semantic guidance only ──
    if progress_cb:
        progress_cb("Identifying source and visual structure...", 7)
    content_identity, multimodal_signals = _run_preplanning_enrichment(
        video_title,
        video_description,
        source_metadata,
        source_path,
        blocks,
        progress_cb,
        reporter,
        interactions,
        multimodal_signals,
    )

    # ── Python: content type detection (hardcoded, committed early) ──
    if progress_cb:
        progress_cb("Detecting content type...", 8)
    if reporter:
        reporter.update_analyzer_phase("content_classification", "running")
    content_type = detect_content_type(
        video_title,
        video_description,
        blocks,
        identifier_genre=content_identity.detected_genre if content_identity else None,
    )

    # ── Python: entity segmentation (all boundaries and labels are deterministic) ──
    entity_segments, entity_grouped = _segment_entities(
        blocks, rich_timeline, multimodal_signals, content_identity, source_duration
    )

    if reporter:
        from backend.config import FAST_MODE, PLAN_MODE
        reporter.update_analyzer_phase(
            "content_classification", "done", progress=100,
            detail={
                "content_type": content_type,
                "entity_grouped": entity_grouped,
                "entity_segment_count": len(entity_segments),
            },
        )
        # Branch is now known — announce the rest of the roadmap (the fixed
        # prefix was already announced at the top of this function).
        full_plan = _build_analyzer_phase_plan(
            entity_grouped=entity_grouped, plan_mode=PLAN_MODE, fast_mode=FAST_MODE,
        )
        reporter.append_analyzer_phases(full_plan[4:])  # skip the already-announced prefix

    # Augment blocks_text with on-screen text (OCR) if available — helps
    # genre planner identify structured content (rounds, scores, names).
    if multimodal_signals and multimodal_signals.on_screen_text:
        ocr_lines = []
        for ost in multimodal_signals.on_screen_text[:30]:  # cap to avoid prompt bloat
            ocr_lines.append(f"  {ost.timestamp:.1f}s: \"{ost.text}\"")
        if ocr_lines:
            blocks_text += "\n\nON-SCREEN TEXT (OCR — may include scores, round numbers, names):\n" + "\n".join(ocr_lines)

    # ── Entity segment selection: keep top N by content quality ──
    max_segments = max_groups * ENTITY_MAX_SEGMENTS_MULTIPLIER

    # ── Entity pre-merge: combine consecutive segments sharing same speaker ──
    if entity_grouped and entity_segments:
        entity_segments = _premerge_entity_segments(entity_segments, blocks)

    if entity_grouped and len(entity_segments) > max_segments:
        logger.info(
            f"Entity segment selection: {len(entity_segments)} segments exceeds "
            f"{max_segments} — selecting top {max_segments} by content quality"
        )
        entity_segments = _select_top_entity_segments(entity_segments, max_segments, blocks)

    # ── Python: audience worth gate ──
    if progress_cb:
        progress_cb("Assessing audience worth...", 10)
    is_worth, worth_verdict, worth_breakdown = assess_audience_worth(blocks, content_type, source_duration)
    # Soft warning — never a hard stop
    if not is_worth:
        logger.warning(f"AUDIENCE WORTH GATE FAILED: {worth_verdict}")

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
    whole_usable = _usable_duration(blocks, 0.0, source_duration)
    usable_hints = f"  whole_video usable≈{whole_usable:.0f}s\n" + usable_hints

    description = (video_description or "")[:500000]
    logger.info(
        f"MULTI-STAGE PLAN source={source_duration:.1f}s groups={min_groups}-{max_groups} "
        f"blocks={len(blocks)} duration_target={reel_dur_min}-{reel_dur_max}s "
        f"content_type={content_type} worth={is_worth}"
    )

    # ── Dispatch: executor mode — LLM plans, Python picks ──
    from backend.config import PLAN_MODE

    if PLAN_MODE == "executor":
        return _select_reel_plan_executor(
            video_title=video_title,
            video_description=description,
            progress_cb=progress_cb,
            reporter=reporter,
            interactions=interactions,
            rich_timeline=rich_timeline,
            source_duration=source_duration,
            min_groups=min_groups,
            max_groups=max_groups,
            reel_dur_min=reel_dur_min,
            reel_dur_max=reel_dur_max,
            blocks=blocks,
            blocks_text=blocks_text,
            usable_hints=usable_hints,
            content_type=content_type,
            worth_verdict=worth_verdict,
            worth_breakdown=worth_breakdown,
            content_identity=content_identity,
            hook_mode=hook_mode,
            multimodal_signals=multimodal_signals,
            entity_segments=entity_segments,
            entity_grouped=entity_grouped,
        )

    # ── Legacy LLM path ──
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
        max_tokens=65536,
    )
    structure = _parse_json_response(raw1)
    sa = structure.get("structure_analysis", structure)
    logger.info(
        f"STRUCTURE: type={sa.get('video_type')} groups={sa.get('final_group_count')} "
        f"reason={sa.get('reasoning')}"
    )

    if progress_cb:
        progress_cb("Selecting clips...", 45)

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

    sa = structure.get("structure_analysis", structure)
    structure_for_prompt = {
        "structure_analysis": sa,
        "identified_units": sa.get("identified_units", structure.get("identified_units", [])),
    }

    raw2 = _call_llm(
        [
            {"role": "system", "content": "Respond with ONLY valid JSON."},
            {"role": "user", "content": _prompt_clip_planner(
                video_title, structure_for_prompt, blocks_text, source_duration, reel_dur_min, reel_dur_max,
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

    for g in groups:
        g.pop("narration_events", None)

    hook_swaps = _rank_hook_candidates(groups, blocks)
    if hook_swaps > 0:
        logger.info(f"Hook ranking: swapped {hook_swaps} hook clip(s)")

    # ── Enforce chronological story flow ──
    groups = _enforce_story_flow(groups)

    if progress_cb:
        progress_cb("Writing narration...", 65)
    raw3 = _call_llm(
        [
            {"role": "system", "content": "Respond with ONLY valid JSON. Do NOT include any reasoning, thinking, or commentary. Output ONLY the JSON object."},
            {"role": "user", "content": _prompt_narration_writer(video_title, groups)},
        ],
        progress_cb, reporter, interactions, stage_name="narration_writer",
        max_tokens=65536,
    )
    narr_plan = _parse_json_response(raw3)
    narr_by_idx = {
        g.get("group_index", i): g.get("narration_events", [])
        for i, g in enumerate(narr_plan.get("reel_groups", []))
    }
    for i, g in enumerate(groups):
        idx = g.get("group_index", i)
        g["narration_events"] = narr_by_idx.get(idx, [])

    # Repair: fill in reel_start/reel_end for narration events missing timestamps
    # (the LLM narration writer is instructed to include them but may omit them)
    from backend.pipeline.plan_executor import place_narration_events
    for g in groups:
        events = g.get("narration_events", [])
        needs_repair = any(
            not e.get("reel_start") and not e.get("reel_end")
            for e in events if isinstance(e, dict)
        )
        if needs_repair:
            g["narration_events"] = place_narration_events(g)

    draft = {
        "structure_analysis": sa,
        "ranked_segments": clips_plan.get("ranked_segments", []),
        "reel_groups": groups,
        "explanations": clips_plan.get("explanations", []) + [worth_verdict],
        "worth_breakdown": worth_breakdown,
        "content_identity": content_identity.model_dump() if content_identity else None,
        "multimodal_signals": multimodal_signals.model_dump() if multimodal_signals else None,
        "entity_segments": [segment.model_dump() for segment in entity_segments],
        "entity_grouped": entity_grouped,
    }

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
                max_tokens=65536,
            )
            revised = _parse_json_response(raw4)
            revised_groups = revised.get("reel_groups", [])
            if isinstance(revised_groups, list) and revised_groups:
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
                    draft["content_identity"] = content_identity.model_dump() if content_identity else None
                    draft["multimodal_signals"] = multimodal_signals.model_dump() if multimodal_signals else None
                    draft["entity_segments"] = [segment.model_dump() for segment in entity_segments]
                    draft["entity_grouped"] = entity_grouped
                    logger.info("Critic applied revisions")
            else:
                logger.info("Critic returned empty groups — keeping draft")
        except Exception as e:
            logger.warning(f"Critic pass failed, keeping draft: {e}")
    else:
        if progress_cb:
            progress_cb("Skipping critic (FAST_MODE)...", 80)
        logger.info("FAST_MODE: skipping critic pass")

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

    # Repair: fill in reel_start/reel_end for narration events missing timestamps
    # (critic pass may have stripped them or LLM may have omitted them)
    for g in groups:
        events = g.get("narration_events", [])
        needs_repair = any(
            not e.get("reel_start") and not e.get("reel_end")
            for e in events if isinstance(e, dict)
        )
        if needs_repair:
            g["narration_events"] = place_narration_events(g)

    logger.info(f"LLM returned {len(groups)} groups (ceiling {max_groups})")

    if progress_cb:
        progress_cb("Validating plan...", 90)
    if reporter:
        reporter.update_analyzer_phase("validation_finalize", "running")
    try:
        reel_plan = finalize_edit(draft, source_duration, min_groups=min_groups, content_type=content_type)
    except Exception as e:
        if reporter:
            reporter.update_analyzer_phase("validation_finalize", "error", error=str(e)[:300])
        raise
    if reporter:
        reporter.update_analyzer_phase(
            "validation_finalize", "done", progress=100,
            detail={"groups": len(reel_plan.reel_groups)},
        )

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
# Executor mode — LLM plans (structure only), Python picks every number
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_ranker(plan: "StoryPlan", blocks: list[SemanticBlock], source_duration: float, content_type: str = "", content_identity: ContentIdentity | None = None, reel_dur_min: int = 90, reel_dur_max: int = 100) -> str:
    """Build the beat-relevance ranking prompt (block IDs only, no timestamps).

    The ranker is the semantic layer of the executor: it tells Python WHICH
    candidate blocks best deliver each beat's intent. Python still owns all
    placement (windows, slots, budgets); the ranker only weights block choice.
    """
    from backend.pipeline.plan_schema import _unit_windows

    windows = _unit_windows(plan, source_duration, content_type=content_type)

    identity_hint = ""
    if content_identity:
        entity_names = ", ".join(content_identity.entity_names) or "none"
        identity_hint = (
            f"\nIDENTIFIER CONTEXT: Creator={content_identity.creator_name or 'unconfirmed'}, "
            f"Structure={content_identity.structure}, Entities={entity_names}\n"
            f"Planning notes: {content_identity.planning_notes}\n"
        )

    lines = [
        "You are a scene-matching assistant for a Shorts editor.",
        "Given story units and their beat intents, rank candidate blocks (by ID) by how well each block DELIVERS the intent.",
        "",
        f"TARGET REEL DURATION: {reel_dur_min}-{reel_dur_max}s per reel. Each unit needs enough blocks to fill this duration.",
        "",
        "RULES",
        "1. Rank by CONTENT MATCH, not importance: a calm setup block is wrong for an escalation intent even if it scored high.",
        '2. One ranked list per beat key: "start", "escalation" (all escalation beats share one list), "payoff".',
        '   If a unit has a "hook" beat, include it as a separate key.',
        "3. Best block first. Only use IDs from the provided BLOCKS lists. Lists may stay empty or short.",
        '4. STRICT JSON ONLY: {"0": {"start": [5], "escalation": [12, 4], "payoff": [18]}, ...}',
        "5. For PAYOFF beats, strongly prefer blocks with the LATEST source timestamps — the winner reveal or final reaction must come LAST.",
        "6. For START beats, prefer the earliest strong block that introduces the scene or person. Prefer blocks mentioning the entity names from the identifier context.",
        "7. For START beats: AVOID blocks with very short text (< 10 words) or low word density (wps < 1.5). These are likely exclamations/reactions, not meaningful intros.",
        "8. Check timestamps: if a block is very early (first 10% of the unit window), verify it contains actual content, not just greeting/intro fluff.",
        identity_hint,
        "",
    ]
    for u in plan.units:
        w = windows.get(u.unit_id, (0.0, source_duration))
        unit_blocks = [
            b for b in blocks
            if b.end > w[0] and b.start < w[1]
            and not b.black_frame and not b.freeze
        ]
        # Entity mode: further restrict to blocks in this unit's entity segments
        if u.entity_segment_ids:
            unit_blocks = [b for b in unit_blocks if b.entity_segment_id in u.entity_segment_ids]
        unit_blocks.sort(key=lambda b: b.importance, reverse=True)
        top = unit_blocks[:50]
        if not top:
            continue
        beats = [b for b in u.arc if b.intent]
        if not beats:
            continue
        lines.append(f"UNIT {u.unit_id} — {u.name} (region {u.region}, priority {u.priority})")
        for b in beats:
            lines.append(f"  {b.beat}: {b.intent}")
        for b in top:
            flags = ", ".join(
                f for f, has in (
                    ("VULGAR", b.has_vulgarity), ("DATING", b.has_dating),
                    ("ROAST", b.has_roast), ("STAKES", b.has_stakes),
                    ("CUR_GAP", b.has_curiosity_gap), ("VIRAL", b.has_viral_trigger),
                    ("INTERRUPT", b.has_pattern_interrupt),
                ) if has
            ) or "no flags"
            text = (b.text or "").strip().replace("\n", " ")
            if len(text) > 500:
                text = text[:500].rstrip() + "..."
            wps_hint = f" wps={b.word_density:.1f}" if b.word_density > 0 else ""
            lines.append(f"  B{b.block_id} [{b.start:.0f}-{b.end:.0f}s]: [{flags}] imp={b.importance:.0f}{wps_hint} \"{text}\"")
        lines.append("")
    return "\n".join(lines)


def _parse_beat_rankings(raw: str, plan: "StoryPlan", blocks: list[SemanticBlock]) -> dict[int, dict[str, list[int]]]:
    """Validate the ranker's JSON into unit_id -> beat -> [block_id, ...]."""
    out: dict[int, dict[str, list[int]]] = {}
    try:
        data = _parse_json_response(raw)
    except (ValueError, TypeError):
        return out
    if not isinstance(data, dict):
        return out
    valid_ids = {b.block_id for b in blocks}
    for unit in plan.units:
        unit_data = data.get(str(unit.unit_id), data.get(unit.unit_id))
        if not isinstance(unit_data, dict):
            continue
        beat_ranks: dict[str, list[int]] = {}
        for beat in ("hook", "start", "escalation", "payoff"):
            lst = unit_data.get(beat)
            if not isinstance(lst, list):
                continue
            seen: list[int] = []
            for x in lst:
                try:
                    bid = int(x)
                except (TypeError, ValueError):
                    continue
                if bid in valid_ids and bid not in seen:
                    seen.append(bid)
            if seen:
                beat_ranks[beat] = seen
        if beat_ranks:
            out[unit.unit_id] = beat_ranks
    return out


def _rank_beats_for_plan(
    plan: "StoryPlan",
    blocks: list[SemanticBlock],
    source_duration: float,
    progress_cb: Callable[[str, float], None] | None,
    reporter: Any,
    interactions: list[LLMInteraction] | None,
    content_type: str = "",
    content_identity: ContentIdentity | None = None,
    reel_dur_min: int = 90,
    reel_dur_max: int = 100,
) -> dict[int, dict[str, list[int]]]:
    """One LLM call: rank block IDs per beat intent. Never raises — {} on failure."""
    try:
        raw = _call_llm(
            [
                {"role": "system", "content": "Respond with ONLY valid JSON."},
                {"role": "user", "content": _prompt_ranker(plan, blocks, source_duration, content_type=content_type, content_identity=content_identity, reel_dur_min=reel_dur_min, reel_dur_max=reel_dur_max)},
            ],
            progress_cb, reporter, interactions, stage_name="moment_beat_matcher",
            max_tokens=65536,
        )
        rankings = _parse_beat_rankings(raw, plan, blocks)
        ranked_units = sum(len(v) for v in rankings.values())
        logger.info(f"EXECUTOR: relevance rankings for {ranked_units} unit-beat(s)")
        return rankings
    except Exception as e:
        logger.warning(f"Relevance ranker failed — falling back to importance-only scoring: {e}")
        return {}


def _select_reel_plan_executor(
    video_title: str,
    video_description: str,
    progress_cb: Callable[[str, float], None] | None,
    reporter: Any,
    interactions: list[LLMInteraction] | None,
    rich_timeline: RichTimeline | None,
    source_duration: float,
    min_groups: int,
    max_groups: int,
    reel_dur_min: int,
    reel_dur_max: int,
    blocks: list[SemanticBlock],
    blocks_text: str,
    usable_hints: str,
    content_type: str = GENRE_GENERAL,
    worth_verdict: str = "",
    worth_breakdown: dict | None = None,
    content_identity: ContentIdentity | None = None,
    hook_mode: HookMode = "skip",
    multimodal_signals: MultimodalSignals | None = None,
    entity_segments: list[EntitySegment] | None = None,
    entity_grouped: bool = False,
) -> ReelPlan:
    """Executor-mode pipeline: 2 LLM calls (genre-specific planner + writer), zero LLM numbers.

    LLM #1 (story planner): genre-specific arc → regions + beats only, no timestamps.
    Python: executes the plan into concrete clips + narration placement.
    LLM #2 (writer): narration line text + reel summaries only — no timestamps.
    Python: validates everything via finalize_edit.
    """
    from backend.pipeline.plan_executor import (
        assign_reel_summary,
        execute_plan,
        place_narration_events,
    )
    from backend.pipeline.plan_schema import (
        _validate_plan_integrity,
        _validate_entity_merges,
        heuristic_story_plan,
        parse_story_plan,
        plan_to_blocks_hint,
        plan_to_structure_analysis,
        StoryPlan,
    )

    logger.info(
        f"EXECUTOR MODE source={source_duration:.1f}s groups={min_groups}-{max_groups} "
        f"blocks={len(blocks)} duration_target={reel_dur_min}-{reel_dur_max}s "
        f"content_type={content_type} entity_grouped={entity_grouped}"
    )

    planner_branch = "unknown"

    # ── Entity grouping: LLM merges adjacent segments into challenge-level reels ──
    plan = None
    if entity_grouped and entity_segments:
        if progress_cb:
            progress_cb("Grouping entity segments into challenge reels (LLM)...", 20)
        for attempt in (1, 2):
            try:
                extra = (
                    "\n\nCRITICAL: Output ONLY the JSON object. No markdown, no comments, "
                    "no text outside the JSON."
                    if attempt == 2 else ""
                )
                raw_plan = _call_llm(
                    [
                        {"role": "system", "content": "Respond with ONLY valid JSON."},
                        {"role": "user", "content": (
                            _prompt_planner_entity(
                                video_title, video_description,
                                entity_segments, blocks, source_duration,
                                content_identity,
                                reel_dur_min=reel_dur_min,
                                reel_dur_max=reel_dur_max,
                                content_type=content_type,
                                min_groups=min_groups,
                                max_groups=max_groups,
                            ) + extra
                        )},
                    ],
                    progress_cb, reporter, interactions, stage_name="entity_group_planner",
                    max_tokens=65536,
                )
                plan = parse_story_plan(
                    _parse_json_response(raw_plan), source_duration,
                    min_groups=min_groups, max_groups=max_groups,
                    hook_mode=hook_mode,
                    entity_segments=entity_segments,
                    blocks=blocks,
                    content_identity=content_identity,
                    content_type=content_type,
                )
                if len(plan.units) > max_groups:
                    plan.units.sort(key=lambda u: (u.priority, u.unit_id))
                    plan.units = plan.units[:max_groups]
                    logger.info(f"Entity happy path: capped {len(plan.units)} units to ceiling {max_groups}")
                planner_branch = "entity_happy_path"
                break
            except Exception as e:
                logger.warning(f"Entity planner attempt {attempt} failed: {e}")
        if plan is None:
            logger.warning("Entity planner failed — creating one unit per segment (capped to max_groups=%d)", max_groups)
            units = _validate_entity_merges(
                [{"segment_ids": [s.entity_segment_id]} for s in entity_segments],
                entity_segments, blocks, source_duration,
                content_identity=content_identity,
                content_type=content_type,
            )
            if len(units) > max_groups:
                units.sort(key=lambda u: (u.priority, -sum(
                    (seg.end - seg.start) for seg in entity_segments
                    if seg.entity_segment_id in u.entity_segment_ids
                )))
                units = units[:max_groups]
                logger.warning(f"Fallback capped from {len(entity_segments)} segments to {max_groups} units")
            plan = StoryPlan(video_type="entity", units=units)
            planner_branch = "entity_fallback"
    else:
        # Genre-specific mode: region-based planning
        if progress_cb:
            progress_cb(f"Planning {content_type} story structure (LLM)...", 20)

        planner_identity = content_identity
        if hook_mode == "required" and content_identity and content_identity.structure == "single_narrative":
            planner_identity = None

        for attempt in (1, 2):
            try:
                extra = (
                    "\n\nCRITICAL: Output ONLY the JSON object. No markdown, no comments, "
                    "no text outside the JSON. No timestamps anywhere."
                    if attempt == 2 else ""
                )
                raw_plan = _call_llm(
                    [
                        {"role": "system", "content": "Respond with ONLY valid JSON."},
                        {"role": "user", "content": (
                            _prompt_planner_for_genre(
                                content_type,
                                video_title, video_description, blocks_text,
                                source_duration, min_groups, max_groups, usable_hints,
                                content_identity=planner_identity,
                                reel_dur_min=reel_dur_min,
                                reel_dur_max=reel_dur_max,
                                hook_mode=hook_mode,
                            ) + extra
                        )},
                    ],
                    progress_cb, reporter, interactions, stage_name="genre_story_planner",
                    max_tokens=65536,
                )
                plan = parse_story_plan(
                    _parse_json_response(raw_plan), source_duration, min_groups, max_groups,
                    hook_mode=hook_mode,
                    content_identity=content_identity,
                )
                planner_branch = "genre_happy_path"
                break
            except Exception as e:
                logger.warning(f"Story planner attempt {attempt} failed: {e}")

        if plan is None:
            logger.warning("Story planner failed twice — using heuristic story plan")
            plan = heuristic_story_plan(source_duration, min_groups, max_groups, hook_mode=hook_mode)
            planner_branch = "genre_fallback"

    _validate_plan_integrity(plan, min_groups, max_groups)

    logger.info(
        f"STORY PLAN: type={plan.video_type} units={len(plan.units)} "
        f"({', '.join(u.name for u in plan.units[:6])})"
    )

    # ── LLM #1.5 Beat Relevance Ranker (semantics only — no timestamps) ──
    if progress_cb:
        progress_cb("Matching beats to moments (LLM)...", 40)
    relevance = _rank_beats_for_plan(
        plan, blocks, source_duration, progress_cb, reporter, interactions, content_type=content_type,
        content_identity=content_identity, reel_dur_min=reel_dur_min, reel_dur_max=reel_dur_max,
    )

    # ── Python: execute the plan into concrete clips ──
    if progress_cb:
        progress_cb("Executing plan (Python)...", 45)
    if reporter:
        reporter.update_analyzer_phase("plan_execution", "running")
    groups = execute_plan(
        plan, blocks, source_duration, reel_dur_min, reel_dur_max, relevance,
        content_type=content_type,
        entity_segments=entity_segments if entity_grouped else None,
        scene_cut_at=multimodal_signals.scene_cut_at if multimodal_signals else None,
        content_identity=content_identity,
    )
    logger.info(f"EXECUTOR: built {len(groups)} groups")
    pre_qa_groups = [g.copy() for g in groups]

    # ── Python: enforce chronological story flow ──
    groups = _enforce_story_flow(groups)
    logger.info(f"STORY FLOW: groups ordered chronologically through source")

    # ── Python: post-execution QA (entity boundaries, payoff position, duration, overlap) ──
    if progress_cb:
        progress_cb("Running QA checks...", 55)
    from backend.pipeline.plan_executor import post_execution_qa
    groups = post_execution_qa(
        groups, source_duration,
        entity_segments=entity_segments if entity_grouped else None,
        reel_dur_min=reel_dur_min, reel_dur_max=reel_dur_max,
    )
    logger.info(f"POST QA: {len(groups)} groups survived QA checks")
    post_qa_groups = [g.copy() for g in groups]
    if reporter:
        reporter.update_analyzer_phase(
            "plan_execution", "done", progress=100,
            detail={"groups": len(groups)},
        )

    # ── LLM #2 Narration Writer (text only — Python owns numbers) ──
    if progress_cb:
        progress_cb("Writing narration (LLM)...", 65)
    sa = plan_to_structure_analysis(
        plan, reasoning=f"Story units chosen by LLM ({content_type}); clips selected deterministically by Python."
    )
    blocks_hint = plan_to_blocks_hint(plan, blocks, source_duration)
    raw_writer = _call_llm(
        [
            {"role": "system", "content": "Respond with ONLY valid JSON. Do NOT include any reasoning, thinking, or commentary. Output ONLY the JSON object."},
            {"role": "user", "content": _prompt_writer(video_title, groups, sa, blocks_hint, content_type, hook_mode=hook_mode, reel_dur_min=reel_dur_min, reel_dur_max=reel_dur_max)},
        ],
        progress_cb, reporter, interactions, stage_name="script_narration_writer",
        max_tokens=65536,
    )
    try:
        writer_plan = _parse_json_response(raw_writer)
        writer_groups = writer_plan.get("reel_groups", []) or []
    except (AttributeError, ValueError, TypeError):
        logger.warning("Narration writer returned malformed JSON — continuing without narration")
        writer_groups = []
    writer_by_idx = {
        g.get("group_index", i): g
        for i, g in enumerate(writer_groups)
        if isinstance(g, dict)
    }
    # Also build a name-based lookup for cross-referencing
    writer_by_name = {
        str(g.get("group_name", "")).strip().lower(): g
        for g in writer_groups
        if isinstance(g, dict) and g.get("group_name")
    }
    # Warn if LLM returned different number of groups
    if len(writer_groups) != len(groups):
        logger.warning(
            f"Narration writer returned {len(writer_groups)} groups but expected {len(groups)} "
            f"— matching by index with fallback to name"
        )
    for i, g in enumerate(groups):
        idx = g.get("group_index", i)
        unit_name = g.get("name", "")
        # Try index match first, then name match
        wg = writer_by_idx.get(idx, {})
        if not wg and unit_name:
            wg = writer_by_name.get(str(unit_name).strip().lower(), {})
        if not wg:
            logger.warning(f"Group {i} (unit_id={idx}, name='{unit_name}'): no matching writer group found")
        events = wg.get("narration_events", [])
        clean = []
        if isinstance(events, list):
            for e in events:
                if not isinstance(e, dict):
                    continue
                ev_type = str(e.get("event_type", "commentary")).strip().lower()
                if ev_type not in ("hook", "start", "commentary"):
                    continue
                text = str(e.get("text", "")).strip()
                if not text:
                    continue
                persona = e.get("persona")
                clean.append({
                    "event_type": ev_type,
                    "text": text,
                    "persona": persona if isinstance(persona, str) else None,
                })
        g["narration_events"] = clean
        assign_reel_summary(g, wg.get("reel_summary"))

    # ── Python: place narration from the final clip layout ──
    if progress_cb:
        progress_cb("Placing narration (Python)...", 80)
    if reporter:
        reporter.update_analyzer_phase("narration_placement", "running")
    for g in groups:
        g["narration_events"] = place_narration_events(g)
    if reporter:
        total_events = sum(len(g.get("narration_events", [])) for g in groups)
        reporter.update_analyzer_phase(
            "narration_placement", "done", progress=100,
            detail={"narration_events": total_events},
        )

    draft = {
        "structure_analysis": sa,
        "ranked_segments": [],
        "reel_groups": groups,
        "plan_mode": "executor",
        "explanations": [
            f"Executor mode: LLM used {content_type}-specific story planning; "
            "Python selected every clip and placed every narration deterministically.",
            worth_verdict,
        ],
        "worth_breakdown": worth_breakdown or {},
        "content_identity": content_identity.model_dump() if content_identity else None,
        "multimodal_signals": multimodal_signals.model_dump() if multimodal_signals else None,
        "entity_segments": [segment.model_dump() for segment in entity_segments or []],
        "entity_grouped": entity_grouped,
    }

    # ── Python validator owns final numbers ──
    if progress_cb:
        progress_cb("Validating plan...", 90)
    if reporter:
        reporter.update_analyzer_phase("validation_finalize", "running")
    # Entity content: no floor enforcement — low-quality entities intentionally dropped
    effective_min_groups = 1 if entity_grouped else min_groups
    try:
        reel_plan = finalize_edit(
            draft, source_duration, min_groups=effective_min_groups, preserve_layout=True, content_type=content_type
        )
    except Exception as e:
        if reporter:
            reporter.update_analyzer_phase("validation_finalize", "error", error=str(e)[:300])
        raise
    if reporter:
        reporter.update_analyzer_phase(
            "validation_finalize", "done", progress=100,
            detail={"groups": len(reel_plan.reel_groups)},
        )

    # ── LLM completeness critic (read-only quality gate) ──
    critic_groups_for_llm = [g.model_dump() for g in reel_plan.reel_groups]
    critic_results = _completeness_critic(
        critic_groups_for_llm, progress_cb, reporter, interactions,
    )
    # Attach verdicts directly to reel_plan groups for downstream UI/API access
    verdicts = {}
    for cg in critic_results:
        if isinstance(cg, dict) and cg.get("completeness_critic"):
            idx = cg.get("group_index")
            verdicts[idx] = cg["completeness_critic"]
            for grp in reel_plan.reel_groups:
                if grp.group_index == idx:
                    grp.completeness_critic = cg["completeness_critic"]
                    break
    if verdicts:
        flagged = [idx for idx, v in verdicts.items() if not v.get("complete", True)]
        if flagged:
            logger.info(f"Completeness critic flagged group(s): {flagged}")

    if progress_cb:
        progress_cb(f"Built reel plan with {len(reel_plan.reel_groups)} group(s)", 100)

    total_clips = sum(len(g.source_clips) for g in reel_plan.reel_groups)
    total_narr = sum(len(g.narration_events) for g in reel_plan.reel_groups)
    avg_dur = sum(g.estimated_duration_seconds for g in reel_plan.reel_groups) / max(
        len(reel_plan.reel_groups), 1
    )
    logger.info(
        f"REEL PLAN STATS (executor): {len(reel_plan.reel_groups)} groups, "
        f"{total_clips} clips, {total_narr} narrations, avg duration {avg_dur:.1f}s, "
        f"content_type={content_type}"
    )

    if reporter and interactions is not None:
        reporter.set_stage_data_key(
            "llm_interactions", [i.model_dump() for i in interactions]
        )

    # ── Consolidated debug artifact dump ──
    _write_debug_artifact(
        job_id=getattr(reporter, "job", None) and reporter.job.id if reporter else None,
        content_identity=content_identity,
        entity_segments=entity_segments,
        planner_branch=planner_branch,
        story_plan=plan,
        relevance=relevance,
        pre_qa_groups=pre_qa_groups,
        post_qa_groups=post_qa_groups,
        final_groups=[g.model_dump() for g in reel_plan.reel_groups],
        completeness_verdicts=verdicts,
        source_duration=source_duration,
        max_groups=max_groups,
    )

    return reel_plan
