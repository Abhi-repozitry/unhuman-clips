"""Planner output schema — the LLM's ONLY deliverable in executor mode.

The LLM produces a story plan with NO timestamps: video type, story units,
their regions (early/mid/late), priorities, and the beats each unit needs
(hook -> start -> escalation -> payoff) plus which engagement flags matter.

Python validates/repairs the plan and executes it deterministically via
``plan_executor.execute_plan``. If the LLM plan is invalid after retries, a
deterministic heuristic plan is generated from the semantic blocks instead —
the pipeline never fails because of a bad LLM plan.
"""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from backend.models import ContentIdentity, EntitySegment, HookMode
from backend.config import ENTITY_MIN_SEGMENT_SECONDS
from backend.pipeline.analyzer import (
    GENRE_COMEDY_SKETCH,
    GENRE_DATING_REALITY,
    GENRE_EXPERIMENT,
    GENRE_GAME_CHALLENGE,
    GENRE_MUSIC_PERF,
    GENRE_NEWS_DOC,
    GENRE_PODCAST,
    GENRE_ROAST_REACTION,
    GENRE_SPORTS_FITNESS,
    GENRE_TUTORIAL,
    GENRE_VLOG_PERSONAL,
    SemanticBlock,
)

__all__ = [
    "Beat",
    "EntitySegment",
    "StoryPlan",
    "Unit",
    "_validate_entity_merges",
    "heuristic_story_plan",
    "parse_story_plan",
    "plan_to_structure_analysis",
]

logger = logging.getLogger(__name__)

Region = Literal["early", "mid", "late"]
BeatType = Literal["hook", "start", "escalation", "payoff"]
Position = Literal["start", "any", "end"]
Flag = Literal["VULGAR", "DATING", "ROAST", "STAKES"]


class Beat(BaseModel):
    beat: BeatType
    position: Position = "any"
    flags: list[Flag] = Field(default_factory=list)
    intent: str = ""


class Unit(BaseModel):
    unit_id: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=120)
    priority: int = Field(default=1, ge=1, le=3)
    region: Region = "mid"
    arc: list[Beat] = Field(default_factory=list)
    entity_segment_ids: list[str] = Field(default_factory=list)
    # When False, _repair_unit skips inserting a payoff beat and execute_plan
    # does not force PAYOFF_MIN_SECONDS — the reel ends after its last real
    # content.  Inferred from ContentIdentity.arc_style ("quiz" -> False).
    requires_payoff: bool = True


class StoryPlan(BaseModel):
    video_type: str = "other"
    units: list[Unit] = Field(default_factory=list)


def _repair_unit(unit: Unit, hook_mode: HookMode = "skip") -> Unit:
    """Normalize a unit's arc: canonical beat ordering for the given hook_mode.

    Ordering:
      required: hook -> start -> escalation* -> payoff
      skip: start -> escalation* -> payoff (hook deterministically stripped)
    """
    has_hook = any(b.beat == "hook" for b in unit.arc)
    has_start = any(b.beat == "start" for b in unit.arc)
    has_payoff = any(b.beat == "payoff" for b in unit.arc)

    # Deduplicate: keep first hook, first start, first payoff
    beats: list[Beat] = []
    hook_seen = False
    start_seen = False
    payoff_seen = False
    for b in unit.arc:
        if b.beat == "hook":
            if hook_seen:
                continue
            hook_seen = True
        elif b.beat == "start":
            if start_seen:
                continue
            start_seen = True
        elif b.beat == "payoff":
            if payoff_seen:
                continue
            payoff_seen = True
        beats.append(b)

    # Enforce mandatory beats per hook_mode
    if hook_mode == "required":
        if not has_hook:
            beats.insert(0, Beat(beat="hook", position="start", intent="curiosity trigger"))
        if not has_start:
            # Insert start after hook (if present) or at beginning
            insert_at = 1 if any(b.beat == "hook" for b in beats) else 0
            beats.insert(insert_at, Beat(beat="start", position="start", intent="scene introduction"))
    elif hook_mode == "skip":
        # Deterministic strip: any hook beat the LLM included is removed.
        beats = [b for b in beats if b.beat != "hook"]
        if not has_start:
            beats.insert(0, Beat(beat="start", position="start", intent="scene introduction"))
    else:
        # Fallback (should not occur after auto removal): treat as skip
        beats = [b for b in beats if b.beat != "hook"]
        if not has_start:
            beats.insert(0, Beat(beat="start", position="start", intent="scene introduction"))

    if not has_payoff and unit.requires_payoff:
        beats.append(Beat(beat="payoff", position="end", intent="resolution / reveal"))

    # Cap escalation beats at 4
    esc_count = 0
    capped: list[Beat] = []
    for b in beats:
        if b.beat == "escalation":
            esc_count += 1
            if esc_count > 4:
                continue
        capped.append(b)

    # Canonical ordering: hook -> start -> escalation* -> payoff
    def _beat_sort_key(b: Beat) -> tuple[int, int]:
        order = {"hook": 0, "start": 1, "escalation": 2, "payoff": 3}
        return (order.get(b.beat, 2), 0)

    hook_beats = sorted([b for b in capped if b.beat == "hook"], key=_beat_sort_key)
    start_beats = sorted([b for b in capped if b.beat == "start"], key=_beat_sort_key)
    esc_beats = [b for b in capped if b.beat == "escalation"]
    payoff_beats = sorted([b for b in capped if b.beat == "payoff"], key=_beat_sort_key)
    capped = hook_beats + start_beats + esc_beats + payoff_beats

    # Positions are role-truthful: hook and start are opening triggers, payoff
    # is the closing reveal — their coarse hints are fixed to the editorial role
    # regardless of what the LLM wrote. Escalation hints are kept verbatim.
    for b in capped:
        b.position = _normalize_beat_position(b.beat, b.position)

    unit.arc = capped
    return unit


def parse_story_plan(
    data: dict,
    source_duration: float,
    min_groups: int,
    max_groups: int,
    hook_mode: HookMode = "skip",
    entity_segments: list[EntitySegment] | None = None,
    blocks: list[SemanticBlock] | None = None,
    content_identity: ContentIdentity | None = None,
    content_type: str = "",
) -> StoryPlan:
    """Validate + repair an LLM story plan. Never raises on shape issues.

    Repairs applied:
    - duplicate hooks/start removed, missing beats inserted per hook_mode
    - priority clamped to 1-3, regions coerced to early|mid|late
    - unit count clamped to [min_groups, max_groups] (sorted by priority)
    - unit_ids renumbered sequentially (duplicate/out-of-order ids from the LLM
      would otherwise collide in the executor's window map)

    Entity mode: when entity_segments and merge_groups are present, uses
    _validate_entity_merges instead of the region-based repair pipeline.
    """
    # Entity mode: LLM returned merge_groups for entity segments
    merge_groups = data.get("merge_groups", []) if isinstance(data, dict) else []
    if entity_segments and merge_groups:
        units = _validate_entity_merges(
            merge_groups, entity_segments, blocks or [], source_duration,
            content_identity=content_identity,
            content_type=content_type,
        )
        if units:
            return StoryPlan(video_type="entity", units=units)
        # Fall through to normal path if merge validation produced nothing

    try:
        plan = StoryPlan.model_validate(data)
    except ValidationError as e:
        logger.warning(f"Story plan failed strict validation, repairing: {e}")
        try:
            plan = StoryPlan.model_validate(_coerce_plan(data))
        except ValidationError as e2:
            logger.warning(f"Story plan unrepairable ({e2}) — using heuristic plan")
            return heuristic_story_plan(source_duration, min_groups, max_groups, hook_mode=hook_mode)

    for i, unit in enumerate(plan.units):
        unit.unit_id = _safe_int(unit.unit_id, i)
        unit.priority = max(1, min(3, _safe_int(unit.priority, 1)))
        if unit.region not in ("early", "mid", "late"):
            unit.region = "mid"
        # Infer requires_payoff from content identity arc_style.
        # "quiz" format (question-answer) has no reveal moment — the reel
        # ends after its last real content, no manufactured payoff.
        if content_identity and content_identity.arc_style == "quiz":
            unit.requires_payoff = False
        _repair_unit(unit, hook_mode=hook_mode)

    if not plan.units:
        logger.warning("Story plan had zero units — using heuristic plan")
        return heuristic_story_plan(source_duration, min_groups, max_groups, hook_mode=hook_mode)

    # Clamp unit count to the deterministic [min, max] window
    plan.units.sort(key=lambda u: (u.priority, u.unit_id))
    if len(plan.units) > max_groups:
        logger.info(f"Story plan: truncating {len(plan.units)} units to ceiling {max_groups}")
        plan.units = plan.units[:max_groups]
    if len(plan.units) < min_groups:
        extra = heuristic_story_plan(source_duration, min_groups, max_groups, hook_mode=hook_mode).units
        taken = {u.name for u in plan.units}
        added = 0
        for u in extra:
            if len(plan.units) >= min_groups:
                break
            if u.name in taken:
                continue
            u.unit_id = max((x.unit_id for x in plan.units), default=-1) + 1
            plan.units.append(u)
            added += 1
        if added:
            logger.info(f"Story plan: padded {added} unit(s) to floor {min_groups}")
    plan.units.sort(key=lambda u: (u.priority, u.unit_id))
    # Renumber sequentially — the LLM may emit duplicate/out-of-order ids, and
    # unit_id doubles as the group_index used to match writer output.
    for i, u in enumerate(plan.units):
        u.unit_id = i

    # Deterministic opening guarantee: the reel viewers see first must open
    # with the video's opening, not its middle. LLMs drift to "mid"/"late"
    # for the top-priority unit; correct that here — this is a layout number,
    # owned by Python (executor philosophy). Safe: the executor only draws
    # usable (non-dead) moments inside the early window, and falls back to
    # the whole source if the opening is genuinely empty.
    if len(plan.units) > 1 and plan.units[0].region != "early":
        logger.warning(
            f"Opening guarantee: top-priority unit was '{plan.units[0].region}' "
            f"('{plan.units[0].name}') — reassigned to 'early' so reel 1 opens the video"
        )
        plan.units[0].region = "early"
    return plan


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_beat_position(beat: str, position) -> str:
    """Deterministic position normalization.

    Positions are coarse hints that steer escalation slot order inside the
    unit window. Hook and start are opening triggers; payoff is the closing
    reveal — their positions are fixed to the editorial role regardless of
    what the LLM writes. Escalations keep whatever valid hint the LLM gave,
    defaulting to "any".
    """
    if beat in ("hook", "start"):
        return "start"
    if beat == "payoff":
        return "end"
    return position if position in ("start", "any", "end") else "any"


def _coerce_plan(data: dict) -> dict:
    """Best-effort coercion of messy LLM output into a StoryPlan-shaped dict."""
    units = data.get("units")
    if not isinstance(units, list):
        units = data.get("identified_units", []) or data.get("reel_groups", [])
    if not isinstance(units, list):
        units = []
    coerced_units = []
    for i, u in enumerate(units or []):
        if not isinstance(u, dict):
            continue
        arc = u.get("arc")
        if not isinstance(arc, list):
            # Beats might be a plain list of strings, or embedded elsewhere
            arc = []
        arc = [
            {
                "beat": b.get("beat", b.get("type", "escalation")),
                "position": _normalize_beat_position(
                    b.get("beat", b.get("type", "escalation")), b.get("position")
                ),
                "flags": b.get("flags", []),
                "intent": b.get("intent", ""),
            }
            for b in arc if isinstance(b, dict)
        ]
        coerced_units.append({
            "unit_id": u.get("unit_id", u.get("id", i)),
            "name": u.get("name", u.get("unit_name", f"Story {i + 1}")),
            "priority": max(1, min(3, int(u.get("priority", 1) or 1))),
            "region": region if (region := u.get("region", u.get("section", "early" if i == 0 else "mid"))) in ("early", "mid", "late") else ("early" if i == 0 else "mid"),
            "arc": arc,
        })
    return {"video_type": data.get("video_type", "other"), "units": coerced_units}


def heuristic_story_plan(
    source_duration: float,
    min_groups: int,
    max_groups: int,
    hook_mode: HookMode = "skip",
) -> StoryPlan:
    """Deterministic fallback plan — no LLM involved.

    Splits the source timeline into evenly spaced regions and gives every
    unit the standard arc.  When hook_mode='required', arc is
    hook -> start -> escalation -> payoff; when 'skip', start -> escalation -> payoff.
    """
    n = max(min_groups, min(max_groups, max(1, int(source_duration / 480) + 1)))
    regions = ["early", "mid", "mid", "late"]
    units = []
    for i in range(n):
        arc: list[Beat] = []
        if hook_mode == "required":
            arc.append(Beat(beat="hook", position="start", intent="curiosity trigger"))
        arc.append(Beat(beat="start", position="start", intent="scene introduction"))
        arc.append(Beat(beat="escalation", position="any", intent="tension builds"))
        arc.append(Beat(beat="payoff", position="end", intent="resolution / reveal"))
        units.append(Unit(
            unit_id=i,
            name=f"Story {i + 1}",
            priority=1,
            region=regions[i % len(regions)],
            arc=arc,
        ))
    return StoryPlan(video_type="other", units=units)


def _validate_plan_integrity(
    plan: "StoryPlan",
    min_groups: int,
    max_groups: int,
) -> None:
    """Warn if the LLM returned a plan with structural issues.

    Checks:
    - Units with no beats (will produce empty/silent reels)
    - Units missing key beats (hook, escalation, payoff)
    - Significantly fewer units than requested
    """
    if not plan.units:
        logger.warning("PLAN VALIDATION: plan has zero units")
        return

    if len(plan.units) < min_groups:
        logger.warning(
            f"PLAN VALIDATION: plan has {len(plan.units)} units but "
            f"min_groups={min_groups} — LLM may have dropped content"
        )

    for i, unit in enumerate(plan.units):
        if not unit.arc:
            logger.warning(
                f"PLAN VALIDATION: unit {i} ('{unit.name}') has no beats — "
                f"will produce empty reel"
            )
            continue
        beats = [b.beat for b in unit.arc]
        if "hook" not in beats and "start" not in beats:
            logger.warning(
                f"PLAN VALIDATION: unit {i} ('{unit.name}') missing "
                f"opening beat (hook/start) — beats: {beats}"
            )
        if "payoff" not in beats:
            logger.info(
                f"PLAN VALIDATION: unit {i} ('{unit.name}') has no payoff beat"
            )


def _segment_requires_payoff(
    seg_start: float,
    seg_end: float,
    blocks: list[SemanticBlock],
    content_identity: ContentIdentity | None,
) -> bool:
    """Infer whether a segment needs a payoff beat.

    Quiz/question-dense content (high has_question ratio) has no reveal
    moment — the reel ends after its last real content.
    """
    if content_identity and content_identity.arc_style == "quiz":
        return False
    seg_blocks = [
        b for b in blocks
        if b.end > seg_start and b.start < seg_end
        and not b.black_frame and not b.freeze
    ]
    if not seg_blocks:
        return True
    question_ratio = sum(1 for b in seg_blocks if b.has_question) / len(seg_blocks)
    return question_ratio < 0.3


# Genre-specific arc templates for entity mode.
# Each template provides start/escalation intents that the beat-relevance
# ranker can match against block content.
_GENRE_ENTITY_ARCS: dict[str, dict] = {
    GENRE_GAME_CHALLENGE: {
        "start_intent": "stakes reveal or challenge rules — the moment that makes you care",
        "escalation_intents": [
            "building pressure — near-misses, competitor reactions, mounting tension",
            "escalating challenge — harder tasks, bigger stakes, dramatic moments",
        ],
        "payoff_intent": "winner reveal or final result — who won and what happened",
    },
    GENRE_SPORTS_FITNESS: {
        "start_intent": "peak action or spectacular play — the moment that demands attention",
        "escalation_intents": [
            "building intensity — increasingly impressive plays or training progression",
            "climax approach — near-record moments, crowd energy rising",
        ],
        "payoff_intent": "winning moment or record broken — the definitive result",
    },
    GENRE_EXPERIMENT: {
        "start_intent": "dramatic hypothesis or result teaser — the question that hooks",
        "escalation_intents": [
            "building process — setup, testing, mounting evidence",
            "tension peak — almost-failures, surprising intermediate results",
        ],
        "payoff_intent": "final result reveal — does it work? what happens?",
    },
    GENRE_DATING_REALITY: {
        "start_intent": "most charged or awkward moment — the chemistry or tension that hooks",
        "escalation_intents": [
            "building flirtation or confrontation — mounting tension between people",
            "dramatic turning point — rejection, confession, or twist",
        ],
        "payoff_intent": "emotional climax — acceptance, rejection, or defining moment",
    },
    GENRE_COMEDY_SKETCH: {
        "start_intent": "comedic setup or premise — the situation that creates expectation",
        "escalation_intents": [
            "absurdity builds — misunderstanding deepens, situation spirals",
            "peak absurdity — the most ridiculous moment before the punchline",
        ],
        "payoff_intent": "punchline or big reveal — the moment that delivers the laugh",
    },
    GENRE_PODCAST: {
        "start_intent": "most provocative or surprising claim — something that demands context",
        "escalation_intents": [
            "story context — mounting details, the journey to the insight",
            "tension peak — the crucial detail that changes everything",
        ],
        "payoff_intent": "insight or reveal — the surprising truth or realization",
    },
    GENRE_ROAST_REACTION: {
        "start_intent": "most shocking claim or brutal opener — the moment that demands a reaction",
        "escalation_intents": [
            "escalating savagery — increasingly intense comments or reactions",
            "peak intensity — the most extreme moment before resolution",
        ],
        "payoff_intent": "hardest punchline or most shocked reaction — the mic-drop moment",
    },
    GENRE_VLOG_PERSONAL: {
        "start_intent": "peak emotional moment — the reveal, surprise, or dramatic beat that hooks",
        "escalation_intents": [
            "journey building — anticipation, effort, near-misses leading to the peak",
            "approach climax — getting closer to the goal, mounting emotion",
        ],
        "payoff_intent": "emotional resolution — the photo, gift, surprise, or personal climax",
    },
    GENRE_TUTORIAL: {
        "start_intent": "finished result or specific problem — show WHY someone should watch",
        "escalation_intents": [
            "key transformation steps — the most interesting or surprising parts",
            "progress visible — things taking shape, visible change happening",
        ],
        "payoff_intent": "completed result — before/after, final product, it actually worked",
    },
    GENRE_MUSIC_PERF: {
        "start_intent": "most powerful vocal or best riff — drop into the climax, not the intro",
        "escalation_intents": [
            "building energy — verse leading to chorus, crowd growing",
            "peak buildup — the moment before the biggest drop or note",
        ],
        "payoff_intent": "final chorus or high note — the performance's emotional peak",
    },
    GENRE_NEWS_DOC: {
        "start_intent": "most shocking fact or key revelation — name the event or stakes",
        "escalation_intents": [
            "evidence building — testimony, developing events, mounting proof",
            "climax approach — the crucial detail that seals the story",
        ],
        "payoff_intent": "conclusion or verdict — the consequence or unanswered question",
    },
}

_GENRE_ENTITY_ARCS_DEFAULT = {
    "start_intent": "the moment that creates the strongest curiosity — a question or dramatic statement",
    "escalation_intents": [
        "tension building — context, details, mounting pressure",
        "peak tension — the most intense moment before resolution",
    ],
    "payoff_intent": "the answer or climax — whatever makes the viewer think 'that was worth watching'",
}


def _entity_arc_for_genre(content_type: str) -> dict:
    """Return genre-specific arc template for entity mode."""
    return _GENRE_ENTITY_ARCS.get(content_type, _GENRE_ENTITY_ARCS_DEFAULT)


def _validate_entity_merges(
    raw_merges: list[dict],
    entity_segments: list[EntitySegment],
    blocks: list[SemanticBlock],
    source_duration: float,
    content_identity: ContentIdentity | None = None,
    content_type: str = "",
) -> list[Unit]:
    """Validate LLM merge groups and convert to Units with entity_segment_ids.

    Each merge group is either one segment or two adjacent segments.
    Invalid two-segment merges split back deterministically into two units.
    Adjacency is index-based: entity segments are contiguous and ordered by start time.
    """
    if not entity_segments:
        return []

    segment_map = {s.entity_segment_id: s for s in entity_segments}
    segments_by_index = sorted(entity_segments, key=lambda s: s.start)

    # Build adjacency index: segment_id -> position in sorted order
    index_of = {s.entity_segment_id: i for i, s in enumerate(segments_by_index)}

    used_ids: set[str] = set()
    units: list[Unit] = []
    arc_tpl = _entity_arc_for_genre(content_type)

    for group in raw_merges:
        seg_ids = group.get("segment_ids", [])
        if not seg_ids or not isinstance(seg_ids, list):
            continue

        # Filter to valid, unused IDs
        valid_ids = [sid for sid in seg_ids if sid in segment_map and sid not in used_ids]
        if not valid_ids:
            continue

        # Validate adjacency: all segments must be consecutive in the sorted list
        indices = [index_of.get(sid) for sid in valid_ids]
        if any(idx is None for idx in indices):
            # Some IDs not found — skip invalid ones
            valid_ids = [sid for sid, idx in zip(valid_ids, indices) if idx is not None]
            indices = [idx for idx in indices if idx is not None]
        if not valid_ids:
            continue

        sorted_pairs = sorted(zip(indices, valid_ids))
        is_consecutive = all(
            sorted_pairs[i + 1][0] - sorted_pairs[i][0] == 1
            for i in range(len(sorted_pairs) - 1)
        )
        if not is_consecutive:
            # Not adjacent — split into individual units
            for sid in valid_ids:
                if sid in used_ids:
                    continue
                used_ids.add(sid)
                seg = segment_map[sid]
                rp = _segment_requires_payoff(seg.start, seg.end, blocks, content_identity)
                unit = Unit(
                    unit_id=len(units),
                    name=seg.entity_name or f"Entity {sid}",
                    priority=1,
                    region="mid",
                    arc=[
                        Beat(beat="start", position="start", intent=arc_tpl["start_intent"]),
                        Beat(beat="escalation", position="any", intent=arc_tpl["escalation_intents"][0]),
                    ] + ([Beat(beat="payoff", position="end", intent=arc_tpl["payoff_intent"])] if rp else []),
                    entity_segment_ids=[sid],
                    requires_payoff=rp,
                )
                units.append(unit)
            continue

        # Check usable duration for merged segments
        merged_start = min(segment_map[sid].start for sid in valid_ids)
        merged_end = max(segment_map[sid].end for sid in valid_ids)
        total_usable = sum(
            b.duration for b in blocks
            if b.end > merged_start and b.start < merged_end
            and not b.black_frame and not b.freeze and b.importance >= 25
        )
        if total_usable < ENTITY_MIN_SEGMENT_SECONDS * 0.5:
            # Too short even as a single segment — skip
            continue

        # Mark all IDs as used
        for sid in valid_ids:
            used_ids.add(sid)

        # Build unit from merge
        primary_seg = segment_map[valid_ids[0]]
        entity_name = primary_seg.entity_name or f"Entity {valid_ids[0]}"
        if len(valid_ids) == 2:
            second_seg = segment_map[valid_ids[1]]
            second_name = second_seg.entity_name or f"Entity {valid_ids[1]}"
            unit_name = f"{entity_name} & {second_name}"
        elif len(valid_ids) > 2:
            # Multiple segments: use primary name + count
            unit_name = f"{entity_name} + {len(valid_ids) - 1} more"
        else:
            unit_name = entity_name

        unit = Unit(
            unit_id=len(units),
            name=unit_name[:120],
            priority=1,
            region="mid",
            arc=[
                Beat(beat="start", position="start", intent=arc_tpl["start_intent"]),
                Beat(beat="escalation", position="any", intent=arc_tpl["escalation_intents"][0]),
            ] + ([Beat(beat="payoff", position="end", intent=arc_tpl["payoff_intent"])] if _segment_requires_payoff(merged_start, merged_end, blocks, content_identity) else []),
            entity_segment_ids=valid_ids,
            requires_payoff=_segment_requires_payoff(merged_start, merged_end, blocks, content_identity),
        )
        units.append(unit)

    # Add any segments the LLM didn't group (orphans)
    for seg in segments_by_index:
        if seg.entity_segment_id not in used_ids:
            used_ids.add(seg.entity_segment_id)
            rp = _segment_requires_payoff(seg.start, seg.end, blocks, content_identity)
            unit = Unit(
                unit_id=len(units),
                name=seg.entity_name or f"Entity {seg.entity_segment_id}",
                priority=1,
                region="mid",
                arc=[
                    Beat(beat="start", position="start", intent=arc_tpl["start_intent"]),
                    Beat(beat="escalation", position="any", intent=arc_tpl["escalation_intents"][0]),
                ] + ([Beat(beat="payoff", position="end", intent=arc_tpl["payoff_intent"])] if rp else []),
                entity_segment_ids=[seg.entity_segment_id],
                requires_payoff=rp,
            )
            units.append(unit)

    # Renumber sequentially
    for i, u in enumerate(units):
        u.unit_id = i

    # Rank by usable duration (longest = highest priority)
    for u in units:
        seg_starts = [segment_map[sid].start for sid in u.entity_segment_ids if sid in segment_map]
        seg_ends = [segment_map[sid].end for sid in u.entity_segment_ids if sid in segment_map]
        if seg_starts and seg_ends:
            dur = max(seg_ends) - min(seg_starts)
            u.priority = 1 if dur >= 40.0 else 2 if dur >= 25.0 else 3

    units.sort(key=lambda u: (u.priority, -sum(
        segment_map[sid].end - segment_map[sid].start
        for sid in u.entity_segment_ids if sid in segment_map
    )))

    # Final renumber after sort
    for i, u in enumerate(units):
        u.unit_id = i

    return units


def plan_to_structure_analysis(plan: StoryPlan, reasoning: str = "") -> dict:
    """Convert a StoryPlan into the legacy structure_analysis dict shape."""
    return {
        "video_type": plan.video_type,
        "identified_units": [
            {
                "name": u.name,
                "approx_start": 0.0,
                "approx_end": 0.0,
                "usable_seconds": 0,
                "kept": True,
            }
            for u in plan.units
        ],
        "final_group_count": len(plan.units),
        "reasoning": reasoning or f"{len(plan.units)} standalone story units from LLM story plan.",
    }


def plan_to_blocks_hint(plan: StoryPlan, blocks: list[SemanticBlock], source_duration: float) -> str:
    """Per-unit top-block hint for the narration writer (no LLM in the loop)."""
    lines = []
    for u in plan.units:
        w = _unit_window(u, source_duration)
        unit_blocks = [
            b for b in blocks
            if b.end > w[0] and b.start < w[1] and not b.black_frame and not b.freeze
        ]
        unit_blocks.sort(key=lambda b: b.importance, reverse=True)
        top5 = unit_blocks[:5]
        if top5:
            block_strs = ", ".join(
                f"Block {b.block_id} (imp={b.importance:.0f}, peak=+{b.peak_offset:.1f}s)"
                for b in top5
            )
            lines.append(f"  {u.name} [{w[0]:.0f}-{w[1]:.0f}s]: {block_strs}")
    return "\n".join(lines)


def _unit_window(unit: Unit, source_duration: float) -> tuple[float, float]:
    """Deterministic region -> source window for a single unit (before subdivision)."""
    if unit.region == "early":
        return 0.0, source_duration * 0.25
    if unit.region == "mid":
        return source_duration * 0.25, source_duration * 0.75
    return source_duration * 0.75, source_duration


# Standard region spans — used for most content types.
# For game/challenge content these are overridden by GAME_REGION_SPANS.
_REGION_SPANS = {"early": (0.0, 0.25), "mid": (0.25, 0.75), "late": (0.75, 1.0)}

# Narrower spans for game/challenge content:
# - early: 0-15% (setup/stakes intro — just the very beginning)
# - late: 70-100% (winner reveal lives here, not buried in credits window)
# - mid: unchanged (the contest/game itself)
_GAME_REGION_SPANS = {"early": (0.0, 0.15), "mid": (0.15, 0.70), "late": (0.70, 1.0)}


def _unit_windows(
    plan: StoryPlan,
    source_duration: float,
    content_type: str = "",
    entity_segments: list | None = None,
) -> dict[int, tuple[float, float]]:
    """Assign each unit a deterministic source window.

    For entity-grouped units, windows come from entity segment boundaries.
    For region-based units, units in the same region subdivide that region's
    span equally, in unit_id order, so two units never compete for the same
    seconds.

    For game/challenge content, narrower region spans keep the winner reveal
    in the true final stretch and the setup in the actual opening.
    """
    # Single unit reel (max_groups = 1): span the FULL video from 0.0 to source_duration
    # so Hook is drawn from start (0-20%), Journey from mid (15-70%), and Payoff from real end (70-100%)!
    # Exception: entity-grouped units always use their segment boundaries.
    has_entity_units = any(u.entity_segment_ids for u in plan.units)
    if len(plan.units) == 1 and not has_entity_units:
        return {plan.units[0].unit_id: (0.0, source_duration)}

    # Entity-grouped mode: windows come from entity segment boundaries.
    # For non-adjacent merged segments (effective_ranges set), use the
    # union of actual block time ranges instead of seg.start..seg.end
    # which would span across other speakers' time.
    if has_entity_units and entity_segments:
        entity_map = {s.entity_segment_id: s for s in entity_segments}
        entity_windows: dict[int, tuple[float, float]] = {}
        for u in plan.units:
            if u.entity_segment_ids:
                all_starts: list[float] = []
                all_ends: list[float] = []
                for sid in u.entity_segment_ids:
                    seg = entity_map.get(sid)
                    if seg is None:
                        continue
                    if seg.effective_ranges:
                        # Non-adjacent merged — use actual sub-ranges
                        for r_start, r_end in seg.effective_ranges:
                            all_starts.append(r_start)
                            all_ends.append(r_end)
                    else:
                        all_starts.append(seg.start)
                        all_ends.append(seg.end)
                if all_starts and all_ends:
                    win_start = min(all_starts)
                    win_end = max(all_ends)
                    # Expand window by 60% to include adjacent content for extension
                    win_dur = win_end - win_start
                    expand = win_dur * 0.6
                    win_start = max(0.0, win_start - expand)
                    win_end = min(source_duration, win_end + expand)
                    entity_windows[u.unit_id] = (round(win_start, 2), round(win_end, 2))
        if entity_windows:
            return entity_windows

    # Region-based mode (default)
    narrow_genres = (GENRE_GAME_CHALLENGE, GENRE_EXPERIMENT, GENRE_SPORTS_FITNESS)
    if content_type in narrow_genres:
        region_spans = _GAME_REGION_SPANS
    else:
        if plan.video_type in ("challenge", "experiment", "sports", *narrow_genres):
            region_spans = _GAME_REGION_SPANS
        else:
            region_spans = _REGION_SPANS

    windows: dict[int, tuple[float, float]] = {}
    for region in region_spans:
        region_units = [u for u in plan.units if u.region == region]
        if not region_units:
            continue
        lo_f, hi_f = region_spans[region]
        lo, hi = source_duration * lo_f, source_duration * hi_f
        span = hi - lo
        k = len(region_units)
        for i, u in enumerate(sorted(region_units, key=lambda x: x.unit_id)):
            w_lo = lo + span * i / k
            w_hi = lo + span * (i + 1) / k
            windows[u.unit_id] = (w_lo, w_hi)
    return windows
