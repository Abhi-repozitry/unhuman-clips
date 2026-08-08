"""Plan executor — deterministically maps a story plan to reel groups.

This module is the "Python picks" half of the LLM-plans/Python-executes
architecture. Given a validated ``StoryPlan`` (regions + beats, no timestamps)
and the pre-scored semantic blocks, it selects concrete source clips using the
same scoring primitives the legacy path used:

- ``_score_clip_as_hook`` for the hook beat (curiosity scoring)
- block importance / energy / engagement flags for escalation beats
- flag-matched, end-anchored importance for the payoff beat

Optionally, an LLM "relevance ranker" (run upstream in analyzer.py) supplies
per-beat ranked block-ID lists. Those act as a semantic layer: ranked blocks
get a score bonus so the picked clip actually DELIVERS the beat's intent
("tension as he walks to the pitch") instead of just being the loudest block.
All placement math (windows, slots, budgets, order) stays fully deterministic;
the ranker only weights *which* block wins inside an already-fixed slot.

Identical input (plan + blocks + fixed relevance) => identical output, always.
"""
from __future__ import annotations

import logging
from typing import Any

from backend.config import MIN_ENTITY_REEL_SECONDS, MIN_USABLE_BLOCK_FRACTION
from backend.models import ContentIdentity
from backend.pipeline.analyzer import (
    CLIP_LONG_MAX_SECONDS,
    CLIP_MEDIUM_MAX_SECONDS,
    CLIP_SHORT_MAX_SECONDS,
    SemanticBlock,
    _score_clip_as_hook,
)
from backend.pipeline.plan_schema import Beat, StoryPlan, Unit, _unit_windows

__all__ = [
    "assign_reel_summary",
    "estimate_speech_duration",
    "execute_plan",
    "place_narration_events",
    "post_execution_qa",
]

logger = logging.getLogger(__name__)

HOOK_MIN_SECONDS = 4.0
HOOK_MAX_SECONDS = 10.0                             # default; overridden per content type
PAYOFF_MIN_SECONDS = 6.0                            # default; overridden per content type
PAYOFF_MAX_SECONDS = CLIP_MEDIUM_MAX_SECONDS       # payoff stays MEDIUM (6-15s)
# Extra runway (on top of PAYOFF_MAX_SECONDS) allowed ONLY for a short,
# tightly-adjacent trailing "conclusion" block (e.g. a result/winner
# announcement that lands in its own block right after the payoff moment,
# often after a short dramatic pause before the reveal line).
PAYOFF_EPILOGUE_MAX_SECONDS = 9.0
# Only bridge a small gap — this absorbs a block split by a pause, not a
# jump to unrelated later content.
PAYOFF_EPILOGUE_GAP_TOLERANCE = 3.5
ESCALATION_MAX_SECONDS = CLIP_MEDIUM_MAX_SECONDS
MIN_CLIP_SECONDS = 3.0
WINDOW_EDGE_GAP = 0.0

# Scene-cut snapping: clip boundaries within this tolerance of a real scene
# cut are nudged to the cut for cleaner editorial transitions.
SCENE_CUT_SNAP_TOLERANCE = 0.5

# Content-type-aware duration overrides.
# Fast-paced content (comedy, roast, sports) needs shorter hooks/payoffs;
# narrative/tutorial content needs longer ones.
_PAYOFF_MIN_BY_CONTENT: dict[str, float] = {
    "comedy_sketch": 3.0,
    "roast_reaction": 3.0,
    "sports_fitness": 3.0,
    "game_challenge": 4.0,
    "podcast_conversational": 8.0,
    "tutorial_educational": 5.0,
    "documentary_narrative": 7.0,
}
_HOOK_MAX_BY_CONTENT: dict[str, float] = {
    "comedy_sketch": 8.0,
    "roast_reaction": 8.0,
    "sports_fitness": 8.0,
    "podcast_conversational": 12.0,
    "tutorial_educational": 12.0,
    "documentary_narrative": 12.0,
}

# Score bonus by relevance rank: rank 0 (best) +25 ... rank 4 +5, rest 0.
RELEVANCE_BONUS_TIERS = (25.0, 18.0, 12.0, 8.0, 5.0)


def _relevance_bonus(ranked: list[int], block_id: int) -> float:
    """Deterministic score bonus for a block's position in a ranked list."""
    try:
        rank = ranked.index(block_id)
    except ValueError:
        return 0.0
    if rank < len(RELEVANCE_BONUS_TIERS):
        return RELEVANCE_BONUS_TIERS[rank]
    return 0.0


# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------

def _snippet(text: str, max_len: int = 60) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _block_has_flag(block: SemanticBlock, flag: str) -> bool:
    if flag == "VULGAR":
        return block.has_vulgarity
    if flag == "DATING":
        return block.has_dating
    if flag == "ROAST":
        return block.has_roast
    if flag == "STAKES":
        return block.has_stakes
    return False


def _block_any_flag(block: SemanticBlock) -> bool:
    return block.has_vulgarity or block.has_dating or block.has_roast or block.has_stakes


# Sponsor / ad-read phrases. Matched case-insensitively as substrings, so
# "use code SAVE20" and "USE CODE save20" both hit "use code". Kept
# deliberately short and high-precision (real ad-read boilerplate) rather
# than broad topical terms, so we don't accidentally filter out genuine
# content that happens to mention a brand in passing.
_SPONSOR_PHRASES = (
    "use code", "promo code", "discount code", "coupon code",
    "% off", "percent off",
    "link in description", "link in bio", "link below",
    "sponsored by", "this video is sponsored", "this episode is sponsored",
    "thanks to our sponsor", "today's sponsor", "brought to you by",
    "download the app", "sign up at", "sign up now",
    "swipe up", "click the link", "check out the link",
    "free shipping", "limited time offer", "use my code",
)


def _is_sponsor_or_offtopic(block: SemanticBlock) -> bool:
    """True if a block's text reads like an ad read / sponsor segment.

    Applied at the candidate-pool level (``_window_blocks``) so ad reads are
    hard-excluded from every beat's pool, not merely scored lower — a
    high-energy sponsor read can otherwise still out-score genuine content
    on importance/energy alone (see: a Mr Beast chocolate-bar ad winning an
    escalation slot).
    """
    text = (block.text or "").lower()
    if not text:
        return False
    return any(phrase in text for phrase in _SPONSOR_PHRASES)


def _trim_block(block: SemanticBlock, max_seconds: float, keep_end: bool = False) -> tuple[float, float]:
    """Trim a block range to at most *max_seconds*, centered on the peak moment.

    With ``keep_end=True`` the reveal tail is preserved (used for payoffs).
    """
    start, end = block.start, block.end
    duration = end - start
    if duration <= max_seconds:
        return start, end
    if keep_end:
        return max(start, end - max_seconds), end
    peak = min(max(start, block.start + block.peak_offset), end)
    new_start = peak - max_seconds / 2.0
    new_end = peak + max_seconds / 2.0
    if new_start < start:
        new_start, new_end = start, start + max_seconds
    if new_end > end:
        new_end, new_start = end, end - max_seconds
    return max(start, new_start), min(end, new_end)


def _snap_to_scene_cut(
    t: float, scene_cuts: list[float], tolerance: float = SCENE_CUT_SNAP_TOLERANCE,
) -> float:
    """Snap a timestamp to the nearest scene cut within *tolerance*.

    Returns the original timestamp unchanged if no scene cut is close enough.
    """
    best_dist = tolerance + 1.0
    best = t
    for cut in scene_cuts:
        d = abs(cut - t)
        if d < best_dist:
            best_dist = d
            best = cut
    return best if best_dist <= tolerance else t


def _snap_clip_boundaries(
    clips: list[dict[str, Any]],
    scene_cuts: list[float],
) -> None:
    """Snap clip start/end boundaries to nearby scene cuts.

    Only nudges boundaries that fall within SCENE_CUT_SNAP_TOLERANCE of a real
    scene cut.  Strictly corrective: never changes which blocks were selected,
    only nudges the exact in/out point for cleaner editorial transitions.

    Mutates clips in-place.
    """
    if not scene_cuts:
        return
    sorted_cuts = sorted(scene_cuts)
    for clip in clips:
        orig_start = clip["source_start"]
        orig_end = clip["source_end"]
        snapped_start = _snap_to_scene_cut(orig_start, sorted_cuts)
        snapped_end = _snap_to_scene_cut(orig_end, sorted_cuts)
        # Respect the original constraint: snapped start must be before snapped end
        if snapped_start < snapped_end:
            clip["source_start"] = round(snapped_start, 3)
            clip["source_end"] = round(snapped_end, 3)
        elif snapped_start != orig_start or snapped_end != orig_end:
            # Snapping would invert the range — keep original boundaries.
            # This can happen when both start and end are near the same cut.
            logger.debug(
                f"Scene-cut snap skipped for clip [{orig_start:.2f}-{orig_end:.2f}]: "
                f"would invert to [{snapped_start:.2f}-{snapped_end:.2f}]"
            )


# ---------------------------------------------------------------------------
# Region windows
# ---------------------------------------------------------------------------

# _unit_windows lives in plan_schema.py (imported at module top) so both the
# executor and the analyzer's ranker prompt use the SAME per-unit windows.


# ---------------------------------------------------------------------------
# Beat -> clip selection
# ---------------------------------------------------------------------------

def _window_blocks(
    window: tuple[float, float],
    blocks: list[SemanticBlock],
    reserved_ranges: list[tuple[float, float]],
    reserved_ids: set[int],
) -> list[SemanticBlock]:
    w_lo, w_hi = window
    raw = []
    for b in blocks:
        if b.end <= w_lo + WINDOW_EDGE_GAP or b.start >= w_hi - WINDOW_EDGE_GAP:
            continue
        if b.block_id in reserved_ids:
            continue
        if any(max(b.start, r0) < min(b.end, r1) - 0.05 for r0, r1 in reserved_ranges):
            continue
        raw.append(b)
    if not raw:
        return []

    usable = [b for b in raw if not (b.black_frame or b.freeze) and not _is_sponsor_or_offtopic(b)]
    strict = [b for b in usable if b.importance >= 25]
    # Soft gate: if the importance filter would leave almost nothing to pick
    # from (e.g. VAD produced no speech energy), relax it instead of starving
    # the beat of all candidates. Black/freeze stays excluded either way.
    if len(usable) and len(strict) / len(usable) < MIN_USABLE_BLOCK_FRACTION:
        return usable
    return strict


def _pick_payoff_block(
    candidates: list[SemanticBlock],
    unit: Unit,
    window: tuple[float, float],
    payoff_ranked: list[int] | None = None,
) -> SemanticBlock | None:
    """Pick the payoff block — strongly prefers the LATEST-occurring moment.

    The payoff (winner reveal, reaction, final punchline) should almost always
    be the chronologically last strong moment in the window. We tighten the
    late_threshold to the last 25% of the window (was 40%) and add a strong
    recency bonus so that, all else equal, the later block wins.
    """
    w_lo, w_hi = window
    span = max(1.0, w_hi - w_lo)
    payoff_flags = {f for b in unit.arc if b.beat == "payoff" for f in b.flags}
    # 25% threshold — payoff must come from the final quarter of the window
    late_threshold = w_lo + span * 0.75
    payoff_ranked = payoff_ranked or []

    def _score(b: SemanticBlock) -> float:
        s = 0.0
        if payoff_flags and any(_block_has_flag(b, f) for f in payoff_flags):
            s += 50.0
        elif payoff_flags and _block_any_flag(b):
            s += 20.0
        s += min(30.0, b.importance * 0.3)
        s += _relevance_bonus(payoff_ranked, b.block_id)
        center = (b.start + b.end) / 2.0
        # Tiered recency bonus — the later in the window, the higher the bonus
        if center >= late_threshold:
            # Map position from late_threshold→w_hi to +15→+30
            recency_frac = min(1.0, (center - late_threshold) / max(1.0, w_hi - late_threshold))
            s += 15.0 + recency_frac * 15.0
        elif center >= w_lo + span * 0.5:
            s += 5.0  # small bonus for mid-window
        if b.silence_before:
            s += 5.0
        # Retention signal bonus for payoffs
        if hasattr(b, "has_viral_trigger") and b.has_viral_trigger:
            s += 10.0
        return s

    ranked = sorted(candidates, key=lambda b: (round(_score(b), 3), -b.start))
    return ranked[-1] if ranked else None


def _pick_hook_block(
    candidates: list[SemanticBlock],
    unit: Unit,
    blocks: list[SemanticBlock],
    payoff_block: SemanticBlock | None,
    hook_ranked: list[int] | None = None,
    entity_names: set[str] | None = None,
) -> SemanticBlock | None:
    if not candidates:
        return None

    hook_flags = {f for b in unit.arc if b.beat == "hook" for f in b.flags}
    hook_ranked = hook_ranked or []

    # Early cutoff for opening hook preference (0-25% of candidates span)
    cand_min = min(b.start for b in candidates)
    cand_max = max(b.end for b in candidates)
    cand_span = max(1.0, cand_max - cand_min)
    early_cutoff = cand_min + cand_span * 0.25

    def _usable(b: SemanticBlock) -> bool:
        if payoff_block is None:
            return True
        return not (b.end > payoff_block.start and b.start < payoff_block.end)

    best: tuple[float, float] | None = None
    best_block: SemanticBlock | None = None
    for b in candidates:
        usable = _usable(b)
        clip = {"source_start": b.start, "source_end": b.end}
        score = _score_clip_as_hook(clip, blocks)
        score += _relevance_bonus(hook_ranked, b.block_id)
        if hook_flags and any(_block_has_flag(b, f) for f in hook_flags):
            score += 10.0
        elif _block_any_flag(b):
            score += 2.0

        # Mild opening position preference: slightly prefer earlier blocks
        # but don't let timing dominate over content quality. A fluff block
        # at 5s should NOT beat a meaningful intro at 30s.
        if (b.start + b.end) / 2.0 <= early_cutoff:
            score += 5.0

        # Entity name bonus: prefer blocks that mention the named entity
        if entity_names and b.text:
            text_lower = b.text.lower()
            for name in entity_names:
                if name.lower() in text_lower:
                    score += 15.0
                    break

        if not usable:
            # Overlaps the payoff block — strongly avoid it, but don't
            # exclude it outright. A hard `continue` here meant that when
            # EVERY candidate overlapped the payoff (a tight window), the
            # hook/start beat silently got no clip at all — the "no start"
            # symptom. A heavy penalty keeps the normal case (plenty of
            # non-overlapping candidates) unaffected while guaranteeing a
            # last-resort pick when nothing else is available.
            score -= 1000.0

        key = (round(score, 3), -b.start)
        if best is None or key > best:
            best = key
            best_block = b
    return best_block


_POS_HINT_FRACTION = {"start": 0.2, "any": 0.5, "end": 0.8}


def _apply_pacing_strategy(clips: list[dict[str, Any]], blocks: list[Any], window: tuple[float, float]) -> None:
    """Adjust clip durations to create short→medium→long pacing.

    Pattern:
    - Hook: SHORT (4-8s)
    - Early escalation: SHORT (3-6s) - quick cuts
    - Middle escalation: MEDIUM (8-15s) - building
    - Late escalation: LONG (15-25s) - sustained
    - Payoff: MEDIUM (6-12s)
    """
    if len(clips) <= 2:
        return

    block_lookup = {b.block_id: b for b in blocks}
    source_duration = window[1] - window[0]

    esc_clips = [c for c in clips if c.get("beat") == "escalation"]
    n_esc = len(esc_clips)
    if n_esc == 0:
        return

    for i, c in enumerate(esc_clips):
        src_start = c["source_start"]
        src_end = c["source_end"]
        max_possible = src_end - src_start

        if n_esc == 1:
            target = min(CLIP_MEDIUM_MAX_SECONDS, max_possible)
        elif n_esc == 2:
            target = min(CLIP_SHORT_MAX_SECONDS if i == 0 else CLIP_MEDIUM_MAX_SECONDS, max_possible)
        else:
            ratio = i / max(1, n_esc - 1)
            if ratio <= 0.33:
                target = min(CLIP_SHORT_MAX_SECONDS, max_possible)
            elif ratio <= 0.66:
                target = min(CLIP_MEDIUM_MAX_SECONDS, max_possible)
            else:
                target = min(CLIP_LONG_MAX_SECONDS, max_possible)

        current_dur = c["source_end"] - c["source_start"]
        if current_dur > target:
            c["source_end"] = round(c["source_start"] + target, 3)


def _pick_escalation_blocks(
    candidates: list[SemanticBlock],
    hook_block: SemanticBlock | None,
    payoff_block: SemanticBlock | None,
    budget: float,
    window: tuple[float, float],
    esc_beats: list[Beat],
    esc_ranked: list[int] | None = None,
    entity_names: set[str] | None = None,
) -> list[SemanticBlock]:
    """Pick escalation blocks wired to POSITIONS across the unit window.

    The window is split into one slot per escalation the reel needs (budget
    driven, capped at 6). Each slot contributes its best block, so escalation
    clips spread across the unit's region instead of clustering around the
    loudest moments. Beat position hints ("start"/"any"/"end") steer slot
    order: hinted slots are picked first so the LLM's coarse staging is
    honored, then the remaining slots fill in source order.

    Within a slot, LLM-ranked blocks (beat relevance) win over mere importance,
    so the clip content matches the beat's intent ("tension as he walks to the
    pitch") instead of just being the loudest moment.
    """
    esc_ranked = esc_ranked or []
    used = {b.block_id for b in (hook_block, payoff_block) if b is not None}
    pool = [
        b for b in candidates
        if b.block_id not in used
        and not (payoff_block is not None and b.end >= payoff_block.start)
    ]
    if not pool:
        return []

    w_lo, w_hi = window
    span = max(1.0, w_hi - w_lo)
    k = max(1, min(13, int(budget // ESCALATION_MAX_SECONDS) + 1))
    slot_w = span / k

    def _slot_index(b: SemanticBlock) -> int:
        idx = int((b.start + b.end) / 2.0 - w_lo) // slot_w
        return max(0, min(k - 1, idx))

    hint_order: list[int] = []
    for beat in esc_beats:
        frac = _POS_HINT_FRACTION.get(str(beat.position), 0.5)
        idx = round(frac * k)
        idx = max(0, min(k - 1, idx))
        if idx not in hint_order:
            hint_order.append(idx)
    slot_order = hint_order + [i for i in range(k) if i not in hint_order]

    by_slot: dict[int, list[SemanticBlock]] = {}
    for b in pool:
        by_slot.setdefault(_slot_index(b), []).append(b)

    picked: list[SemanticBlock] = []
    total = 0.0
    for i in slot_order:
        slot_pool = by_slot.get(i, [])
        if not slot_pool:
            continue

        def _esc_score(b: SemanticBlock) -> float:
            score = _relevance_bonus(esc_ranked, b.block_id) + b.importance * 0.3
            if payoff_block is not None and b.start < payoff_block.start:
                dist = payoff_block.start - b.end
                if 0.0 <= dist <= 25.0:
                    score += 15.0 * (1.0 - dist / 25.0)
            # Gameplay/action content bonus: STAKES, VIRAL, and INTERRUPT flags
            # mark moments with real substance — near-misses, dramatic reveals,
            # action peaks — that make escalation feel like it's building.
            if b.has_stakes:
                score += 12.0
            if b.has_viral_trigger:
                score += 10.0
            if b.has_pattern_interrupt:
                score += 8.0
            # On-topic bonus: prefer blocks that actually mention the named
            # entity/action over adjacent banter or sponsor-adjacent chatter
            # that happens to score well on energy alone.
            if entity_names and b.text:
                text_lower = b.text.lower()
                if any(name.lower() in text_lower for name in entity_names):
                    score += 15.0
            return score

        best = max(
            slot_pool,
            key=lambda b: (_esc_score(b), -b.start),
        )
        dur = min(best.duration, ESCALATION_MAX_SECONDS)
        if total + dur > budget and total > 0.0:
            continue
        picked.append(best)
        total += dur
    return picked


def _clip_from_block(block: SemanticBlock, max_seconds: float, keep_end: bool) -> dict[str, Any]:
    start, end = _trim_block(block, max_seconds, keep_end=keep_end)
    if end - start < MIN_CLIP_SECONDS:
        end = min(block.end, start + MIN_CLIP_SECONDS)
    return {"source_start": round(start, 3), "source_end": round(end, 3)}


def _clamp_clip_to_entity_segments(
    clip: dict[str, Any],
    entity_segments: list[EntitySegment] | None,
    entity_segment_ids: list[str] | None,
) -> None:
    """Clamp a clip's time range to the boundaries of its entity segment.

    For non-adjacent merged segments (effective_ranges set), finds the
    specific sub-range containing the clip's midpoint.  For contiguous
    segments, uses seg.start/seg.end directly.  Mutates clip in-place.
    """
    if not entity_segments or not entity_segment_ids:
        return
    mid = (clip["source_start"] + clip["source_end"]) / 2.0
    for seg in entity_segments:
        if seg.entity_segment_id not in entity_segment_ids:
            continue
        # Determine which time range to clamp against
        if seg.effective_ranges:
            # Non-adjacent merged segment — find the sub-range containing mid
            for r_start, r_end in seg.effective_ranges:
                if r_start - 0.1 <= mid <= r_end + 0.1:
                    if clip["source_end"] > r_end + 0.1:
                        clip["source_end"] = round(r_end, 3)
                    if clip["source_start"] < r_start - 0.1:
                        clip["source_start"] = round(r_start, 3)
                    return
            # Mid not in any effective range — don't clamp (clip is in a gap)
            return
        else:
            # Contiguous segment — use start/end directly
            if seg.start - 0.1 <= mid <= seg.end + 0.1:
                if clip["source_end"] > seg.end + 0.1:
                    clip["source_end"] = round(seg.end, 3)
                if clip["source_start"] < seg.start - 0.1:
                    clip["source_start"] = round(seg.start, 3)
            return


def _extend_short_clip(
    clip: dict[str, Any],
    blocks: list[SemanticBlock],
    window: tuple[float, float],
    target_seconds: float,
    block_ids_used: set[int],
    reserved_ranges: list[tuple[float, float]] | None = None,
    entity_segment_ids: list[str] | None = None,
    entity_segments: list[EntitySegment] | None = None,
) -> dict[str, Any]:
    """Grow a too-short clip by absorbing adjacent blocks within the window.

    Blocks that overlap a higher-priority unit's reserved range are never
    absorbed, so cross-group overlap can't sneak in via extension.
    When entity_segment_ids is provided, only blocks from those segments
    are absorbed — clips cannot cross entity boundaries.
    When entity_segments is provided, the new endpoint is verified to stay
    within the entity's effective_ranges (prevents crossing gaps).
    """
    reserved_ranges = reserved_ranges or []

    # Precompute entity time boundaries for gap checks
    entity_ranges: list[tuple[float, float]] = []
    if entity_segments and entity_segment_ids:
        for seg in entity_segments:
            if seg.entity_segment_id in entity_segment_ids:
                if seg.effective_ranges:
                    entity_ranges.extend(seg.effective_ranges)
                else:
                    entity_ranges.append((seg.start, seg.end))

    while clip["source_end"] - clip["source_start"] < target_seconds - 0.01:
        candidates = [
            b for b in blocks
            if b.block_id not in block_ids_used
            and not b.black_frame and not b.freeze
            and b.start >= clip["source_start"] and b.start <= clip["source_end"] + 0.5
            and b.end <= window[1] + 0.1
            and not any(max(b.start, r0) < min(b.end, r1) - 0.05 for r0, r1 in reserved_ranges)
        ]
        # Entity boundary enforcement: only absorb blocks from the same entity segments
        if entity_segment_ids:
            candidates = [b for b in candidates if b.entity_segment_id in entity_segment_ids]
        if not candidates:
            break
        nxt = min(candidates, key=lambda b: b.start)
        new_end = round(max(clip["source_end"], nxt.end), 3)
        # Gap check: verify new endpoint stays within an entity range
        if entity_ranges:
            in_range = any(
                r_start - 0.1 <= new_end <= r_end + 0.1
                for r_start, r_end in entity_ranges
            )
            if not in_range:
                # Extending would cross into a gap — clamp to nearest range boundary
                break
        clip["source_end"] = new_end
        block_ids_used.add(nxt.block_id)
    return clip


def _extend_payoff_epilogue(
    clip: dict[str, Any],
    blocks: list[SemanticBlock],
    window: tuple[float, float],
    block_ids_used: set[int],
    reserved_ranges: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Absorb ONE short, tightly-adjacent trailing block past PAYOFF_MAX_SECONDS.

    A payoff's real conclusion ("...and the gold medal goes to...") often
    sits in its own short semantic block immediately after the climax
    moment — sometimes split further by a dramatic pause into two short
    blocks (the suspense beat, then the announcement line) — exactly where
    the PAYOFF_MAX_SECONDS cap would otherwise cut the clip off. This
    bridges gaps of at most PAYOFF_EPILOGUE_GAP_TOLERANCE between
    consecutive blocks, looping to chain more than one short trailing block
    together, but only ever while the resulting clip stays within
    PAYOFF_MAX_SECONDS + PAYOFF_EPILOGUE_MAX_SECONDS, so it can't turn into
    another unbounded absorption.
    """
    reserved_ranges = reserved_ranges or []
    hard_cap = clip["source_start"] + PAYOFF_MAX_SECONDS + PAYOFF_EPILOGUE_MAX_SECONDS
    while True:
        candidates = [
            b for b in blocks
            if b.block_id not in block_ids_used
            and not b.black_frame and not b.freeze
            and b.start >= clip["source_end"] - 0.1
            and b.start <= clip["source_end"] + PAYOFF_EPILOGUE_GAP_TOLERANCE
            and b.end <= window[1] + 0.1
            and b.end <= hard_cap
            and not any(max(b.start, r0) < min(b.end, r1) - 0.05 for r0, r1 in reserved_ranges)
        ]
        if not candidates:
            break
        nxt = min(candidates, key=lambda b: b.start)
        clip["source_end"] = round(max(clip["source_end"], nxt.end), 3)
        block_ids_used.add(nxt.block_id)
    return clip


def _extend_clips_to_fill(
    clips: list[dict[str, Any]],
    target: float,
    source_duration: float,
    window: tuple[float, float],
    blocks: list[SemanticBlock],
    used_block_ids: set[int],
    reserved_ranges: list[tuple[float, float]] | None = None,
    entity_block_ids: set[int] | None = None,
) -> int:
    """Extend clips to reach *target* total by absorbing valuable content.

    Strategy (in priority order):
    1. Absorb unselected high-importance blocks adjacent to existing clips.
    2. Extend into gaps only as a last resort (lower-value filler content).

    When entity_block_ids is provided, only blocks from those segments are
    eligible — clips cannot cross entity boundaries.

    Returns the number of clips that were extended.
    """
    reserved_ranges = reserved_ranges or []
    total = sum(c["source_end"] - c["source_start"] for c in clips)
    if total >= target - 0.5:
        return 0
    deficit = target - total
    w_lo, w_hi = window
    extended = 0

    # --- Pass 1: absorb unselected blocks adjacent to clips, sorted by importance ---
    unused_blocks = sorted(
        (b for b in blocks
         if b.block_id not in used_block_ids
         and not b.black_frame and not b.freeze
         and b.start >= w_lo and b.end <= w_hi + 0.1
         and not any(max(b.start, r0) < min(b.end, r1) - 0.05
                     for r0, r1 in reserved_ranges)
         and (entity_block_ids is None or b.block_id in entity_block_ids)),
        key=lambda b: -b.importance,
    )
    for blk in unused_blocks:
        if deficit <= 0.5:
            break
        # Find the clip whose range is closest to this block
        best_clip = None
        best_gap = float("inf")
        for clip in clips:
            gap_to_end = blk.start - clip["source_end"]
            gap_from_start = clip["source_start"] - blk.end
            if 0.0 <= gap_to_end <= 5.0 and gap_to_end < best_gap:
                best_gap = gap_to_end
                best_clip = (clip, "end")
            if 0.0 <= gap_from_start <= 5.0 and gap_from_start < best_gap:
                best_gap = gap_from_start
                best_clip = (clip, "start")
        if best_clip is None:
            continue
        clip, side = best_clip
        if side == "end":
            raw_end = min(blk.end, w_hi, source_duration)
            # Don't extend past the next clip's start
            clip_idx = clips.index(clip)
            if clip_idx + 1 < len(clips):
                next_start = clips[clip_idx + 1]["source_start"]
                raw_end = min(raw_end, next_start - 0.1)
            # Cap extension to deficit so we don't absorb oversized blocks
            capped_end = min(raw_end, clip["source_end"] + deficit)
            actual = capped_end - clip["source_end"]
            if actual > 0.1:
                clip["source_end"] = round(capped_end, 3)
                deficit -= actual
                extended += 1
                used_block_ids.add(blk.block_id)
        else:
            raw_start = max(blk.start, w_lo, 0.0)
            # Don't extend past the previous clip's end
            clip_idx = clips.index(clip)
            if clip_idx > 0:
                prev_end = clips[clip_idx - 1]["source_end"]
                raw_start = max(raw_start, prev_end + 0.1)
            capped_start = max(raw_start, clip["source_start"] - deficit)
            actual = clip["source_start"] - capped_start
            if actual > 0.1:
                clip["source_start"] = round(capped_start, 3)
                deficit -= actual
                extended += 1
                used_block_ids.add(blk.block_id)

    # --- Pass 2: fill remaining deficit by extending into gaps (lower-value) ---
    # Skip for entity mode — extending into gaps would cross entity boundaries
    if deficit > 0.5 and entity_block_ids is None:
        per_clip = deficit / max(len(clips), 1)
        for clip in clips:
            if deficit <= 0.5:
                break
            extend = min(per_clip, deficit)
            new_end = min(clip["source_end"] + extend, w_hi, source_duration)
            actual = new_end - clip["source_end"]
            if actual > 0.1 and not any(
                max(clip["source_end"], r0) < min(new_end, r1) - 0.05
                for r0, r1 in reserved_ranges
            ):
                clip["source_end"] = round(new_end, 3)
                deficit -= actual
                extended += 1
                continue
            new_start = max(clip["source_start"] - extend, w_lo, 0.0)
            actual = clip["source_start"] - new_start
            if actual > 0.1 and not any(
                max(new_start, r0) < min(clip["source_start"], r1) - 0.05
                for r0, r1 in reserved_ranges
            ):
                clip["source_start"] = round(new_start, 3)
                deficit -= actual
                extended += 1
    return extended


def _beat_intent(unit: Unit, beat: str) -> str:
    for b in unit.arc:
        if b.beat == beat and b.intent:
            return b.intent
    return ""


# ---------------------------------------------------------------------------
# Group assembly
# ---------------------------------------------------------------------------

def _build_group_dict(
    unit: Unit,
    clips: list[dict[str, Any]],
    window: tuple[float, float],
    source_duration: float,
) -> dict[str, Any]:
    total = sum(c["source_end"] - c["source_start"] for c in clips)
    counts = {"Short": 0, "Medium": 0, "Long": 0}
    for c in clips:
        dur = c["source_end"] - c["source_start"]
        if dur <= CLIP_SHORT_MAX_SECONDS:
            counts["Short"] += 1
        elif dur <= CLIP_MEDIUM_MAX_SECONDS:
            counts["Medium"] += 1
        else:
            counts["Long"] += 1
    arc_desc = " -> ".join(b.beat for b in unit.arc)
    hook_reason = next((c["reason"] for c in clips if c.get("is_hook_clip")), "")
    return {
        "group_index": unit.unit_id,
        "name": unit.name,
        "group_reasoning": (
            f"{counts['Short']} Short:{counts['Medium']} Medium:{counts['Long']} Long, "
            f"{total:.1f}s total. Arc: {arc_desc}. {hook_reason}"
        ),
        "estimated_duration_seconds": round(total + 2.0, 1),
        "reel_summary": {
            "title": unit.name[:60],
            "short_description": "",
            "source_understanding": ", ".join(b.intent for b in unit.arc if b.intent)[:300] or "story unit",
            "narrative_angle": f"{unit.region} region, priority {unit.priority}",
            "key_moment": _beat_intent(unit, "payoff") or "the payoff / reveal",
        },
        "source_clips": clips,
        "narration_events": [],
    }


def _reorder_reel_clips(clips: list[dict[str, Any]]) -> None:
    """Restore reel order after deterministic helpers re-sorted the list:
    hook first, escalation by source time, payoff last.

    The hook/start clip always opens the reel regardless of source time.
    Escalation clips play in chronological order for smooth temporal flow.
    Payoff always closes the reel.
    """
    beat_rank = {"hook": 0, "start": 0, "escalation": 1, "payoff": 2}

    def _key(c: dict[str, Any]) -> tuple[int, float]:
        return (beat_rank.get(c.get("beat") or c.get("_beat", "escalation"), 1), c.get("source_start", 0.0))

    clips.sort(key=_key)


def _trim_clips_to_budget(
    clips: list[dict[str, Any]],
    budget: float,
    payoff_min: float = PAYOFF_MIN_SECONDS,
) -> None:
    """Shrink clip ranges until total <= budget.

    The payoff clip keeps its END (the reveal); escalation/hook clips shrink
    from the end so their clean entry stays intact.
    """
    total = sum(c["source_end"] - c["source_start"] for c in clips)
    if total <= budget or len(clips) <= 1:
        return

    over = total - budget

    payoff = clips[-1]
    payoff_trim = min(over, max(0.0, payoff["source_end"] - payoff["source_start"] - payoff_min))
    payoff["source_start"] = round(payoff["source_start"] + payoff_trim, 3)
    over -= payoff_trim

    for c in reversed(clips[1:-1]):
        if over <= 0:
            break
        trim = min(over, max(0.0, c["source_end"] - c["source_start"] - MIN_CLIP_SECONDS))
        c["source_end"] = round(c["source_end"] - trim, 3)
        over -= trim

    if over > 0:
        hook = clips[0]
        trim = min(over, max(0.0, hook["source_end"] - hook["source_start"] - MIN_CLIP_SECONDS))
        hook["source_end"] = round(hook["source_end"] - trim, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_plan(
    plan: StoryPlan,
    blocks: list[SemanticBlock],
    source_duration: float,
    reel_dur_min: int,
    reel_dur_max: int,
    relevance: dict[int, dict[str, list[int]]] | None = None,
    content_type: str = "",
    entity_segments: list | None = None,
    scene_cut_at: list[float] | None = None,
    content_identity: ContentIdentity | None = None,
) -> list[dict[str, Any]]:
    """Convert a validated story plan into reel_groups dicts (deterministic).

    ``relevance`` optionally maps unit_id -> beat ("hook"/"escalation"/"payoff")
    -> ranked block IDs (best first, from the LLM beat-relevance ranker). The
    ranking weights which block wins inside each fixed window/slot — every
    number (windows, slots, budgets, reel order) stays computed by Python.

    Groups appear in priority order; unit windows are exclusive per unit so
    source timestamps are never shared between groups.

    ``content_type`` is passed to _unit_windows to apply genre-appropriate
    region span tables (e.g. narrower early/late for game_challenge).

    ``entity_segments`` when provided enables entity-boundary-aware clip
    selection: candidates are restricted to the unit's entity segment blocks,
    and clips cannot cross entity boundaries.
    """
    relevance = relevance or {}
    windows = _unit_windows(plan, source_duration, content_type=content_type, entity_segments=entity_segments)
    reserved_ranges: list[tuple[float, float]] = []
    reserved_ids: set[int] = set()

    # Resolve content-type-aware duration constants
    payoff_min = _PAYOFF_MIN_BY_CONTENT.get(content_type, PAYOFF_MIN_SECONDS)
    hook_max = _HOOK_MAX_BY_CONTENT.get(content_type, HOOK_MAX_SECONDS)

    groups: list[dict[str, Any]] = []
    # Process units in CHRONOLOGICAL (source-time) order — NOT plan.units'
    # own list order, which is priority-sorted (see plan_schema.parse_story_
    # plan) purely to decide which units survive the max_groups cap.
    # Reserved-range bookkeeping below must accumulate in time order: a
    # unit's payoff/epilogue often needs to reach into the small gap AFTER
    # its own footage to catch a trailing conclusion line, and the only
    # unit that should legitimately win that gap is whichever one's own
    # footage the gap actually follows. Processing in priority order let a
    # later-in-time-but-higher-priority unit's window run first and claim
    # that gap out from under the earlier one — exactly what produced
    # "group opens with the previous group's payoff" / "group's real
    # conclusion got cut off". Priority still fully governs which units
    # were kept; it should not also decide who wins a race for shared
    # boundary seconds.
    processing_order = sorted(
        plan.units, key=lambda u: windows.get(u.unit_id, (0.0, source_duration))[0]
    )
    for unit in processing_order:
        window = windows.get(unit.unit_id, (0.0, source_duration))
        candidates = _window_blocks(window, blocks, reserved_ranges, reserved_ids)

        if not candidates:
            # No usable blocks in this unit's window — skip it rather than
            # risk pulling blocks from other regions (which would corrupt
            # other groups' content).  The min_groups floor is enforced
            # later by the heuristic fallback if needed.
            logger.warning(
                f"Unit {unit.unit_id} ('{unit.name}'): no usable blocks in "
                f"[{window[0]:.0f}-{window[1]:.0f}]s — skipping (isolated groups)"
            )
            continue

        if not candidates:
            logger.warning(f"Unit {unit.unit_id} ('{unit.name}'): no usable blocks in window — skipping")
            continue

        # Entity mode: restrict HIGH-QUALITY candidates to entity segments
        # But allow extension to use ALL blocks in window (gap content)
        entity_block_ids: set[int] | None = None
        entity_window_only = False
        if unit.entity_segment_ids:
            entity_block_ids = set()
            for seg in (entity_segments or []):
                if seg.entity_segment_id in unit.entity_segment_ids:
                    entity_block_ids.update(seg.block_ids)
            entity_candidates = [b for b in candidates if b.block_id in entity_block_ids]
            if entity_candidates:
                candidates = entity_candidates
            # If multiple segments merged, allow extension beyond entity blocks
            if len(unit.entity_segment_ids) > 1:
                entity_window_only = False  # Allow gap content for merged groups
            else:
                entity_window_only = True

        # Duration targets for this unit
        unit_reel_dur_min = reel_dur_min
        unit_reel_dur_max = reel_dur_max

        # For merged entity groups, don't clamp to entity boundaries (allow gap content)
        should_clamp_entity = entity_window_only
        if entity_window_only:
            ext_entity_ids = unit.entity_segment_ids or None
            ext_entity_segs = entity_segments
        else:
            ext_entity_ids = None
            ext_entity_segs = None

        logger.debug(
            f"Unit {unit.unit_id}: duration target {unit_reel_dur_min}-{unit_reel_dur_max}s"
        )

        u_rel = relevance.get(unit.unit_id, {})
        hook_ranked = u_rel.get("hook", [])
        start_ranked = u_rel.get("start", [])
        esc_ranked = u_rel.get("escalation", [])
        payoff_ranked = u_rel.get("payoff", [])

        # Extract entity names for opening clip preference
        entity_names = set(content_identity.entity_names) if content_identity and content_identity.entity_names else None

        # Determine opening clip: hook (when present) or start (when no hook)
        has_hook_beat = any(b.beat == "hook" for b in unit.arc)
        has_start_beat = any(b.beat == "start" for b in unit.arc)

        payoff_block = _pick_payoff_block(candidates, unit, window, payoff_ranked) if unit.requires_payoff else None
        hook_block = _pick_hook_block(candidates, unit, blocks, payoff_block, hook_ranked, entity_names=entity_names) if has_hook_beat else None

        # When no hook beat exists (hook_mode="skip"), the start beat fills the opening slot
        start_block = None
        if not has_hook_beat and has_start_beat:
            start_block = _pick_hook_block(candidates, unit, blocks, payoff_block, start_ranked, entity_names=entity_names)

        # Escalation fills the space between opening + payoff and the max reel
        # budget. Overshoot is trimmed later; undershoot is topped up by the
        # compositor (last-clip extension) — never extend the payoff end here.
        payoff_reserve = PAYOFF_MAX_SECONDS if unit.requires_payoff else 0.0
        budget = max(0.0, unit_reel_dur_max - (hook_max + payoff_reserve))
        esc_beats = [b for b in unit.arc if b.beat == "escalation"]
        esc_blocks = _pick_escalation_blocks(candidates, hook_block or start_block, payoff_block, budget, window, esc_beats, esc_ranked, entity_names=entity_names)

        # Build concrete clips --------------------------------------------------
        used_blocks = {b.block_id for b in (hook_block, start_block, payoff_block, *esc_blocks) if b is not None}
        hook_clip = None
        if hook_block is not None:
            # Target 4.0 - {hook_max}s for hook clip
            target_dur = max(HOOK_MIN_SECONDS, min(hook_max, hook_block.duration))
            hook_clip = _clip_from_block(hook_block, target_dur, keep_end=False)
            hook_clip = _extend_short_clip(hook_clip, blocks, window, HOOK_MIN_SECONDS, used_blocks, reserved_ranges, entity_segment_ids=ext_entity_ids, entity_segments=ext_entity_segs)

            # Clamp hook clip duration strictly to HOOK_MIN_SECONDS - {hook_max}s range
            h_dur = hook_clip["source_end"] - hook_clip["source_start"]
            if h_dur < HOOK_MIN_SECONDS:
                hook_clip["source_end"] = round(min(window[1], hook_clip["source_start"] + HOOK_MIN_SECONDS), 3)
            elif h_dur > hook_max:
                hook_clip["source_end"] = round(hook_clip["source_start"] + hook_max, 3)
            # Entity boundary: don't let hook extend past its segment
            if should_clamp_entity:
                _clamp_clip_to_entity_segments(hook_clip, entity_segments, unit.entity_segment_ids or None)

            hook_clip["is_hook_clip"] = True
            hook_clip["beat"] = "hook"
            hook_clip["_beat"] = "hook"
            hook_clip["reason"] = (
                f"HOOK: {_beat_intent(unit, 'hook') or 'curiosity trigger'} — \"{_snippet(hook_block.text)}\""
            )
        elif start_block is not None:
            # Start beat fills the opening slot (no hook present)
            target_dur = max(HOOK_MIN_SECONDS, min(hook_max, start_block.duration))
            hook_clip = _clip_from_block(start_block, target_dur, keep_end=False)
            hook_clip = _extend_short_clip(hook_clip, blocks, window, HOOK_MIN_SECONDS, used_blocks, reserved_ranges, entity_segment_ids=ext_entity_ids, entity_segments=ext_entity_segs)

            h_dur = hook_clip["source_end"] - hook_clip["source_start"]
            if h_dur < HOOK_MIN_SECONDS:
                hook_clip["source_end"] = round(min(window[1], hook_clip["source_start"] + HOOK_MIN_SECONDS), 3)
            elif h_dur > hook_max:
                hook_clip["source_end"] = round(hook_clip["source_start"] + hook_max, 3)
            _clamp_clip_to_entity_segments(hook_clip, entity_segments, unit.entity_segment_ids or None)

            hook_clip["is_hook_clip"] = False
            hook_clip["beat"] = "start"
            hook_clip["_beat"] = "start"
            hook_clip["reason"] = (
                f"START: {_beat_intent(unit, 'start') or 'scene introduction'} — \"{_snippet(start_block.text)}\""
            )

        payoff_clip = None
        if payoff_block is not None:
            payoff_clip = _clip_from_block(payoff_block, PAYOFF_MAX_SECONDS, keep_end=True)
            payoff_clip = _extend_short_clip(payoff_clip, blocks, window, payoff_min, used_blocks, reserved_ranges, entity_segment_ids=ext_entity_ids, entity_segments=ext_entity_segs)
            # _extend_short_clip only checks the target (floor) before
            # absorbing each block — it has no ceiling, so a single large
            # adjacent block can push the payoff well past PAYOFF_MAX_SECONDS.
            p_dur = payoff_clip["source_end"] - payoff_clip["source_start"]
            if p_dur > PAYOFF_MAX_SECONDS:
                payoff_clip["source_start"] = round(payoff_clip["source_end"] - PAYOFF_MAX_SECONDS, 3)
            # Entity boundary clamp runs BEFORE the epilogue extension, not
            # after. A genuine conclusion line ("...and the gold medal goes
            # to...") routinely sits just past the entity segment's own
            # detected boundary — clamping AFTER the epilogue step undid all
            # of its work by cutting straight back to that boundary, which
            # is exactly why a payoff could look complete right up until a
            # max-duration cut and then simply lose its ending. Clamping
            # first (so the "core" pick+extend-to-minimum stays honest to
            # the segment), then deliberately letting the epilogue step
            # reach past that boundary — bounded by its own small,
            # self-contained cap — is what lets the real conclusion survive.
            if should_clamp_entity:
                _clamp_clip_to_entity_segments(payoff_clip, entity_segments, unit.entity_segment_ids or None)
            # Epilogue: a genuine conclusion line ("...and the gold medal
            # goes to...") often lands in its own short block immediately
            # after the payoff moment, right where PAYOFF_MAX_SECONDS cuts
            # off. Absorb short, tightly-adjacent trailing blocks (a
            # suspense pause + the announcement line are often two blocks)
            # even if it pushes past the normal cap, bounded by
            # PAYOFF_EPILOGUE_MAX_SECONDS so it can never run away.
            payoff_clip = _extend_payoff_epilogue(payoff_clip, blocks, window, used_blocks, reserved_ranges)
            payoff_clip["is_hook_clip"] = False
            payoff_clip["beat"] = "payoff"
            payoff_clip["_beat"] = "payoff"
            payoff_clip["reason"] = (
                f"PAYOFF: {_beat_intent(unit, 'payoff') or 'resolution / reveal'} — \"{_snippet(payoff_block.text)}\""
            )

        esc_clips = []
        esc_intent = _beat_intent(unit, "escalation") or "tension builds"
        for i, b in enumerate(esc_blocks):
            clip = _clip_from_block(b, ESCALATION_MAX_SECONDS, keep_end=False)
            clip = _extend_short_clip(clip, blocks, window, MIN_CLIP_SECONDS, used_blocks, reserved_ranges, entity_segment_ids=ext_entity_ids, entity_segments=ext_entity_segs)
            # Same no-ceiling issue as payoff above: clamp back down if a
            # single large adjacent block got absorbed whole.
            e_dur = clip["source_end"] - clip["source_start"]
            if e_dur > ESCALATION_MAX_SECONDS:
                clip["source_end"] = round(clip["source_start"] + ESCALATION_MAX_SECONDS, 3)
            if should_clamp_entity:
                _clamp_clip_to_entity_segments(clip, entity_segments, unit.entity_segment_ids or None)
            clip["is_hook_clip"] = False
            clip["beat"] = "escalation"
            clip["_beat"] = "escalation"
            label = f"ESCALATION {i + 1}" if len(esc_blocks) > 1 else "ESCALATION"
            clip["reason"] = f"{label}: {esc_intent} — \"{_snippet(b.text)}\""
            esc_clips.append(clip)

        if hook_clip is None and payoff_clip is None and not esc_clips:
            logger.warning(f"Unit {unit.unit_id} ('{unit.name}'): no clips could be built — skipping")
            continue

        # Hardcoded Story Flow: Hook (Start, 4-10s) -> Journey (Mid escalation) -> Payoff (End climax)
        esc_clips.sort(key=lambda c: c["source_start"])
        # Renumber escalations chronologically after sorting
        if len(esc_clips) > 1:
            for i, c in enumerate(esc_clips):
                old_reason = c.get("reason", "")
                # Replace "ESCALATION N" with the correct chronological number
                import re as _re
                c["reason"] = _re.sub(r"ESCALATION \d+", f"ESCALATION {i + 1}", old_reason)
        clips: list[dict[str, Any]] = []
        if hook_clip:
            clips.append(hook_clip)
        clips.extend(esc_clips)
        if payoff_clip:
            clips.append(payoff_clip)

        # Tag clips with entity segment IDs for QA boundary enforcement
        if unit.entity_segment_ids:
            for c in clips:
                c["entity_segment_ids"] = list(unit.entity_segment_ids)

        # Anti-mixup enforcement: ensure non-hook clips never jump backward in source time
        if len(clips) > 1:
            has_hook = clips[0].get("is_hook_clip", False)
            hook = clips[0] if has_hook else None
            body = clips[1:] if has_hook else clips
            body.sort(key=lambda c: c["source_start"])
            clips = ([hook] + body) if hook else body

        # Apply short→medium→long pacing strategy for engaging reels
        _apply_pacing_strategy(clips, blocks, window)

        # Seamless pre-payoff buildup connection: eliminate small gaps between final escalation and payoff
        if esc_clips and payoff_clip:
            last_esc = max(esc_clips, key=lambda c: c["source_start"])
            gap = payoff_clip["source_start"] - last_esc["source_end"]
            if 0.0 < gap <= 5.0 and last_esc["source_end"] + gap <= window[1]:
                last_esc["source_end"] = round(payoff_clip["source_start"], 3)

        # ── Minimum 3-clip floor: hook → escalation → payoff ──────────────────
        # If we only have 1-2 clips, pad with the next-best importance block from
        # the same window so every reel has a proper arc.
        # Entity content: skip floor padding — low-quality entities intentionally dropped.
        if len(clips) < 3 and len(candidates) > len(used_blocks) and not unit.entity_segment_ids:
            pad_pool = [
                b for b in candidates
                if b.block_id not in used_blocks
                and not b.black_frame and not b.freeze
                and b.importance >= 20
            ]
            # Sort by importance descending, then by temporal proximity to existing clips
            # (prefer blocks that fill gaps between existing clips rather than distant blocks)
            existing_ends = sorted(c.get("source_end", 0) for c in clips)
            existing_starts = sorted(c.get("source_start", 0) for c in clips)
            def _temporal_proximity(b: SemanticBlock) -> float:
                """Distance to nearest existing clip boundary (lower = closer)."""
                b_mid = (b.start + b.end) / 2.0
                min_dist = float("inf")
                for e in existing_ends:
                    min_dist = min(min_dist, abs(b_mid - e))
                for s in existing_starts:
                    min_dist = min(min_dist, abs(b_mid - s))
                return min_dist
            pad_pool.sort(key=lambda b: (-b.importance, _temporal_proximity(b)))
            for pad_b in pad_pool:
                if len(clips) >= 3:
                    break
                pad_clip = _clip_from_block(pad_b, ESCALATION_MAX_SECONDS, keep_end=False)
                pad_clip = _extend_short_clip(pad_clip, blocks, window, MIN_CLIP_SECONDS, used_blocks, reserved_ranges, entity_segment_ids=ext_entity_ids, entity_segments=ext_entity_segs)
                if should_clamp_entity:
                    _clamp_clip_to_entity_segments(pad_clip, entity_segments, unit.entity_segment_ids or None)
                pad_clip["is_hook_clip"] = False
                pad_clip["beat"] = "escalation"
                pad_clip["_beat"] = "escalation"
                pad_clip["reason"] = f"ESCALATION (padded): tension — \"{_snippet(pad_b.text)}\""
                # Insert before payoff clip
                insert_pos = len(clips) - 1 if payoff_clip else len(clips)
                clips.insert(insert_pos, pad_clip)
                used_blocks.add(pad_b.block_id)
                logger.debug(f"Unit {unit.unit_id}: padded clip from block {pad_b.block_id} to reach 3-clip floor")

        # Ensure escalation clips remain chronologically sorted after any padding
        if len(clips) >= 3:
            esc_portion = clips[1:-1] if payoff_clip else clips[1:]
            esc_portion.sort(key=lambda c: c["source_start"])
            if payoff_clip:
                clips = [clips[0]] + esc_portion + [clips[-1]]
            else:
                clips = [clips[0]] + esc_portion

        # Duration budget --------------------------------------------------------
        total = sum(c["source_end"] - c["source_start"] for c in clips)
        if total > unit_reel_dur_max:
            _trim_clips_to_budget(clips, unit_reel_dur_max, payoff_min=payoff_min)

        # Short source: extend clips to reach minimum duration target
        total = sum(c["source_end"] - c["source_start"] for c in clips)
        if total < unit_reel_dur_min - 0.5:
            prev_total = total
            soft_target = unit_reel_dur_min
            # For merged entity groups, allow extension using ALL blocks in window
            ext_block_ids = entity_block_ids if entity_window_only else None
            n_ext = _extend_clips_to_fill(
                clips, soft_target, source_duration, window,
                blocks, used_blocks, reserved_ranges,
                entity_block_ids=ext_block_ids,
            )
            if n_ext > 0:
                total = sum(c["source_end"] - c["source_start"] for c in clips)
                logger.info(
                    f"Unit {unit.unit_id} ('{unit.name}'): clips short "
                    f"({prev_total:.1f}s) — "
                    f"extended {n_ext} clips to {total:.1f}s "
                    f"(target {soft_target}s)"
                )

        # Scene-cut snapping: nudge clip boundaries to real scene cuts when
        # available and within tolerance.  Strictly corrective — never changes
        # which blocks were selected, only nudges the exact in/out points.
        if scene_cut_at:
            _snap_clip_boundaries(clips, scene_cut_at)

        _reorder_reel_clips(clips)

        for c in clips:
            c["source_start"] = round(max(0.0, c["source_start"]), 3)
            c["source_end"] = round(min(source_duration, c["source_end"]), 3)
            # Keep _beat on clips — QA and downstream need it for payoff
            # detection and trim protection.

        group = _build_group_dict(unit, clips, window, source_duration)
        # Stash per-unit duration targets so post_execution_qa can honour them
        # instead of re-flattening every entity group to the global bounds.
        group["unit_dur_min"] = unit_reel_dur_min
        group["unit_dur_max"] = unit_reel_dur_max
        groups.append(group)

        reserved_ranges.extend((c["source_start"], c["source_end"]) for c in clips)
        # Also reserve the unit's FULL window, not just the seconds it
        # actually picked. The window was already computed conservatively
        # (padding capped at a fraction of the real gap to each neighbor —
        # see _unit_windows), so it can never reach into another unit's own
        # footage. Reserving it in full means a later-processed unit can't
        # pick up leftover, unused seconds inside an earlier unit's fair
        # territory (e.g. a bit of trailing crowd reaction the earlier unit
        # chose not to use) and present it as its own opening — closing the
        # other half of the cross-group bleed that chronological processing
        # order alone doesn't cover.
        reserved_ranges.append(window)
        reserved_ids |= {b.block_id for b in (hook_block, start_block, payoff_block, *esc_blocks) if b is not None}

        logger.info(
            f"EXECUTOR unit {unit.unit_id} ('{unit.name}', {unit.region}, prio {unit.priority}): "
            f"{len(clips)} clips, {total:.1f}s in [{window[0]:.0f}-{window[1]:.0f}]s"
        )

    if not groups:
        raise RuntimeError("Plan executor produced zero groups — no usable content in any unit window.")
    return groups


# ---------------------------------------------------------------------------
# Post-execution QA — deterministic Python repairs after execute_plan()
# ---------------------------------------------------------------------------

def _resolve_intra_group_overlaps(clips: list[dict[str, Any]]) -> bool:
    """Trim overlapping source ranges between clips in the SAME group.

    Different beats (hook/start, escalation, payoff) are selected somewhat
    independently and each can be extended on its own, so two clips can end
    up covering the same source seconds even though they never share a
    block id — the reel then visibly repeats that footage. The payoff
    keeps its tail (the reveal) since that's the moment the reel is
    building to; the earlier-in-source-time clip gives up the overlapping
    head. Clips that shrink below MIN_CLIP_SECONDS are dropped.

    Note: `clips` list ORDER is the reel's playback order (hook first,
    then escalations chronologically, then payoff last) — that is NOT
    necessarily source-chronological order, so this function must not
    reorder the list. It only mutates the shared clip dicts (by walking
    a source-time-sorted VIEW) and drops any that shrink too far,
    filtering the original list in its original order.

    Mutates `clips` in place (may remove entries, never reorders the
    survivors). Returns True if anything changed.
    """
    if len(clips) < 2:
        return False
    ordered_view = sorted(clips, key=lambda c: c["source_start"])
    changed = False
    for i in range(1, len(ordered_view)):
        prev = ordered_view[i - 1]
        cur = ordered_view[i]
        overlap = prev["source_end"] - cur["source_start"]
        if overlap <= 0.05:
            continue
        changed = True
        cur_is_payoff = (cur.get("beat") or cur.get("_beat")) == "payoff"
        prev_is_payoff = (prev.get("beat") or prev.get("_beat")) == "payoff"
        if cur_is_payoff and not prev_is_payoff:
            # Payoff keeps its range — the earlier clip gives up its tail.
            prev["source_end"] = round(max(prev["source_start"], cur["source_start"]), 3)
        else:
            # Default: the later-starting clip gives up the overlapping head.
            cur["source_start"] = round(min(cur["source_end"], prev["source_end"]), 3)
    survivors = [c for c in clips if c["source_end"] - c["source_start"] >= MIN_CLIP_SECONDS - 0.01]
    if len(survivors) != len(clips):
        changed = True
    clips[:] = survivors
    return changed


def post_execution_qa(
    groups: list[dict[str, Any]],
    source_duration: float,
    entity_segments: list | None = None,
    reel_dur_min: int = 30,
    reel_dur_max: int = 50,
) -> list[dict[str, Any]]:
    """Run deterministic QA checks after execute_plan() and before finalize_edit().

    Checks (in order):
    1. Trim/drop clips that cross entity boundaries.
    2. Resolve overlapping source ranges within a group (repeated footage,
       or an escalation clip's extension creeping into the payoff's own
       territory).
    3. Ensure payoff is the latest source-time clip in each group.
    4. Enforce output duration bounds using actual clip duration + 2s pad.
    5. Resolve cross-group source overlaps globally.
    6. Recalculate estimates for every modified group.
    7. Drop only unsalvageable groups (zero usable clips); retain everything else.

    Returns the (possibly pruned) groups list.
    """
    if not groups:
        return groups

    entity_map: dict[str, Any] = {}
    if entity_segments:
        entity_map = {s.entity_segment_id: s for s in entity_segments}

    modified_groups: set[int] = set()

    # --- 1. Entity boundary trimming ---
    for g in groups:
        clips = g.get("source_clips", [])
        if not clips:
            continue
        trimmed = False
        for c in clips:
            seg_ids = c.get("entity_segment_ids") or []
            if not seg_ids:
                continue
            mid = (c["source_start"] + c["source_end"]) / 2.0
            # Find the entity segment this clip belongs to
            for sid in seg_ids:
                seg = entity_map.get(sid)
                if seg is None:
                    continue
                # Use effective_ranges for non-adjacent merged segments
                if seg.effective_ranges:
                    for r_start, r_end in seg.effective_ranges:
                        if r_start - 0.1 <= mid <= r_end + 0.1:
                            if c["source_end"] > r_end + 0.1:
                                c["source_end"] = round(r_end, 3)
                                trimmed = True
                            if c["source_start"] < r_start - 0.1:
                                c["source_start"] = round(r_start, 3)
                                trimmed = True
                            break
                else:
                    if seg.start - 0.1 <= mid <= seg.end + 0.1:
                        if c["source_end"] > seg.end + 0.1:
                            c["source_end"] = round(seg.end, 3)
                            trimmed = True
                        if c["source_start"] < seg.start - 0.1:
                            c["source_start"] = round(seg.start, 3)
                            trimmed = True
        # Remove clips that became too short after entity boundary trimming
        clips[:] = [c for c in clips if c["source_end"] - c["source_start"] >= MIN_CLIP_SECONDS - 0.01]
        if trimmed:
            modified_groups.add(g.get("group_index", -1))
            logger.info(
                f"QA: trimmed entity-boundary crossings in group {g.get('group_index')}"
            )

    # --- 2. Resolve overlapping source ranges WITHIN a group ---
    # Beats (hook/start, escalation, payoff) are picked semi-independently
    # and each can be independently extended, so two clips in the same
    # group can end up covering the same source seconds — the reel then
    # visibly repeats that footage, or an escalation clip's extension
    # creeps past the payoff block's own start into the payoff's own
    # territory. This runs BEFORE the "payoff must be latest" check below:
    # if an escalation clip were left overrunning into the payoff's range,
    # its (wrongly) later source_end could fool that check into swapping
    # the two clips' identities — crowning the overrun escalation clip as
    # "the payoff" and demoting the real payoff to an escalation slot. Only
    # this raw, mislabeled ordering issue is distinct from step 5 below,
    # which resolves overlaps ACROSS groups instead of within one.
    for g in groups:
        clips = g.get("source_clips", [])
        if len(clips) < 2:
            continue
        before = [(c["source_start"], c["source_end"]) for c in clips]
        if _resolve_intra_group_overlaps(clips):
            g["source_clips"] = clips
            modified_groups.add(g.get("group_index", -1))
            logger.info(
                f"QA: resolved intra-group overlap in group {g.get('group_index')} "
                f"({before} -> {[(c['source_start'], c['source_end']) for c in clips]})"
            )

    # --- 3. Payoff must be the latest source-time clip in its group ---
    for g in groups:
        clips = g.get("source_clips", [])
        if len(clips) < 2:
            continue
        payoff_idx = None
        for i, c in enumerate(clips):
            beat = c.get("beat") or c.get("_beat", "")
            reason = (c.get("reason") or "").upper()
            if beat == "payoff" or reason.startswith("PAYOFF"):
                payoff_idx = i
                break
        if payoff_idx is None:
            continue
        payoff = clips[payoff_idx]
        latest_idx = max(range(len(clips)), key=lambda i: clips[i].get("source_end", 0.0))
        if latest_idx != payoff_idx:
            # Swap timestamps so the payoff clip gets the latest source range.
            # Only swap timestamps and is_hook_clip — keep reason/_beat on the
            # original clips so metadata stays consistent.
            other = clips[latest_idx]
            payoff["source_start"], other["source_start"] = other["source_start"], payoff["source_start"]
            payoff["source_end"], other["source_end"] = other["source_end"], payoff["source_end"]
            payoff["is_hook_clip"], other["is_hook_clip"] = other.get("is_hook_clip", False), payoff.get("is_hook_clip", False)
            modified_groups.add(g.get("group_index", -1))
            logger.info(
                f"QA: moved payoff timestamps to latest source-time in group {g.get('group_index')}"
            )

    # --- 4. Duration bounds enforcement (actual clip duration + 2.0s pad) ---
    for g in groups:
        clips = g.get("source_clips", [])
        if not clips:
            continue
        # Use per-unit targets stashed by execute_plan when available,
        # instead of re-flattening entity groups to the global bounds.
        g_min = g.get("unit_dur_min", reel_dur_min) or reel_dur_min
        g_max = g.get("unit_dur_max", reel_dur_max) or reel_dur_max
        total_clip = sum(c["source_end"] - c["source_start"] for c in clips)
        estimated = total_clip + 2.0
        if estimated > g_max + 0.5:
            # Trim excess from the longest non-hook, non-payoff clip.
            # Use g_max directly — the executor already accounted for the
            # +2s pad in its budget math, so trimming to g_max (not g_max-2)
            # avoids double-trimming clips that were already correctly sized.
            _trim_to_duration(clips, g_max)
            modified_groups.add(g.get("group_index", -1))
            logger.info(
                f"QA: trimmed group {g.get('group_index')} to fit {g_max}s max "
                f"(per-unit target)"
            )
        elif estimated < g_min - 0.5:
            g["_below_min"] = True
            logger.info(
                f"QA: group {g.get('group_index')} estimated {estimated:.1f}s "
                f"< {g_min}s min (marked as below minimum)"
            )

    # --- 3b. Drop weakest below-minimum groups when enough strong groups exist ---
    below_min = [g for g in groups if g.get("_below_min")]
    at_or_above = [g for g in groups if not g.get("_below_min")]
    if below_min and len(at_or_above) >= 3:
        below_min.sort(key=lambda g: g.get("estimated_duration_seconds", 0.0))
        survivors_needed = max(1, len(at_or_above) // 3)
        kept_weak = below_min[-survivors_needed:]
        dropped_weak = below_min[:len(below_min) - survivors_needed]
        for g in dropped_weak:
            g["source_clips"] = []
            logger.info(
                f"QA: pruned below-min group {g.get('group_index')} "
                f"({g.get('estimated_duration_seconds', 0.0):.1f}s) "
                f"in favor of stronger groups"
            )
        for g in kept_weak:
            g.pop("_below_min", None)
        for g in at_or_above:
            g.pop("_below_min", None)

    # --- 4. Cross-group source overlap resolution ---
    # Sort groups by priority (lower index = higher priority)
    sorted_groups = sorted(groups, key=lambda g: g.get("group_index", 999))
    claimed_ranges: list[tuple[float, float]] = []
    for g in sorted_groups:
        clips = g.get("source_clips", [])
        kept: list[dict[str, Any]] = []
        for c in clips:
            cs, ce = c["source_start"], c["source_end"]
            overlaps_higher = any(
                max(cs, r0) < min(ce, r1) - 0.05
                for r0, r1 in claimed_ranges
            )
            if overlaps_higher:
                # Trim to the earliest available gap
                trimmed_clip = _trim_to_gap(c, claimed_ranges, source_duration)
                if trimmed_clip is not None:
                    kept.append(trimmed_clip)
                modified_groups.add(g.get("group_index", -1))
                logger.info(
                    f"QA: trimmed cross-group overlap in group {g.get('group_index')}"
                )
            else:
                kept.append(c)
        if len(kept) < len(clips):
            g["source_clips"] = kept
        for c in kept:
            claimed_ranges.append((c["source_start"], c["source_end"]))

    # --- 5. Recalculate estimates for modified groups ---
    for g in groups:
        idx = g.get("group_index", -1)
        if idx in modified_groups:
            clips = g.get("source_clips", [])
            total = sum(c["source_end"] - c["source_start"] for c in clips)
            g["estimated_duration_seconds"] = round(total + 2.0, 1)

    # --- 6. Drop only unsalvageable groups ---
    surviving = [g for g in groups if g.get("source_clips")]
    dropped = len(groups) - len(surviving)
    if dropped > 0:
        logger.warning(f"QA: dropped {dropped} unsalvageable group(s)")
    return surviving


def _trim_to_duration(clips: list[dict[str, Any]], budget: float) -> None:
    """Trim excess clip duration to fit within budget.

    Trims from the longest non-hook, non-payoff clips first.
    """
    total = sum(c["source_end"] - c["source_start"] for c in clips)
    over = total - budget
    if over <= 0:
        return

    # Sort candidates: trimmable clips (not hook, not payoff) by duration desc
    candidates = []
    for i, c in enumerate(clips):
        if c.get("is_hook_clip") or (c.get("beat") or c.get("_beat")) == "payoff":
            continue
        dur = c["source_end"] - c["source_start"]
        if dur > MIN_CLIP_SECONDS:
            candidates.append((i, c, dur))
    candidates.sort(key=lambda x: -x[2])

    for _, c, dur in candidates:
        if over <= 0:
            break
        trim = min(over, dur - MIN_CLIP_SECONDS)
        c["source_end"] = round(c["source_end"] - trim, 3)
        over -= trim

    # If still over budget, trim payoff (respect PAYOFF_MIN_SECONDS)
    if over > 0:
        for c in clips:
            if (c.get("beat") or c.get("_beat")) == "payoff" or c.get("reason", "").startswith("PAYOFF"):
                dur = c["source_end"] - c["source_start"]
                trim = min(over, max(0.0, dur - PAYOFF_MIN_SECONDS))
                c["source_start"] = round(c["source_start"] + trim, 3)
                over -= trim
                break


def _trim_to_gap(
    clip: dict[str, Any],
    claimed: list[tuple[float, float]],
    source_duration: float,
) -> dict[str, Any] | None:
    """Trim a clip so it fits in the closest available gap to its original position.

    Unlike the old "earliest gap" approach, this finds the gap whose center
    is closest to the clip's original center. This prevents the payoff clip
    from being relocated to a completely different part of the source video.

    Returns None if no gap can accommodate the clip.
    """
    cs, ce = clip["source_start"], clip["source_end"]
    clip_mid = (cs + ce) / 2.0
    clip_len = ce - cs

    # Sort claimed ranges and find gaps
    gaps: list[tuple[float, float]] = []
    prev_end = 0.0
    for r0, r1 in sorted(claimed):
        if r0 > prev_end + 0.05:
            gaps.append((prev_end, r0))
        prev_end = max(prev_end, r1)
    if prev_end < source_duration - 0.05:
        gaps.append((prev_end, source_duration))

    if not gaps:
        return None

    # Score each gap by distance from clip's original center (prefer closest)
    # and whether it can fit the full clip (prefer full-fit over partial)
    scored_gaps: list[tuple[float, float, float, bool]] = []
    for g_start, g_end in gaps:
        gap_len = g_end - g_start
        gap_mid = (g_start + g_end) / 2.0
        dist = abs(gap_mid - clip_mid)
        fits_full = gap_len >= clip_len - 0.05
        fits_min = gap_len >= MIN_CLIP_SECONDS
        if fits_min:
            # priority_key: (fits_full, -dist) — True > False, higher dist score better
            scored_gaps.append((g_start, g_end, dist, fits_full))

    if not scored_gaps:
        return None

    # Sort: prefer gaps that fit the full clip, then by closest distance
    scored_gaps.sort(key=lambda x: (not x[3], x[2]))

    g_start, g_end, _, fits_full = scored_gaps[0]
    if fits_full:
        clip["source_start"] = round(g_start, 3)
        clip["source_end"] = round(g_start + clip_len, 3)
    else:
        clip["source_start"] = round(g_start, 3)
        clip["source_end"] = round(g_end, 3)
    return clip


# ---------------------------------------------------------------------------
# Narration placement — Python owns all numbers
# ---------------------------------------------------------------------------

def estimate_speech_duration(text: str, words_per_second: float = 2.5) -> float:
    """Rough deterministic speaking duration for a narration line."""
    words = max(1, len((text or "").split()))
    return round(min(8.0, max(2.0, words / words_per_second)), 2)


def place_narration_events(
    group: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assign reel_start/reel_end to narration events from the clip layout.

    Rules (deterministic):
    - hook/start event at 0.0 (dropped at TTS when a hook clip exists)
    - commentary 1 starts right after the opening clip ends (+0.8s gap)
    - commentary 2 starts 2s before the payoff clip, never before
      commentary 1 end + 0.8s
    - last 3s of the reel stay narration-free; events capped at 2 commentaries
    """
    if events is None:
        events = list(group.get("narration_events", []))
    est = float(group.get("estimated_duration_seconds", 0.0)) or 0.0
    if not events or est <= 0:
        return events

    hooks = [e for e in events if str(e.get("event_type", "")).lower() in ("hook", "start")]
    commentaries = [e for e in events if str(e.get("event_type", "")).lower() == "commentary"]

    clip_ends: list[float] = []
    acc = 0.0
    for c in group.get("source_clips", []):
        acc += float(c.get("source_end", 0.0)) - float(c.get("source_start", 0.0))
        clip_ends.append(acc)
    hook_clip_reel_end = clip_ends[0] if clip_ends else 0.0
    payoff_clip_reel_start = clip_ends[-2] if len(clip_ends) >= 2 else est

    prev_end = 0.0
    if hooks:
        h = hooks[0]
        dur = estimate_speech_duration(h.get("text", ""))
        h["reel_start"] = 0.0
        h["reel_end"] = round(min(4.0, max(2.0, dur)), 2)
        prev_end = h["reel_end"]

    placed: list[dict[str, Any]] = [e for e in events if str(e.get("event_type", "")).lower() in ("hook", "start")][:1]
    free_tail = 3.0
    for i, c in enumerate(commentaries[:2]):
        dur = estimate_speech_duration(c.get("text", ""))
        if i == 0:
            start = hook_clip_reel_end + 0.8
        else:
            start = max(payoff_clip_reel_start - 2.0, prev_end + 0.8)
        max_start = max(prev_end + 0.8, est - dur - free_tail)
        start = round(min(start, max_start), 2)
        end = round(min(start + dur, est - free_tail), 2)
        if end <= start:
            end = round(start + min(dur, 1.0), 2)
        c["reel_start"] = start
        c["reel_end"] = end
        prev_end = end
        placed.append(c)

    return placed


def assign_reel_summary(group: dict[str, Any], summary: dict[str, Any] | None) -> None:
    """Fill reel_summary from the LLM writer output (keys validated downstream)."""
    if not isinstance(summary, dict):
        return
    current = group.get("reel_summary", {})
    for key in ("title", "short_description", "source_understanding", "narrative_angle", "key_moment"):
        val = summary.get(key)
        if isinstance(val, str) and val.strip():
            current[key] = val.strip()
    group["reel_summary"] = current