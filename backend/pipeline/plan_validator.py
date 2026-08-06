"""Plan validation — all deterministic validation and repair for LLM output.

Every deterministic operation belongs in Python:
- JSON repair
- Timestamp validation
- Clip bounds enforcement
- Overlap detection and repair
- Narration validation
- Duration validation
- Caption validation
- Final integrity validation

The LLM must NEVER become responsible for these operations.
"""
from __future__ import annotations

import logging

from backend.config import MAX_OUTPUT_DURATION, MIN_CONTENT_DURATION, MIN_OUTPUT_DURATION
from backend.models import ReelPlan
from backend.pipeline.sanitize import sanitize_text

__all__ = [
    "enforce_clip_pacing",
    "finalize_edit",
    "remove_overlaps",
    "repair_clip_diversity",
    "validate_clip_bounds",
    "validate_clip_diversity",
    "validate_narration",
    "validate_timing",
    "verify_captions",
    "verify_duration",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clip Bounds Validation
# ---------------------------------------------------------------------------

def validate_clip_bounds(
    groups: list[dict],
    source_duration: float,
    min_clip_duration: float = 3.0,
) -> int:
    """Clamp clip timestamps to [0, source_duration] and enforce minimum duration.

    Returns the number of clips that were adjusted.
    """
    adjusted = 0
    for group in groups:
        for clip in group.get("source_clips", []):
            s = clip.get("source_start", 0.0)
            e = clip.get("source_end", 0.0)
            new_s = max(0.0, min(s, source_duration))
            new_e = max(0.0, min(e, source_duration))

            # Enforce minimum clip duration
            if new_e - new_s < min_clip_duration and source_duration >= min_clip_duration:
                new_e = min(source_duration, new_s + min_clip_duration)
                if new_e - new_s < min_clip_duration:
                    new_s = max(0.0, new_e - min_clip_duration)

            if new_s != s or new_e != e:
                adjusted += 1
                clip["source_start"] = round(new_s, 3)
                clip["source_end"] = round(new_e, 3)

            # Ensure start < end
            if clip["source_start"] >= clip["source_end"]:
                clip["source_end"] = min(clip["source_start"] + min_clip_duration, source_duration)

    if adjusted > 0:
        logger.info(f"Adjusted {adjusted} clip timestamps to valid bounds")
    return adjusted


# ---------------------------------------------------------------------------
# Timing Validation
# ---------------------------------------------------------------------------

def validate_timing(groups: list[dict], source_duration: float) -> None:
    """Validate and log timing consistency for each group."""
    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        clips_total = sum(c.get("source_end", 0) - c.get("source_start", 0) for c in clips)
        nar_events = group.get("narration_events", [])
        nar_total = sum(e.get("reel_end", 0) - e.get("reel_start", 0) for e in nar_events)

        # Narration overlaps clips, so estimated = clips_total + 2.0 pad
        computed_estimated = clips_total + 2.0
        llm_estimate = group.get("estimated_duration_seconds", 0)

        if llm_estimate < computed_estimated:
            logger.info(
                f"Group {i}: Raising estimated_duration from {llm_estimate:.1f}s "
                f"to {computed_estimated:.1f}s (computed from clips + pad)"
            )
            group["estimated_duration_seconds"] = round(computed_estimated, 1)

        logger.info(
            f"Group {i} timing: clips={clips_total:.1f}s ({len(clips)} clips), "
            f"narration={nar_total:.1f}s ({len(nar_events)} events), "
            f"estimated={group['estimated_duration_seconds']:.1f}s"
        )


# ---------------------------------------------------------------------------
# Overlap Detection and Repair
# ---------------------------------------------------------------------------

# Keywords in clip reason that signal editorial importance (higher = more valuable)
_HIGH_IMPORTANCE_KEYWORDS = {"hook", "climax", "peak", "reveal", "payoff", "surprise", "shock", "reaction"}
_MEDIUM_IMPORTANCE_KEYWORDS = {"escalation", "build", "tension", "drama", "conflict"}


def _estimate_clip_importance(clip: dict) -> float:
    """Estimate a clip's editorial importance (0-100) from its reason text.

    Higher scores = more editorially valuable. Used to decide which clip
    to keep when two clips overlap.
    """
    reason = clip.get("reason", "").lower()
    score = 30.0  # baseline

    # Hook clip bonus
    if clip.get("is_hook_clip", False):
        score += 25.0

    # Keyword matching
    for kw in _HIGH_IMPORTANCE_KEYWORDS:
        if kw in reason:
            score += 15.0
            break
    for kw in _MEDIUM_IMPORTANCE_KEYWORDS:
        if kw in reason:
            score += 8.0
            break

    # Duration bonus — very short clips are less valuable as primary content
    duration = clip.get("source_end", 0) - clip.get("source_start", 0)
    if duration >= 10.0:
        score += 10.0
    elif duration >= 5.0:
        score += 5.0

    return min(100.0, score)


def remove_overlaps(groups: list[dict]) -> int:
    """Detect and remove overlapping clips within each group.

    Uses importance-weighted resolution: when two clips overlap, the more
    editorially valuable clip is kept (not just the longer one).

    Returns the number of clips removed due to overlap.
    """
    removed = 0
    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        if len(clips) <= 1:
            continue

        # Sort by start time
        clips.sort(key=lambda c: c.get("source_start", 0))

        filtered = [clips[0]]
        for clip in clips[1:]:
            prev = filtered[-1]
            if clip.get("source_start", 0) < prev.get("source_end", 0):
                # Overlap detected — keep the more important clip
                prev_imp = _estimate_clip_importance(prev)
                curr_imp = _estimate_clip_importance(clip)
                if curr_imp > prev_imp:
                    filtered[-1] = clip
                    removed += 1
                    logger.info(
                        f"Group {i}: Replaced lower-importance clip "
                        f"(imp={prev_imp:.0f}) with higher-importance clip (imp={curr_imp:.0f})"
                    )
                else:
                    removed += 1
                    logger.info(
                        f"Group {i}: Removed lower-importance clip (imp={curr_imp:.0f}), "
                        f"keeping higher-importance clip (imp={prev_imp:.0f})"
                    )
            else:
                filtered.append(clip)

        if len(filtered) < len(clips):
            group["source_clips"] = filtered
            logger.info(f"Group {i}: {len(clips)} -> {len(filtered)} clips after overlap removal")

    return removed


# ---------------------------------------------------------------------------
# Narration Validation
# ---------------------------------------------------------------------------

def validate_narration(groups: list[dict]) -> None:
    """Validate narration events: types, hook placement, distribution, text sanitization."""
    for i, group in enumerate(groups):
        # Sanitize narration text
        for event in group.get("narration_events", []):
            if "text" in event and isinstance(event["text"], str):
                event["text"] = sanitize_text(event["text"])

        # Validate event types and hook placement
        hook_seen = False
        usable_count = 0
        for j, event in enumerate(group.get("narration_events", [])):
            ev_type = str(event.get("event_type", "unknown")).strip().lower()

            if ev_type == "hook":
                if hook_seen:
                    logger.info(f"Group {i}: Duplicate hook event {j}; converting to 'commentary'")
                    event["event_type"] = "commentary"
                    ev_type = "commentary"
                else:
                    hook_seen = True

            if ev_type not in ("hook", "start", "commentary"):
                logger.warning(
                    f"Group {i} narration event {j} has unrecognized type '{ev_type}' — "
                    f"will be SILENTLY DROPPED before TTS"
                )
            else:
                usable_count += 1

            # Opening events (hook or start) must start at 0.0
            if ev_type in ("hook", "start") and event.get("reel_start", 0) != 0.0:
                logger.warning(f"Group {i}: {ev_type.capitalize()} must start at 0.0, correcting")
                event["reel_start"] = 0.0

        if usable_count == 0:
            logger.warning(f"Group {i}: ZERO usable narration events — reel will have NO narration")

        # Cap at 6 narration events (1 opening + up to 5 commentaries)
        usable_events = [
            e for e in group.get("narration_events", [])
            if str(e.get("event_type", "")).strip().lower() in ("hook", "start", "commentary")
        ]
        if len(usable_events) > 6:
            # Keep opening (hook or start) + first 5 commentaries, drop the rest
            opening = next((e for e in usable_events if e.get("event_type") in ("hook", "start")), None)
            commentary = [e for e in usable_events if e.get("event_type") == "commentary"]
            keep_events = ([opening] if opening else []) + commentary[:5]
            drop_events = [e for e in usable_events if e not in keep_events]
            for e in drop_events:
                group.get("narration_events", []).remove(e)
            if drop_events:
                logger.info(
                    f"Group {i}: Capped narration from {len(usable_events)} to "
                    f"{len(keep_events)} events (max 6 allowed)"
                )

        # Distribution check: ensure commentary is spread across the reel
        est_dur = group.get("estimated_duration_seconds", 120)
        commentary_events = [
            e for e in group.get("narration_events", [])
            if str(e.get("event_type", "")).strip().lower() == "commentary"
            and (e.get("reel_end", 0) - e.get("reel_start", 0)) >= 0.3
        ]
        if len(commentary_events) >= 2:
            last_40_start = est_dur * 0.6
            all_in_tail = all(e.get("reel_start", 0) >= last_40_start for e in commentary_events)
            if all_in_tail:
                logger.warning(
                    f"Group {i}: ALL {len(commentary_events)} commentary events clustered in "
                    f"last 40%. Redistributing..."
                )
                targets = [0.40, 0.75]
                for idx, event in enumerate(commentary_events):
                    fraction = targets[idx % len(targets)]
                    new_start = round(est_dur * fraction, 2)
                    dur = event.get("reel_end", 0) - event.get("reel_start", 0)
                    event["reel_start"] = new_start
                    event["reel_end"] = round(new_start + dur, 2)


# ---------------------------------------------------------------------------
# Duration Verification
# ---------------------------------------------------------------------------

def _expand_clips_to_fill_gap(
    clips: list[dict], source_duration: float, target_total: float
) -> int:
    """Expand existing clips so their total duration reaches target_total.

    Strategy:
    1. First pass: extend each clip up to CLIP_DURATION_SOFT_MAX
    2. If deficit remains: second pass, extend beyond soft max
    3. If still short: merge adjacent/overlapping clips to reclaim gap space

    Returns the number of clips that were expanded or removed.
    """
    from backend.config import CLIP_DURATION_SOFT_MAX

    clips_total = sum(c.get("source_end", 0) - c.get("source_start", 0) for c in clips)
    deficit = target_total - clips_total
    if deficit <= 0:
        return 0

    # Sort by importance (hook clips first, then by start time)
    clips.sort(key=lambda c: (-float(c.get("is_hook_clip", False)), c.get("source_start", 0)))

    expanded = 0

    for pass_num in (1, 2):
        if deficit <= 0:
            break
        for clip in clips:
            if deficit <= 0:
                break

            cur_start = clip.get("source_start", 0.0)
            cur_end = clip.get("source_end", 0.0)
            cur_dur = cur_end - cur_start

            if pass_num == 1 and cur_dur >= CLIP_DURATION_SOFT_MAX:
                continue

            # Pass 1: up to soft max. Pass 2: up to deficit per clip.
            if pass_num == 1:
                max_extra = CLIP_DURATION_SOFT_MAX - cur_dur
            else:
                max_extra = deficit

            extra_each_side = max_extra / 2.0

            # Extension must never cross another clip's interval. A clip that
            # straddles the current range's left/right edge blocks that side
            # entirely; clips fully before/after bound the usable space.
            overlap_back = any(
                o is not clip and o["source_start"] < cur_start < o["source_end"]
                for o in clips
            )
            overlap_fwd = any(
                o is not clip and o["source_start"] < cur_end < o["source_end"]
                for o in clips
            )
            prev_end = max(
                (o["source_end"] for o in clips if o is not clip and o["source_end"] <= cur_start),
                default=0.0,
            )
            next_start = min(
                (o["source_start"] for o in clips if o is not clip and o["source_start"] >= cur_end),
                default=source_duration,
            )
            avail_back = 0.0 if overlap_back else max(0.0, cur_start - prev_end)
            avail_fwd = 0.0 if overlap_fwd else max(0.0, next_start - cur_end)

            # Extend backward and forward, allocating unused budget to the other side
            extend_back = min(extra_each_side, avail_back, deficit)
            remaining_after_back = deficit - extend_back
            extend_fwd = min(extra_each_side + (extra_each_side - extend_back), avail_fwd, remaining_after_back)

            if extend_back > 0 or extend_fwd > 0:
                clip["source_start"] = round(cur_start - extend_back, 3)
                clip["source_end"] = round(cur_end + extend_fwd, 3)
                deficit -= (extend_back + extend_fwd)
                expanded += 1
                logger.info(
                    f"Expanded clip [{cur_start:.1f}-{cur_end:.1f}] -> "
                    f"[{clip['source_start']:.1f}-{clip['source_end']:.1f}] "
                f"(+{extend_back + extend_fwd:.1f}s)"
            )

    # Pass 3: merge overlapping/adjacent clips to reclaim gap space
    if deficit > 0 and len(clips) > 1:
        clips.sort(key=lambda c: c.get("source_start", 0))
        merged = []
        for clip in clips:
            if merged and clip["source_start"] <= merged[-1]["source_end"] + 1.0:
                # Merge: extend previous clip's end to cover this clip
                reclaimed = clip["source_end"] - merged[-1]["source_end"]
                if reclaimed > 0:
                    merged[-1]["source_end"] = clip["source_end"]
                    deficit -= reclaimed
                    expanded += 1
                    logger.info(
                        f"Merged clips into [{merged[-1]['source_start']:.1f}-"
                        f"{merged[-1]['source_end']:.1f}] (+{reclaimed:.1f}s reclaimed)"
                    )
            else:
                merged.append(dict(clip))
        if len(merged) < len(clips):
            clips.clear()
            clips.extend(merged)

    return expanded


def verify_duration(groups: list[dict], source_duration: float) -> None:
    """Verify and enforce duration constraints on each group.

    For executor-mode groups with unit_dur_min/max, uses those as bounds
    instead of the global MIN_CONTENT_DURATION/MIN_OUTPUT_DURATION.
    """
    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        clips_total = sum(c.get("source_end", 0) - c.get("source_start", 0) for c in clips)

        # Use per-unit targets when available (executor mode), fall back to globals
        unit_min = group.get("unit_dur_min", 0) or 0
        unit_max = group.get("unit_dur_max", 0) or 0
        has_unit_targets = unit_min > 0 and unit_max > 0

        # If clip content is too short, expand clips as much as source allows
        target_content = unit_min if has_unit_targets else MIN_CONTENT_DURATION
        if clips_total < target_content and source_duration > clips_total:
            logger.warning(
                f"Group {i}: clip content {clips_total:.1f}s below minimum "
                f"{target_content:.1f}s — expanding clips"
            )
            expanded = _expand_clips_to_fill_gap(clips, source_duration, target_content)
            if expanded > 0:
                clips_total = sum(
                    c.get("source_end", 0) - c.get("source_start", 0) for c in clips
                )
                logger.info(
                    f"Group {i}: expanded {expanded} clips, new total {clips_total:.1f}s"
                )

        actual_estimated = clips_total + 2.0

        llm_estimate = group.get("estimated_duration_seconds", 0)
        dur_floor = unit_min if has_unit_targets else MIN_OUTPUT_DURATION
        if llm_estimate < dur_floor and source_duration >= dur_floor:
            logger.warning(
                f"Group {i}: estimated {llm_estimate:.1f}s below target {dur_floor}s. "
                f"Computed: {actual_estimated:.1f}s"
            )

        # Bump estimated to at least computed actual
        if actual_estimated > llm_estimate:
            group["estimated_duration_seconds"] = round(actual_estimated, 1)

        # Cap at MAX_OUTPUT_DURATION (or per-unit max if available)
        dur_cap = unit_max if has_unit_targets else MAX_OUTPUT_DURATION
        if group["estimated_duration_seconds"] > dur_cap:
            logger.warning(
                f"Group {i}: capping estimated {group['estimated_duration_seconds']:.1f}s "
                f"to {dur_cap}s"
            )
            group["estimated_duration_seconds"] = float(dur_cap)


# ---------------------------------------------------------------------------
# Caption Validation
# ---------------------------------------------------------------------------

def verify_captions(groups: list[dict]) -> None:
    """Verify narration events will produce valid captions."""
    for i, group in enumerate(groups):
        for j, event in enumerate(group.get("narration_events", [])):
            text = event.get("text", "")
            if not text or not text.strip():
                logger.warning(f"Group {i} event {j}: empty text — will produce silent caption")
            if len(text) > 200:
                logger.warning(f"Group {i} event {j}: text length {len(text)} may exceed caption limits")


# ---------------------------------------------------------------------------
# Clip Diversity Validation
# ---------------------------------------------------------------------------

MIN_TEMPORAL_GAP = 1.0
MIN_TIMELINE_SPREAD_FRACTION = 0.15


def validate_clip_diversity(groups: list[dict], source_duration: float) -> None:
    """Check clip diversity within each group and log warnings for poor diversity.

    Checks:
    1. Temporal gaps — clips too close together (< MIN_TEMPORAL_GAP) get flagged
    2. Timeline spread — all clips clustered in a small fraction of source duration
    3. Duration variety — all clips the same approximate length

    This is a soft check (warnings only). Actual enforcement happens in the
    assemble stage and overlap removal.
    """
    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        if len(clips) < 3:
            continue

        # Sort by start time
        sorted_clips = sorted(clips, key=lambda c: c.get("source_start", 0))

        # 1. Temporal gap checks
        tight_pairs = 0
        for j in range(1, len(sorted_clips)):
            gap = sorted_clips[j]["source_start"] - sorted_clips[j - 1]["source_end"]
            if gap < MIN_TEMPORAL_GAP:
                tight_pairs += 1

        if tight_pairs > len(sorted_clips) // 2:
            logger.info(
                f"Group {i}: {tight_pairs}/{len(sorted_clips)-1} clip pairs have "
                f"<{MIN_TEMPORAL_GAP}s gap — consider more temporal spread"
            )

        # 2. Timeline spread
        first_start = sorted_clips[0]["source_start"]
        last_end = sorted_clips[-1]["source_end"]
        span = last_end - first_start
        if source_duration > 30 and span < source_duration * MIN_TIMELINE_SPREAD_FRACTION:
            logger.info(
                f"Group {i}: clips span only {span:.0f}s/{source_duration:.0f}s "
                f"({span/source_duration*100:.0f}%) — consider wider timeline spread"
            )

        # 3. Duration variety
        durations = [c["source_end"] - c["source_start"] for c in sorted_clips]
        avg_dur = sum(durations) / len(durations)
        uniform = all(abs(d - avg_dur) < 3.0 for d in durations)
        if uniform and len(durations) >= 4:
            logger.info(
                f"Group {i}: all {len(durations)} clips are ~{avg_dur:.0f}s — "
                f"consider mixing SHORT/MEDIUM/LONG clip lengths"
            )


# ---------------------------------------------------------------------------
# Pacing Enforcement
# ---------------------------------------------------------------------------

def _classify_clip_duration(duration: float) -> str:
    """Classify a clip as SHORT, MEDIUM, or LONG based on duration."""
    from backend.pipeline.analyzer import classify_clip_duration

    return classify_clip_duration(duration)


FAST_PACED_GENRES = {"comedy_sketch", "roast_reaction", "sports_fitness", "game_challenge"}


def enforce_clip_pacing(groups: list[dict], content_type: str = "", source_duration: float = 0.0) -> int:
    """Enforce deterministic pacing rules on clip selections.

    Rules:
    - No back-to-back LONG clips (merge or trim the weaker one)
    - No 3+ SHORT clips in a row (merge the two weakest adjacent)
    - Final clip must be MEDIUM (payoff moments need enough time for the reveal)
      — skipped for fast-paced genres (comedy, roast, sports, game) where SHORT
      payoffs are natural.

    Returns the number of clips modified or removed.
    """
    adjustments = 0

    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        if len(clips) < 2:
            continue

        clips.sort(key=lambda c: c.get("source_start", 0))

        # 1. Final clip must be MEDIUM (payoff needs time for the reveal)
        #    Skip for fast-paced genres where SHORT payoffs are natural.
        if clips and content_type not in FAST_PACED_GENRES:
            last = clips[-1]
            last_dur = last.get("source_end", 0) - last.get("source_start", 0)
            if _classify_clip_duration(last_dur) != "MEDIUM" and len(clips) >= 2:
                # Find nearest MEDIUM clip to swap with
                for j in range(len(clips) - 2, -1, -1):
                    j_dur = clips[j].get("source_end", 0) - clips[j].get("source_start", 0)
                    if _classify_clip_duration(j_dur) == "MEDIUM":
                        clips[-1], clips[j] = clips[j], clips[-1]
                        adjustments += 1
                        logger.info(
                            f"Group {i}: Swapped final {_classify_clip_duration(last_dur)} clip with "
                            f"clip at position {j} to end on MEDIUM payoff"
                        )
                        break

        # Re-classify after potential swap
        classified = [
            (k, _classify_clip_duration(c.get("source_end", 0) - c.get("source_start", 0)))
            for k, c in enumerate(clips)
        ]

        # 2. No back-to-back LONG clips — trim the one with lower importance
        k = 0
        while k < len(classified) - 1:
            if classified[k][1] == "LONG" and classified[k + 1][1] == "LONG":
                # Trim the less important one to MEDIUM range (max 15s)
                imp_a = _estimate_clip_importance(clips[classified[k][0]])
                imp_b = _estimate_clip_importance(clips[classified[k + 1][0]])
                trim_idx = classified[k][0] if imp_a <= imp_b else classified[k + 1][0]
                clip = clips[trim_idx]
                clip_dur = clip.get("source_end", 0) - clip.get("source_start", 0)
                # Trim to 15s but respect 3s minimum, clamp to source duration
                target_dur = max(3.0, min(15.0, clip_dur))
                if clip_dur > target_dur:
                    new_end = clip["source_start"] + target_dur
                    if source_duration > 0:
                        new_end = min(new_end, source_duration)
                    clip["source_end"] = round(new_end, 3)
                    adjustments += 1
                    logger.info(
                        f"Group {i}: Trimmed back-to-back LONG clip at "
                        f"{clip['source_start']:.1f}s from {clip_dur:.1f}s to 15.0s"
                    )
                    # Reclassify
                    new_dur = clip["source_end"] - clip["source_start"]
                    classified[trim_idx] = (trim_idx, _classify_clip_duration(new_dur))
            k += 1

        # 3. No 3+ SHORT in a row — merge the two weakest adjacent SHORTs
        #    Skip for fast-paced genres where rapid cuts are natural.
        if content_type not in FAST_PACED_GENRES:
            classified = [
                (k, _classify_clip_duration(c.get("source_end", 0) - c.get("source_start", 0)))
                for k, c in enumerate(clips)
            ]
            run_start = 0
            while run_start < len(classified):
                # Find run of SHORT
                run_end = run_start
                while run_end < len(classified) and classified[run_end][1] == "SHORT":
                    run_end += 1
                run_len = run_end - run_start
                if run_len >= 3:
                    # Find the two weakest adjacent SHORTs in this run
                    best_merge_pair = None
                    best_combined_imp = float("inf")
                    for m in range(run_start, run_end - 1):
                        imp_a = _estimate_clip_importance(clips[classified[m][0]])
                        imp_b = _estimate_clip_importance(clips[classified[m + 1][0]])
                        combined = imp_a + imp_b
                        if combined < best_combined_imp:
                            best_combined_imp = combined
                            best_merge_pair = (m, m + 1)
                    if best_merge_pair:
                        idx_a = classified[best_merge_pair[0]][0]
                        idx_b = classified[best_merge_pair[1]][0]
                        clip_a = clips[idx_a]
                        clip_b = clips[idx_b]
                        # Merge by extending clip_a to cover clip_b
                        clip_a["source_end"] = clip_b.get("source_end", clip_a["source_end"])
                        # Remove clip_b
                        clips.pop(idx_b)
                        adjustments += 1
                        logger.info(
                            f"Group {i}: Merged 3+ SHORT run by combining clips at "
                            f"{clip_a['source_start']:.1f}s and {clip_b['source_start']:.1f}s"
                        )
                        # Rebuild classified after removal
                        classified = [
                            (k, _classify_clip_duration(c.get("source_end", 0) - c.get("source_start", 0)))
                            for k, c in enumerate(clips)
                        ]
                        # Don't advance run_start — re-check from same position
                        continue
                # Advance past the current run (at least 1 position)
                run_start = max(run_end, run_start + 1)

        # 4. Re-check: final clip must be MEDIUM (merge may have changed classification)
        if clips:
            last = clips[-1]
            last_dur = last.get("source_end", 0) - last.get("source_start", 0)
            if _classify_clip_duration(last_dur) != "MEDIUM" and len(clips) >= 2:
                for j in range(len(clips) - 2, -1, -1):
                    j_dur = clips[j].get("source_end", 0) - clips[j].get("source_start", 0)
                    if _classify_clip_duration(j_dur) == "MEDIUM":
                        clips[-1], clips[j] = clips[j], clips[-1]
                        adjustments += 1
                        logger.info(
                            f"Group {i}: Re-swapped final clip after merge to end on MEDIUM"
                        )
                        break

        group["source_clips"] = clips

    return adjustments


# ---------------------------------------------------------------------------
# Diversity Repair
# ---------------------------------------------------------------------------

def repair_clip_diversity(groups: list[dict], source_duration: float) -> int:
    """Actively repair diversity issues flagged by validate_clip_diversity.

    Repairs:
    - Tight temporal gaps: expand the tighter clip outward by MIN_TEMPORAL_GAP
    - Narrow timeline spread: replace most-redundant clip with one from
      the least-represented region of the group

    Returns the number of clips repaired.
    """
    repairs = 0

    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        if len(clips) < 3:
            continue

        clips.sort(key=lambda c: c.get("source_start", 0))

        # 1. Fix tight temporal gaps
        for j in range(1, len(clips)):
            gap = clips[j]["source_start"] - clips[j - 1]["source_end"]
            if 0 < gap < MIN_TEMPORAL_GAP:
                deficit = MIN_TEMPORAL_GAP - gap
                # Expand the earlier clip's end outward, but don't create overlap
                expandable = deficit / 2.0
                max_expand = clips[j]["source_start"] - clips[j - 1]["source_end"] - 0.1
                expandable = min(expandable, max(0.0, max_expand))
                if expandable > 0:
                    old_end = clips[j - 1]["source_end"]
                    new_end = old_end + expandable
                    if source_duration > 0:
                        new_end = min(new_end, source_duration)
                    clips[j - 1]["source_end"] = round(new_end, 3)
                    repairs += 1
                    logger.info(
                        f"Group {i}: Expanded clip at {clips[j-1]['source_start']:.1f}s "
                        f"end by {expandable:.1f}s to fix tight gap"
                    )

        # 2. Fix narrow timeline spread
        if source_duration > 30:
            first_start = clips[0]["source_start"]
            last_end = clips[-1]["source_end"]
            span = last_end - first_start
            if span < source_duration * MIN_TIMELINE_SPREAD_FRACTION:
                # Find the least-represented region (gaps between clips)
                # and the most-redundant clip (highest pairwise overlap)
                # For simplicity: find the clip with highest overlap with another
                max_overlap = 0.0
                redundant_idx = -1
                for j in range(len(clips)):
                    for k in range(len(clips)):
                        if j == k:
                            continue
                        overlap_start = max(clips[j]["source_start"], clips[k]["source_start"])
                        overlap_end = min(clips[j]["source_end"], clips[k]["source_end"])
                        overlap = max(0.0, overlap_end - overlap_start)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            redundant_idx = j

                if redundant_idx >= 0 and max_overlap > 0:
                    # Find the largest gap in the group's coverage
                    gaps = []
                    for j in range(1, len(clips)):
                        gap_start = clips[j - 1]["source_end"]
                        gap_end = clips[j]["source_start"]
                        if gap_end - gap_start > 2.0:
                            gaps.append((gap_start, gap_end, gap_end - gap_start))
                    gaps.sort(key=lambda g: g[2], reverse=True)

                    if gaps:
                        # Place a short clip (≤6s) in the largest gap
                        gap_start, gap_end, gap_size = gaps[0]
                        clip_dur = min(6.0, gap_size * 0.5)
                        clip_center = (gap_start + gap_end) / 2.0
                        clips[redundant_idx]["source_start"] = round(clip_center - clip_dur / 2, 3)
                        clips[redundant_idx]["source_end"] = round(clip_center + clip_dur / 2, 3)
                        repairs += 1
                        logger.info(
                            f"Group {i}: Relocated redundant clip to gap at "
                            f"{clips[redundant_idx]['source_start']:.1f}s for wider spread"
                        )

        group["source_clips"] = clips

    return repairs


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _compute_clip_overlap_ratio(group_a: dict, group_b: dict) -> float:
    """Compute what fraction of clip timeline overlaps between two groups.

    Uses total overlapping seconds / total unique seconds to measure similarity.
    A ratio of 0.0 means no overlap; 1.0 means identical clip coverage.

    Intervals within each group are merged first to avoid double-counting
    overlapping clips (e.g., group B has [5-15, 10-25] against group A's [0-20]).
    """
    clips_a = group_a.get("source_clips", [])
    clips_b = group_b.get("source_clips", [])
    if not clips_a or not clips_b:
        return 0.0

    def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not intervals:
            return []
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    # Collect and merge clip intervals within each group
    intervals_a = _merge_intervals(sorted(
        [(c.get("source_start", 0), c.get("source_end", 0)) for c in clips_a]
    ))
    intervals_b = _merge_intervals(sorted(
        [(c.get("source_start", 0), c.get("source_end", 0)) for c in clips_b]
    ))

    total_overlap = 0.0
    for sa, ea in intervals_a:
        for sb, eb in intervals_b:
            overlap_start = max(sa, sb)
            overlap_end = min(ea, eb)
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start

    total_a = sum(e - s for s, e in intervals_a)
    total_b = sum(e - s for s, e in intervals_b)
    total_unique = total_a + total_b - total_overlap

    if total_unique <= 0:
        return 0.0

    # Return overlap relative to the smaller group (how much of it is duplicated)
    min_total = min(total_a, total_b)
    return total_overlap / min_total if min_total > 0 else 0.0


def deduplicate_groups(groups: list[dict]) -> list[dict]:
    """Remove groups that are too similar based on clip timeline overlap.

    Uses fuzzy overlap scoring: if two groups share more than
    GROUP_OVERLAP_THRESHOLD of their clip time, the weaker one is pruned.
    """
    from backend.config import GROUP_OVERLAP_THRESHOLD

    if len(groups) <= 1:
        return groups

    # Remove empty groups first
    valid = []
    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        if not clips:
            logger.warning(f"Pruning Group {i}: No source clips")
            continue
        valid.append(group)

    if not valid:
        logger.warning("All groups filtered out! Keeping first group.")
        return [groups[0]]

    # Pairwise overlap check — keep the stronger group when similarity is high
    keep = []
    pruned_indices = set()

    for i in range(len(valid)):
        if i in pruned_indices:
            continue
        for j in range(i + 1, len(valid)):
            if j in pruned_indices:
                continue
            ratio = _compute_clip_overlap_ratio(valid[i], valid[j])
            if ratio > GROUP_OVERLAP_THRESHOLD:
                # Prune the weaker group (fewer clips, or shorter estimated duration)
                wi = len(valid[i].get("source_clips", []))
                wj = len(valid[j].get("source_clips", []))
                di = valid[i].get("estimated_duration_seconds", 0)
                dj = valid[j].get("estimated_duration_seconds", 0)

                if (wi, di) >= (wj, dj):
                    pruned_indices.add(j)
                    logger.warning(
                        f"Pruning Group {j} ({wj} clips, {dj:.0f}s): "
                        f"{ratio*100:.0f}% overlap with Group {i}"
                    )
                else:
                    pruned_indices.add(i)
                    logger.warning(
                        f"Pruning Group {i} ({wi} clips, {di:.0f}s): "
                        f"{ratio*100:.0f}% overlap with Group {j}"
                    )
                    break  # Group i is gone, no need to check more

    keep = [g for idx, g in enumerate(valid) if idx not in pruned_indices]

    if not keep:
        logger.warning("All groups filtered out by overlap check! Keeping first group.")
        return [valid[0]]

    if len(keep) < len(valid):
        logger.info(
            f"Group dedup: {len(valid)} -> {len(keep)} groups "
            f"(threshold={GROUP_OVERLAP_THRESHOLD*100:.0f}%)"
        )

    return keep


# ---------------------------------------------------------------------------
# Final Integrity Validation
# ---------------------------------------------------------------------------

def _transfer_per_unit_targets(groups: list[dict]) -> None:
    """Transfer stashed per-unit duration targets from dict keys to ReelGroup fields.

    execute_plan stashes _unit_reel_dur_min/_unit_reel_dur_max on each group
    dict. These must be moved to unit_dur_min/unit_dur_max before Pydantic
    converts the dicts to ReelGroup models (unknown keys cause validation errors).
    """
    for g in groups:
        if isinstance(g, dict):
            g["unit_dur_min"] = g.pop("_unit_reel_dur_min", 0.0)
            g["unit_dur_max"] = g.pop("_unit_reel_dur_max", 0.0)


def finalize_edit(
    plan_dict: dict,
    source_duration: float,
    min_groups: int = 1,
    preserve_layout: bool = False,
    content_type: str = "",
) -> ReelPlan:
    """Run all validation steps and return a validated ReelPlan.

    This is the single entry point for post-LLM validation.

    ``preserve_layout`` skips every destructive repair (overlap removal,
    clip expansion, pacing swaps/merges, diversity relocation, group
    deduplication). The executor mode passes it because its clip ranges,
    reel order and overlap-freedom are already guaranteed deterministically
    — the legacy repairs fight those guarantees (they re-sort clips, stretch
    them toward CLIP_DURATION_SOFT_MAX and relocate them into "gaps").
    """
    groups = plan_dict.get("reel_groups", [])
    if not groups:
        raise RuntimeError("No reel_groups in plan")

    entity_grouped = plan_dict.get("entity_grouped", False)

    # 1. Clip bounds
    adjusted = validate_clip_bounds(groups, source_duration)
    if adjusted > 0:
        logger.info(f"Adjusted {adjusted} clip timestamps to valid bounds")

    # 2. Timing validation
    validate_timing(groups, source_duration)

    if preserve_layout:
        # 3. Narration validation
        validate_narration(groups)

        # 4. Caption verification
        verify_captions(groups)

        # Floor enforcement — the executor already guarantees >= min_groups
        # Entity content: skip floor check (low-quality entities intentionally dropped)
        if not entity_grouped and len(groups) < min_groups:
            raise RuntimeError(
                f"Group count ({len(groups)}) fell below minimum ({min_groups})"
            )

        total_clips = sum(len(g.get("source_clips", [])) for g in groups)
        total_narrations = sum(len(g.get("narration_events", [])) for g in groups)
        logger.info(
            f"Plan validated (layout preserved): {len(groups)} groups, "
            f"{total_clips} clips, {total_narrations} narrations"
        )

        # Last-resort repair: fill in missing reel_start/reel_end on any
        # narration events that slipped through without timestamps
        for g in groups:
            for ev in g.get("narration_events", []):
                if isinstance(ev, dict) and "reel_start" not in ev:
                    ev["reel_start"] = 0.0
                if isinstance(ev, dict) and "reel_end" not in ev:
                    ev["reel_end"] = 0.0

        _transfer_per_unit_targets(groups)
        return ReelPlan(**plan_dict)

    # 3. Overlap removal
    removed = remove_overlaps(groups)
    if removed > 0:
        logger.info(f"Removed {removed} overlapping clips")

    # 4. Narration validation
    validate_narration(groups)

    # 5. Duration verification
    verify_duration(groups, source_duration)

    # 6. Caption verification
    verify_captions(groups)

    # 7. Clip diversity validation (soft checks — warnings only)
    validate_clip_diversity(groups, source_duration)

    # 8. Repair diversity issues
    diversity_repairs = repair_clip_diversity(groups, source_duration)
    if diversity_repairs > 0:
        logger.info(f"Repaired {diversity_repairs} diversity issues")

    # 9. Enforce pacing rules
    pacing_adjustments = enforce_clip_pacing(groups, content_type=content_type, source_duration=source_duration)
    if pacing_adjustments > 0:
        logger.info(f"Made {pacing_adjustments} pacing adjustments")

    # 10. Deduplication
    deduplicated = deduplicate_groups(groups)
    plan_dict["reel_groups"] = deduplicated

    # 11. Floor enforcement — fail if dedup dropped below minimum
    # Entity content: skip floor check (low-quality entities intentionally dropped)
    if not entity_grouped and len(deduplicated) < min_groups:
        raise RuntimeError(
            f"Group count ({len(deduplicated)}) fell below minimum ({min_groups}) "
            f"after deduplication. Need at least {min_groups} standalone groups."
        )

    # 12. Log summary
    total_clips = sum(len(g.get("source_clips", [])) for g in deduplicated)
    total_narrations = sum(len(g.get("narration_events", [])) for g in deduplicated)
    avg_duration = sum(g.get("estimated_duration_seconds", 0) for g in deduplicated) / max(len(deduplicated), 1)
    logger.info(
        f"Plan validated: {len(deduplicated)} groups, {total_clips} clips, "
        f"{total_narrations} narrations, avg {avg_duration:.1f}s"
    )

    # Last-resort repair: fill in missing reel_start/reel_end on any
    # narration events that slipped through without timestamps
    for g in deduplicated:
        for ev in g.get("narration_events", []):
            if isinstance(ev, dict) and "reel_start" not in ev:
                ev["reel_start"] = 0.0
            if isinstance(ev, dict) and "reel_end" not in ev:
                    ev["reel_end"] = 0.0

    _transfer_per_unit_targets(deduplicated)
    return ReelPlan(**plan_dict)
