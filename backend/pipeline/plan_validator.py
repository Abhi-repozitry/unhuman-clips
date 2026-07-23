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

import json
import logging
import re
from typing import Any

from backend.config import MAX_OUTPUT_DURATION, MIN_OUTPUT_DURATION
from backend.models import ReelPlan
from backend.pipeline.sanitize import sanitize_text

__all__ = [
    "repair_json",
    "validate_clip_bounds",
    "validate_timing",
    "remove_overlaps",
    "validate_narration",
    "verify_duration",
    "verify_captions",
    "finalize_edit",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Repair
# ---------------------------------------------------------------------------

def repair_json(text: str) -> str:
    """Repair truncated or malformed JSON from LLM output.

    Handles: trailing commas, unclosed strings, unbalanced braces/brackets,
    markdown fences, and partial JSON extraction.
    """
    if not text:
        return ""

    # Strip markdown fences
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

    # Try direct parse first
    try:
        json.loads(t)
        return t
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    repaired = re.sub(r',\s*([}\]])', r'\1', t)

    # Close unclosed string quotes
    unescaped_quotes = len(re.findall(r'(?<!\\)"', repaired))
    if unescaped_quotes % 2 != 0:
        repaired += '"'

    # Balance braces and brackets
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

    # Scan backwards for last complete JSON object
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

    # Last resort: scan for any valid JSON substring
    try:
        for start_pos in range(len(repaired)):
            if repaired[start_pos] in ('{', '['):
                for end_pos in range(len(repaired), start_pos, -1):
                    candidate = repaired[start_pos:end_pos]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue
    except (json.JSONDecodeError, IndexError):
        pass

    return ""


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
    for i, group in enumerate(groups):
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

def remove_overlaps(groups: list[dict]) -> int:
    """Detect and remove overlapping clips within each group.

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
                # Overlap detected — keep the longer clip
                prev_dur = prev.get("source_end", 0) - prev.get("source_start", 0)
                curr_dur = clip.get("source_end", 0) - clip.get("source_start", 0)
                if curr_dur > prev_dur:
                    filtered[-1] = clip
                    removed += 1
                    logger.info(f"Group {i}: Replaced shorter overlapping clip with longer one")
                else:
                    removed += 1
                    logger.info(f"Group {i}: Removed shorter overlapping clip")
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

            if ev_type not in ("hook", "commentary"):
                logger.warning(
                    f"Group {i} narration event {j} has unrecognized type '{ev_type}' — "
                    f"will be SILENTLY DROPPED before TTS"
                )
            else:
                usable_count += 1

            # Hook must start at 0.0
            if ev_type == "hook" and event.get("reel_start", 0) != 0.0:
                logger.warning(f"Group {i}: Hook must start at 0.0, correcting")
                event["reel_start"] = 0.0

        if usable_count == 0:
            logger.warning(f"Group {i}: ZERO usable narration events — reel will have NO narration")

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
                targets = [0.25, 0.50, 0.75]
                for idx, event in enumerate(commentary_events):
                    fraction = targets[idx % len(targets)]
                    new_start = round(est_dur * fraction, 2)
                    dur = event.get("reel_end", 0) - event.get("reel_start", 0)
                    event["reel_start"] = new_start
                    event["reel_end"] = round(new_start + dur, 2)


# ---------------------------------------------------------------------------
# Duration Verification
# ---------------------------------------------------------------------------

def verify_duration(groups: list[dict], source_duration: float) -> None:
    """Verify and enforce duration constraints on each group."""
    min_reel = min(MIN_OUTPUT_DURATION, int(source_duration * 0.6)) if source_duration < 120 else MIN_OUTPUT_DURATION

    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        clips_total = sum(c.get("source_end", 0) - c.get("source_start", 0) for c in clips)
        nar_events = group.get("narration_events", [])
        nar_total = sum(e.get("reel_end", 0) - e.get("reel_start", 0) for e in nar_events)
        actual_estimated = clips_total + 2.0

        llm_estimate = group.get("estimated_duration_seconds", 0)
        if llm_estimate < min_reel and source_duration >= min_reel:
            logger.warning(
                f"Group {i}: estimated {llm_estimate:.1f}s below target {min_reel}s. "
                f"Computed: {actual_estimated:.1f}s"
            )

        # Bump estimated to at least computed actual
        if actual_estimated > llm_estimate:
            group["estimated_duration_seconds"] = round(actual_estimated, 1)

        # Cap at MAX_OUTPUT_DURATION
        if group["estimated_duration_seconds"] > MAX_OUTPUT_DURATION:
            logger.warning(
                f"Group {i}: capping estimated {group['estimated_duration_seconds']:.1f}s "
                f"to {MAX_OUTPUT_DURATION}s"
            )
            group["estimated_duration_seconds"] = float(MAX_OUTPUT_DURATION)


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
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_groups(groups: list[dict]) -> list[dict]:
    """Remove duplicate groups by clip fingerprint."""
    if len(groups) <= 1:
        return groups

    filtered = []
    seen_fingerprints = set()

    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        if not clips:
            logger.warning(f"Pruning Group {i}: No source clips")
            continue

        fingerprint = tuple(
            sorted(
                (round(c.get("source_start", 0.0), 1), round(c.get("source_end", 0.0), 1))
                for c in clips
            )
        )
        if fingerprint in seen_fingerprints:
            logger.warning(f"Pruning Group {i}: Duplicate clip selection")
            continue

        seen_fingerprints.add(fingerprint)
        filtered.append(group)

    if not filtered:
        logger.warning("All groups filtered out! Keeping first group.")
        return [groups[0]]

    return filtered


# ---------------------------------------------------------------------------
# Final Integrity Validation
# ---------------------------------------------------------------------------

def finalize_edit(plan_dict: dict, source_duration: float) -> ReelPlan:
    """Run all validation steps and return a validated ReelPlan.

    This is the single entry point for post-LLM validation.
    """
    groups = plan_dict.get("reel_groups", [])
    if not groups:
        raise RuntimeError("No reel_groups in plan")

    # 1. Clip bounds
    adjusted = validate_clip_bounds(groups, source_duration)
    if adjusted > 0:
        logger.info(f"Adjusted {adjusted} clip timestamps to valid bounds")

    # 2. Timing validation
    validate_timing(groups, source_duration)

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

    # 7. Deduplication
    deduplicated = deduplicate_groups(groups)
    plan_dict["reel_groups"] = deduplicated

    # 8. Log summary
    total_clips = sum(len(g.get("source_clips", [])) for g in deduplicated)
    total_narrations = sum(len(g.get("narration_events", [])) for g in deduplicated)
    avg_duration = sum(g.get("estimated_duration_seconds", 0) for g in deduplicated) / max(len(deduplicated), 1)
    logger.info(
        f"Plan validated: {len(deduplicated)} groups, {total_clips} clips, "
        f"{total_narrations} narrations, avg {avg_duration:.1f}s"
    )

    return ReelPlan(**plan_dict)
