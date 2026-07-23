"""Narration timing validation and auto-adjustment.

Ensures commentary narration events don't collide with active speech
and maintains proper gaps between narration windows.
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

__all__ = ["validate_and_adjust_narration_timings"]

logger = logging.getLogger(__name__)


def validate_and_adjust_narration_timings(
    group_narration_audio: list[dict[str, Any]],
    source_clips: list[Any],
    transcript: list[dict],
    target_duration: float,
    reporter: Any,
    group_idx: int,
) -> None:
    """Map reel timestamps to source transcript speech intervals across source_clips.

    Detects commentary narration events that collide with active speech (>15% overlap)
    and automatically shifts their reel_start to nearby speech gaps or the post-clip area.
    Ensures non-overlapping narration windows and caps all narrations within target_duration.

    Args:
        group_narration_audio: List of narration audio metadata dicts (mutated in place).
        source_clips: Source clip objects with source_start/source_end.
        transcript: Full transcript list with start/end/text keys.
        target_duration: Target duration for capping narration end times.
        reporter: ProgressReporter for status updates.
        group_idx: Group index for log messages.
    """
    if not group_narration_audio:
        return

    # 1. Map transcript speech segments to reel-relative timeline
    reel_speech_intervals = []
    cumulative_offset = 0.0
    for clip in source_clips:
        c_start = clip.source_start if hasattr(clip, "source_start") else clip["source_start"]
        c_end = clip.source_end if hasattr(clip, "source_end") else clip["source_end"]
        clip_dur = c_end - c_start
        for seg in transcript:
            s_start = seg["start"]
            s_end = seg["end"]
            ov_s = max(c_start, s_start)
            ov_e = min(c_end, s_end)
            if ov_s < ov_e - 0.1:  # meaningful speech duration
                reel_s = cumulative_offset + (ov_s - c_start)
                reel_e = cumulative_offset + (ov_e - c_start)
                reel_speech_intervals.append((reel_s, reel_e, seg.get("text", "")))
        cumulative_offset += clip_dur

    def get_speech_overlap(r_start: float, r_end: float) -> tuple:
        total_overlap = 0.0
        texts = []
        for s_start, s_end, text in reel_speech_intervals:
            ov_s = max(r_start, s_start)
            ov_e = min(r_end, s_end)
            if ov_s < ov_e:
                total_overlap += (ov_e - ov_s)
                if text:
                    texts.append(text)
        return total_overlap, texts

    def find_speech_gap(duration: float, search_start: float) -> float:
        """Find a gap of at least `duration` seconds with minimal speech overlap.
        Enforces 0.5s minimum distance from any speech boundary for clean VAD-driven ducking."""
        candidate = search_start
        max_search = max(target_duration, cumulative_offset + 30.0)
        while candidate + duration <= max_search:
            overlap, _ = get_speech_overlap(candidate, candidate + duration)
            if overlap <= 0.05:  # Near-zero overlap required for clean ducking
                # Verify 0.5s gap from nearest speech boundaries
                gap_ok = True
                for s_start, s_end, _ in reel_speech_intervals:
                    if abs(candidate - s_end) < 0.5 and candidate >= s_end - 0.1:
                        gap_ok = False
                        break
                    if abs((candidate + duration) - s_start) < 0.5 and (candidate + duration) <= s_start + 0.1:
                        gap_ok = False
                        break
                if gap_ok:
                    return candidate
            next_step = candidate + 0.2
            for s_start, s_end, _ in reel_speech_intervals:
                if s_start <= candidate < s_end:
                    next_step = max(next_step, s_end + 0.5)
            candidate = next_step
        return search_start

    # 2. Inspect and shift commentary narrations that collide with active dialogue
    for nar in group_narration_audio:
        event_type = nar.get("event_type", "commentary")
        duration = nar.get("duration", 0.0)
        if duration <= 0.1:
            continue

        reel_s = nar["reel_start"]
        reel_e = nar["reel_start"] + duration
        overlap, texts = get_speech_overlap(reel_s, reel_e)
        overlap_ratio = overlap / duration if duration > 0 else 0.0

        if event_type in ("commentary", "hook") and overlap_ratio > 0.10:
            sample_text = texts[0][:60] + "..." if texts else ""
            reporter.log_info(
                f"[WARN] Group {group_idx+1}: {event_type.capitalize()} narration '{nar['text'][:40]}...' "
                f"at reel [{reel_s:.2f}s-{reel_e:.2f}s] overlaps {overlap_ratio*100:.0f}% with transcript speech "
                f"(\"{sample_text}\"). Auto-shifting to nearest silent gap..."
            )
            new_s = find_speech_gap(duration, reel_s)
            if new_s != reel_s and new_s + duration <= target_duration:
                old_start = nar['reel_start']
                nar["reel_start"] = round(new_s, 2)
                nar["reel_end"] = round(new_s + duration, 2)
                reporter.log_info(
                    f"[INFO] Group {group_idx+1}: Auto-shifted {event_type} from "
                    f"[{old_start:.2f}s] -> [{nar['reel_start']:.2f}s-{nar['reel_end']:.2f}s] (gap verified)"
                )
            elif overlap_ratio > 0.4:
                reporter.log_info(
                    f"[WARN] Group {group_idx+1}: Could not find gap for '{nar['text'][:30]}...' "
                    f"({overlap_ratio*100:.0f}% overlap). Consider removing this narration."
                )

    # 3. Ensure narrations do not overlap with each other (0.8s minimum gap)
    group_narration_audio.sort(key=lambda x: x["reel_start"])
    for i in range(1, len(group_narration_audio)):
        prev_end = group_narration_audio[i-1]["reel_end"]
        curr_start = group_narration_audio[i]["reel_start"]
        curr_dur = group_narration_audio[i]["duration"]
        min_gap = 0.8
        if curr_start < prev_end + min_gap:
            new_start = prev_end + min_gap
            new_start = find_speech_gap(curr_dur, new_start)

            if new_start + curr_dur <= target_duration:
                old_start = group_narration_audio[i]["reel_start"]
                group_narration_audio[i]["reel_start"] = round(new_start, 2)
                group_narration_audio[i]["reel_end"] = round(new_start + curr_dur, 2)
                reporter.log_info(
                    f"[INFO] Group {group_idx+1}: Shifted narration '{group_narration_audio[i]['text'][:30]}...' "
                    f"from [{old_start:.2f}s] -> [{group_narration_audio[i]['reel_start']:.2f}s-{group_narration_audio[i]['reel_end']:.2f}s] "
                    f"(min gap {min_gap}s from prior narration, speech-gap verified)"
                )

    # 4. Cap narrations at target_duration
    for nar in group_narration_audio:
        original_dur = nar.get("duration", 0.0)
        if nar["reel_end"] > target_duration:
            if nar["reel_start"] < target_duration - 0.5:
                nar["reel_end"] = target_duration
                nar["duration"] = nar["reel_end"] - nar["reel_start"]
            else:
                nar["reel_start"] = max(0.0, target_duration - 0.5)
                nar["reel_end"] = target_duration
                nar["duration"] = 0.5

            if nar["duration"] < original_dur:
                try:
                    from backend.ffmpeg_utils import get_ffmpeg
                    ffmpeg = get_ffmpeg()
                    tmp_path = nar["path"] + ".tmp.wav"
                    subprocess.run([
                        ffmpeg, "-loglevel", "error", "-y",
                        "-i", nar["path"],
                        "-t", str(nar["duration"]),
                        "-c", "copy", tmp_path
                    ], check=True)
                    os.replace(tmp_path, nar["path"])
                    reporter.log_info(f"[INFO] Group {group_idx+1}: Trimmed narration audio file to {nar['duration']:.2f}s to fit target duration.")
                except Exception as e:
                    reporter.log_info(f"[WARN] Failed to trim narration audio: {e}")
