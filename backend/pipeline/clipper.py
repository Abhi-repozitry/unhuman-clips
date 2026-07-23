"""Video clip cutting module — extracts segments from source video.

Provides cut_clips() and cut_group_clips() with parallel execution
via ThreadPoolExecutor and fast input seeking (-ss before -i).
"""
from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from backend.config import CLIPS_DIR
from backend.ffmpeg_utils import get_encoder, get_ffmpeg

__all__ = ["cut_clips", "cut_group_clips"]


def _get_encoder_opts() -> list[str]:
    """Return ffmpeg encoder flags using the cached encoder detection."""
    encoder = get_encoder()
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p7", "-rc", "vbr", "-cq", "23"]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "23"]


def _validate_clip(source_path: str, start: float, end: float, source_duration: float = 0.0) -> None:
    """Validate clip timestamps before cutting.

    Args:
        source_path: Path to the source video file.
        start: Clip start time in seconds.
        end: Clip end time in seconds.
        source_duration: Total source duration for bounds checking.

    Raises:
        FileNotFoundError: If source file does not exist.
        ValueError: If timestamps are invalid.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source video not found: {source_path}")
    if start < 0:
        raise ValueError(f"Clip start time {start}s is negative")
    if end <= start:
        raise ValueError(f"Clip end time {end}s must be after start time {start}s")
    if source_duration > 0 and end > source_duration:
        raise ValueError(f"Clip end time {end}s exceeds source duration {source_duration}s")


def cut_clips(
    source_path: str,
    clip_windows: list[dict[str, float]],
    job_id: str,
    progress_cb: Callable[[str, float], None] | None = None,
    reporter: Any = None,
) -> list[str]:
    """Cut clips from source video with parallel execution and fast input seeking.

    Args:
        source_path: Path to the source video file.
        clip_windows: List of dicts with 'start' and 'end' keys.
        job_id: Job identifier for output naming.
        progress_cb: Optional progress callback.
        reporter: Optional ProgressReporter instance.

    Returns:
        List of output clip file paths in order.
    """
    total = len(clip_windows)
    if total == 0:
        return []

    encoder_opts = _get_encoder_opts()
    clip_paths = [None] * total

    def _cut_one(i: int, window: dict) -> str:
        start = window["start"]
        end = window["end"]
        out_path = str(CLIPS_DIR / f"{job_id}_clip_{i}.mp4")
        duration = max(0.1, end - start)
        cmd = [
            get_ffmpeg(), "-ss", f"{start:.6f}", "-i", source_path,
            "-t", f"{duration:.6f}",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
        ] + encoder_opts + ["-c:a", "aac", "-b:a", "192k", "-y", out_path]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg clip failed: {e.stderr.decode('utf-8', errors='replace')}") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FFmpeg clip timed out after 300s for clip {i}")

        if reporter:
            reporter.update_clip_progress(i, "clipping_done", 100)
        return out_path

    max_workers = min(4, total)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_cut_one, i, w): i for i, w in enumerate(clip_windows)}
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            clip_paths[idx] = future.result()
            completed += 1
            if progress_cb:
                progress_cb(f"Cut {completed}/{total} clips", (completed / total) * 100)

    if progress_cb:
        progress_cb(f"Cut {total} clips complete", 100)

    return clip_paths


def cut_group_clips(
    source_path: str,
    source_clips: list[dict[str, float]],
    job_id: str,
    group_idx: int,
    progress_cb: Callable[[str, float], None] | None = None,
    reporter: Any = None,
) -> list[str]:
    """Cut clips for a single reel group with parallel execution.

    Uses fast input seeking (-ss before -i) and ThreadPoolExecutor for speed.

    Args:
        source_path: Path to the source video file.
        source_clips: List of dicts with 'source_start' and 'source_end' keys.
        job_id: Job identifier for output naming.
        group_idx: Group index for output naming.
        progress_cb: Optional progress callback.
        reporter: Optional ProgressReporter instance.

    Returns:
        List of clip file paths in original order.
    """
    total = len(source_clips)
    if total == 0:
        return []

    encoder_opts = _get_encoder_opts()
    clip_paths = [None] * total

    def _cut_one(i: int, clip: dict) -> str:
        start = clip["source_start"]
        end = clip["source_end"]
        out_path = str(CLIPS_DIR / f"{job_id}_group{group_idx}_clip_{i}.mp4")
        duration = max(0.1, end - start)
        cmd = [
            get_ffmpeg(), "-ss", f"{start:.6f}", "-i", source_path,
            "-t", f"{duration:.6f}",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
        ] + encoder_opts + ["-c:a", "aac", "-b:a", "192k", "-y", out_path]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg clip failed: {e.stderr.decode('utf-8', errors='replace')}") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FFmpeg clip timed out after 300s for group {group_idx} clip {i}")

        if reporter:
            reporter.update_clip_progress(i, "clipping_done", 100)
        return out_path

    # Use up to 4 worker threads for parallel clip cutting
    max_workers = min(4, total)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_cut_one, i, clip): i for i, clip in enumerate(source_clips)}
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            clip_paths[idx] = future.result()
            completed += 1
            if progress_cb:
                progress_cb(f"Group {group_idx+1}: Cut {completed}/{total} clips", (completed / total) * 100)

    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Cut {total} clips complete", 100)

    return clip_paths
