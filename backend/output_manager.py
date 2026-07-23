"""Output file management — final edit, duration probe, file staging.

Provides OutputManager for validating, duration-capping, and staging
final output videos into the OUTPUTS_DIR with descriptive filenames.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

from backend.ffmpeg_utils import get_ffmpeg, get_ffprobe

__all__ = ["OutputManager"]


class OutputManager:
    """Handles final output validation, duration capping, and file staging."""

    def final_edit_group(self, input_path: str, group: Any, working_dir: Path, job_id: str) -> str:
        """Validate and stage the final output for a group.

        Ensures the output file exists inside OUTPUTS_DIR, duration is capped
        at MAX_OUTPUT_DURATION, and has a descriptive filename based on the
        group title.

        Args:
            input_path: Path to the composited video to stage.
            group: ReelGroup object with group_index and reel_summary.
            working_dir: Working directory for this job.
            job_id: Unique job identifier.

        Returns:
            Final output file path in OUTPUTS_DIR.
        """
        from backend.config import OUTPUTS_DIR, MAX_OUTPUT_DURATION

        ffmpeg = get_ffmpeg()
        ffprobe = get_ffprobe()

        title_slug = "".join(
            c for c in (group.reel_summary.title or "reel")
            if c.isalnum() or c in (' ', '_', '-')
        ).strip()
        title_slug = title_slug.replace(' ', '_')[:40]
        output_filename = f"{job_id}_reel_{group.group_index}_{title_slug}.mp4"
        output_path = OUTPUTS_DIR / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration = self._probe_duration_sync(input_path, ffprobe)

        if duration > float(MAX_OUTPUT_DURATION):
            subprocess.run(
                [ffmpeg, "-loglevel", "error", "-i", input_path,
                 "-t", str(MAX_OUTPUT_DURATION), "-c", "copy", "-y", str(output_path)],
                check=True
            )
        else:
            shutil.copy2(input_path, output_path)

        return str(output_path)

    def _probe_duration_sync(self, path: str, ffprobe: str | None = None) -> float:
        """Probe the duration of a media file (synchronous).

        Args:
            path: Path to the media file.
            ffprobe: Optional ffprobe path; resolves automatically if not provided.

        Returns:
            Duration in seconds, or 0.0 if probing fails.
        """
        if ffprobe is None:
            ffprobe = get_ffprobe()
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True
        )
        if probe.returncode == 0:
            try:
                return float(probe.stdout.strip())
            except ValueError:
                pass
        return 0.0

    async def probe_duration(self, path: str) -> float:
        """Probe the duration of a media file (async wrapper).

        Args:
            path: Path to the media file.

        Returns:
            Duration in seconds, or 0.0 if probing fails.
        """
        return await asyncio.to_thread(self._probe_duration_sync, path)
