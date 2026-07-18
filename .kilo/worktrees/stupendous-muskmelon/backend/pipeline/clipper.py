import subprocess
import os
from backend.config import CLIPS_DIR
from typing import Callable, Optional, Any


def _get_encoder_opts() -> list[str]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
    except Exception as e:
        raise RuntimeError(f"Could not inspect ffmpeg encoders: {e}") from e

    if "h264_nvenc" in result.stdout:
        return ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p7", "-rc", "vbr", "-cq", "23"]

    if os.environ.get("ALLOW_CPU_FFMPEG_FALLBACK") == "1":
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "23"]

    raise RuntimeError("h264_nvenc encoder is not available. Install/configure NVIDIA ffmpeg support or set ALLOW_CPU_FFMPEG_FALLBACK=1.")


def cut_clips(source_path: str, clip_windows: list, job_id: str, 
              progress_cb: Optional[Callable[[str, float], None]] = None,
              reporter: Optional[Any] = None) -> list[str]:
    clip_paths = []
    total = len(clip_windows)
    encoder_opts = _get_encoder_opts()

    for i, window in enumerate(clip_windows):
        start = window["start"]
        end = window["end"]
        out_path = str(CLIPS_DIR / f"{job_id}_clip_{i}.mp4")

        if progress_cb:
            progress_cb(f"Cutting clip {i+1}/{total}", ((i + 1) / total) * 100)
        if reporter:
            reporter.update_clip_progress(i, "clipping", ((i + 1) / total) * 100)

        cmd = [
            "ffmpeg", "-i", source_path,
            "-ss", str(start), "-to", str(end),
        ] + encoder_opts + ["-c:a", "aac", "-y", out_path]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg clip failed: {e.stderr.decode()}") from e

        clip_paths.append(out_path)
        if reporter:
            reporter.update_clip_progress(i, "clipping_done", 100)

    if progress_cb:
        progress_cb(f"Cut {total} clips complete", 100)

    return clip_paths
