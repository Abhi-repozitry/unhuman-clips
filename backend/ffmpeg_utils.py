"""Centralized ffmpeg/ffprobe path resolution and encoder detection.

Provides get_ffmpeg(), get_ffprobe(), and get_encoder() that:
1. Check config.py hardcoded path (if it exists on disk)
2. Fall back to shutil.which("ffmpeg") / shutil.which("ffprobe")
3. Raise RuntimeError if neither is found
4. Cache encoder detection (h264_nvenc vs libx264) as a singleton
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from functools import lru_cache

__all__ = ["get_ffmpeg", "get_ffprobe", "get_encoder"]


@lru_cache(maxsize=1)
def get_ffmpeg() -> str:
    """Return the path to the ffmpeg binary.

    Resolution order:
      1. FFMPEG_PATH from backend.config (if the file exists on disk)
      2. First 'ffmpeg' found on system PATH via shutil.which
      3. RuntimeError if not found
    """
    from backend.config import FFMPEG_PATH

    if os.path.isfile(FFMPEG_PATH):
        return FFMPEG_PATH

    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    raise RuntimeError(
        f"ffmpeg not found. Config path '{FFMPEG_PATH}' does not exist "
        f"and 'ffmpeg' is not on PATH."
    )


@lru_cache(maxsize=1)
def get_ffprobe() -> str:
    """Return the path to the ffprobe binary.

    Resolution order:
      1. FFPROBE_PATH from backend.config (if the file exists on disk)
      2. First 'ffprobe' found on system PATH via shutil.which
      3. Derive from get_ffmpeg() by replacing 'ffmpeg' with 'ffprobe' in the directory
      4. RuntimeError if not found
    """
    from backend.config import FFPROBE_PATH

    if os.path.isfile(FFPROBE_PATH):
        return FFPROBE_PATH

    on_path = shutil.which("ffprobe")
    if on_path:
        return on_path

    # Try deriving from ffmpeg location
    ffmpeg_dir = os.path.dirname(get_ffmpeg())
    candidate = os.path.join(ffmpeg_dir, "ffprobe.exe" if os.name == "nt" else "ffprobe")
    if os.path.isfile(candidate):
        return candidate

    raise RuntimeError(
        f"ffprobe not found. Config path '{FFPROBE_PATH}' does not exist, "
        f"'ffprobe' is not on PATH, and could not be derived from ffmpeg location."
    )


_encoder_cache: dict[str, str | None] = {}
_encoder_lock = threading.Lock()


def get_encoder() -> str:
    """Return the best available H.264 encoder name, cached after first call.

    Resolution order:
      1. h264_nvenc (GPU) if available in ffmpeg encoders
      2. libx264 (CPU) if ALLOW_CPU_FFMPEG_FALLBACK=1
      3. RuntimeError if neither is available

    The result is cached for the process lifetime — no repeated subprocess calls.
    """
    with _encoder_lock:
        if "encoder" in _encoder_cache:
            return _encoder_cache["encoder"]

    try:
        result = subprocess.run(
            [get_ffmpeg(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        if "h264_nvenc" in result.stdout:
            enc = "h264_nvenc"
        elif os.environ.get("ALLOW_CPU_FFMPEG_FALLBACK") == "1":
            enc = "libx264"
        else:
            raise RuntimeError(
                "h264_nvenc encoder is not available. "
                "Install/configure NVIDIA ffmpeg support or set ALLOW_CPU_FFMPEG_FALLBACK=1."
            )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not inspect ffmpeg encoders: {e}") from e

    with _encoder_lock:
        _encoder_cache["encoder"] = enc
    return enc
