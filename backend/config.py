"""Application configuration — loads env vars, validates, and exports constants.

All configurable values are read from environment variables with sensible defaults.
validate_config() is called automatically on import to check critical paths and ranges.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

__all__ = [
    "BASE_DIR", "DOWNLOADS_DIR", "WORKING_DIR", "OUTPUTS_DIR", "CLIPS_DIR",
    "get_job_working_dir", "validate_config", "cleanup_stale_files",
    "FFMPEG_PATH", "FFPROBE_PATH",
    "NVIDIA_API_KEY", "NVIDIA_BASE_URL", "NVIDIA_MODEL", "NVIDIA_MODEL_FALLBACK",
    "WHISPER_MODEL_SIZE", "WHISPER_COMPUTE_TYPE_CUDA", "WHISPER_COMPUTE_TYPE_CPU",
    "TTS_VOICE",
]

logger = logging.getLogger(__name__)

# Load .env from backend/ directory with absolute path to be robust regardless of CWD
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

DOWNLOADS_DIR = BASE_DIR / "storage" / "downloads"
WORKING_DIR = BASE_DIR / "storage" / "working"
OUTPUTS_DIR = BASE_DIR / "storage" / "outputs"
CLIPS_DIR = BASE_DIR / "storage" / "clips"

DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
WORKING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def get_job_working_dir(job_id: str) -> Path:
    path = WORKING_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3-turbo")
WHISPER_COMPUTE_TYPE_CUDA = os.environ.get("WHISPER_COMPUTE_TYPE_CUDA", "float16")
WHISPER_COMPUTE_TYPE_CPU = os.environ.get("WHISPER_COMPUTE_TYPE_CPU", "int8")

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "openai/gpt-oss-120b")
NVIDIA_MODEL_FALLBACK = os.environ.get("NVIDIA_MODEL_FALLBACK", "nvidia/llama-3.3-nemotron-super-49b-v1.5")

CLIP_COUNT_MIN = int(os.environ.get("CLIP_COUNT_MIN", "6"))
CLIP_COUNT_MAX = int(os.environ.get("CLIP_COUNT_MAX", "12"))
CLIP_DURATION_SOFT_MIN = float(os.environ.get("CLIP_DURATION_SOFT_MIN", "10"))
CLIP_DURATION_SOFT_MAX = float(os.environ.get("CLIP_DURATION_SOFT_MAX", "30"))
HOOK_SECONDS = float(os.environ.get("HOOK_SECONDS", "3"))
INSIGHT_SECONDS_MAX = float(os.environ.get("INSIGHT_SECONDS_MAX", "4"))
MIN_OUTPUT_DURATION = int(os.environ.get("MIN_OUTPUT_DURATION", "90"))
MAX_OUTPUT_DURATION = int(os.environ.get("MAX_OUTPUT_DURATION", "180"))

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30

DOWNLOAD_MAX_HEIGHT = int(os.environ.get("DOWNLOAD_MAX_HEIGHT", "1080"))

FFMPEG_PATH = r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffprobe.exe"

TTS_VOICE = os.environ.get("TTS_VOICE", "en-US-ChristopherNeural")

CAPTION_FONT_SIZE = 64
CAPTION_FONT = "Arial"

# VAD-based audio ducking configuration
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
VAD_PRE_BUFFER_SECONDS = float(os.environ.get("VAD_PRE_BUFFER_SECONDS", "0.4"))
VAD_POST_BUFFER_SECONDS = float(os.environ.get("VAD_POST_BUFFER_SECONDS", "0.25"))
VAD_SCURVE_RAMP_SECONDS = float(os.environ.get("VAD_SCURVE_RAMP_SECONDS", "0.15"))
VAD_DUCKING_DEPTH = float(os.environ.get("VAD_DUCKING_DEPTH", "0.97"))
VAD_SILENCE_THRESHOLD = float(os.environ.get("VAD_SILENCE_THRESHOLD", "0.3"))

# Audio mixing constants — narration must be LOUD to be clearly audible over background
NARRATION_VOLUME_BOOST = float(os.environ.get("NARRATION_VOLUME_BOOST", "2.5"))
ALIMITER_LIMIT = float(os.environ.get("ALIMITER_LIMIT", "0.95"))
ALIMITER_ATTACK_MS = int(os.environ.get("ALIMITER_ATTACK_MS", "3"))
ALIMITER_RELEASE_MS = int(os.environ.get("ALIMITER_RELEASE_MS", "50"))

# Concurrency limits
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))
GPU_SEMAPHORE_SIZE = int(os.environ.get("GPU_SEMAPHORE_SIZE", "1"))
MAX_GROUP_RETRIES = int(os.environ.get("MAX_GROUP_RETRIES", "2"))

# Fast mode: skip expensive operations for faster iteration during development
FAST_MODE = os.environ.get("FAST_MODE", "0") == "1"


def validate_config() -> list[str]:
    """Validate critical configuration values at startup.

    Checks paths, environment variables, and numeric ranges for correctness.
    Returns warnings for non-fatal issues; logs info for successful checks.

    Returns:
        List of warning messages for non-fatal issues.
    """
    warnings = []

    # --- Check ffmpeg ---
    ffmpeg_path = Path(FFMPEG_PATH)
    if not ffmpeg_path.exists():
        import shutil
        if shutil.which("ffmpeg"):
            logger.info("Config: Using ffmpeg from PATH (config path does not exist)")
        else:
            warnings.append(
                f"ffmpeg not found at {FFMPEG_PATH} and not on PATH. "
                f"Downloads requiring muxing will fail."
            )
    else:
        logger.info("Config: ffmpeg found at %s", FFMPEG_PATH)

    # --- Check NVIDIA API key ---
    if not NVIDIA_API_KEY:
        warnings.append(
            "NVIDIA_API_KEY not set. LLM analysis will use fallback heuristic plan."
        )
    else:
        logger.info("Config: NVIDIA API key is set")

    # --- Check Whisper model ---
    logger.info("Config: Whisper model=%s, compute_type_cuda=%s",
                WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE_CUDA)

    # --- Validate duration constraints ---
    if MIN_OUTPUT_DURATION >= MAX_OUTPUT_DURATION:
        warnings.append(
            f"MIN_OUTPUT_DURATION ({MIN_OUTPUT_DURATION}) >= "
            f"MAX_OUTPUT_DURATION ({MAX_OUTPUT_DURATION}). "
            f"Output duration will be capped incorrectly."
        )

    # --- Validate numeric ranges ---
    if CLIP_COUNT_MIN < 1:
        warnings.append(f"CLIP_COUNT_MIN ({CLIP_COUNT_MIN}) must be >= 1.")
    if CLIP_COUNT_MAX < CLIP_COUNT_MIN:
        warnings.append(
            f"CLIP_COUNT_MAX ({CLIP_COUNT_MAX}) < CLIP_COUNT_MIN ({CLIP_COUNT_MIN})."
        )
    if not (0.0 <= VAD_THRESHOLD <= 1.0):
        warnings.append(f"VAD_THRESHOLD ({VAD_THRESHOLD}) must be in [0.0, 1.0].")
    if not (0.0 <= VAD_DUCKING_DEPTH <= 1.0):
        warnings.append(f"VAD_DUCKING_DEPTH ({VAD_DUCKING_DEPTH}) must be in [0.0, 1.0].")
    if MAX_WORKERS < 1:
        warnings.append(f"MAX_WORKERS ({MAX_WORKERS}) must be >= 1.")
    if GPU_SEMAPHORE_SIZE < 1:
        warnings.append(f"GPU_SEMAPHORE_SIZE ({GPU_SEMAPHORE_SIZE}) must be >= 1.")

    # --- Validate output dimensions ---
    if OUTPUT_WIDTH <= 0 or OUTPUT_HEIGHT <= 0:
        warnings.append(
            f"Invalid output dimensions: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}."
        )
    if OUTPUT_FPS <= 0 or OUTPUT_FPS > 120:
        warnings.append(f"OUTPUT_FPS ({OUTPUT_FPS}) should be in range 1-120.")

    # --- Check storage directories exist ---
    for name, path in [("DOWNLOADS_DIR", DOWNLOADS_DIR), ("WORKING_DIR", WORKING_DIR),
                       ("OUTPUTS_DIR", OUTPUTS_DIR), ("CLIPS_DIR", CLIPS_DIR)]:
        if not path.exists():
            warnings.append(f"{name} ({path}) does not exist and could not be created.")

    return warnings


# Run validation on import
_validation_warnings = validate_config()
for _w in _validation_warnings:
    logger.warning("Config: %s", _w)


def cleanup_stale_files(max_age_hours: int = 24) -> int:
    """Remove temp/working files older than max_age_hours.

    Cleans up:
      - Working directory intermediate files (group_*.mp4, group_*.wav, etc.)
      - Stale concat file lists (_concat_*.txt)
      - Empty directories

    Returns the number of files removed.
    """
    import time
    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    removed = 0

    for directory in (WORKING_DIR, CLIPS_DIR):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError:
                    pass

    # Remove empty subdirectories in working dir
    if WORKING_DIR.exists():
        for d in sorted(WORKING_DIR.iterdir(), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir():
                try:
                    if not any(d.iterdir()):
                        d.rmdir()
                except OSError:
                    pass

    if removed:
        logger.info("Cleanup: removed %d stale files (max age %dh)", removed, max_age_hours)
    return removed