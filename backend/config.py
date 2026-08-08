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
    "AI_PROVIDER",
    "BASE_DIR",
    "CLIPS_DIR",
    "DOWNLOADS_DIR",
    "ENTITY_MIN_SEGMENT_SECONDS",
    "ENTITY_MAX_SEGMENTS_MULTIPLIER",
    "FFMPEG_PATH",
    "FFPROBE_PATH",
    "MAX_INPUT_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "MIN_CONTENT_DURATION",
    "MIN_ENTITY_REEL_SECONDS",
    "MIN_USABLE_BLOCK_FRACTION",
    "MULTIMODAL_ENABLED",
    "OCR_MAX_CONCURRENCY",
    "OCR_MAX_FRAMES",
    "OCR_MODE",
    "OCR_SAMPLE_INTERVAL_SECONDS",
    "OPENCODE_API_KEY",
    "OPENCODE_BASE_URL",
    "OPENCODE_MODEL",
    "OUTPUTS_DIR",
    "PLAN_MODE",
    "SCENE_CUT_THRESHOLD",
    "SCENE_SAMPLE_FPS",
    "TTS_VOICE",
    "VISION_API_KEY",
    "VISION_BASE_URL",
    "VISION_MODEL",
    "VISION_OCR_ENABLED",
    "VISION_TIMEOUT_SECONDS",
    "WHISPER_COMPUTE_TYPE_CPU",
    "WHISPER_COMPUTE_TYPE_CUDA",
    "WHISPER_MODEL_SIZE",
    "WORKING_DIR",
    "cleanup_job_files",
    "get_job_working_dir",
    "validate_config",
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


def cleanup_job_files(job_id: str) -> None:
    """Remove all temporary files for a job: working dir, clips, and LLM debug logs."""
    import shutil

    # 1. Remove the job's working directory (checkpoints, intermediate files, downloads)
    job_work_dir = WORKING_DIR / job_id
    if job_work_dir.exists():
        try:
            shutil.rmtree(job_work_dir)
            logger.info("Cleaned up working dir: %s", job_work_dir)
        except Exception as e:
            logger.warning("Failed to clean working dir %s: %s", job_work_dir, e)

    # 2. Remove clip files for this job
    for clip_path in CLIPS_DIR.glob(f"{job_id}_*.mp4"):
        try:
            clip_path.unlink()
            logger.info("Cleaned up clip: %s", clip_path)
        except Exception as e:
            logger.warning("Failed to clean clip %s: %s", clip_path, e)

    # 3. Remove LLM debug logs from WORKING_DIR root
    for debug_file in WORKING_DIR.glob("llm_debug_*.txt"):
        try:
            debug_file.unlink()
        except Exception:
            pass


WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3-turbo")
WHISPER_COMPUTE_TYPE_CUDA = os.environ.get("WHISPER_COMPUTE_TYPE_CUDA", "float16")
WHISPER_COMPUTE_TYPE_CPU = os.environ.get("WHISPER_COMPUTE_TYPE_CPU", "int8")

OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY")
OPENCODE_BASE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
OPENCODE_MODEL = os.environ.get("OPENCODE_MODEL", "mimo-v2.5-free")

# AI Provider: "mimo" for OpenCode API
AI_PROVIDER = os.environ.get("AI_PROVIDER", "mimo")

MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "134464"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "65536"))
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "max")

CLIP_COUNT_MIN = int(os.environ.get("CLIP_COUNT_MIN", "6"))
CLIP_COUNT_MAX = int(os.environ.get("CLIP_COUNT_MAX", "12"))
CLIP_DURATION_SOFT_MIN = float(os.environ.get("CLIP_DURATION_SOFT_MIN", "6"))
CLIP_DURATION_SOFT_MAX = float(os.environ.get("CLIP_DURATION_SOFT_MAX", "30"))
HOOK_SECONDS = float(os.environ.get("HOOK_SECONDS", "3"))
INSIGHT_SECONDS_MAX = float(os.environ.get("INSIGHT_SECONDS_MAX", "4"))
MIN_OUTPUT_DURATION = int(os.environ.get("MIN_OUTPUT_DURATION", "90"))
MAX_OUTPUT_DURATION = int(os.environ.get("MAX_OUTPUT_DURATION", "100"))
# Entity-mode reel minimum: distinct from global MIN_OUTPUT_DURATION.
# A candidate with Ns of usable content should produce ~N*1.3s, not be
# clamped to the video-level 90s floor.
MIN_ENTITY_REEL_SECONDS = int(os.environ.get("MIN_ENTITY_REEL_SECONDS", "15"))
# Minimum seconds of actual clip content required per group.
# Compositor extends last clip into source if clips fall short.
MIN_CONTENT_DURATION = float(os.environ.get("MIN_CONTENT_DURATION", "90"))

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1350
OUTPUT_FPS = 60

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
VAD_DUCKING_DEPTH = float(os.environ.get("VAD_DUCKING_DEPTH", "0.85"))
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

# Group deduplication: if two groups share more than this fraction of clip
# timeline overlap, the weaker group is pruned
GROUP_OVERLAP_THRESHOLD = float(os.environ.get("GROUP_OVERLAP_THRESHOLD", "0.5"))

# If fewer than this fraction of blocks in a window pass the importance>=25
# usability gate (e.g. VAD produced no speech energy), the gate is relaxed to
# avoid starving the planner of pickable content. 0.0 disables the fallback.
MIN_USABLE_BLOCK_FRACTION = float(os.environ.get("MIN_USABLE_BLOCK_FRACTION", "0.3"))

# Fast mode: skip expensive operations for faster iteration during development
FAST_MODE = os.environ.get("FAST_MODE", "0") == "1"

# Plan mode:
#   "executor" (default) — LLM produces a story plan (regions/beats only, no
#       timestamps); Python deterministically maps beats to clips. Consistent
#       output for identical input.
#   "llm" — legacy multi-stage path: LLM picks exact clip timestamps.
PLAN_MODE = os.environ.get("PLAN_MODE", "executor")

# Multimodal enrichment is CPU/hosted only: scene detection uses OpenCV on the
# CPU, and OCR is sent through an OpenAI-compatible vision endpoint.
MULTIMODAL_ENABLED = os.environ.get("MULTIMODAL_ENABLED", "1") == "1"
ENTITY_MIN_SEGMENT_SECONDS = float(os.environ.get("ENTITY_MIN_SEGMENT_SECONDS", "20"))
ENTITY_MAX_SEGMENTS_MULTIPLIER = int(os.environ.get("ENTITY_MAX_SEGMENTS_MULTIPLIER", "3"))
SCENE_SAMPLE_FPS = float(os.environ.get("SCENE_SAMPLE_FPS", "2"))
SCENE_CUT_THRESHOLD = float(os.environ.get("SCENE_CUT_THRESHOLD", "0.45"))
OCR_SAMPLE_INTERVAL_SECONDS = float(os.environ.get("OCR_SAMPLE_INTERVAL_SECONDS", "4"))
OCR_MAX_FRAMES = int(os.environ.get("OCR_MAX_FRAMES", "240"))
OCR_MAX_CONCURRENCY = int(os.environ.get("OCR_MAX_CONCURRENCY", "20"))
VISION_OCR_ENABLED = os.environ.get("VISION_OCR_ENABLED", "1") == "1"
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "http://localhost:20128/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "mimo-v2.5-free")
VISION_API_KEY = os.environ.get("VISION_API_KEY") or OPENCODE_API_KEY
VISION_TIMEOUT_SECONDS = float(os.environ.get("VISION_TIMEOUT_SECONDS", "30"))
# OCR mode: "keep" (use OCR results) or "skip" (skip OCR, default)
OCR_MODE = os.environ.get("OCR_MODE", "skip")


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

    # --- Check OpenCode API key ---
    if not OPENCODE_API_KEY:
        warnings.append(
            "OPENCODE_API_KEY not set. LLM analysis will use fallback heuristic plan."
        )
    else:
        logger.info("Config: OpenCode API key is set")

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
    if MIN_CONTENT_DURATION > MIN_OUTPUT_DURATION:
        warnings.append(
            f"MIN_CONTENT_DURATION ({MIN_CONTENT_DURATION}) > "
            f"MIN_OUTPUT_DURATION ({MIN_OUTPUT_DURATION}). "
            f"Clip expansion may not reach target."
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
    if not (0.0 <= MIN_USABLE_BLOCK_FRACTION <= 1.0):
        warnings.append(f"MIN_USABLE_BLOCK_FRACTION ({MIN_USABLE_BLOCK_FRACTION}) must be in [0.0, 1.0].")
    if MAX_WORKERS < 1:
        warnings.append(f"MAX_WORKERS ({MAX_WORKERS}) must be >= 1.")
    if GPU_SEMAPHORE_SIZE < 1:
        warnings.append(f"GPU_SEMAPHORE_SIZE ({GPU_SEMAPHORE_SIZE}) must be >= 1.")
    if SCENE_SAMPLE_FPS <= 0:
        warnings.append(f"SCENE_SAMPLE_FPS ({SCENE_SAMPLE_FPS}) must be > 0.")
    if ENTITY_MIN_SEGMENT_SECONDS <= 0:
        warnings.append(f"ENTITY_MIN_SEGMENT_SECONDS ({ENTITY_MIN_SEGMENT_SECONDS}) must be > 0.")
    if not (0.0 < SCENE_CUT_THRESHOLD <= 1.0):
        warnings.append(f"SCENE_CUT_THRESHOLD ({SCENE_CUT_THRESHOLD}) must be in (0.0, 1.0].")
    if OCR_SAMPLE_INTERVAL_SECONDS <= 0:
        warnings.append(f"OCR_SAMPLE_INTERVAL_SECONDS ({OCR_SAMPLE_INTERVAL_SECONDS}) must be > 0.")
    if OCR_MAX_FRAMES < 1:
        warnings.append(f"OCR_MAX_FRAMES ({OCR_MAX_FRAMES}) must be >= 1.")
    if OCR_MAX_CONCURRENCY < 1:
        warnings.append(f"OCR_MAX_CONCURRENCY ({OCR_MAX_CONCURRENCY}) must be >= 1.")
    if VISION_TIMEOUT_SECONDS <= 0:
        warnings.append(f"VISION_TIMEOUT_SECONDS ({VISION_TIMEOUT_SECONDS}) must be > 0.")
    if OCR_MODE not in ("keep", "skip"):
        warnings.append(f"OCR_MODE ({OCR_MODE}) must be 'keep' or 'skip'.")

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
