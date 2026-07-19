import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

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


WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "medium.en")
WHISPER_COMPUTE_TYPE_CUDA = os.environ.get("WHISPER_COMPUTE_TYPE_CUDA", "int8_float16")
WHISPER_COMPUTE_TYPE_CPU = os.environ.get("WHISPER_COMPUTE_TYPE_CPU", "int8")

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "poolside/laguna-xs-2.1")
NVIDIA_MODEL_FALLBACK = os.environ.get("NVIDIA_MODEL_FALLBACK", "stepfun-ai/step-3.7-flash")

CLIP_COUNT_MIN = int(os.environ.get("CLIP_COUNT_MIN", "4"))
CLIP_COUNT_MAX = int(os.environ.get("CLIP_COUNT_MAX", "5"))
CLIP_DURATION_SOFT_MIN = float(os.environ.get("CLIP_DURATION_SOFT_MIN", "10"))
CLIP_DURATION_SOFT_MAX = float(os.environ.get("CLIP_DURATION_SOFT_MAX", "15"))
HOOK_SECONDS = float(os.environ.get("HOOK_SECONDS", "3"))
INSIGHT_SECONDS_MAX = float(os.environ.get("INSIGHT_SECONDS_MAX", "4"))
TARGET_TOTAL_SECONDS = int(os.environ.get("TARGET_TOTAL_SECONDS", "80"))
MAX_OUTPUT_DURATION = int(os.environ.get("MAX_OUTPUT_DURATION", "90"))

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30

DOWNLOAD_MAX_HEIGHT = int(os.environ.get("DOWNLOAD_MAX_HEIGHT", "1080"))

FFMPEG_PATH = r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffprobe.exe"

TTS_VOICE = "en-US-ChristopherNeural"

CAPTION_FONT_SIZE = 64
CAPTION_FONT = "Arial"
