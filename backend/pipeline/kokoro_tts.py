"""Kokoro TTS module — English text-to-speech via hexgrad/Kokoro-82M.

Lazy-loads the pipeline on first use to avoid startup latency.
Singleton pattern: pipeline loaded once and reused across calls.
Config-driven via config/models.yaml (tts.kokoro section).
"""
from __future__ import annotations

import gc
import logging
import os
import subprocess
import time
from collections.abc import Callable

import numpy as np
import soundfile as sf
import torch

from backend.ffmpeg_utils import get_ffprobe

__all__ = ["synthesize_kokoro"]

logger = logging.getLogger(__name__)

# Lazy-loaded pipeline state
_pipeline = None

# Config defaults
MODEL_ID = "hexgrad/Kokoro-82M"
DEFAULT_VOICE = "am_adam"
LANG_CODE = "a"
DEFAULT_SPEED = 1.0
OUTPUT_SAMPLE_RATE = 16000

MIN_VALID_AUDIO_BYTES = 500
MAX_TTS_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.5
KOKORO_NATIVE_SR = 24000


def _clear_gpu_memory() -> None:
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    free_mem = torch.cuda.mem_get_info(0)[0] / 1024**2
    logger.info("GPU memory after cleanup: %.0f MB free", free_mem)


def _load_pipeline() -> None:
    global _pipeline
    if _pipeline is not None:
        return
    _clear_gpu_memory()
    from kokoro import KPipeline
    logger.info("Loading Kokoro TTS pipeline (lang=%s)...", LANG_CODE)
    t0 = time.time()
    _pipeline = KPipeline(lang_code=LANG_CODE, repo_id=MODEL_ID)
    logger.info("Kokoro TTS loaded in %.1fs", time.time() - t0)


def _resample(audio_arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio_arr
    import torchaudio
    wav = torch.from_numpy(audio_arr).unsqueeze(0).float()
    resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr)
    return resampler(wav).squeeze().numpy()


def _sanitize_tts_text(text: str) -> str:
    """Clean text for TTS to avoid phonemizer warnings.

    Replaces or removes characters that confuse the phonemizer: special
    unicode, excessive punctuation, abbreviations with periods, etc.
    """
    import re

    cleaned = text.strip()

    cleaned = cleaned.replace("…", "...")
    cleaned = cleaned.replace(""", '"').replace(""", '"')
    cleaned = cleaned.replace("'", "'").replace("'", "'")
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")
    cleaned = cleaned.replace("\u202f", " ").replace("\u00a0", " ")

    cleaned = re.sub(r"\.{2,}", ".", cleaned)

    cleaned = re.sub(r"([!?]){2,}", r"\1", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def synthesize_kokoro(
    text: str,
    out_path: str,
    progress_cb: Callable[[str, float], None] | None = None,
    voice: str | None = None,
    speed: float | None = None,
) -> float:
    """Synthesize English text to speech using Kokoro TTS.

    Returns duration of the generated audio in seconds.
    """
    if not text or not text.strip():
        raise RuntimeError("synthesize_kokoro called with empty text")

    text = _sanitize_tts_text(text)
    if not text:
        raise RuntimeError("sanitize_tts_text produced empty output")

    if progress_cb:
        progress_cb("Loading English TTS model...", 5)

    _load_pipeline()

    if progress_cb:
        progress_cb("Generating English TTS audio...", 20)

    tts_voice = voice or DEFAULT_VOICE
    tts_speed = speed if speed is not None else DEFAULT_SPEED

    last_error = None
    for attempt in range(1, MAX_TTS_ATTEMPTS + 1):
        try:
            audio_chunks = []
            for _, _, audio in _pipeline(text, voice=tts_voice, speed=tts_speed):
                audio_chunks.append(audio)

            if not audio_chunks:
                raise RuntimeError(f"Kokoro produced no audio for text: {text[:60]!r}")

            audio_arr = np.concatenate(audio_chunks)
            audio_arr = _resample(audio_arr, KOKORO_NATIVE_SR, OUTPUT_SAMPLE_RATE)
            sf.write(out_path, audio_arr, OUTPUT_SAMPLE_RATE)

            if not os.path.exists(out_path) or os.path.getsize(out_path) < MIN_VALID_AUDIO_BYTES:
                size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                raise RuntimeError(
                    f"Kokoro produced a suspiciously small file "
                    f"({size} bytes) for text: {text[:60]!r}"
                )
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_TTS_ATTEMPTS:
                logger.warning("Kokoro TTS attempt %d/%d failed (%s); retrying...", attempt, MAX_TTS_ATTEMPTS, e)
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            else:
                raise RuntimeError(f"Kokoro TTS failed after {MAX_TTS_ATTEMPTS} attempts: {last_error}") from last_error

    if progress_cb:
        progress_cb("Getting audio duration...", 80)

    try:
        result = subprocess.run(
            [get_ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out_path],
            capture_output=True, check=True, text=True, timeout=30,
        )
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get audio duration: {e}") from e

    if duration < 0.05:
        raise RuntimeError(
            f"Kokoro produced near-zero-duration file ({duration:.3f}s) for text: {text[:60]!r}"
        )

    if progress_cb:
        progress_cb("English TTS complete", 100)

    return duration
