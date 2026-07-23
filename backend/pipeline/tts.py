"""TTS module — text-to-speech via Microsoft Edge TTS (edge-tts).

Provides synthesize_commentary() with retry logic for transient failures
and audio validation to catch empty/truncated responses.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from typing import Callable

import edge_tts

from backend.config import TTS_VOICE
from backend.ffmpeg_utils import get_ffprobe

__all__ = ["synthesize_commentary"]

logger = logging.getLogger(__name__)

# TTS rate — configurable via environment variable
TTS_RATE = os.environ.get("TTS_RATE", "+10%")

# A successful edge-tts save should never be this small. Anything below this
# is almost certainly a truncated/empty response from the service (network
# hiccup, throttling, etc.) that would otherwise ship as silent narration
# with no error raised.
MIN_VALID_AUDIO_BYTES = 500

# Retry transient edge-tts/network failures instead of failing the whole job
# on a single blip.
MAX_TTS_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.5


def synthesize_commentary(
    text: str,
    out_path: str,
    progress_cb: Callable[[str, float], None] | None = None,
    rate: str | None = None,
) -> float:
    """Synthesize text to speech using edge-tts with retry logic.

    Args:
        text: Text to synthesize (must be non-empty).
        out_path: Output WAV file path.
        progress_cb: Optional progress callback.
        rate: TTS rate override (e.g., '+10%'). Defaults to TTS_RATE env var.

    Returns:
        Duration of the generated audio in seconds.

    Raises:
        RuntimeError: On empty text, TTS failure after retries, or invalid audio.
    """
    if not text or not text.strip():
        raise RuntimeError("synthesize_commentary called with empty text — refusing to synthesize silent audio.")

    tts_rate = rate or TTS_RATE

    if progress_cb:
        progress_cb("Generating TTS audio...", 10)

    # edge-tts's Communicate() XML-escapes its text input before building SSML,
    # so manually injected <break> tags would be escaped into inert literal text.
    # edge-tts's built-in SentenceBoundary handling already provides natural pacing
    # at sentence punctuation — no manual SSML break injection needed.
    last_error = None
    for attempt in range(1, MAX_TTS_ATTEMPTS + 1):
        try:
            async def _run_tts():
                communicate = edge_tts.Communicate(text, TTS_VOICE, rate=tts_rate)
                await communicate.save(out_path)

            asyncio.run(_run_tts())

            # Sanity check: a real synthesized line is never this small. Catch
            # silent/empty edge-tts responses here instead of shipping dead
            # air all the way to the final composed video.
            if not os.path.exists(out_path) or os.path.getsize(out_path) < MIN_VALID_AUDIO_BYTES:
                size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                raise RuntimeError(
                    f"edge-tts produced a suspiciously small/empty file "
                    f"({size} bytes) for text: {text[:60]!r}"
                )
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_TTS_ATTEMPTS:
                logger.warning(f"TTS attempt {attempt}/{MAX_TTS_ATTEMPTS} failed ({e}); retrying...")
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            else:
                raise RuntimeError(f"edge-tts failed after {MAX_TTS_ATTEMPTS} attempts: {last_error}") from last_error

    if progress_cb:
        progress_cb("Getting audio duration...", 80)

    try:
        result = subprocess.run(
            [get_ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out_path],
            capture_output=True, check=True, text=True
        )
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get audio duration: {e}") from e

    if duration < 0.05:
        raise RuntimeError(
            f"edge-tts produced a near-zero-duration file ({duration:.3f}s) for text: {text[:60]!r}"
        )

    if progress_cb:
        progress_cb("TTS complete", 100)

    return duration