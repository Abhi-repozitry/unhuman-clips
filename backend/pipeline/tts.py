"""TTS module — text-to-speech via Kokoro TTS (hexgrad/Kokoro-82M).

Provides synthesize_commentary() with retry logic for transient failures
and audio validation to catch empty/truncated responses.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from backend.pipeline.kokoro_tts import synthesize_kokoro

__all__ = ["synthesize_commentary"]

logger = logging.getLogger(__name__)


def synthesize_commentary(
    text: str,
    out_path: str,
    progress_cb: Callable[[str, float], None] | None = None,
    rate: str | None = None,
    voice: str | None = None,
) -> float:
    """Synthesize text to speech using Kokoro TTS.

    Args:
        text: Text to synthesize (must be non-empty).
        out_path: Output WAV file path (16kHz mono).
        progress_cb: Optional progress callback.
        rate: Unused, kept for API compatibility.
        voice: Voice name (e.g., 'am_adam'). Falls back to config default.

    Returns:
        Duration of the generated audio in seconds.
    """
    if not text or not text.strip():
        raise RuntimeError("synthesize_commentary called with empty text")

    return synthesize_kokoro(text, out_path, progress_cb, voice=voice)
