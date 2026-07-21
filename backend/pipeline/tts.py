"""
TTS Module - edge-tts Implementation

Uses Microsoft Edge's free TTS service (no server required).
edge-tts's built-in SentenceBoundary handling provides natural pacing.
"""

import asyncio
import edge_tts
import subprocess
import os

from typing import Callable, Optional


# Default voice - can be changed
TTS_VOICE = "en-US-ChristopherNeural"

# FFmpeg/ffprobe path (same as compositor.py)
FFMPEG_DIR = r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin"
FFPROBE_PATH = os.path.join(FFMPEG_DIR, "ffprobe.exe")

# Rate at which the TTS speaks (default 1.0, lower = slower, higher = faster)
TTS_RATE = "+10%"  # Slightly faster for energetic delivery


def synthesize_commentary(text: str, out_path: str, progress_cb: Optional[Callable[[str, float], None]] = None) -> float:
    if progress_cb:
        progress_cb("Generating TTS audio...", 10)

    # edge-tts's Communicate() XML-escapes its text input before building SSML,
    # so manually injected <break> tags would be escaped into inert literal text.
    # edge-tts's built-in SentenceBoundary handling already provides natural pacing
    # at sentence punctuation — no manual SSML break injection needed.
    try:
        async def _run_tts():
            communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
            await communicate.save(out_path)

        asyncio.run(_run_tts())
    except Exception as e:
        raise RuntimeError(f"edge-tts failed: {e}") from e

    if progress_cb:
        progress_cb("Getting audio duration...", 80)

    try:
        result = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out_path],
            capture_output=True, check=True, text=True
        )
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get audio duration: {e}") from e

    if progress_cb:
        progress_cb("TTS complete", 100)

    return duration