"""
TTS Module - edge-tts Implementation

Uses Microsoft Edge's free TTS service (no server required).
Enhanced with SSML for natural pauses between sentences.
"""

import asyncio
import edge_tts
import subprocess
import os
import re
from typing import Callable, Optional


# Default voice - can be changed
TTS_VOICE = "en-US-ChristopherNeural"

# FFmpeg/ffprobe path (same as compositor.py)
FFMPEG_DIR = r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin"
FFPROBE_PATH = os.path.join(FFMPEG_DIR, "ffprobe.exe")

# Rate at which the TTS speaks (default 1.0, lower = slower, higher = faster)
TTS_RATE = "+10%"  # Slightly faster for energetic delivery


def _add_natural_pauses(text: str) -> str:
    """
    Add SSML timing tags to create natural rhythm.
    - Pause (200ms) after each sentence-ending period/question/exclamation.
    - Pause (100ms) after commas for natural breath.
    - Keeps overall delivery smooth and varied.
    """
    if not text:
        return text

    # Convert to SSML with breaks for natural speech rhythm
    # Break after sentences: . ! ?
    text = re.sub(r'\.(?!\d)\s*', '. <break time="280ms"/> ', text)
    text = re.sub(r'\?\s*', '? <break time="300ms"/> ', text)
    text = re.sub(r'!\s*', '! <break time="300ms"/> ', text)
    text = re.sub(r',\s*', ', <break time="120ms"/> ', text)
    text = re.sub(r';\s*', '; <break time="150ms"/> ', text)

    # Clean up double spaces and trailing breaks
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'<break[^>]*/>\s*<break[^>]*/>', '<break time="300ms"/>', text)

    return text


def synthesize_commentary(text: str, out_path: str, progress_cb: Optional[Callable[[str, float], None]] = None) -> float:
    if progress_cb:
        progress_cb("Generating TTS audio...", 10)

    try:
        # Add natural pauses for better rhythm
        ssml_text = _add_natural_pauses(text)

        # edge-tts is async, so we need to run it in a new event loop
        async def _run_tts():
            communicate = edge_tts.Communicate(ssml_text, TTS_VOICE, rate=TTS_RATE)
            await communicate.save(out_path)

        asyncio.run(_run_tts())
    except Exception as e:
        # If SSML fails, fall back to plain text
        print(f"[WARN] edge-tts SSML failed: {e}, retrying with plain text")
        try:
            async def _run_tts_plain():
                communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
                await communicate.save(out_path)
            asyncio.run(_run_tts_plain())
        except Exception as e2:
            raise RuntimeError(f"edge-tts failed: {e2}") from e2

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