"""
TTS Module - edge-tts Implementation

Uses Microsoft Edge's free TTS service (no server required).
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


def synthesize_commentary(text: str, out_path: str, progress_cb: Optional[Callable[[str, float], None]] = None) -> float:
    if progress_cb:
        progress_cb("Generating TTS audio...", 10)

    try:
        # edge-tts is async, so we need to run it in a new event loop
        async def _run_tts():
            communicate = edge_tts.Communicate(text, TTS_VOICE)
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