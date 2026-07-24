import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import os
import re
import shutil
import tempfile
import logging

from backend.ffmpeg_utils import get_ffmpeg, get_ffprobe

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        ffprobe_path = get_ffprobe()
        if ffprobe_path:
            cmd = [
                ffprobe_path, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return float(result.stdout.strip())

        result = subprocess.run(
            [get_ffmpeg(), "-i", str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return 0.0
    except Exception:
        return 0.0


def detect_silence_with_vad(
    video_path: str,
    threshold: float = 0.5,
    min_silence_duration: float = 0.3,
    window_size: float = 0.1
) -> List[Dict[str, float]]:
    """Detect silent segments using Silero VAD.

    Returns list of {start, end, duration} for non-speech (silent) sections.
    More aggressive: catches shorter silences (0.3s+) for tighter pacing.
    """
    try:
        import torch
        import soundfile as sf
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError:
        print("[WARN] Silero VAD not available, falling back to ffmpeg silencedetect")
        return detect_silence_ffmpeg(video_path, -35.0, min_silence_duration)

    # Extract audio to temp WAV, then load with soundfile.
    # silero_vad.read_audio() wraps torchaudio.load() which fails on
    # torchaudio >=2.9 without torchcodec. FFmpeg already produces a valid
    # 16kHz mono WAV, so soundfile is the most reliable loader.
    sampling_rate = 16000
    tmp_wav = None
    try:
        tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="editor_vad_")
        os.close(tmp_fd)
        ffmpeg = get_ffmpeg()
        subprocess.run([
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", str(sampling_rate), "-ac", "1",
            tmp_wav
        ], capture_output=True, text=True, timeout=60)
        if not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
            return detect_silence_ffmpeg(video_path, -35.0, min_silence_duration)

        # Load WAV directly with soundfile — no torchaudio dependency
        wav_np, _ = sf.read(tmp_wav, dtype='float32')
        wav = torch.from_numpy(wav_np)

        # Load the Silero VAD model (required in silero_vad v6+)
        model = load_silero_vad()

        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=threshold,
            min_silence_duration_ms=int(min_silence_duration * 1000),
            window_size_samples=int(window_size * sampling_rate),
            return_seconds=True
        )

        if not speech_timestamps:
            return []

        silence_segments = []
        prev_end = 0.0

        for ts in speech_timestamps:
            speech_start = ts["start"]
            speech_end = ts["end"]
            if speech_start > prev_end:
                silence_duration = speech_start - prev_end
                if silence_duration >= min_silence_duration:
                    silence_segments.append({
                        "start": prev_end,
                        "end": speech_start,
                        "duration": silence_duration
                    })
            prev_end = speech_end

        total_duration = len(wav) / sampling_rate
        if total_duration > prev_end:
            silence_duration = total_duration - prev_end
            if silence_duration >= min_silence_duration:
                silence_segments.append({
                    "start": prev_end,
                    "end": total_duration,
                    "duration": silence_duration
                })

        return silence_segments

    except Exception as e:
        logger.warning(f"Silero VAD detection failed: {type(e).__name__}: {e}, falling back to ffmpeg")
        return detect_silence_ffmpeg(video_path, -35.0, min_silence_duration)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass


def detect_silence_ffmpeg(
    video_path: str,
    silence_threshold_db: float = -35.0,
    silence_duration: float = 0.3
) -> List[Dict[str, float]]:
    """
    Detect silent segments using ffmpeg silencedetect filter.
    Parses stateful start/end lines from FFmpeg output.
    """
    import re

    cmd = [
        get_ffmpeg(), "-loglevel", "info",
        "-y",
        "-i", str(video_path),
        "-af", f"silencedetect=n={silence_threshold_db}dB:d={silence_duration}",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        stderr = result.stderr
        silence_start_re = re.compile(r"silence_start:\s*(\d+\.?\d*)")
        silence_end_re = re.compile(r"silence_end:\s*(\d+\.?\d*)\s*\|\s*silence_duration:\s*(\d+\.?\d*)")

        silence_segments = []
        current_start = None

        for line in stderr.splitlines():
            start_match = silence_start_re.search(line)
            if start_match:
                current_start = float(start_match.group(1))
                continue

            end_match = silence_end_re.search(line)
            if end_match and current_start is not None:
                end = float(end_match.group(1))
                dur = float(end_match.group(2))
                silence_segments.append({"start": current_start, "end": end, "duration": dur})
                current_start = None

        return silence_segments

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []


def detect_silence(
    video_path: str,
    silence_threshold_db: float = -35.0,
    silence_duration: float = 0.3
) -> List[Dict[str, float]]:
    """Detect silent segments using Silero VAD (preferred) or ffmpeg fallback.
    Converts dB threshold to VAD probability threshold (0.0-1.0)."""
    # Convert dB to VAD threshold: -35dB ≈ 0.5, -20dB ≈ 0.8, -50dB ≈ 0.2
    vad_threshold = max(0.1, min(0.9, 10 ** (silence_threshold_db / 20.0)))
    return detect_silence_with_vad(video_path, threshold=vad_threshold, min_silence_duration=silence_duration)


def trim_silence(
    input_path: str,
    output_path: str,
    silence_segments: List[Dict[str, float]],
    min_audio_gap: float = 0.2  # Reduced from 0.3 to trim more aggressively
) -> Tuple[bool, float]:
    """
    Trim silence from video. Returns (success, time_saved_seconds).
    Strategy: remove silent segments shorter than min_audio_gap at clip boundaries.
    For simplicity, we trim leading/trailing silence only.
    """
    duration = get_video_duration(input_path)
    if duration <= 0:
        return False, 0.0

    if not silence_segments:
        # No silence detected, just copy
        cmd_copy = [get_ffmpeg(), "-loglevel", "error", "-i", str(input_path), "-c", "copy", "-y", str(output_path)]
        try:
            subprocess.run(cmd_copy, capture_output=True, check=True, timeout=120)
            return True, 0.0
        except subprocess.CalledProcessError:
            return False, 0.0

    # Sort by start time
    silence_segments.sort(key=lambda s: s["start"])

    # Find leading silence (starts at 0)
    leading_silence = None
    for seg in silence_segments:
        if seg["start"] <= 0.1:
            leading_silence = seg
            break

    # Find trailing silence (ends near video end)
    trailing_silence = None
    for seg in silence_segments:
        if seg["end"] >= duration - 0.1:
            trailing_silence = seg
            break

    # Calculate trim points
    start_trim = leading_silence["end"] if leading_silence else 0.0
    end_trim = trailing_silence["start"] if trailing_silence else duration

    time_saved = (leading_silence["duration"] if leading_silence else 0) + \
                 (trailing_silence["duration"] if trailing_silence else 0)

    if time_saved <= min_audio_gap:
        # Not worth trimming
        cmd_copy = [get_ffmpeg(), "-loglevel", "error", "-i", str(input_path), "-c", "copy", "-y", str(output_path)]
        try:
            subprocess.run(cmd_copy, capture_output=True, check=True, timeout=120)
            return True, 0.0
        except subprocess.CalledProcessError:
            return False, 0.0

    # Apply trim
    cmd_trim = [
        get_ffmpeg(), "-loglevel", "error",
        "-i", str(input_path),
        "-ss", str(start_trim),
        "-to", str(end_trim),
        "-c", "copy",
        "-y", str(output_path)
    ]
    try:
        subprocess.run(cmd_trim, capture_output=True, check=True, timeout=120)
        return True, time_saved
    except subprocess.CalledProcessError:
        return False, 0.0


def apply_edits(
    input_path: str,
    working_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply editing operations to a video.
    Returns dict with output_path, edits_applied, time_saved_seconds, durations.
    Enhanced with more aggressive silence trimming and subtle speed adjustment.
    """
    if config is None:
        config = {
            "silence_threshold_db": -35.0,  # More sensitive
            "silence_duration": 0.3,         # Catch shorter pauses
            "min_audio_gap": 0.2,            # Trim more aggressively
            "enable_silence_trim": True,
            "enable_speed_adjust": False,    # Disabled: natural pacing at 1.0x
            "speed_target": 1.0,             # No speed adjustment
            "max_speed_ratio": 1.0,
        }

    working_path = Path(working_dir)
    working_path.mkdir(parents=True, exist_ok=True)

    input_path_obj = Path(input_path)
    original_input_path = input_path  # Save before trim modifies it
    output_path = working_path / f"edited_{input_path_obj.name}"
    temp_path = working_path / f"temp_trim_{input_path_obj.name}"

    original_duration = get_video_duration(input_path)
    edits_applied = []
    total_time_saved = 0.0

    # Step 1: Silence detection and trim
    if config.get("enable_silence_trim", True):
        silence_segments = detect_silence(
            input_path,
            config.get("silence_threshold_db", -35.0),
            config.get("silence_duration", 0.3)
        )

        if silence_segments:
            success, time_saved = trim_silence(
                input_path,
                str(temp_path),
                silence_segments,
                config.get("min_audio_gap", 0.2)
            )
            if success and time_saved > 0:
                edits_applied.append({
                    "type": "silence_trim",
                    "segments_removed": len([s for s in silence_segments if s["duration"] >= config.get("min_audio_gap", 0.2)]),
                    "time_saved": time_saved
                })
                total_time_saved += time_saved
                # Use trimmed output as new input for next steps
                input_path = str(temp_path)

    # Step 2: Final copy (no speed adjustment — natural pacing at 1.0x)
    final_cmd = [get_ffmpeg(), "-loglevel", "error", "-i", str(input_path), "-c", "copy", "-y", str(output_path)]
    try:
        subprocess.run(final_cmd, capture_output=True, check=True, timeout=60)
    except subprocess.CalledProcessError:
        # Fallback: just copy original untrimmed file
        output_path = Path(original_input_path)

    final_duration = get_video_duration(str(output_path))

    # Cleanup temp
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass

    return {
        "output_path": str(output_path),
        "edits_applied": edits_applied,
        "time_saved_seconds": round(total_time_saved, 2),
        "original_duration": round(original_duration, 2),
        "final_duration": round(final_duration, 2)
    }


def edit_final_video(video_path: str, job_id: str) -> Dict[str, Any]:
    """Edit the final composed video: trim silence, subtle speed up, finalize.

    This is the entry point called by queue_manager.py.
    Delegates to apply_edits with enhanced default config.
    """
    working_dir = Path(video_path).parent
    return apply_edits(video_path, str(working_dir))