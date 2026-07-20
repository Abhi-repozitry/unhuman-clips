import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import os
import re
import shutil


def _get_ffmpeg_path() -> str:
    candidates = [
        "C:\\Users\\starr\\.vscode\\extensions\\kilocode.kilo-code-7.4.11-win32-x64\\bin\\ffmpeg.exe",
        "C:\\Users\\starr\\.vscode\\extensions\\kilocode.kilo-code-7.4.9-win32-x64\\bin\\ffmpeg.exe",
        "C:\\Users\\starr\\.vscode\\extensions\\.8abe98e8-0d96-44cc-a28c-f78f911caf54\\bin\\ffmpeg.exe",
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError("ffmpeg not found. Please install ffmpeg and add it to PATH.")


def _get_ffprobe_path() -> Optional[str]:
    path = shutil.which("ffprobe")
    if path:
        return path
    candidates = [
        "C:\\ffmpeg\\bin\\ffprobe.exe",
        "C:\\Program Files\\ffmpeg\\bin\\ffprobe.exe",
        "C:\\Program Files (x86)\\ffmpeg\\bin\\ffprobe.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        ffprobe_path = _get_ffprobe_path()
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
            [_get_ffmpeg_path(), "-i", str(video_path)],
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
    min_silence_duration: float = 0.3,  # Reduced from 0.5 to catch more silence
    window_size: float = 0.1
) -> List[Dict[str, float]]:
    """Detect silent segments using Silero VAD.

    Returns list of {start, end, duration} for non-speech (silent) sections.
    More aggressive: catches shorter silences (0.3s+) for tighter pacing.
    """
    try:
        import torch
        import torchaudio
        from silero_vad import get_speech_timestamps, read_audio, VADIterator
    except ImportError:
        print("[WARN] Silero VAD not available, falling back to ffmpeg silencedetect")
        return detect_silence_ffmpeg(video_path, -35.0, min_silence_duration)

    try:
        wav, sr = read_audio(video_path, sampling_rate=16000)

        speech_timestamps = get_speech_timestamps(
            wav,
            sr,
            threshold=threshold,
            min_silence_duration_ms=int(min_silence_duration * 1000),
            window_size_samples=int(window_size * sr),
            return_seconds=True
        )

        if not speech_timestamps:
            return []

        silence_segments = []
        prev_end = 0.0

        for speech_start, speech_end in speech_timestamps:
            if speech_start > prev_end:
                silence_duration = speech_start - prev_end
                if silence_duration >= min_silence_duration:
                    silence_segments.append({
                        "start": prev_end,
                        "end": speech_start,
                        "duration": silence_duration
                    })
            prev_end = speech_end

        total_duration = len(wav) / sr
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
        print(f"[WARN] Silero VAD detection failed: {e}, falling back to ffmpeg")
        return detect_silence_ffmpeg(video_path, -35.0, min_silence_duration)


def detect_silence_ffmpeg(
    video_path: str,
    silence_threshold_db: float = -35.0,  # More sensitive (was -40)
    silence_duration: float = 0.3  # Shorter minimum (was 0.5)
) -> List[Dict[str, float]]:
    """
    Detect silent segments using ffmpeg silencedetect filter.
    More aggressive: lower threshold and shorter duration to catch more pauses.
    """
    import re

    cmd = [
        _get_ffmpeg_path(), "-loglevel", "error",
        "-i", str(video_path),
        "-af", f"silencedetect=n={silence_threshold_db}dB:d={silence_duration}",
        "-f", "null", "-",
        "-y"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        stderr = result.stderr
        silence_pattern = re.compile(
            r"silence_start: (\d+\.?\d*) \| silence_end: (\d+\.?\d*) \| silence_duration: (\d+\.?\d*)"
        )

        silence_segments = []
        for match in silence_pattern.finditer(stderr):
            start = float(match.group(1))
            end = float(match.group(2))
            duration = float(match.group(3))
            silence_segments.append({"start": start, "end": end, "duration": duration})

        return silence_segments

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []


def detect_silence(
    video_path: str,
    silence_threshold_db: float = -35.0,
    silence_duration: float = 0.3
) -> List[Dict[str, float]]:
    """Detect silent segments using Silero VAD (preferred) or ffmpeg fallback.
    More aggressive defaults for tighter pacing."""
    return detect_silence_with_vad(video_path, min_silence_duration=silence_duration)


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
        cmd_copy = [_get_ffmpeg_path(), "-loglevel", "error", "-i", str(input_path), "-c", "copy", "-y", str(output_path)]
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
        cmd_copy = [_get_ffmpeg_path(), "-loglevel", "error", "-i", str(input_path), "-c", "copy", "-y", str(output_path)]
        try:
            subprocess.run(cmd_copy, capture_output=True, check=True, timeout=120)
            return True, 0.0
        except subprocess.CalledProcessError:
            return False, 0.0

    # Apply trim
    cmd_trim = [
        _get_ffmpeg_path(), "-loglevel", "error",
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
            "enable_speed_adjust": True,     # New: subtle speed up of slow parts
            "speed_target": 1.05,            # 5% speed up for energy (very subtle)
            "max_speed_ratio": 1.10,         # Max 10% speed up on slowest segments
        }

    working_path = Path(working_dir)
    working_path.mkdir(parents=True, exist_ok=True)

    input_path_obj = Path(input_path)
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

    # Step 2: Subtle speed adjustment for energy
    if config.get("enable_speed_adjust", True) and total_time_saved < 5.0:
        current_duration = get_video_duration(input_path)
        if current_duration > 30.0:
            # Apply very subtle speed up (1.05x) to give the reel more energy
            speed_ratio = config.get("speed_target", 1.05)
            speed_path = working_path / f"temp_speed_{input_path_obj.name}"

            cmd_speed = [
                _get_ffmpeg_path(), "-loglevel", "error",
                "-i", str(input_path),
                "-filter_complex",
                f"[0:v]setpts={1.0/speed_ratio:.4f}*PTS[v];[0:a]atempo={speed_ratio:.4f}[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                "-y", str(speed_path)
            ]
            try:
                subprocess.run(cmd_speed, capture_output=True, check=True, timeout=120)
                new_duration = get_video_duration(str(speed_path))
                time_saved_speed = current_duration - new_duration
                edits_applied.append({
                    "type": "speed_adjust",
                    "ratio": speed_ratio,
                    "time_saved": round(time_saved_speed, 2)
                })
                total_time_saved += time_saved_speed
                input_path = str(speed_path)
                print(f"[INFO] Speed adjusted {speed_ratio}x: {current_duration:.1f}s -> {new_duration:.1f}s")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Speed adjustment failed: {e.stderr.decode() if e.stderr else 'unknown'}")

    # Step 3: Final copy
    final_cmd = [_get_ffmpeg_path(), "-loglevel", "error", "-i", str(input_path), "-c", "copy", "-y", str(output_path)]
    try:
        subprocess.run(final_cmd, capture_output=True, check=True, timeout=60)
    except subprocess.CalledProcessError:
        # Fallback: just copy original
        output_path = Path(input_path)

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