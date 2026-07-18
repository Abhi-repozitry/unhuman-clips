import subprocess
import os
from pathlib import Path
from backend.config import HOOK_SECONDS, OUTPUTS_DIR, get_job_working_dir
from typing import Callable, Optional, List, Dict, Any

FFMPEG_PATH = r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"


def _get_video_encoder(fallback=False):
    """Return the best available H.264 encoder.
    Prefer h264_nvenc (GPU), fall back to libx264 (CPU) if NVENC fails or is forced."""
    if not fallback:
        try:
            result = subprocess.run(
                [FFMPEG_PATH, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10
            )
            if "h264_nvenc" in result.stdout:
                return "h264_nvenc"
        except Exception:
            pass
    if os.environ.get("ALLOW_CPU_FFMPEG_FALLBACK") == "1" or fallback:
        return "libx264"
    raise RuntimeError("h264_nvenc encoder is not available. Install/configure NVIDIA ffmpeg support or set ALLOW_CPU_FFMPEG_FALLBACK=1.")


def _build_encoder_opts(encoder: str) -> list:
    """Return encoder-specific ffmpeg arguments."""
    if encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
            "-preset", "p7", "-rc", "vbr", "-cq", "23",
        ]
    else:
        return [
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "23",
        ]


def _run_ffmpeg(cmd: list, description: str, attempt: int = 1, max_attempts: int = 2):
    """Run ffmpeg, retrying with CPU encoder if NVENC fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        if attempt < max_attempts and "h264_nvenc" in " ".join(cmd) and os.environ.get("ALLOW_CPU_FFMPEG_FALLBACK") == "1":
            stderr = e.stderr.decode() if e.stderr else ""
            print(f"[WARN] NVENC failed for {description}, retrying with libx264: {stderr[:200]}")
            new_cmd = []
            skip_next = False
            for j, arg in enumerate(cmd):
                if skip_next:
                    skip_next = False
                    continue
                if arg == "h264_nvenc":
                    new_cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
                    skip_next = True
                elif arg in ("-preset", "-rc", "-cq"):
                    skip_next = True
                elif arg == "p7":
                    pass
                elif arg == "vbr":
                    pass
                elif arg == "23" and j > 0 and cmd[j-1] == "-cq":
                    pass
                else:
                    new_cmd.append(arg)
            return _run_ffmpeg(new_cmd, description, attempt + 1, max_attempts)
        raise RuntimeError(f"{description} failed: {e.stderr.decode() if e.stderr else 'unknown'}") from e


def _ass_filter(path: str) -> str:
    escaped = str(path).replace("\\", "/").replace("'", r"\'")
    return f"ass='{escaped}'"


def _build_ducking_expression(narration_events: List[Dict[str, Any]]) -> str:
    """
    Build ffmpeg volume filter expression with smooth ducking ramps.
    For each narration event [s, e]:
      - Ramp-in: s to s+0.12  (volume 1.0 -> 0.125)
      - Flat:    s+0.12 to e-0.18  (volume 0.125)
      - Ramp-out: e-0.18 to e  (volume 0.125 -> 1.0)
    Overall duck = max(duck_per_event) across all events.
    Volume = 1.0 - 0.875 * max_duck
    """
    if not narration_events:
        return "1.0"

    event_exprs = []
    for ev in narration_events:
        s = ev["reel_start"]
        e = ev["reel_end"]
        # Skip if event is too short for ramps
        if e - s < 0.3:
            continue
        # duck_event = min(1, max(0, (t-s)/0.12)) * (1 - min(1, max(0, (t-(e-0.18))/0.18)))
        ramp_in = f"min(1,max(0,(t-{s})/0.12))"
        ramp_out = f"min(1,max(0,(t-({e}-0.18))/0.18))"
        duck_expr = f"({ramp_in})*(1-{ramp_out})"
        event_exprs.append(duck_expr)

    if not event_exprs:
        return "1.0"

    if len(event_exprs) == 1:
        max_duck = event_exprs[0]
    else:
        max_duck = "max(" + ",".join(event_exprs) + ")"

    return f"1.0-0.875*{max_duck}"


def compose_group(
    job_id: str,
    group_idx: int,
    group_clip_paths: List[str],
    source_clips: List[Dict[str, float]],
    narration_audio: List[Dict[str, Any]],
    clip_caption_paths: List[str],
    narration_caption_paths: List[str],
    source_path: str,
    working_dir: Path,
    progress_cb: Optional[Callable[[str, float], None]] = None
) -> str:
    """
    Build a single group's output reel.
    - Continuous video from concatenated source_clips (trimmed from source video)
    - Clip captions at BOTTOM (alignment=2, margin_v=80)
    - Narration captions at TOP (alignment=8, margin_v=60)
    - Audio: clip audio ducked during narration windows + narration audio mixed at full volume
    """
    from backend.config import MAX_OUTPUT_DURATION
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    encoder = _get_video_encoder()
    encoder_opts = _build_encoder_opts(encoder)
    print(f"[INFO] compose_group {group_idx}: Using video encoder: {encoder}")

    # 1. Build continuous video from source_clips (trim from source video)
    n_clips = len(source_clips)
    if n_clips == 0:
        raise RuntimeError("No source clips in group")

    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Building continuous video from {n_clips} clips...", 5)

    # Build filter_complex for video: trim each clip, concat, then overlay captions
    video_filter_parts = []
    for i, clip in enumerate(source_clips):
        start = clip["source_start"]
        end = clip["source_end"]
        duration = end - start
        video_filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]"
        )

    concat_inputs = "".join(f"[v{i}]" for i in range(n_clips))
    video_filter_parts.append(f"{concat_inputs}concat=n={n_clips}:v=1:a=0[v]")

    video_filter = ";".join(video_filter_parts)

    # Add clip captions (bottom)
    clip_caption_filters = []
    for i, cap_path in enumerate(clip_caption_paths):
        clip_caption_filters.append(f"[v]{_ass_filter(cap_path)}[v{i+1}]")
    video_filter += ";" + ";".join(clip_caption_filters)
    last_v = f"v{len(clip_caption_filters)}"

    # Add narration captions (top) - overlay on top of clip captions
    for i, cap_path in enumerate(narration_caption_paths):
        video_filter += f";[{last_v}]{_ass_filter(cap_path)}[v{i+len(clip_caption_filters)+1}]"
        last_v = f"v{i+len(clip_caption_filters)+1}"

    video_output = working_dir / f"group_{group_idx}_video.mp4"
    ffmpeg_video = [
        FFMPEG_PATH, "-loglevel", "error",
        "-i", source_path,
        "-filter_complex", video_filter,
        "-map", f"[{last_v}]",
    ] + encoder_opts + [
        "-r", "30", "-y", str(video_output)
    ]
    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Rendering video with captions...", 25)
    _run_ffmpeg(ffmpeg_video, f"Group {group_idx} video")

    # 2. Build continuous clip audio from source_clips
    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Building continuous clip audio...", 40)

    audio_filter_parts = []
    for i, clip in enumerate(source_clips):
        start = clip["source_start"]
        end = clip["source_end"]
        audio_filter_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]"
        )

    concat_audio_inputs = "".join(f"[a{i}]" for i in range(n_clips))
    audio_filter_parts.append(f"{concat_audio_inputs}concat=n={n_clips}:v=0:a=1[clip_audio]")

    audio_filter = ";".join(audio_filter_parts)

    clip_audio_output = working_dir / f"group_{group_idx}_clip_audio.wav"
    ffmpeg_clip_audio = [
        FFMPEG_PATH, "-loglevel", "error",
        "-i", source_path,
        "-filter_complex", audio_filter,
        "-map", "[clip_audio]",
        "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
        "-y", str(clip_audio_output)
    ]
    _run_ffmpeg(ffmpeg_clip_audio, f"Group {group_idx} clip audio")

    # 3. Build narration audio track (narration events positioned at reel_start)
    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Building narration audio track...", 55)

    narration_filter_parts = []
    for i, nar in enumerate(narration_audio):
        path = nar["path"]
        reel_start = nar["reel_start"]
        # Use adelay to position narration at correct time
        delay_ms = int(reel_start * 1000)
        narration_filter_parts.append(
            f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[nar{i}]"
        )

    narration_inputs = "".join(f"[nar{i}]" for i in range(len(narration_audio)))
    narration_filter_parts.append(f"{narration_inputs}amix=inputs={len(narration_audio)}:duration=first:dropout_transition=0.1[narration_mix]")

    narration_audio_output = working_dir / f"group_{group_idx}_narration.wav"
    ffmpeg_narration = [
        FFMPEG_PATH, "-loglevel", "error",
        "-i", clip_audio_output,  # dummy input for filter chain
    ]
    # Add narration audio files as inputs
    for nar in narration_audio:
        ffmpeg_narration.extend(["-i", nar["path"]])
    ffmpeg_narration.extend([
        "-filter_complex", ";".join(narration_filter_parts),
        "-map", "[narration_mix]",
        "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
        "-y", str(narration_audio_output)
    ])
    _run_ffmpeg(ffmpeg_narration, f"Group {group_idx} narration audio")

    # 4. Apply ducking to clip audio and mix with narration
    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Applying audio ducking and mixing...", 70)

    # Build ducking expression from narration events
    duck_expr = _build_ducking_expression(narration_audio)

    mixed_audio_output = working_dir / f"group_{group_idx}_mixed_audio.wav"
    ffmpeg_mix = [
        FFMPEG_PATH, "-loglevel", "error",
        "-i", str(clip_audio_output),
        "-i", str(narration_audio_output),
        "-filter_complex",
        f"[0:a]volume={duck_expr}:eval=frame[ducked];[ducked][1:a]amix=inputs=2:duration=first:dropout_transition=0.1[mixed]",
        "-map", "[mixed]",
        "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
        "-y", str(mixed_audio_output)
    ]
    _run_ffmpeg(ffmpeg_mix, f"Group {group_idx} audio mix")

    # 5. Final mux: video + mixed audio
    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Final mux...", 85)

    output_path = OUTPUTS_DIR / f"{job_id}_group{group_idx}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_final = [
        FFMPEG_PATH, "-loglevel", "error",
        "-i", str(video_output),
        "-i", str(mixed_audio_output),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-shortest",
        "-y", str(output_path)
    ]
    try:
        subprocess.run(ffmpeg_final, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Final mux failed: {e.stderr.decode() if e.stderr else 'unknown'}") from e

    # Check duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
        capture_output=True, text=True
    )
    if probe.returncode == 0:
        try:
            actual_duration = float(probe.stdout.strip())
            if actual_duration > 90:
                print(f"[WARN] Group {group_idx} output {actual_duration:.1f}s exceeds 90s cap")
        except ValueError:
            pass

    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Complete!", 100)

    return str(output_path)


# Legacy function kept for backward compatibility (not used in new pipeline)
def build_final_video(
    job_id: str,
    clip_paths: list,
    clip_windows: list,
    commentary_audio: list,
    caption_paths_commentary: list,
    caption_paths_clips: list,
    progress_cb: Optional[Callable[[str, float], None]] = None
) -> str:
    raise RuntimeError("build_final_video is deprecated. Use compose_group() per group.")