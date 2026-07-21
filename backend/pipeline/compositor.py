import subprocess
import os
from pathlib import Path
from backend.config import HOOK_SECONDS, OUTPUTS_DIR, get_job_working_dir, FFMPEG_PATH, FFPROBE_PATH
from typing import Callable, Optional, List, Dict, Any


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


def _run_ffmpeg(cmd: list, description: str, attempt: int = 1, max_attempts: int = 2, cwd: str = None):
    """Run ffmpeg, retrying with CPU encoder if NVENC fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, cwd=cwd)
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
            return _run_ffmpeg(new_cmd, description, attempt + 1, max_attempts, cwd=cwd)
        raise RuntimeError(f"{description} failed: {e.stderr.decode() if e.stderr else 'unknown'}") from e


def _ass_filter(path: str) -> str:
    filename = Path(path).name
    return f"ass=filename={filename}"


def _build_ducking_filter_chain(narration_events, input_label="0:a", output_label="ducked"):
    """
    Build a robust single-filter ducking expression.
    Ducks audio to ~0.03 during narration events — original audio is near-silent, avoiding voice overlap.
    Uses pre-duck buffer (0.3s before) and post-duck buffer (0.2s after) to prevent bleed.
    Uses fast ramps (0.08s) for snappy transitions.
    """
    valid_events = [ev for ev in narration_events if ev.get("reel_end", 0) - ev.get("reel_start", 0) >= 0.3]
    if not valid_events:
        return f"[{input_label}]anull[{output_label}]"

    print(f"[INFO] Audio ducking: {len(valid_events)} narration windows to duck")
    duck_terms = []
    for i, ev in enumerate(valid_events):
        # Apply pre-duck buffer (0.3s before) and post-duck buffer (0.2s after)
        s = max(0.0, ev["reel_start"] - 0.3)
        e = ev["reel_end"] + 0.2
        print(f"[INFO]   Duck window {i+1}: [{s:.3f}s - {e:.3f}s] "
              f"(narration [{ev['reel_start']:.3f}s - {ev['reel_end']:.3f}s], "
              f"pre-buf=0.3s, post-buf=0.2s)")
        # Fast 0.08s ramps for snappy ducking transitions
        ramp_in = f"min(1,max(0,(t-{s:.3f})/0.08))"
        ramp_out = f"min(1,max(0,(t-({e:.3f}-0.08))/0.08))"
        duck_terms.append(f"if(between(t,{s:.3f},{e:.3f}),({ramp_in})*(1-({ramp_out})),0)")

    if len(duck_terms) == 1:
        duck_expr = duck_terms[0]
    else:
        duck_expr = f"min(1.0,{'+'.join(duck_terms)})"

    # Ducking factor 0.97 -> original audio drops to 0.03 (3%) during narration
    vol_expr = f"1.0-({duck_expr}*0.97)"
    print(f"[INFO] Audio ducking depth: 0.97 (original audio -> 3% volume during narration)")
    return f"[{input_label}]volume='{vol_expr}':eval=frame[{output_label}]"


def _get_video_duration_seconds(video_path: str) -> float:
    """Get duration of a video file using ffprobe."""
    try:
        result = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=15
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, OSError):
        return 0.0


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
    estimated_duration_seconds: float = 0.0,
    progress_cb: Optional[Callable[[str, float], None]] = None
) -> str:
    """
    Build a single group's output reel.
    - Continuous video from pre-cut group_clip_paths (if available) or trimmed from source_clips
    - Video is padded with last-frame freeze to fill target_duration
    - Clip captions at BOTTOM (alignment=2, margin_v=80)
    - Narration captions at TOP (alignment=8, margin_v=60)
    - Audio: clip audio padded to target_duration & ducked during narration windows + narration audio mixed
    """
    from backend.config import MAX_OUTPUT_DURATION, MIN_OUTPUT_DURATION
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    encoder = _get_video_encoder()
    encoder_opts = _build_encoder_opts(encoder)
    print(f"[INFO] compose_group {group_idx}: Using video encoder: {encoder}")

    n_clips = len(source_clips)
    if n_clips == 0:
        raise RuntimeError("No source clips in group")

    # Check whether to use pre-cut clip files from CLIPPING stage
    use_precut = bool(
        group_clip_paths
        and len(group_clip_paths) == n_clips
        and all(os.path.exists(p) for p in group_clip_paths)
    )
    if use_precut:
        print(f"[INFO] Group {group_idx}: Using {n_clips} pre-cut clip files from CLIPPING stage.")
    else:
        print(f"[INFO] Group {group_idx}: Pre-cut clips unavailable or incomplete; trimming from source video.")

    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Building continuous video from {n_clips} clips...", 5)

    total_clip_duration = sum(clip["source_end"] - clip["source_start"] for clip in source_clips)
    
    max_narration_end = 0.0
    if narration_audio:
        max_narration_end = max(nar.get("reel_end", 0) for nar in narration_audio)
    
    target_duration = max(total_clip_duration, max_narration_end, estimated_duration_seconds, float(MIN_OUTPUT_DURATION))
    target_duration = min(target_duration, float(MAX_OUTPUT_DURATION))
    pad_duration = target_duration - total_clip_duration

    # Cap the tpad freeze: a freeze longer than 3 s is the video-freeze bug.
    # This happens when the analyzer selects far fewer clips than MIN_OUTPUT_DURATION
    # or estimated_duration_seconds requires.  We allow a tiny 3-second grace freeze
    # (e.g. for a last-scene hold) but refuse to freeze for longer.
    MAX_FREEZE_PAD = 3.0
    if pad_duration > MAX_FREEZE_PAD:
        print(f"[WARN] Group {group_idx}: clip content ({total_clip_duration:.1f}s) far short of "
              f"target ({target_duration:.1f}s) — capping freeze pad to {MAX_FREEZE_PAD:.1f}s "
              f"(was {pad_duration:.1f}s).  Audio will be padded with silence instead.")
        pad_duration = MAX_FREEZE_PAD
        # Recompute target_duration so audio padding (apad=whole_dur=...) uses the same
        # corrected value, not the original inflated target.
        target_duration = total_clip_duration + pad_duration

    print(f"[INFO] Group {group_idx}: total_clip_duration={total_clip_duration:.1f}s, "
          f"max_narration_end={max_narration_end:.1f}s, est={estimated_duration_seconds:.1f}s, target={target_duration:.1f}s, pad={pad_duration:.1f}s")

    # 1. Build video filter complex
    if use_precut:
        ffmpeg_video_inputs = []
        for p in group_clip_paths:
            ffmpeg_video_inputs.extend(["-i", str(p)])
        concat_inputs = "".join(f"[{i}:v]" for i in range(n_clips))
        video_filter_parts = [f"{concat_inputs}concat=n={n_clips}:v=1:a=0[base_v]"]
    else:
        ffmpeg_video_inputs = ["-i", source_path]
        video_filter_parts = []
        for i, clip in enumerate(source_clips):
            start = clip["source_start"]
            end = clip["source_end"]
            video_filter_parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]"
            )
        concat_inputs = "".join(f"[v{i}]" for i in range(n_clips))
        video_filter_parts.append(f"{concat_inputs}concat=n={n_clips}:v=1:a=0[base_v]")

    # Freeze last frame to pad video to target duration
    if pad_duration > 0.5:
        video_filter_parts.append(
            f"[base_v]tpad=stop_mode=clone:stop_duration={pad_duration:.2f}[padded_v]"
        )
        last_video_label = "padded_v"
    else:
        last_video_label = "base_v"

    video_filter = ";".join(video_filter_parts)

    # Add clip captions (bottom)
    clip_caption_filters = []
    last_v = last_video_label
    for i, cap_path in enumerate(clip_caption_paths):
        clip_caption_filters.append(f"[{last_v}]{_ass_filter(cap_path)}[v{i+1}]")
        last_v = f"v{i+1}"
    if clip_caption_filters:
        video_filter += ";" + ";".join(clip_caption_filters)

    # Add narration captions (top)
    for i, cap_path in enumerate(narration_caption_paths):
        video_filter += f";[{last_v}]{_ass_filter(cap_path)}[v{i+len(clip_caption_filters)+1}]"
        last_v = f"v{i+len(clip_caption_filters)+1}"

    video_output = working_dir / f"group_{group_idx}_video.mp4"
    ffmpeg_video = [
        FFMPEG_PATH, "-loglevel", "error"
    ] + ffmpeg_video_inputs + [
        "-filter_complex", video_filter,
        "-map", f"[{last_v}]",
    ] + encoder_opts + [
        "-r", "30", "-y", str(video_output)
    ]
    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Rendering video ({total_clip_duration:.0f}s+{pad_duration:.0f}s pad)...", 25)
    _run_ffmpeg(ffmpeg_video, f"Group {group_idx} video", cwd=str(working_dir))

    # Verify video duration
    vid_dur = _get_video_duration_seconds(str(video_output))
    print(f"[INFO] Group {group_idx}: video output duration: {vid_dur:.1f}s")

    # 2. Build continuous clip audio & pad with silence to target_duration
    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Building continuous clip audio (padded to {target_duration:.1f}s)...", 40)

    if use_precut:
        ffmpeg_audio_inputs = []
        for p in group_clip_paths:
            ffmpeg_audio_inputs.extend(["-i", str(p)])
        concat_audio_inputs = "".join(f"[{i}:a]" for i in range(n_clips))
        audio_filter = f"{concat_audio_inputs}concat=n={n_clips}:v=0:a=1[raw_audio];[raw_audio]apad=whole_dur={target_duration:.2f},atrim=end={target_duration:.2f}[clip_audio]"
    else:
        ffmpeg_audio_inputs = ["-i", source_path]
        audio_filter_parts = []
        for i, clip in enumerate(source_clips):
            start = clip["source_start"]
            end = clip["source_end"]
            audio_filter_parts.append(
                f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]"
            )
        concat_audio_inputs = "".join(f"[a{i}]" for i in range(n_clips))
        audio_filter_parts.append(
            f"{concat_audio_inputs}concat=n={n_clips}:v=0:a=1[raw_audio];"
            f"[raw_audio]apad=whole_dur={target_duration:.2f},atrim=end={target_duration:.2f}[clip_audio]"
        )
        audio_filter = ";".join(audio_filter_parts)

    clip_audio_output = working_dir / f"group_{group_idx}_clip_audio.wav"
    ffmpeg_clip_audio = [
        FFMPEG_PATH, "-loglevel", "error"
    ] + ffmpeg_audio_inputs + [
        "-filter_complex", audio_filter,
        "-map", "[clip_audio]",
        "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
        "-y", str(clip_audio_output)
    ]
    _run_ffmpeg(ffmpeg_clip_audio, f"Group {group_idx} clip audio")

    clip_audio_dur = _get_video_duration_seconds(str(clip_audio_output))
    print(f"[INFO] Group {group_idx}: clip audio output duration (padded): {clip_audio_dur:.1f}s")
    if abs(clip_audio_dur - target_duration) > 1.0:
        print(f"[WARN] Group {group_idx}: clip audio duration {clip_audio_dur:.1f}s deviates from target {target_duration:.1f}s by {abs(clip_audio_dur - target_duration):.1f}s!")

    # 3. Build narration audio track (padded to target_duration)
    if narration_audio:
        if progress_cb:
            progress_cb(f"Group {group_idx+1}: Building narration audio track...", 55)

        narration_filter_parts = []
        for i, nar in enumerate(narration_audio):
            reel_start = nar["reel_start"]
            delay_ms = int(reel_start * 1000)
            narration_filter_parts.append(
                f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[nar{i}]"
            )

        narration_inputs = "".join(f"[nar{i}]" for i in range(len(narration_audio)))
        narration_filter_parts.append(
            f"{narration_inputs}amix=inputs={len(narration_audio)}:duration=longest:dropout_transition=0.1:normalize=0[raw_narration_mix];"
            f"[raw_narration_mix]apad=whole_dur={target_duration:.2f},atrim=end={target_duration:.2f}[narration_mix]"
        )

        narration_audio_output = working_dir / f"group_{group_idx}_narration.wav"
        ffmpeg_narration = [
            FFMPEG_PATH, "-loglevel", "error",
            "-i", str(clip_audio_output),  # Input 0 placeholder for sample rate matching
        ]
        for nar in narration_audio:
            ffmpeg_narration.extend(["-i", nar["path"]])
        ffmpeg_narration.extend([
            "-filter_complex", ";".join(narration_filter_parts),
            "-map", "[narration_mix]",
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            "-y", str(narration_audio_output)
        ])
        _run_ffmpeg(ffmpeg_narration, f"Group {group_idx} narration audio")

        narr_dur = _get_video_duration_seconds(str(narration_audio_output))
        print(f"[INFO] Group {group_idx}: narration audio output duration: {narr_dur:.1f}s")

        # 4. Apply ducking to clip audio and mix with narration
        if progress_cb:
            progress_cb(f"Group {group_idx+1}: Applying audio ducking and mixing...", 70)

        duck_chain = _build_ducking_filter_chain(narration_audio, input_label="0:a", output_label="ducked")

        mixed_audio_output = working_dir / f"group_{group_idx}_mixed_audio.wav"
        ffmpeg_mix = [
            FFMPEG_PATH, "-loglevel", "error",
            "-i", str(clip_audio_output),
            "-i", str(narration_audio_output),
            "-filter_complex",
            f"{duck_chain};"
            f"[1:a]volume=1.15[narr];"
            f"[ducked][narr]amix=inputs=2:duration=first:dropout_transition=0.1:normalize=0,"
            f"apad=whole_dur={target_duration:.2f},atrim=end={target_duration:.2f}[mixed]",
            "-map", "[mixed]",
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            "-y", str(mixed_audio_output)
        ]
        _run_ffmpeg(ffmpeg_mix, f"Group {group_idx} audio mix")

        mix_dur = _get_video_duration_seconds(str(mixed_audio_output))
        print(f"[INFO] Group {group_idx}: mixed audio output duration: {mix_dur:.1f}s (target: {target_duration:.1f}s)")
        if abs(mix_dur - target_duration) > 0.5:
            print(f"[WARN] Group {group_idx}: mixed audio duration {mix_dur:.1f}s deviates from target {target_duration:.1f}s by {abs(mix_dur - target_duration):.1f}s — re-padding...")
            # Re-pad mixed audio to exactly match target_duration
            repadded_output = working_dir / f"group_{group_idx}_mixed_audio_repadded.wav"
            ffmpeg_repad = [
                FFMPEG_PATH, "-loglevel", "error",
                "-i", str(mixed_audio_output),
                "-af", f"apad=whole_dur={target_duration:.2f},atrim=end={target_duration:.2f}",
                "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
                "-y", str(repadded_output)
            ]
            _run_ffmpeg(ffmpeg_repad, f"Group {group_idx} audio re-pad")
            import shutil
            shutil.move(str(repadded_output), str(mixed_audio_output))
            repad_dur = _get_video_duration_seconds(str(mixed_audio_output))
            print(f"[INFO] Group {group_idx}: re-padded mixed audio duration: {repad_dur:.1f}s")
    else:
        mixed_audio_output = clip_audio_output

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

    actual_duration = _get_video_duration_seconds(str(output_path))
    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Done ({actual_duration:.1f}s)", 100)
    print(f"[INFO] Group {group_idx} output: {output_path.name} (final video duration: {actual_duration:.1f}s)")
    if abs(actual_duration - target_duration) > 2.0:
        print(f"[WARN] Group {group_idx}: FINAL OUTPUT duration {actual_duration:.1f}s deviates from target {target_duration:.1f}s by {abs(actual_duration - target_duration):.1f}s — check audio/video alignment!")

    return str(output_path)