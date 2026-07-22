import subprocess
import os
from pathlib import Path
from backend.config import HOOK_SECONDS, FFMPEG_PATH, FFPROBE_PATH
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
    """Run ffmpeg with a 600-second timeout, retrying with CPU encoder if NVENC fails.
    Uses stderr=PIPE, stdout=DEVNULL to prevent pipe buffer deadlock on Windows."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            cwd=cwd,
            timeout=600,
        )
        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{description} timed out after 600 seconds. FFmpeg may be deadlocked or the input is too large.")
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode(errors="replace") if e.stderr else "(no stderr)"
        if attempt < max_attempts and "h264_nvenc" in " ".join(cmd) and os.environ.get("ALLOW_CPU_FFMPEG_FALLBACK") == "1":
            print(f"[WARN] NVENC failed for {description}, retrying with libx264: {stderr_text[:200]}")
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
        raise RuntimeError(f"{description} failed: {stderr_text[:1000]}") from e


def _ass_filter(path: str) -> str:
    filename = Path(path).name
    return f"ass=filename={filename}"


def _build_ducking_filter_chain(narration_events, input_label="0:a", output_label="ducked",
                                 target_duration: float = 0.0, key_moment_end: float = 0.0):
    """
    Build a robust single-filter ducking expression.
    Ducks audio to ~0.03 during narration events — original audio is near-silent, avoiding voice overlap.
    Uses tighter pre-duck buffer (0.4s before) and post-duck buffer (0.25s after) to prevent bleed.
    Uses smooth 0.1s ramps for natural transitions.
    SKIPS ducking during the final key moment (last 8s of the group) to let the payoff breathe.
    """
    valid_events = [ev for ev in narration_events if ev.get("reel_end", 0) - ev.get("reel_start", 0) >= 0.3]
    if not valid_events:
        return f"[{input_label}]anull[{output_label}]"

    # Identify the payoff zone: the final key moment (last 8s of target duration or last narration window)
    payoff_start = max(0.0, (key_moment_end if key_moment_end > 0 else target_duration) - 8.0)
    
    print(f"[INFO] Audio ducking: {len(valid_events)} narration windows (payoff silence zone starts at {payoff_start:.1f}s)")
    duck_terms = []
    PRE_BUF = 0.4   # Tighter pre-buffer: 0.4s (was 0.3s)
    POST_BUF = 0.25 # Tighter post-buffer: 0.25s (was 0.2s)
    RAMP = 0.1      # Smoother ramp: 0.1s (was 0.08s)
    
    for i, ev in enumerate(valid_events):
        # Skip ducking if event falls entirely within the payoff zone
        if ev["reel_start"] >= payoff_start:
            print(f"[INFO]   Skipping duck window {i+1}: narration at [{ev['reel_start']:.3f}s-{ev['reel_end']:.3f}s] is in payoff zone (after {payoff_start:.1f}s)")
            continue
            
        # Apply pre-duck buffer and post-duck buffer
        s = max(0.0, ev["reel_start"] - PRE_BUF)
        e = ev["reel_end"] + POST_BUF
        
        # If the duck window would extend into payoff zone, cap it there
        if s < payoff_start < e:
            e = payoff_start
            print(f"[INFO]   Capped duck window {i+1} at payoff boundary: [{s:.3f}s - {e:.3f}s]")
        
        # Only add if we have a valid window after capping
        if e - s < 0.2:
            continue
            
        print(f"[INFO]   Duck window {i+1}: [{s:.3f}s - {e:.3f}s] "
              f"(narration [{ev['reel_start']:.3f}s - {ev['reel_end']:.3f}s], "
              f"pre-buf={PRE_BUF}s, post-buf={POST_BUF}s, ramp={RAMP}s)")
        ramp_in = f"min(1,max(0,(t-{s:.3f})/{RAMP}))"
        ramp_out = f"min(1,max(0,(t-({e:.3f}-{RAMP}))/{RAMP}))"
        duck_terms.append(f"if(between(t,{s:.3f},{e:.3f}),({ramp_in})*(1-({ramp_out})),0)")

    if not duck_terms:
        # All events were in the payoff zone or too short — no ducking needed
        return f"[{input_label}]anull[{output_label}]"

    if len(duck_terms) == 1:
        duck_expr = duck_terms[0]
    else:
        duck_expr = f"min(1.0,{'+'.join(duck_terms)})"

    # Ducking factor 0.97 -> original audio drops to 0.03 (3%) during narration
    vol_expr = f"1.0-({duck_expr}*0.97)"
    print(f"[INFO] Audio ducking depth: 0.97 (original audio -> 3% volume during narration). "
          f"{len(duck_terms)} active duck windows, payoff zone protected after {payoff_start:.1f}s")
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

    # ===== STRICT GROUP ISOLATION VALIDATION =====
    # Ensure this group only uses its own clips, narration, and captions.
    # Cross-group contamination is a critical bug — verify every input is self-contained.
    n_clips = len(source_clips)
    if n_clips == 0:
        raise RuntimeError(f"Group {group_idx}: No source clips in group — cannot render.")

    # Validate clip paths match source_clips count
    if group_clip_paths and len(group_clip_paths) != n_clips:
        raise RuntimeError(
            f"Group {group_idx}: GROUP ISOLATION VIOLATION — group_clip_paths count ({len(group_clip_paths)}) "
            f"does not match source_clips count ({n_clips}). This indicates cross-group data contamination."
        )

    # Validate narration audio paths exist and belong to this group
    for i, nar in enumerate(narration_audio):
        nar_path = nar.get("path", "")
        if not nar_path or not os.path.exists(nar_path):
            raise RuntimeError(
                f"Group {group_idx}: GROUP ISOLATION VIOLATION — narration audio {i} path missing or invalid: {nar_path}"
            )
        # Verify the path contains this group's identifier to prevent cross-group file usage
        if f"group_{group_idx}_narration_" not in str(nar_path):
            raise RuntimeError(
                f"Group {group_idx}: GROUP ISOLATION VIOLATION — narration audio {i} path '{nar_path}' "
                f"does not belong to this group (missing 'group_{group_idx}_narration_' prefix). "
                f"This indicates cross-group data contamination."
            )

    # Validate caption paths belong to this group
    for i, cap_path in enumerate(clip_caption_paths):
        if f"group_{group_idx}_clip_caption_" not in str(cap_path):
            raise RuntimeError(
                f"Group {group_idx}: GROUP ISOLATION VIOLATION — clip caption {i} path '{cap_path}' "
                f"does not belong to this group. Cross-group contamination detected."
            )
    for i, cap_path in enumerate(narration_caption_paths):
        if f"group_{group_idx}_narr_caption_" not in str(cap_path):
            raise RuntimeError(
                f"Group {group_idx}: GROUP ISOLATION VIOLATION — narration caption {i} path '{cap_path}' "
                f"does not belong to this group. Cross-group contamination detected."
            )

    print(f"[INFO] Rendering isolated Group {group_idx} with {n_clips} clips and {len(narration_audio)} narration events — all paths validated for group isolation.")

    encoder = _get_video_encoder()
    encoder_opts = _build_encoder_opts(encoder)
    print(f"[INFO] compose_group {group_idx}: Using video encoder: {encoder}")

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

    # TARGET DURATION CALCULATION (IMPROVED):
    # We now respect estimated_duration_seconds as a strong signal, not a weak hint.
    # MAX_FREEZE_PAD increased to 12s to allow longer freeze-padding on valid long reels.
    # The analyzer now produces 90-150s groups, so we need to allow proportionally more pad.
    #
    # Strategy:
    # 1. Start with max(total_clip_duration, max_narration_end, estimated_duration_seconds, MIN_OUTPUT_DURATION)
    # 2. Allow freeze-pad up to MAX_FREEZE_PAD to fill the gap
    # 3. Clamp to MAX_OUTPUT_DURATION
    # 4. If pad exceeds MAX_FREEZE_PAD, cap pad to MAX_FREEZE_PAD and let audio run longer via silence
    
    MAX_FREEZE_PAD = 12.0  # Increased from 3.0 to allow longer freeze-padding for 90-150s target reels
    
    if pad_duration > MAX_FREEZE_PAD:
        # narration_tail is how far narration reaches past the last video frame.
        narration_tail = max(0.0, max_narration_end - total_clip_duration)
        # Allow more pad when there's narration tail to cover, up to MAX_FREEZE_PAD
        allowed_pad = min(MAX_FREEZE_PAD, max(narration_tail, MAX_FREEZE_PAD * 0.5))

        print(f"[INFO] Group {group_idx}: clip content ({total_clip_duration:.1f}s) short of "
              f"target ({target_duration:.1f}s) — using freeze pad {allowed_pad:.1f}s "
              f"(was {pad_duration:.1f}s cap, narration_tail={narration_tail:.1f}s). "
              f"Audio will be padded with silence beyond freeze.")

        # Only drop narration events if they'd be lost beyond video+freeze end.
        # Previously we capped at total_clip_duration + allowed_pad, but now with
        # larger pad allowance we can accommodate more narration without dropping.
        capped_limit = total_clip_duration + min(MAX_FREEZE_PAD, max(narration_tail, 3.0))
        truncated_events = [
            nar for nar in narration_audio
            if nar.get("reel_end", 0) > capped_limit
        ]
        if truncated_events:
            for nar in truncated_events:
                print(
                    f"[WARN] Group {group_idx}: dropping narration event that would be truncated — "
                    f"type={nar.get('event_type', '?')!r}, "
                    f"reel=[{nar.get('reel_start', 0):.2f}s–{nar.get('reel_end', 0):.2f}s], "
                    f"capped_limit={capped_limit:.2f}s, "
                    f"text={str(nar.get('text', nar.get('narration_text', '')))[:80]!r}"
                )
            # narration_audio and narration_caption_paths are index-correlated (built in
            # the same order in queue_manager).  Zip them together, apply the same
            # reel_end <= capped_limit filter, then unzip so both lists stay in sync.
            paired = list(zip(narration_audio, narration_caption_paths))
            paired = [(nar, cap) for nar, cap in paired if nar.get("reel_end", 0) <= capped_limit]
            if paired:
                narration_audio, narration_caption_paths = map(list, zip(*paired))
            else:
                narration_audio, narration_caption_paths = [], []
            # Recompute max_narration_end after dropping over-limit events.
            max_narration_end = max((nar.get("reel_end", 0) for nar in narration_audio), default=0.0)

        pad_duration = allowed_pad
        # Recompute target_duration so audio padding (apad=whole_dur=...) uses the same
        # corrected value, not the original inflated target.
        target_duration = total_clip_duration + pad_duration

    print(f"[INFO] Group {group_idx}: total_clip_duration={total_clip_duration:.1f}s, "
          f"max_narration_end={max_narration_end:.1f}s, est={estimated_duration_seconds:.1f}s, target={target_duration:.1f}s, pad={pad_duration:.1f}s")

    # 1. Build video filter complex
    if use_precut:
        ffmpeg_video_inputs = []
        video_filter_parts = []
        for i, p in enumerate(group_clip_paths):
            ffmpeg_video_inputs.extend(["-i", str(p)])
            video_filter_parts.append(f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v{i}]")
        concat_inputs = "".join(f"[v{i}]" for i in range(n_clips))
        video_filter_parts.append(f"{concat_inputs}concat=n={n_clips}:v=1:a=0[base_v]")
    else:
        ffmpeg_video_inputs = ["-i", source_path]
        video_filter_parts = []
        for i, clip in enumerate(source_clips):
            start = clip["source_start"]
            end = clip["source_end"]
            video_filter_parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v{i}]"
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
    all_caption_filters = []
    last_v = last_video_label
    caption_label_idx = 1
    for i, cap_path in enumerate(clip_caption_paths):
        next_label = f"vc{caption_label_idx}"
        all_caption_filters.append(f"[{last_v}]{_ass_filter(cap_path)}[{next_label}]")
        last_v = next_label
        caption_label_idx += 1

    # Add narration captions (top)
    for i, cap_path in enumerate(narration_caption_paths):
        next_label = f"vc{caption_label_idx}"
        all_caption_filters.append(f"[{last_v}]{_ass_filter(cap_path)}[{next_label}]")
        last_v = next_label
        caption_label_idx += 1

    if all_caption_filters:
        video_filter += ";" + ";".join(all_caption_filters)

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

        # Identify the payoff moment (last narration event's end) so ducking skips it
        key_moment_end = 0.0
        if narration_audio:
            # Use the final narration event's reel_end as the key moment boundary
            key_moment_end = max(nar.get("reel_end", 0) for nar in narration_audio)
        
        duck_chain = _build_ducking_filter_chain(
            narration_audio, input_label="0:a", output_label="ducked",
            target_duration=target_duration, key_moment_end=key_moment_end
        )

        mixed_audio_output = working_dir / f"group_{group_idx}_mixed_audio.wav"
        ffmpeg_mix = [
            FFMPEG_PATH, "-loglevel", "error",
            "-i", str(clip_audio_output),
            "-i", str(narration_audio_output),
            "-filter_complex",
            f"{duck_chain};"
            f"[1:a]volume=1.15[narr];"
            f"[ducked][narr]amix=inputs=2:duration=first:dropout_transition=0.1:normalize=0,"
            f"alimiter=limit=0.95:attack=5:release=50,"
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

    # Write intermediate output to working_dir; _final_edit_group in queue_manager owns OUTPUTS_DIR placement.
    output_path = working_dir / f"group_{group_idx}_output.mp4"
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
    _run_ffmpeg(ffmpeg_final, f"Group {group_idx} final mux")

    actual_duration = _get_video_duration_seconds(str(output_path))
    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Done ({actual_duration:.1f}s)", 100)
    print(f"[INFO] Group {group_idx} output: {output_path.name} (final video duration: {actual_duration:.1f}s)")
    if abs(actual_duration - target_duration) > 2.0:
        print(f"[WARN] Group {group_idx}: FINAL OUTPUT duration {actual_duration:.1f}s deviates from target {target_duration:.1f}s by {abs(actual_duration - target_duration):.1f}s — check audio/video alignment!")

    return str(output_path)