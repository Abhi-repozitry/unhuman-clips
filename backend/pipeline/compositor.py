import subprocess
import os
from pathlib import Path
from backend.config import (
    FFMPEG_PATH, FFPROBE_PATH,
    OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FPS,
    NARRATION_VOLUME_BOOST, ALIMITER_LIMIT, ALIMITER_ATTACK_MS, ALIMITER_RELEASE_MS,
    VAD_THRESHOLD, VAD_PRE_BUFFER_SECONDS, VAD_POST_BUFFER_SECONDS,
    VAD_SCURVE_RAMP_SECONDS, VAD_DUCKING_DEPTH, VAD_SILENCE_THRESHOLD,
)
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
            # Rebuild command from scratch for CPU encoding (avoids arg patching bugs)
            new_cmd = [FFMPEG_PATH, "-loglevel", "error"]
            i = 1  # skip FFMPEG_PATH
            while i < len(cmd):
                arg = cmd[i]
                if arg in ("-ss", "-i", "-t"):
                    new_cmd.extend([arg, cmd[i + 1]]); i += 2
                elif arg in ("-y",):
                    new_cmd.append(arg); i += 1
                elif arg == "-filter_complex":
                    new_cmd.extend([arg, cmd[i + 1]]); i += 2
                elif arg.startswith("[") and cmd[i + 1] == "-map" if i + 1 < len(cmd) else False:
                    new_cmd.extend([arg, cmd[i + 1]]); i += 2
                elif arg == "-map":
                    new_cmd.extend([arg, cmd[i + 1]]); i += 2
                else:
                    i += 1  # skip all encoder-specific args
            new_cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "23"])
            new_cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            new_cmd.extend(["-movflags", "+faststart", "-avoid_negative_ts", "make_zero"])
            new_cmd.extend(["-y", cmd[-1]])  # output path is last arg
            return _run_ffmpeg(new_cmd, description, attempt + 1, max_attempts, cwd=cwd)
        raise RuntimeError(f"{description} failed: {stderr_text[:1000]}") from e


def _ass_filter(path: str) -> str:
    filename = Path(path).name
    return f"ass=filename={filename}"


def get_speech_timestamps_from_narration(
    narration_path: str,
    threshold: float = VAD_THRESHOLD,
    min_speech_duration_ms: int = 100,
    min_silence_duration_ms: int = 200,
) -> List[Dict[str, float]]:
    """Run Silero VAD on a narration audio file to detect precise speech timestamps.

    Returns list of {"start": float, "end": float} dicts for each detected
    speech segment within the narration audio. These are used to drive
    intelligent ducking — original audio is only ducked during actual TTS
    speech, not during silence or breath pauses within the narration.

    Falls back to a single speech window spanning the full file if VAD fails.
    Uses the same API pattern as editor.py for consistency.
    """
    fallback = [{"start": 0.0, "end": _get_audio_duration(narration_path)}]

    try:
        import torch
        import torchaudio
        from silero_vad import get_speech_timestamps, read_audio
    except ImportError:
        print(f"[WARN] silero-vad or torch not available, using full-window fallback for {Path(narration_path).name}")
        return fallback

    try:
        wav, sr = read_audio(narration_path, sampling_rate=16000)

        if len(wav) == 0:
            return fallback

        speech_timestamps = get_speech_timestamps(
            wav,
            sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=True,
        )

        if not speech_timestamps:
            return fallback

        return [{"start": ts["start"], "end": ts["end"]} for ts in speech_timestamps]

    except Exception as e:
        print(f"[WARN] Silero VAD failed on {Path(narration_path).name}: {e}, using full-window fallback")
        return fallback


def _get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file using ffprobe."""
    try:
        result = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=15
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, OSError):
        return 0.0


def _build_ducking_filter_chain(
    narration_events: List[Dict[str, Any]],
    narration_vad_timestamps: Optional[List[List[Dict[str, float]]]] = None,
    input_label: str = "0:a",
    output_label: str = "ducked",
    target_duration: float = 0.0,
    key_moment_end: float = 0.0,
) -> str:
    """Build a VAD-driven ducking filter chain using Silero VAD speech timestamps.

    Instead of ducking during the entire narration event window (which includes
    silence/breath pauses), this function uses per-narration VAD timestamps to
    duck ONLY during actual TTS speech. This produces much more natural ducking
    that preserves the original audio during narration pauses.

    Features:
    - VAD-precise ducking: only ducks during detected speech, not silence
    - S-curve ramps: smooth 3x²-2x³ Hermite transitions (no clicks)
    - Pre/post buffers: tight 0.4s pre, 0.25s post around each speech segment
    - Payoff zone protection: skips ducking for the final 8s key moment
    - Depth: ducks original audio to ~3% (configurable via VAD_DUCKING_DEPTH)
    """
    if not narration_events:
        return f"[{input_label}]anull[{output_label}]"

    # Filter valid narration events
    valid_events = []
    valid_vad = []
    for i, ev in enumerate(narration_events):
        dur = ev.get("reel_end", 0) - ev.get("reel_start", 0)
        if dur >= 0.3:
            valid_events.append(ev)
            if narration_vad_timestamps and i < len(narration_vad_timestamps):
                valid_vad.append(narration_vad_timestamps[i])
            else:
                # No VAD data for this event — use full window as fallback
                valid_vad.append([{"start": ev["reel_start"], "end": ev["reel_end"]}])

    if not valid_events:
        return f"[{input_label}]anull[{output_label}]"

    # Payoff zone: the final 8s of the reel or after key_moment_end
    payoff_start = max(0.0, (key_moment_end if key_moment_end > 0 else target_duration) - 8.0)

    PRE_BUF = VAD_PRE_BUFFER_SECONDS
    POST_BUF = VAD_POST_BUFFER_SECONDS
    RAMP = VAD_SCURVE_RAMP_SECONDS
    DEPTH = VAD_DUCKING_DEPTH

    print(f"[INFO] VAD-driven ducking: {len(valid_events)} narration events, "
          f"payoff zone starts at {payoff_start:.1f}s, ramp={RAMP}s, depth={DEPTH}")

    duck_terms = []

    for ev_idx, (ev, vad_segments) in enumerate(zip(valid_events, valid_vad)):
        # Process each VAD-detected speech segment within this narration event
        for seg_idx, seg in enumerate(vad_segments):
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            seg_dur = seg_end - seg_start
            if seg_dur < 0.1:
                continue

            # Convert VAD-relative timestamps to reel-absolute timestamps
            # VAD timestamps are within the individual narration audio file,
            # so we offset by the narration event's reel_start
            reel_offset = ev["reel_start"]
            abs_start = reel_offset + seg_start
            abs_end = reel_offset + seg_end

            # Apply pre/post buffers
            duck_start = max(0.0, abs_start - PRE_BUF)
            duck_end = abs_end + POST_BUF

            # Skip if entirely in payoff zone
            if duck_start >= payoff_start:
                print(f"[INFO]   VAD skip (payoff): narr {ev_idx+1} seg {seg_idx+1} "
                      f"[{abs_start:.3f}-{abs_end:.3f}s] in payoff zone")
                continue

            # Cap at payoff boundary if it overlaps
            if duck_start < payoff_start < duck_end:
                duck_end = payoff_start

            # Skip tiny windows
            if duck_end - duck_start < 0.15:
                continue

            # Build S-curve duck expression using Hermite 3x²-2x³
            # ramp_in: ease from 0→1 over RAMP seconds at duck_start
            # ramp_out: ease from 1→0 over RAMP seconds before duck_end
            ramp_in_start = duck_start
            ramp_in_end = duck_start + RAMP
            ramp_out_start = duck_end - RAMP
            ramp_out_end = duck_end

            # S-curve expression: smooth ease-in-ease-out
            # In the ramp-in zone: sigmoid curve from 0 to 1
            # In the sustained zone: full duck (1.0)
            # In the ramp-out zone: sigmoid curve from 1 to 0
            # Outside all zones: 0
            ri_s = f"{ramp_in_start:.4f}"
            ri_e = f"{ramp_in_end:.4f}"
            ro_s = f"{ramp_out_start:.4f}"
            ro_e = f"{ramp_out_end:.4f}"
            r = f"{RAMP:.4f}"

            expr = (
                f"if(between(t,{ri_s},{ro_e}),"
                f"if(lt(t,{ri_e}),"
                # Ramp-in: Hermite via xn*xn*(3-2*xn) where xn=(t-ri_s)/RAMP
                f"((t-{ri_s})/{r})*((t-{ri_s})/{r})*(3-2*(t-{ri_s})/{r}),"
                f"if(lt(t,{ro_s}),"
                # Sustained zone
                f"1.0,"
                # Ramp-out: 1 - Hermite
                f"(1.0-((t-{ro_s})/{r})*((t-{ro_s})/{r})*(3-2*(t-{ro_s})/{r}))"
                f")))"
            )

            duck_terms.append(expr)
            print(f"[INFO]   VAD duck seg: narr {ev_idx+1} seg {seg_idx+1} "
                  f"[{abs_start:.3f}-{abs_end:.3f}s] -> duck window "
                  f"[{duck_start:.3f}-{duck_end:.3f}s]")

    if not duck_terms:
        return f"[{input_label}]anull[{output_label}]"

    if len(duck_terms) == 1:
        duck_expr = duck_terms[0]
    else:
        duck_expr = f"max({','.join(duck_terms)})"

    vol_expr = f"1.0-({duck_expr}*{DEPTH:.2f})"
    print(f"[INFO] VAD ducking: {len(duck_terms)} speech segments, "
          f"depth={DEPTH*100:.0f}% reduction during TTS speech only")
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
) -> Dict[str, Any]:
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
    
    # IMPROVED TARGET DURATION CALCULATION:
    # estimated_duration_seconds is the most reliable signal — it represents the
    # analyzer's intent for how long this reel should be. We respect it as a
    # strong target, not a hint.
    #
    # Priority: estimated_duration_seconds > max_narration_end > total_clip_duration > MIN_OUTPUT_DURATION
    # We use the HIGHEST of these to ensure nothing gets cut off.
    target_duration = max(
        estimated_duration_seconds,    # Analyzer's intended duration (primary signal)
        max_narration_end,             # Don't cut off narration
        total_clip_duration,           # Don't cut off clip content
        float(MIN_OUTPUT_DURATION)     # Never go below minimum
    )
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
        capped_limit = total_clip_duration + allowed_pad
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
            video_filter_parts.append(f"[{i}:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},setsar=1[v{i}]")
        concat_inputs = "".join(f"[v{i}]" for i in range(n_clips))
        video_filter_parts.append(f"{concat_inputs}concat=n={n_clips}:v=1:a=0[base_v]")
    else:
        ffmpeg_video_inputs = ["-i", source_path]
        video_filter_parts = []
        for i, clip in enumerate(source_clips):
            start = clip["source_start"]
            end = clip["source_end"]
            video_filter_parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},setsar=1[v{i}]"
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
        "-r", str(OUTPUT_FPS), "-y", str(video_output)
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
    # Initialize VAD defaults in case narration_audio is empty
    vad_stats = {"active": False}
    vad_analysis_entries = []

    if narration_audio:
        if progress_cb:
            progress_cb(f"Group {group_idx+1}: Building narration audio track...", 55)

        narration_filter_parts = []
        for i, nar in enumerate(narration_audio):
            reel_start = nar["reel_start"]
            # Use round() instead of int() to avoid truncation precision loss
            delay_ms = round(reel_start * 1000)
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

    # 4. Apply VAD-driven ducking to clip audio and mix with narration
        if progress_cb:
            progress_cb(f"Group {group_idx+1}: Running VAD on narration + applying intelligent ducking...", 65)

        # Run Silero VAD on each narration audio file to get precise speech timestamps
        narration_vad_timestamps = []
        vad_analysis_entries = []
        for i, nar in enumerate(narration_audio):
            vad_segs = get_speech_timestamps_from_narration(nar["path"])
            narration_vad_timestamps.append(vad_segs)
            total_speech_dur = sum(s["end"] - s["start"] for s in vad_segs)
            vad_analysis_entries.append({
                "segments": len(vad_segs),
                "speech_duration": round(total_speech_dur, 2),
                "total_duration": round(nar.get("duration", 0), 2),
            })
            print(f"[INFO] VAD narr {i+1}: {len(vad_segs)} speech segments, "
                  f"total speech={total_speech_dur:.2f}s of {nar.get('duration', 0):.2f}s audio")

        # Aggregate VAD stats for frontend display
        total_vad_segments = sum(e["segments"] for e in vad_analysis_entries)
        total_vad_speech = round(sum(e["speech_duration"] for e in vad_analysis_entries), 2)
        vad_stats = {
            "active": True,
            "total_segments": total_vad_segments,
            "total_speech_duration": total_vad_speech,
            "narration_count": len(narration_audio),
        }

        if progress_cb:
            progress_cb(f"Group {group_idx+1}: Applying VAD-driven audio ducking...", 70)

        # Identify the payoff moment (last narration event's end) so ducking skips it
        key_moment_end = 0.0
        if narration_audio:
            # Use the final narration event's reel_end as the key moment boundary
            key_moment_end = max(nar.get("reel_end", 0) for nar in narration_audio)
        
        duck_chain = _build_ducking_filter_chain(
            narration_audio,
            narration_vad_timestamps=narration_vad_timestamps,
            input_label="0:a",
            output_label="ducked",
            target_duration=target_duration,
            key_moment_end=key_moment_end
        )

        mixed_audio_output = working_dir / f"group_{group_idx}_mixed_audio.wav"
        ffmpeg_mix = [
            FFMPEG_PATH, "-loglevel", "error",
            "-i", str(clip_audio_output),
            "-i", str(narration_audio_output),
            "-filter_complex",
            f"{duck_chain};"
            f"[1:a]volume={NARRATION_VOLUME_BOOST}[narr];"
            f"[ducked][narr]amix=inputs=2:duration=first:dropout_transition=0.1:normalize=0,"
            f"alimiter=limit={ALIMITER_LIMIT}:attack={ALIMITER_ATTACK_MS}:release={ALIMITER_RELEASE_MS}:level=disabled,"
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

    return {
        "output_path": str(output_path),
        "vad_stats": vad_stats if narration_audio else {"active": False},
        "vad_analysis": vad_analysis_entries if narration_audio else [],
    }