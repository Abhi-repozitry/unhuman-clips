import subprocess
import os
from pathlib import Path
from backend.config import HOOK_SECONDS, OUTPUTS_DIR, get_job_working_dir
from typing import Callable, Optional


def _get_video_encoder(fallback=False):
    """Return the best available H.264 encoder.
    Prefer h264_nvenc (GPU), fall back to libx264 (CPU) if NVENC fails or is forced."""
    if not fallback:
        # Check NVENC availability once
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
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
            # Swap encoder from NVENC to libx264
            new_cmd = []
            skip_next = False
            for j, arg in enumerate(cmd):
                if skip_next:
                    skip_next = False
                    continue
                if arg == "h264_nvenc":
                    new_cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
                    skip_next = True  # skip the next arg which was pix_fmt
                elif arg in ("-preset", "-rc", "-cq"):
                    skip_next = True  # skip NVENC-specific flags and their values
                elif arg == "p7":
                    pass  # NVENC-specific preset value
                elif arg == "vbr":
                    pass  # NVENC-specific RC mode
                elif arg == "23" and j > 0 and cmd[j-1] == "-cq":
                    pass  # NVENC-specific CQ value
                else:
                    new_cmd.append(arg)
            return _run_ffmpeg(new_cmd, description, attempt + 1, max_attempts)
        raise RuntimeError(f"{description} failed: {e.stderr.decode() if e.stderr else 'unknown'}") from e


def _ass_filter(path: str) -> str:
    escaped = str(path).replace("\\", "/").replace("'", r"\'")
    return f"ass='{escaped}'"


def _portrait_filter(source: str, caption_path: str, label: str = "v") -> str:
    return (
        f"[{source}]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,boxblur=20:10[bg];"
        f"[{source}]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[comp];"
        f"[comp]{_ass_filter(caption_path)}[{label}]"
    )


def _blurred_frame_filter(caption_path: str) -> str:
    return (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:10,"
        f"{_ass_filter(caption_path)}[v]"
    )


def build_final_video(
    job_id: str,
    clip_paths: list,
    clip_windows: list,
    commentary_audio: list,
    caption_paths_commentary: list,
    caption_paths_clips: list,
    progress_cb: Optional[Callable[[str, float], None]] = None
) -> str:
    from backend.config import MAX_OUTPUT_DURATION
    working_dir = get_job_working_dir(job_id)
    working_dir = Path(working_dir)

    # Detect encoder - prefer NVENC but fallback to libx264
    encoder = _get_video_encoder()
    encoder_opts = _build_encoder_opts(encoder)
    print(f"[INFO] Using video encoder: {encoder}")

    n = len(clip_paths)
    total_steps = n * 4 + 1  # hook, body, frame, insight per moment + concat
    
    for i in range(n):
        step = i * 4
        if progress_cb:
            progress_cb(f"Processing moment {i+1}/{n}: rendering hook...", (step / total_steps) * 100)

        clip_duration = max(0.1, clip_windows[i]["end"] - clip_windows[i]["start"])
        hook_duration = min(HOOK_SECONDS, clip_duration)
        hook_out = working_dir / f"seg_{i}_hook.mp4"
        hook_filter = _portrait_filter("0:v", caption_paths_commentary[i]["hook"])
        ffmpeg_hook = [
            "ffmpeg", "-loglevel", "error",
            "-i", clip_paths[i],
            "-i", commentary_audio[i]["hook"]["path"],
            "-filter_complex", hook_filter,
            "-map", "[v]", "-map", "1:a",
            "-t", str(hook_duration),
        ] + encoder_opts + [
            "-r", "30", "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-y", str(hook_out)
        ]
        _run_ffmpeg(ffmpeg_hook, f"Hook segment {i}")

        step = i * 4 + 1
        if progress_cb:
            progress_cb(f"Processing moment {i+1}/{n}: rendering clip body...", (step / total_steps) * 100)
        
        clip_out = working_dir / f"seg_{i}_clip.mp4"
        body_duration = max(0.1, clip_duration - hook_duration)
        clip_filter = _portrait_filter("0:v", caption_paths_clips[i])
        ffmpeg_clip = [
            "ffmpeg", "-loglevel", "error",
            "-ss", str(hook_duration),
            "-i", clip_paths[i],
            "-filter_complex", clip_filter,
            "-map", "[v]", "-map", "0:a",
            "-t", str(body_duration),
        ] + encoder_opts + [
            "-r", "30", "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-y", str(clip_out)
        ]
        _run_ffmpeg(ffmpeg_clip, f"Clip segment {i}")

        step = i * 4 + 2
        if progress_cb:
            progress_cb(f"Processing moment {i+1}/{n}: extracting insight frame...", (step / total_steps) * 100)

        frame_path = working_dir / f"frame_{i}.jpg"
        ffmpeg_frame = [
            "ffmpeg", "-loglevel", "error",
            "-sseof", "-0.1",
            "-i", clip_paths[i],
            "-vframes", "1",
            "-y", str(frame_path)
        ]
        try:
            subprocess.run(ffmpeg_frame, capture_output=True, check=True)
        except subprocess.CalledProcessError:
            fallback_frame = [
                "ffmpeg", "-loglevel", "error",
                "-ss", str(max(0, hook_duration)),
                "-i", clip_paths[i],
                "-vframes", "1",
                "-y", str(frame_path)
            ]
            try:
                subprocess.run(fallback_frame, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Frame extraction failed for segment {i}: {e.stderr.decode() if e.stderr else 'unknown'}") from e

        step = i * 4 + 3
        if progress_cb:
            progress_cb(f"Processing moment {i+1}/{n}: rendering insight...", (step / total_steps) * 100)

        insight_out = working_dir / f"seg_{i}_insight.mp4"
        insight_filter = _blurred_frame_filter(caption_paths_commentary[i]["insight"])
        insight_duration = commentary_audio[i]["insight"]["duration"]
        ffmpeg_insight = [
            "ffmpeg", "-loglevel", "error",
            "-loop", "1",
            "-i", str(frame_path),
            "-i", commentary_audio[i]["insight"]["path"],
            "-filter_complex", insight_filter,
            "-map", "[v]", "-map", "1:a",
            "-t", str(insight_duration),
        ] + encoder_opts + [
            "-r", "30", "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-shortest", "-y", str(insight_out)
        ]
        _run_ffmpeg(ffmpeg_insight, f"Insight segment {i}")

    if progress_cb:
        progress_cb("Concatenating final video...", 95)
    
    concat_list_path = working_dir / "concat_list.txt"
    with open(concat_list_path, "w") as f:
        for i in range(n):
            hook_segment = working_dir / f"seg_{i}_hook.mp4"
            clip_segment = working_dir / f"seg_{i}_clip.mp4"
            insight_segment = working_dir / f"seg_{i}_insight.mp4"
            f.write(f"file '{hook_segment}'\n")
            f.write(f"file '{clip_segment}'\n")
            f.write(f"file '{insight_segment}'\n")

    output_path = OUTPUTS_DIR / f"{job_id}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ffmpeg_concat = [
        "ffmpeg", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list_path),
        "-c", "copy",
        "-t", str(MAX_OUTPUT_DURATION),
        "-y", str(output_path)
    ]
    try:
        result = subprocess.run(ffmpeg_concat, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Final concatenation failed: {e.stderr.decode() if e.stderr else 'unknown'}") from e

    if not output_path.exists():
        raise RuntimeError(f"Output file was not created: {output_path}")
    
    # Check actual duration and warn if capped
    probe_duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
        capture_output=True, text=True
    )
    if probe_duration.returncode == 0:
        try:
            actual_duration = float(probe_duration.stdout.strip())
            if actual_duration >= MAX_OUTPUT_DURATION:
                print(f"[WARN] Output video capped at {MAX_OUTPUT_DURATION}s (actual duration would have been longer)")
        except ValueError:
            pass

    if progress_cb:
        progress_cb("Video generation complete!", 100)

    return str(output_path)
