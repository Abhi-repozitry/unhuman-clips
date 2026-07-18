from backend.config import CAPTION_FONT, CAPTION_FONT_SIZE
from typing import Callable, Optional, List, Dict, Any
from backend.pipeline.ocr import detect_existing_captions


def _escape_ass_text(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("\n", " ")
    text = text.replace(",", "\\,")
    return text


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _wrap_text_ass(text: str, max_chars: int = 28) -> str:
    """Wrap text for 9:16 portrait — narrower lines for readability."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line = (current_line + " " + word).strip()
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\\N".join(lines)


def _ass_header(style_lines: str, dialogue_lines: str) -> str:
    return f"""[Script Info]
Title: Unhuman Clips Captions
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_lines}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{dialogue_lines}"""


def _make_style(name: str, font: str, size: int, alignment: int, margin_v: int,
                bold: int = 0, outline: int = 3, shadow: int = 2,
                primary: str = "&H00FFFFFF", back: str = "&H80000000") -> str:
    """Build a single ASS Style line.
    Alignment: 2=bottom center, 8=top center
    """
    return (f"Style: {name},{font},{size},{primary},&H00000008,"
            f"&H00000000,{back},{bold},0,0,0,100,100,0,0,"
            f"1,{outline},{shadow},{alignment},40,40,{margin_v},0")


def generate_clip_ass(transcript: list, clip_start: float, clip_end: float,
                      out_path: str,
                      progress_cb: Optional[Callable[[str, float], None]] = None) -> str:
    """Generate captions for original-clip segments — bottom-aligned with background box."""
    if progress_cb:
        progress_cb("Filtering transcript for clip...", 20)

    filtered = []
    for entry in transcript:
        entry_start = entry["start"]
        entry_end = entry["end"]
        if entry_end < clip_start or entry_start > clip_end:
            continue
        adjusted_start = max(0.0, entry_start - clip_start)
        adjusted_end = min(clip_end - clip_start, entry_end - clip_start)
        if adjusted_end > adjusted_start:
            filtered.append({
                "start": adjusted_start,
                "end": adjusted_end,
                "text": entry["text"],
            })

    style_line = _make_style(
        "ClipCaption", CAPTION_FONT, CAPTION_FONT_SIZE,
        alignment=2,   # bottom center
        margin_v=80,   # 80px from bottom edge
        outline=3, shadow=2,
        primary="&H00FFFFFF", back="&H80000000"
    )

    dialogues = []
    for entry in filtered:
        wrapped = _wrap_text_ass(entry["text"], max_chars=28)
        text = _escape_ass_text(wrapped)
        start_ts = _format_timestamp(entry["start"])
        end_ts = _format_timestamp(entry["end"])
        dialogues.append(
            f"Dialogue: 0,{start_ts},{end_ts},ClipCaption,,0,0,0,,"
            f"{{\\bord3\\shad2\\fn{CAPTION_FONT}}}{text}"
        )

    ass_content = _ass_header(style_line, "\n".join(dialogues))

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
    except IOError as e:
        raise RuntimeError(f"Failed to write ASS file: {e}") from e

    if progress_cb:
        progress_cb("Clip caption generated", 100)

    return out_path


def generate_commentary_ass(text: str, duration: float, out_path: str,
                            progress_cb: Optional[Callable[[str, float], None]] = None) -> str:
    """Generate white captions for TTS hook/insight segments."""
    if progress_cb:
        progress_cb("Wrapping commentary text...", 30)

    font_size = CAPTION_FONT_SIZE + 8
    wrapped = _wrap_text_ass(text, max_chars=26)

    if progress_cb:
        progress_cb("Generating ASS format...", 70)

    style_line = _make_style(
        "CommentaryCaption", CAPTION_FONT, font_size,
        alignment=2,   # bottom center (match ClipCaption placement)
        margin_v=80,   # 80px from bottom edge (same as ClipCaption)
        outline=3, shadow=2,
        primary="&H00FFFFFF",
        back="&H80000000"
    )

    start_ts = _format_timestamp(0.0)
    end_ts = _format_timestamp(duration)
    text_escaped = _escape_ass_text(wrapped)

    dialogue = (
        f"Dialogue: 0,{start_ts},{end_ts},CommentaryCaption,,0,0,0,,"
        f"{{\\bord3\\shad2\\fn{CAPTION_FONT}}}{text_escaped}"
    )

    ass_content = _ass_header(style_line, dialogue)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
    except IOError as e:
        raise RuntimeError(f"Failed to write ASS file: {e}") from e

    if progress_cb:
        progress_cb("Commentary caption generated", 100)

    return out_path


def generate_group_captions(
    transcript: list,
    source_clips: List[Dict[str, float]],
    narration_events: List[Dict[str, Any]],
    working_dir: str,
    group_idx: int,
    source_path: str,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, List[str]]:
    """
    Generate all captions for a single reel group.
    
    Returns dict with:
    - "clip_captions": list of paths (bottom zone, alignment=2, margin_v=80)
    - "narration_captions": list of paths (top zone, alignment=8, margin_v=60)
    - "has_existing_captions": list of bool per source_clip (from OCR)
    
    OCR Integration: Runs detect_existing_captions on source_clips.
    Skips clip caption generation where has_captions=True.
    """
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb(f"Group {group_idx+1}: Checking for existing burned-in captions...", 10)

    # OCR: detect existing captions on source clips
    caption_results = detect_existing_captions(
        video_path=source_path,
        clip_windows=[{"start": c["source_start"], "end": c["source_end"]} for c in source_clips],
        working_dir=str(working_dir / "ocr"),
        progress_cb=lambda msg, p: progress_cb(f"OCR: {msg}", 10 + p * 0.2) if progress_cb else None,
    )

    has_existing = [r.get("has_captions", False) for r in caption_results]
    skipped_count = sum(1 for h in has_existing if h)
    if skipped_count:
        print(f"[INFO] Group {group_idx}: Skipping clip caption generation for {skipped_count} clip(s) with existing captions")

    # Clip captions (bottom zone) - only for clips WITHOUT existing captions
    clip_caption_paths = []
    for i, clip in enumerate(source_clips):
        if has_existing[i]:
            clip_caption_paths.append(None)  # placeholder for skipped
            continue

        out_path = working_dir / f"group_{group_idx}_clip_caption_{i}.ass"

        if progress_cb:
            progress_cb(f"Group {group_idx+1}: Generating clip caption {i+1}/{len(source_clips)}...", 
                       30 + (i / len(source_clips)) * 30)

        generate_clip_ass(
            transcript,
            clip["source_start"],
            clip["source_end"],
            str(out_path),
            progress_cb=lambda msg, p: progress_cb(f"Clip {i+1} caption: {msg}", 30 + (i + p/100) / len(source_clips) * 30) if progress_cb else None,
        )
        clip_caption_paths.append(str(out_path))

    # Narration captions (top zone) - reel-relative timing
    narration_caption_paths = []
    for i, event in enumerate(narration_events):
        if not event.get("voice_id"):
            continue  # skip non-TTS events

        out_path = working_dir / f"group_{group_idx}_narr_caption_{i}.ass"

        if progress_cb:
            progress_cb(f"Group {group_idx+1}: Generating narration caption {i+1}/{len(narration_events)}...",
                       60 + (i / len(narration_events)) * 30)

        generate_narration_ass(
            event["text"],
            event["reel_end"] - event["reel_start"],
            str(out_path),
            progress_cb=lambda msg, p: progress_cb(f"Narr {i+1} caption: {msg}", 60 + (i + p/100) / len(narration_events) * 30) if progress_cb else None,
        )
        narration_caption_paths.append({
            "event_type": event["event_type"],
            "reel_start": event["reel_start"],
            "reel_end": event["reel_end"],
            "path": str(out_path),
        })

    if progress_cb:
        progress_cb(f"Group {group_idx+1}: All captions generated", 100)

    return {
        "clip_captions": clip_caption_paths,
        "narration_captions": narration_caption_paths,
        "has_existing_captions": has_existing,
    }


def generate_narration_ass(text: str, duration: float, out_path: str,
                           progress_cb: Optional[Callable[[str, float], None]] = None) -> str:
    """Generate narration caption — TOP zone (alignment=8, margin_v=60)."""
    if progress_cb:
        progress_cb("Wrapping narration text...", 30)

    font_size = CAPTION_FONT_SIZE + 8
    wrapped = _wrap_text_ass(text, max_chars=26)

    if progress_cb:
        progress_cb("Generating ASS format...", 70)

    style_line = _make_style(
        "NarrationCaption", CAPTION_FONT, font_size,
        alignment=8,   # top center
        margin_v=60,   # 60px from top edge
        outline=3, shadow=2,
        primary="&H00FFFFFF",
        back="&H80000000"
    )

    start_ts = _format_timestamp(0.0)
    end_ts = _format_timestamp(duration)
    text_escaped = _escape_ass_text(wrapped)

    dialogue = (
        f"Dialogue: 0,{start_ts},{end_ts},NarrationCaption,,0,0,0,,"
        f"{{\\bord3\\shad2\\fn{CAPTION_FONT}}}{text_escaped}"
    )

    ass_content = _ass_header(style_line, dialogue)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
    except IOError as e:
        raise RuntimeError(f"Failed to write ASS file: {e}") from e

    if progress_cb:
        progress_cb("Narration caption generated", 100)

    return out_path