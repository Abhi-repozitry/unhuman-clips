from backend.config import CAPTION_FONT, CAPTION_FONT_SIZE
from typing import Callable, Optional


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
        # Use \bord and \shad override tags in case style gets overridden; also add \fn for font consistency
        # Add a semi-transparent background box via \3c&H00000000\3a&H80 (black outline with alpha)
        # and \1c for primary color, \3c for outline color = black so border acts as background
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
        alignment=5,   # center; safe for hook/insight cards and portrait clips
        margin_v=0,
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
