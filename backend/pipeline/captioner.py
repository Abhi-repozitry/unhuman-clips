from backend.config import CAPTION_FONT
from typing import Callable, Optional
from backend.pipeline.sanitize import sanitize_text


# Caption sizes (9:16 portrait, 1080x1920)
CLIP_CAPTION_SIZE = 56       # Larger than before (was 48 default)
COMMENTARY_CAPTION_SIZE = 64  # Larger than before (was 48+8=56)

# Key words to highlight with color
KEY_WORDS = {
    "wait", "what", "why", "how", "never", "always", "secret", "hidden",
    "truth", "reveal", "shock", "insane", "crazy", "best", "worst",
    "first", "last", "ever", "impossible", "possible",
    "breakthrough", "discover", "invent", "create", "change",
    "amazing", "incredible", "unbelievable", "stunning", "remarkable",
    "win", "lose", "beat", "victory", "defeat", "champion",
    "dangerous", "risky", "safe", "protect", "save", "avoid",
    "love", "hate", "fear", "scared", "excited", "terrible", "beautiful",
    "shocked", "thrilled", "devastated", "hilarious", "intense",
    "literally", "actually", "honestly", "absolutely", "guaranteed",
    "not", "no", "yes", "wrong", "right", "stop", "go", "look", "watch",
    "boom", "bang", "pow", "wow", "whoa", "oh", "no way",
}

# Highlight color
HIGHLIGHT_COLOR = "&H00FFFF66"  # Warm yellow highlight


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


def _wrap_text_ass(text: str, max_chars: int = 24) -> str:
    """Wrap text for 9:16 portrait — narrower lines for readability. Was 28, tightened to 24 for larger font."""
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
                bold: int = 0, outline: int = 4, shadow: int = 3,
                primary: str = "&H00FFFFFF", back: str = "&H80000000") -> str:
    """Build a single ASS Style line.
    Alignment: 2=bottom center, 8=top center
    Increased outline from 3->4 and shadow from 2->3 for better readability.
    """
    return (f"Style: {name},{font},{size},{primary},&H00000008,"
            f"&H00000000,{back},{bold},0,0,0,100,100,0,0,"
            f"1,{outline},{shadow},{alignment},40,40,{margin_v},0")


def _highlight_key_words(text: str) -> str:
    """
    Wrap key words in ASS override tags to highlight them with a different color.
    Returns text with {\\c&Hxxxxxx&} tags around key words.
    Text is already ASS-escaped — do NOT call _escape_ass_text here.
    """
    words = text.split()
    result_parts = []
    for word in words:
        # Strip ASS tags for checking
        clean = word.replace("\\N", "").strip(".,!?;:'\"()[]{}")
        if clean.lower() in KEY_WORDS:
            result_parts.append(f"{{\\c{HIGHLIGHT_COLOR}}}{word}{{\\c}}")
        else:
            result_parts.append(word)
    return " ".join(result_parts)


def generate_clip_ass(transcript: list, clip_start: float, clip_end: float,
                      out_path: str,
                      progress_cb: Optional[Callable[[str, float], None]] = None,
                      start_time: float = 0.0) -> str:
    """Generate captions for original-clip segments — bottom-aligned with background box.
    Enhanced with larger font, key word highlighting, and better positioning."""
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
                "text": sanitize_text(entry["text"]),
            })

    # Use larger clip caption size
    style_line = _make_style(
        "ClipCaption", CAPTION_FONT, CLIP_CAPTION_SIZE,
        alignment=2,   # bottom center
        margin_v=100,  # 100px from bottom edge (more breathing room for mobile)
        outline=4, shadow=3,  # thicker outline for readability
        bold=1,  # bold for clip captions too
        primary="&H00FFFFFF", back="&H80000000"
    )

    dialogues = []
    for entry in filtered:
        escaped = _escape_ass_text(entry["text"])  # Escape BEFORE wrapping
        wrapped = _wrap_text_ass(escaped, max_chars=22)  # Wrap adds \N line breaks
        highlighted = _highlight_key_words(wrapped)
        start_ts = _format_timestamp(entry["start"] + start_time)
        end_ts = _format_timestamp(entry["end"] + start_time)
        dialogues.append(
            f"Dialogue: 0,{start_ts},{end_ts},ClipCaption,,0,0,0,,"
            f"{{\\bord4\\shad3\\b1\\fn{CAPTION_FONT}}}{highlighted}"
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
                            progress_cb: Optional[Callable[[str, float], None]] = None,
                            start_time: float = 0.0) -> str:
    """Generate commentary (hook/insight) captions — top center (alignment=8, margin_v=60), larger, with key word highlights."""
    if progress_cb:
        progress_cb("Wrapping commentary text...", 30)

    text = sanitize_text(text)
    escaped = _escape_ass_text(text)  # Escape BEFORE wrapping
    wrapped = _wrap_text_ass(escaped, max_chars=22)  # Wrap adds \N line breaks
    highlighted = _highlight_key_words(wrapped)

    if progress_cb:
        progress_cb("Generating ASS format...", 70)

    style_line = _make_style(
        "CommentaryCaption", CAPTION_FONT, COMMENTARY_CAPTION_SIZE,
        alignment=8,     # top center
        margin_v=60,     # 60px from top
        outline=4, shadow=3,
        bold=1,          # Bold for emphasis
        primary="&H00FFFFFF",
        back="&H80000000"
    )

    start_ts = _format_timestamp(start_time)
    end_ts = _format_timestamp(start_time + duration)
    text_escaped = highlighted

    dialogue = (
        f"Dialogue: 0,{start_ts},{end_ts},CommentaryCaption,,0,0,0,,"
        f"{{\\bord4\\shad3\\b1\\fn{CAPTION_FONT}}}{text_escaped}"
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


