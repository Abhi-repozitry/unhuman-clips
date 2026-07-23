from backend.pipeline.captioner import _escape_ass_text

cases = [
    ("line1\nline2", "simple newline"),
    ("hello", "no specials"),
    ("path \\foo\\bar", "backslash path"),
    ('a {b} c, d', "braces & comma"),
    ("first\np\\ath\nsecond", "everything"),
    ("", "empty"),
]

for text, label in cases:
    out = _escape_ass_text(text)
    print(f"[{label}] input={text!r}\n         output={out!r}\n")
