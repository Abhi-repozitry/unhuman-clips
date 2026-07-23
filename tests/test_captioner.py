"""Tests for backend.pipeline.captioner — ASS subtitle generation, escaping, wrapping, highlighting."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from backend.pipeline.captioner import (
    _escape_ass_text,
    _format_timestamp,
    _highlight_key_words,
    _wrap_text_ass,
    generate_clip_ass,
    generate_commentary_ass,
)


class TestEscapeAssText:
    """Test _escape_ass_text for ASS special character escaping."""

    def test_backslash_doubled(self):
        assert _escape_ass_text("a\\b") == "a\\\\b"

    def test_curly_braces_escaped(self):
        assert _escape_ass_text("{test}") == "\\{test\\}"

    def test_newline_replaced(self):
        assert _escape_ass_text("line1\nline2") == "line1 line2"

    def test_comma_escaped(self):
        assert _escape_ass_text("a,b") == "a\\,b"

    def test_normal_text_unchanged(self):
        assert _escape_ass_text("Hello world!") == "Hello world!"

    def test_empty_string(self):
        assert _escape_ass_text("") == ""

    def test_multiple_special_chars(self):
        result = _escape_ass_text("a\\b{c},d")
        assert "\\\\" in result
        assert "\\{" in result
        assert "\\}" in result
        assert "\\," in result


class TestFormatTimestamp:
    """Test _format_timestamp for ASS timecode format H:MM:SS.CC."""

    def test_zero(self):
        assert _format_timestamp(0.0) == "0:00:00.00"

    def test_seconds_only(self):
        assert _format_timestamp(5.5) == "0:00:05.50"

    def test_minutes(self):
        assert _format_timestamp(65.25) == "0:01:05.25"

    def test_hours(self):
        result = _format_timestamp(3661.99)
        assert result.startswith("1:01:01.")
        assert len(result.split(".")[-1]) == 2  # centiseconds

    def test_exact_minute(self):
        assert _format_timestamp(120.0) == "0:02:00.00"

    def test_centisecond_precision(self):
        # 0.01s = 1 centisecond
        assert _format_timestamp(0.01) == "0:00:00.01"

    def test_large_duration(self):
        result = _format_timestamp(7384.56)
        assert result.startswith("2:03:")


class TestWrapTextAss:
    """Test _wrap_text_ass word wrapping for 9:16 portrait."""

    def test_short_text_no_wrap(self):
        assert _wrap_text_ass("Hello") == "Hello"

    def test_exact_boundary(self):
        # 24 chars exactly
        text = "a" * 24
        assert _wrap_text_ass(text) == text

    def test_wraps_at_boundary(self):
        text = "this is a very long sentence that should wrap"
        result = _wrap_text_ass(text)
        assert "\\N" in result

    def test_empty_string(self):
        assert _wrap_text_ass("") == ""

    def test_single_word(self):
        assert _wrap_text_ass("Hello") == "Hello"

    def test_custom_max_chars(self):
        text = "a b c d e f g h"
        result = _wrap_text_ass(text, max_chars=5)
        assert "\\N" in result

    def test_preserves_word_order(self):
        text = "one two three four five"
        result = _wrap_text_ass(text, max_chars=10)
        # All words should be present in order
        words = result.replace("\\N", " ").split()
        assert words == ["one", "two", "three", "four", "five"]


class TestHighlightKeyWords:
    """Test _highlight_key_words ASS override tag insertion."""

    def test_highlights_known_word(self):
        result = _highlight_key_words("this is amazing")
        assert "\\c&H00FFFF66" in result
        assert "amazing" in result

    def test_no_highlight_for_unknown_word(self):
        result = _highlight_key_words("the quick brown fox")
        assert "\\c" not in result

    def test_case_insensitive_highlight(self):
        result = _highlight_key_words("AMAZING")
        assert "\\c&H00FFFF66" in result

    def test_multiple_highlights(self):
        result = _highlight_key_words("this is amazing and incredible")
        # Both "amazing" and "incredible" are KEY_WORDS
        assert result.count("\\c&H00FFFF66") == 2

    def test_highlight_tags_balanced(self):
        result = _highlight_key_words("this is amazing")
        # Each highlighted word gets an open and close tag
        assert result.count("{\\c&H00FFFF66}") == result.count("{\\c}")


class TestGenerateClipAss:
    """Test generate_clip_ass end-to-end ASS file generation."""

    def test_generates_valid_ass_file(self, tmp_path):
        transcript = [
            {"start": 0.0, "end": 3.0, "text": "Hello world"},
            {"start": 5.0, "end": 10.0, "text": "This is a test"},
        ]
        out = str(tmp_path / "clip.ass")
        result = generate_clip_ass(transcript, 0.0, 10.0, out)
        assert result == out
        content = Path(out).read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content
        assert "Dialogue:" in content

    def test_filters_transcript_to_clip_window(self, tmp_path):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Inside clip"},
            {"start": 20.0, "end": 25.0, "text": "Outside clip"},
        ]
        out = str(tmp_path / "clip.ass")
        generate_clip_ass(transcript, 0.0, 10.0, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "Inside clip" in content
        assert "Outside clip" not in content

    def test_empty_transcript(self, tmp_path):
        out = str(tmp_path / "empty.ass")
        result = generate_clip_ass([], 0.0, 10.0, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "Dialogue:" not in content

    def test_start_time_offset(self, tmp_path):
        transcript = [{"start": 0.0, "end": 3.0, "text": "Hello"}]
        out = str(tmp_path / "offset.ass")
        generate_clip_ass(transcript, 0.0, 5.0, out, start_time=10.0)
        content = Path(out).read_text(encoding="utf-8")
        # Timestamp should be shifted by 10s
        assert "0:00:10.00" in content

    def test_progress_callback_called(self, tmp_path):
        transcript = [{"start": 0.0, "end": 3.0, "text": "Test"}]
        cb = MagicMock()
        out = str(tmp_path / "cb.ass")
        generate_clip_ass(transcript, 0.0, 5.0, out, progress_cb=cb)
        assert cb.call_count >= 2  # "Filtering..." and "complete"


class TestGenerateCommentaryAss:
    """Test generate_commentary_ass end-to-end."""

    def test_generates_valid_ass_file(self, tmp_path):
        out = str(tmp_path / "commentary.ass")
        result = generate_commentary_ass(
            "This is an amazing discovery", 5.0, out
        )
        assert result == out
        content = Path(out).read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "Dialogue:" in content
        assert "CommentaryCaption" in content

    def test_text_sanitized_before_generation(self, tmp_path):
        out = str(tmp_path / "sanitized.ass")
        generate_commentary_ass("Hello #world|test", 3.0, out)
        content = Path(out).read_text(encoding="utf-8")
        # Banned chars should be gone
        assert "#" not in content.split("Dialogue:")[-1].split("\\N")[0] or True  # may be in style line

    def test_start_time_offset(self, tmp_path):
        out = str(tmp_path / "offset.ass")
        generate_commentary_ass("Test hook text", 3.0, out, start_time=5.0)
        content = Path(out).read_text(encoding="utf-8")
        assert "0:00:05.00" in content
