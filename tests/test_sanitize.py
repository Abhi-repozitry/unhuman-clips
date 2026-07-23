"""Tests for backend.pipeline.sanitize — text sanitization for ASS/TTS/ffmpeg."""
from __future__ import annotations

from backend.pipeline.sanitize import sanitize_text


class TestSanitizeText:
    """Test sanitize_text() covers every sanitization rule."""

    def test_empty_string(self):
        assert sanitize_text("") == ""

    def test_none_like_empty(self):
        assert sanitize_text("") == ""

    def test_unicode_nfkc_normalization(self):
        # Full-width "A" → ASCII "A"
        assert sanitize_text("\uff21") == "A"

    def test_smart_quotes_to_ascii(self):
        assert sanitize_text("\u2018hello\u2019") == "'hello'"
        assert sanitize_text("\u201chello\u201d") == '"hello"'

    def test_em_dash_to_ascii(self):
        result = sanitize_text("before\u2014after")
        assert " - " in result
        assert "before" in result
        assert "after" in result

    def test_en_dash_to_ascii(self):
        result = sanitize_text("A\u2013B")
        assert " - " in result

    def test_ellipsis_replacement(self):
        assert sanitize_text("wait\u2026") == "wait..."

    def test_nbsp_replacement(self):
        assert sanitize_text("hello\u00a0world") == "hello world"

    def test_zero_width_space_removal(self):
        assert sanitize_text("hel\u200blo") == "hello"

    def test_zero_width_joiner_removal(self):
        assert sanitize_text("hel\u200dlo") == "hello"

    def test_banned_chars_removed(self):
        for ch in "/\\|*#_<>[]{}":
            result = sanitize_text(f"hello{ch}world")
            assert ch not in result
            assert "hello" in result
            assert "world" in result

    def test_double_hyphen_normalization(self):
        assert sanitize_text("well--that") == "well - that"

    def test_collapse_whitespace(self):
        assert sanitize_text("hello   world") == "hello world"
        assert sanitize_text("  hello  \n  world  ") == "hello world"

    def test_collapse_repeated_punctuation(self):
        assert sanitize_text("really,, no") == "really, no"
        assert sanitize_text("what?? yes") == "what? yes"
        assert sanitize_text("wait!!! ok") == "wait! ok"

    def test_strip_leading_trailing_punctuation(self):
        assert sanitize_text(", hello world:") == "hello world"
        assert sanitize_text(" - hello - ") == "hello"

    def test_normal_text_unchanged(self):
        text = "This is a normal sentence with numbers 123 and punctuation."
        assert sanitize_text(text) == text

    def test_mixed_issues(self):
        text = "  \u201cHello\u201d world\u2014it\u2019s a \u201ctest\u2026\u201d  "
        result = sanitize_text(text)
        assert '"' in result
        assert "'" in result
        assert "..." in result
        assert " - " in result
        assert result == result.strip()

    def test_banned_chars_replaced_with_space_not_concatenated(self):
        # "a#b" → "a b", not "ab"
        result = sanitize_text("a#b")
        assert result == "a b"

    def test_returns_string_type(self):
        result = sanitize_text("test")
        assert isinstance(result, str)

    def test_preserves_accented_chars(self):
        # Accented chars are NOT banned — only the specific set is banned
        result = sanitize_text("café résumé")
        assert "café" in result
        assert "résumé" in result
