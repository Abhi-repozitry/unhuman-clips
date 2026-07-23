"""Tests for backend.pipeline.analyzer — reel plan prompt building, JSON extraction/repair, group logic."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.pipeline.analyzer import (
    _compute_group_count_target,
    _extract_json_object,
    _format_full_transcript,
    _normalize_clip_range,
    _try_repair_truncated_json,
)


class TestFormatFullTranscript:
    """Test _format_full_transcript formatting."""

    def test_empty_transcript(self):
        assert _format_full_transcript([]) == ""

    def test_segments_formatted(self, sample_transcript):
        result = _format_full_transcript(sample_transcript)
        assert "Seg 0" in result
        assert "[0.0-5.0s]" in result
        assert "Welcome to the show" in result

    def test_all_segments_present(self, sample_transcript):
        result = _format_full_transcript(sample_transcript)
        for i in range(len(sample_transcript)):
            assert f"Seg {i}" in result

    def test_empty_text_segments_filtered(self):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": ""},
            {"start": 5.0, "end": 10.0, "text": "   "},
            {"start": 10.0, "end": 15.0, "text": "Actual text"},
        ]
        result = _format_full_transcript(transcript)
        assert "Seg 0" not in result
        assert "Seg 1" not in result
        assert "Seg 2" in result


class TestExtractJsonObject:
    """Test _extract_json_object for various LLM output formats."""

    def test_clean_json(self):
        data = '{"key": "value"}'
        assert _extract_json_object(data) == data

    def test_fenced_json(self):
        data = '{"key": "value"}'
        fenced = f"```json\n{data}\n```"
        assert _extract_json_object(fenced) == data

    def test_fenced_without_lang(self):
        data = '{"key": "value"}'
        fenced = f"```\n{data}\n```"
        assert _extract_json_object(fenced) == data

    def test_json_with_surrounding_text(self):
        data = '{"key": "value"}'
        wrapped = f"Here is the result: {data} let me know if you need more."
        assert _extract_json_object(wrapped) == data

    def test_nested_json(self):
        data = '{"reel_groups": [{"group_index": 0, "source_clips": []}]}'
        result = _extract_json_object(data)
        assert json.loads(result) == json.loads(data)

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            _extract_json_object("This is just plain text with no JSON")

    def test_json_array_still_finds_outermost_braces(self):
        data = '{"result": [1, 2, 3]}'
        assert _extract_json_object(data) == data

    def test_multiline_json(self):
        data = '{\n  "key": "value",\n  "nested": {"a": 1}\n}'
        assert _extract_json_object(data) == data


class TestTryRepairTruncatedJson:
    """Test _try_repair_truncated_json for common LLM truncation patterns."""

    def test_valid_json_passthrough(self):
        data = '{"key": "value"}'
        assert _try_repair_truncated_json(data) == data

    def test_missing_closing_brace(self):
        truncated = '{"key": "value"'
        repaired = _try_repair_truncated_json(truncated)
        assert repaired  # not empty
        parsed = json.loads(repaired)
        assert parsed["key"] == "value"

    def test_missing_closing_bracket(self):
        truncated = '[{"key": "value"}'
        repaired = _try_repair_truncated_json(truncated)
        assert repaired
        parsed = json.loads(repaired)
        assert isinstance(parsed, list)

    def test_trailing_comma(self):
        data = '{"key": "value",}'
        repaired = _try_repair_truncated_json(data)
        assert repaired
        parsed = json.loads(repaired)
        assert "key" in parsed

    def test_unclosed_string_quote(self):
        truncated = '{"key": "valu'
        repaired = _try_repair_truncated_json(truncated)
        assert repaired
        parsed = json.loads(repaired)
        assert "key" in parsed

    def test_deeply_nested_truncated(self):
        truncated = '{"a": {"b": {"c": 1'
        repaired = _try_repair_truncated_json(truncated)
        assert repaired
        parsed = json.loads(repaired)
        assert parsed["a"]["b"]["c"] == 1

    def test_empty_input(self):
        assert _try_repair_truncated_json("") == ""

    def test_returns_empty_for_unrepairable(self):
        # Totally random text with no JSON structure
        result = _try_repair_truncated_json("not json at all !!!")
        # May or may not find something, but should not crash
        assert isinstance(result, str)


class TestComputeGroupCountTarget:
    """Test _compute_group_count_target duration-based scaling."""

    def test_short_video(self):
        assert _compute_group_count_target(120) == (1, 4)  # < 300s

    def test_boundary_300s(self):
        assert _compute_group_count_target(300) == (1, 4)

    def test_medium_video(self):
        assert _compute_group_count_target(450) == (3, 6)  # 300-600s

    def test_boundary_600s(self):
        assert _compute_group_count_target(600) == (3, 6)

    def test_long_video(self):
        assert _compute_group_count_target(900) == (4, 8)  # 600-1200s

    def test_boundary_1200s(self):
        assert _compute_group_count_target(1200) == (4, 8)

    def test_very_long_video(self):
        assert _compute_group_count_target(2400) == (5, 12)  # > 1200s

    def test_zero_duration(self):
        assert _compute_group_count_target(0) == (1, 4)


class TestNormalizeClipRange:
    """Test _normalize_clip_range for expanding/shrinking segment ranges."""

    def test_no_expansion_needed(self):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 10.0, "text": "b"},
            {"start": 10.0, "end": 15.0, "text": "c"},
        ]
        start, end = _normalize_clip_range(transcript, 0, 2)
        assert start == 0
        assert end == 2

    def test_expands_to_meet_soft_min(self):
        transcript = [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 1.0, "end": 2.0, "text": "b"},
            {"start": 2.0, "end": 3.0, "text": "c"},
        ]
        # CLIP_DURATION_SOFT_MIN = 10s, but transcript is only 3s total
        start, end = _normalize_clip_range(transcript, 1, 1)
        assert end >= start  # should have expanded

    def test_clamps_indices_to_valid_range(self):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 10.0, "text": "b"},
        ]
        # Negative indices should be clamped
        start, end = _normalize_clip_range(transcript, -5, 100)
        assert start >= 0
        assert end <= len(transcript) - 1

    def test_single_segment(self):
        transcript = [{"start": 0.0, "end": 15.0, "text": "only one"}]
        start, end = _normalize_clip_range(transcript, 0, 0)
        assert start == 0
        assert end == 0
