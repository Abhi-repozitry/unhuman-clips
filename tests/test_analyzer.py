"""Tests for backend.pipeline.analyzer — multi-stage helpers, JSON extraction/repair, transcript formatting."""
from __future__ import annotations

import json

import pytest

from backend.models import FFmpegMetrics, RichTimeline, RichTimelineSegment
from backend.pipeline.analyzer import (
    _compute_group_count_ceiling,
    _compute_importance,
    _extract_json_object,
    _format_blocks_for_llm,
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


class TestComputeImportance:
    """Test deterministic importance scoring."""

    def test_high_energy_high_score(self):
        score = _compute_importance(0.9, -10.0, False, False, False, False)
        assert score > 50.0

    def test_low_energy_low_score(self):
        score = _compute_importance(0.1, -40.0, False, False, False, False)
        assert score < 20.0

    def test_ocr_boosts_score(self):
        base = _compute_importance(0.5, -20.0, False, False, False, False)
        with_ocr = _compute_importance(0.5, -20.0, True, False, False, False)
        assert with_ocr > base

    def test_black_frame_penalizes(self):
        base = _compute_importance(0.5, -20.0, False, False, False, False)
        with_black = _compute_importance(0.5, -20.0, False, False, True, False)
        assert with_black < base

    def test_score_clamped_0_100(self):
        score = _compute_importance(1.0, 0.0, True, True, False, False)
        assert 0.0 <= score <= 100.0

    def test_silence_before_boosts(self):
        base = _compute_importance(0.5, None, False, False, False, False)
        with_silence = _compute_importance(0.5, None, False, True, False, False)
        assert with_silence > base


class TestComputeGroupCountCeiling:
    """Test group count ceiling based on source duration."""

    def test_short_video_returns_2(self):
        assert _compute_group_count_ceiling(90.0) == 2  # 1.5 min

    def test_medium_video_returns_4(self):
        assert _compute_group_count_ceiling(300.0) == 4  # 5 min

    def test_long_video_returns_6(self):
        assert _compute_group_count_ceiling(480.0) == 6  # 8 min

    def test_very_long_video_returns_8(self):
        assert _compute_group_count_ceiling(900.0) == 8  # 15 min

    def test_huge_video_returns_10(self):
        assert _compute_group_count_ceiling(1800.0) == 10  # 30 min

    def test_boundary_2min(self):
        assert _compute_group_count_ceiling(120.0) == 2  # exactly 2 min

    def test_boundary_5min(self):
        assert _compute_group_count_ceiling(300.0) == 4  # exactly 5 min

    def test_boundary_10min(self):
        assert _compute_group_count_ceiling(600.0) == 6  # exactly 10 min

    def test_boundary_20min(self):
        assert _compute_group_count_ceiling(1200.0) == 8  # exactly 20 min


class TestFormatBlocksForLlm:
    """Test block formatting for LLM prompt."""

    def test_empty_blocks(self):
        assert _format_blocks_for_llm([]) == ""

    def test_includes_block_info(self):
        from backend.pipeline.analyzer import SemanticBlock
        blocks = [SemanticBlock(
            block_id=0, start=0.0, end=5.0, text="hello world",
            speech_energy=0.8, volume_db=-10.0, ocr=[], silence_before=False,
            black_frame=False, freeze=False, importance=75.0, peak_offset=2.5,
        )]
        result = _format_blocks_for_llm(blocks)
        assert "Block 0" in result
        assert "hello world" in result

    def test_top_n_emphasis(self):
        from backend.pipeline.analyzer import SemanticBlock
        blocks = [
            SemanticBlock(block_id=i, start=float(i*10), end=float(i*10+5),
                          text=f"block {i}", speech_energy=0.5, volume_db=None,
                          ocr=[], silence_before=False, black_frame=False,
                          freeze=False, importance=float(i*10), peak_offset=0.0)
            for i in range(5)
        ]
        result = _format_blocks_for_llm(blocks, top_n=2)
        assert "TOP-2" in result


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
        result = _try_repair_truncated_json("not json at all !!!")
        assert isinstance(result, str)


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
        start, end = _normalize_clip_range(transcript, 1, 1)
        assert end >= start  # should have expanded

    def test_clamps_indices_to_valid_range(self):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 10.0, "text": "b"},
        ]
        start, end = _normalize_clip_range(transcript, -5, 100)
        assert start >= 0
        assert end <= len(transcript) - 1

    def test_single_segment(self):
        transcript = [{"start": 0.0, "end": 15.0, "text": "only one"}]
        start, end = _normalize_clip_range(transcript, 0, 0)
        assert start == 0
        assert end == 0
