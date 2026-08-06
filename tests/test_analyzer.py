"""Tests for backend.pipeline.analyzer — multi-stage helpers, JSON extraction/repair, transcript formatting."""
from __future__ import annotations

import json

import pytest

from backend.models import SourceMetadata
from backend.pipeline.analyzer import (
    _compute_group_count_ceiling,
    _compute_group_count_floor,
    _compute_importance,
    _extract_json_object,
    _format_blocks_for_llm,
    _format_full_transcript,
    _identify_content,
    _normalize_clip_range,
    _rank_hook_candidates,
    _score_clip_as_hook,
    _try_repair_truncated_json,
    detect_content_type,
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
        score = _compute_importance(0.9, -10.0, False, False, False)
        assert score > 50.0

    def test_low_energy_low_score(self):
        score = _compute_importance(0.1, -40.0, False, False, False)
        assert score < 20.0

    def test_black_frame_penalizes(self):
        base = _compute_importance(0.5, -20.0, False, False, False)
        with_black = _compute_importance(0.5, -20.0, False, True, False)
        assert with_black < base

    def test_score_clamped_0_100(self):
        score = _compute_importance(1.0, 0.0, True, False, False)
        assert 0.0 <= score <= 100.0

    def test_silence_before_boosts(self):
        base = _compute_importance(0.5, None, False, False, False)
        with_silence = _compute_importance(0.5, None, True, False, False)
        assert with_silence > base


class TestComputeGroupCountCeiling:
    """Test group count ceiling based on source duration."""

    def test_short_video_returns_1(self):
        assert _compute_group_count_ceiling(90.0) == 1  # 1.5 min

    def test_5min_video_returns_1(self):
        assert _compute_group_count_ceiling(300.0) == 1  # 5 min

    def test_8min_video_returns_2(self):
        assert _compute_group_count_ceiling(480.0) == 2  # 8 min

    def test_15min_video_returns_6(self):
        assert _compute_group_count_ceiling(900.0) == 6  # 15 min

    def test_30min_video_returns_8(self):
        assert _compute_group_count_ceiling(1800.0) == 8  # 30 min

    def test_boundary_2min(self):
        assert _compute_group_count_ceiling(120.0) == 1  # exactly 2 min

    def test_boundary_5min(self):
        assert _compute_group_count_ceiling(300.0) == 1  # exactly 5 min

    def test_boundary_10min(self):
        assert _compute_group_count_ceiling(600.0) == 2  # exactly 10 min

    def test_boundary_25min(self):
        assert _compute_group_count_ceiling(1500.0) == 6  # exactly 25 min

    def test_boundary_35min(self):
        assert _compute_group_count_ceiling(2100.0) == 8  # exactly 35 min

    def test_40min_video_returns_10(self):
        assert _compute_group_count_ceiling(2400.0) == 10  # 40 min


class TestComputeGroupCountFloor:
    """Test group count floor based on source duration."""

    def test_short_video_returns_1(self):
        assert _compute_group_count_floor(90.0) == 1  # 1.5 min

    def test_5min_video_returns_1(self):
        assert _compute_group_count_floor(300.0) == 1  # 5 min

    def test_8min_video_returns_1(self):
        assert _compute_group_count_floor(480.0) == 1  # 8 min

    def test_15min_video_returns_3(self):
        assert _compute_group_count_floor(900.0) == 3  # 15 min

    def test_30min_video_returns_4(self):
        assert _compute_group_count_floor(1800.0) == 4  # 30 min

    def test_boundary_6min(self):
        assert _compute_group_count_floor(360.0) == 1  # exactly 6 min

    def test_boundary_10min(self):
        assert _compute_group_count_floor(600.0) == 1  # exactly 10 min

    def test_boundary_25min(self):
        assert _compute_group_count_floor(1500.0) == 3  # exactly 25 min

    def test_boundary_35min(self):
        assert _compute_group_count_floor(2100.0) == 4  # exactly 35 min

    def test_40min_video_returns_5(self):
        assert _compute_group_count_floor(2400.0) == 5  # 40 min


class TestFormatBlocksForLlm:
    """Test block formatting for LLM prompt."""

    def test_empty_blocks(self):
        assert _format_blocks_for_llm([]) == ""

    def test_includes_block_info(self):
        from backend.pipeline.analyzer import SemanticBlock
        blocks = [SemanticBlock(
            block_id=0, start=0.0, end=5.0, text="hello world",
            speech_energy=0.8, volume_db=-10.0, silence_before=False,
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
                          silence_before=False, black_frame=False,
                          freeze=False, importance=float(i*10), peak_offset=0.0)
            for i in range(5)
        ]
        result = _format_blocks_for_llm(blocks, top_n=2)
        assert "TOP-2" in result

    def test_top_n_excludes_lower_importance_blocks(self):
        from backend.pipeline.analyzer import SemanticBlock

        blocks = [
            SemanticBlock(
                block_id=i,
                start=float(i * 10),
                end=float(i * 10 + 5),
                text=f"block {i}",
                speech_energy=0.5,
                volume_db=None,
                silence_before=False,
                black_frame=False,
                freeze=False,
                importance=float(i * 10),
                peak_offset=0.0,
            )
            for i in range(4)
        ]

        result = _format_blocks_for_llm(blocks, top_n=2)

        assert "block 0" not in result
        assert "block 1" not in result
        assert "block 2" in result
        assert "block 3" in result


class TestIdentifier:
    @staticmethod
    def _blocks():
        from backend.pipeline.analyzer import SemanticBlock

        return [
            SemanticBlock(
                block_id=0,
                start=0.0,
                end=8.0,
                text="Contestant Maya enters the challenge.",
                speech_energy=0.8,
                volume_db=-12.0,
                silence_before=True,
                black_frame=False,
                freeze=False,
                importance=80.0,
                peak_offset=4.0,
            )
        ]

    def test_identifier_uses_confirmed_channel_metadata(self, monkeypatch):
        from backend.pipeline import analyzer as analyzer_module

        calls = []

        def fake_call_llm(*args, **kwargs):
            messages = args[0]
            calls.append((messages, kwargs))
            return json.dumps({
                "creator_name": "Guessed Creator",
                "content_format": "multi-contestant challenge",
                "detected_genre": "game_challenge",
                "structure": "multi_entity",
                "entity_names": ["Maya", "Maya", "Chris"],
                "hook_recommendation": "skip",
                "planning_notes": "Keep each contestant self-contained. Preserve each outcome.",
            })

        monkeypatch.setattr(analyzer_module, "_call_llm", fake_call_llm)
        identity = _identify_content(
            "Secret millionaire",
            "A contestant challenge",
            SourceMetadata(channel_name="KSI", channel_description="Challenge creator"),
            self._blocks(),
            None,
            None,
            [],
        )

        assert identity is not None
        assert identity.creator_name == "KSI"
        assert identity.entity_names == ["Maya", "Chris"]
        assert calls[0][1]["stage_name"] == "identifier"
        assert "Channel name: KSI" in calls[0][0][1]["content"]

    def test_identifier_failure_does_not_raise(self, monkeypatch):
        from backend.pipeline import analyzer as analyzer_module

        monkeypatch.setattr(analyzer_module, "_call_llm", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))

        assert _identify_content("Title", "", None, self._blocks(), None, None, []) is None

    def test_identifier_genre_can_corroborate_borderline_python_signal(self):
        assert detect_content_type("challenge", "", [], identifier_genre=None) == "general"
        assert detect_content_type("challenge", "", [], identifier_genre="game_challenge") == "game_challenge"


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


class TestSemanticBlockFields:
    """Test new SemanticBlock fields and summary_line output."""

    def test_question_flag_in_summary(self):
        from backend.pipeline.analyzer import SemanticBlock
        block = SemanticBlock(
            block_id=0, start=0.0, end=5.0, text="What happens next?",
            speech_energy=0.8, volume_db=-10.0, silence_before=False,
            black_frame=False, freeze=False, importance=75.0, peak_offset=2.5,
            has_question=True,
        )
        result = block.summary_line()
        assert "[Q]" in result

    def test_exclamation_flag_in_summary(self):
        from backend.pipeline.analyzer import SemanticBlock
        block = SemanticBlock(
            block_id=0, start=0.0, end=5.0, text="Unbelievable!",
            speech_energy=0.8, volume_db=-10.0, silence_before=False,
            black_frame=False, freeze=False, importance=75.0, peak_offset=2.5,
            has_exclamation=True,
        )
        result = block.summary_line()
        assert "[!]" in result

    def test_emphasis_flag_in_summary(self):
        from backend.pipeline.analyzer import SemanticBlock
        block = SemanticBlock(
            block_id=0, start=0.0, end=5.0, text="This is INCREDIBLE",
            speech_energy=0.8, volume_db=-10.0, silence_before=False,
            black_frame=False, freeze=False, importance=75.0, peak_offset=2.5,
            has_emphasis=True,
        )
        result = block.summary_line()
        assert "[CAPS]" in result

    def test_word_density_in_summary(self):
        from backend.pipeline.analyzer import SemanticBlock
        block = SemanticBlock(
            block_id=0, start=0.0, end=5.0, text="fast talking here",
            speech_energy=0.8, volume_db=-10.0, silence_before=False,
            black_frame=False, freeze=False, importance=75.0, peak_offset=2.5,
            word_density=3.2,
        )
        result = block.summary_line()
        assert "wps=3.2" in result

    def test_multiple_flags_combined(self):
        from backend.pipeline.analyzer import SemanticBlock
        block = SemanticBlock(
            block_id=0, start=0.0, end=5.0, text="WHAT?!",
            speech_energy=0.8, volume_db=-10.0, silence_before=True,
            black_frame=False, freeze=False, importance=75.0, peak_offset=2.5,
            has_question=True, has_exclamation=True, has_emphasis=True, word_density=2.0,
        )
        result = block.summary_line()
        assert "Q" in result
        assert "!" in result
        assert "CAPS" in result
        assert "SILENCE_BEFORE" in result
        assert "wps=2.0" in result


class TestScoreClipAsHook:
    """Test hook clip scoring."""

    def _make_block(self, **kwargs):
        from backend.pipeline.analyzer import SemanticBlock
        defaults = dict(
            block_id=0, start=0.0, end=5.0, text="test",
            speech_energy=0.5, volume_db=None, silence_before=False,
            black_frame=False, freeze=False, importance=50.0, peak_offset=0.0,
        )
        defaults.update(kwargs)
        return SemanticBlock(**defaults)

    def test_question_block_scores_higher(self):
        clip = {"source_start": 0.0, "source_end": 5.0}
        block_no_q = self._make_block(has_question=False)
        block_with_q = self._make_block(has_question=True)
        assert _score_clip_as_hook(clip, [block_with_q]) > _score_clip_as_hook(clip, [block_no_q])

    def test_silence_before_boosts(self):
        clip = {"source_start": 0.0, "source_end": 5.0}
        block_no_silence = self._make_block(silence_before=False)
        block_with_silence = self._make_block(silence_before=True)
        assert _score_clip_as_hook(clip, [block_with_silence]) > _score_clip_as_hook(clip, [block_no_silence])

    def test_shorter_clip_scores_higher(self):
        block = self._make_block()
        clip_short = {"source_start": 0.0, "source_end": 3.0}
        clip_long = {"source_start": 0.0, "source_end": 20.0}
        assert _score_clip_as_hook(clip_short, [block]) > _score_clip_as_hook(clip_long, [block])

    def test_non_start_position_scores_higher(self):
        from backend.pipeline.analyzer import SemanticBlock
        # Block spans 5-15s so it overlaps both clips
        block = SemanticBlock(
            block_id=0, start=5.0, end=15.0, text="test",
            speech_energy=0.5, volume_db=None, silence_before=False,
            black_frame=False, freeze=False, importance=50.0, peak_offset=0.0,
        )
        # Clip at start (0-5s) doesn't overlap the block
        clip_start = {"source_start": 0.0, "source_end": 5.0}
        # Clip later (5-15s) overlaps the block and starts after 3.0s
        clip_later = {"source_start": 5.0, "source_end": 15.0}
        score_start = _score_clip_as_hook(clip_start, [block])
        score_later = _score_clip_as_hook(clip_later, [block])
        assert score_later > score_start

    def test_no_blocks_returns_zero(self):
        clip = {"source_start": 0.0, "source_end": 5.0}
        assert _score_clip_as_hook(clip, []) == 0.0


class TestRankHookCandidates:
    """Test hook clip re-ranking."""

    def test_swaps_to_better_hook(self):
        from backend.pipeline.analyzer import SemanticBlock
        blocks = [
            SemanticBlock(
                block_id=0, start=0.0, end=5.0, text="boring intro",
                speech_energy=0.3, volume_db=None, silence_before=False,
                black_frame=False, freeze=False, importance=30.0, peak_offset=0.0,
            ),
            SemanticBlock(
                block_id=1, start=10.0, end=15.0, text="What happens next?!",
                speech_energy=0.9, volume_db=-10.0, silence_before=True,
                black_frame=False, freeze=False, importance=85.0, peak_offset=2.0,
                has_question=True, has_exclamation=True,
            ),
        ]
        groups = [{
            "group_index": 0,
            "source_clips": [
                {"source_start": 0.0, "source_end": 5.0, "is_hook_clip": True},
                {"source_start": 10.0, "source_end": 15.0, "is_hook_clip": False},
            ],
        }]
        swaps = _rank_hook_candidates(groups, blocks)
        assert swaps == 1
        assert groups[0]["source_clips"][0]["is_hook_clip"] is False
        assert groups[0]["source_clips"][1]["is_hook_clip"] is True

    def test_no_swap_when_current_is_best(self):
        from backend.pipeline.analyzer import SemanticBlock
        blocks = [
            SemanticBlock(
                block_id=0, start=0.0, end=5.0, text="Amazing hook?!",
                speech_energy=0.9, volume_db=-10.0, silence_before=True,
                black_frame=False, freeze=False, importance=90.0, peak_offset=1.0,
                has_question=True, has_exclamation=True,
            ),
            SemanticBlock(
                block_id=1, start=10.0, end=15.0, text="boring middle",
                speech_energy=0.3, volume_db=None, silence_before=False,
                black_frame=False, freeze=False, importance=30.0, peak_offset=0.0,
            ),
        ]
        groups = [{
            "group_index": 0,
            "source_clips": [
                {"source_start": 0.0, "source_end": 5.0, "is_hook_clip": True},
                {"source_start": 10.0, "source_end": 15.0, "is_hook_clip": False},
            ],
        }]
        swaps = _rank_hook_candidates(groups, blocks)
        assert swaps == 0
        assert groups[0]["source_clips"][0]["is_hook_clip"] is True

    def test_single_clip_group_skipped(self):
        groups = [{
            "group_index": 0,
            "source_clips": [
                {"source_start": 0.0, "source_end": 5.0, "is_hook_clip": True},
            ],
        }]
        swaps = _rank_hook_candidates(groups, [])
        assert swaps == 0

    def test_no_hook_clip_in_group(self):
        groups = [{
            "group_index": 0,
            "source_clips": [
                {"source_start": 0.0, "source_end": 5.0, "is_hook_clip": False},
                {"source_start": 10.0, "source_end": 15.0, "is_hook_clip": False},
            ],
        }]
        swaps = _rank_hook_candidates(groups, [])
        assert swaps == 0

    def test_empty_groups_list(self):
        swaps = _rank_hook_candidates([], [])
        assert swaps == 0

    def test_empty_clip_list_in_group(self):
        groups = [{"group_index": 0, "source_clips": []}]
        swaps = _rank_hook_candidates(groups, [])
        assert swaps == 0
