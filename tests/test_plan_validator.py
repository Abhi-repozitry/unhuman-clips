"""Tests for backend.pipeline.plan_validator — deterministic validation and repair."""
from __future__ import annotations

import json

import pytest

from backend.models import ReelPlan
from backend.pipeline.plan_validator import (
    deduplicate_groups,
    finalize_edit,
    remove_overlaps,
    repair_json,
    validate_clip_bounds,
    validate_narration,
    verify_captions,
    verify_duration,
)


class TestRepairJson:
    """Test JSON repair for LLM output."""

    def test_valid_json_passthrough(self):
        data = '{"key": "value"}'
        assert repair_json(data) == data

    def test_fenced_json(self):
        data = '{"key": "value"}'
        fenced = f"```json\n{data}\n```"
        assert repair_json(fenced) == data

    def test_missing_closing_brace(self):
        truncated = '{"key": "value"'
        repaired = repair_json(truncated)
        assert repaired
        parsed = json.loads(repaired)
        assert parsed["key"] == "value"

    def test_trailing_comma(self):
        data = '{"key": "value",}'
        repaired = repair_json(data)
        assert repaired
        parsed = json.loads(repaired)
        assert "key" in parsed

    def test_empty_input(self):
        assert repair_json("") == ""


class TestValidateClipBounds:
    """Test clip bounds clamping."""

    def test_clamps_to_source_duration(self):
        groups = [{"source_clips": [{"source_start": -5.0, "source_end": 200.0}]}]
        adjusted = validate_clip_bounds(groups, source_duration=100.0)
        assert adjusted > 0
        clip = groups[0]["source_clips"][0]
        assert clip["source_start"] >= 0.0
        assert clip["source_end"] <= 100.0

    def test_enforces_minimum_duration(self):
        groups = [{"source_clips": [{"source_start": 10.0, "source_end": 11.0}]}]
        validate_clip_bounds(groups, source_duration=60.0, min_clip_duration=3.0)
        clip = groups[0]["source_clips"][0]
        assert clip["source_end"] - clip["source_start"] >= 3.0

    def test_valid_clip_unchanged(self):
        groups = [{"source_clips": [{"source_start": 10.0, "source_end": 20.0}]}]
        validate_clip_bounds(groups, source_duration=60.0)
        clip = groups[0]["source_clips"][0]
        assert clip["source_start"] == 10.0
        assert clip["source_end"] == 20.0


class TestRemoveOverlaps:
    """Test overlap detection and removal."""

    def test_no_overlap(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 5.0},
            {"source_start": 10.0, "source_end": 15.0},
        ]}]
        removed = remove_overlaps(groups)
        assert removed == 0
        assert len(groups[0]["source_clips"]) == 2

    def test_overlap_keeps_longer(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 10.0},
            {"source_start": 5.0, "source_end": 7.0},
        ]}]
        removed = remove_overlaps(groups)
        assert removed > 0
        assert len(groups[0]["source_clips"]) == 1
        assert groups[0]["source_clips"][0]["source_end"] == 10.0


class TestValidateNarration:
    """Test narration validation."""

    def test_hook_at_nonzero_corrected(self):
        groups = [{"estimated_duration_seconds": 100, "narration_events": [
            {"event_type": "hook", "reel_start": 5.0, "reel_end": 8.0, "text": "test"},
        ]}]
        validate_narration(groups)
        assert groups[0]["narration_events"][0]["reel_start"] == 0.0

    def test_duplicate_hook_converted(self):
        groups = [{"estimated_duration_seconds": 100, "narration_events": [
            {"event_type": "hook", "reel_start": 0.0, "reel_end": 3.0, "text": "first"},
            {"event_type": "hook", "reel_start": 10.0, "reel_end": 13.0, "text": "second"},
        ]}]
        validate_narration(groups)
        assert groups[0]["narration_events"][0]["event_type"] == "hook"
        assert groups[0]["narration_events"][1]["event_type"] == "commentary"


class TestDeduplicateGroups:
    """Test group deduplication."""

    def test_no_duplicates(self):
        groups = [
            {"source_clips": [{"source_start": 0.0, "source_end": 5.0}]},
            {"source_clips": [{"source_start": 10.0, "source_end": 15.0}]},
        ]
        result = deduplicate_groups(groups)
        assert len(result) == 2

    def test_exact_duplicate_removed(self):
        groups = [
            {"source_clips": [{"source_start": 0.0, "source_end": 5.0}]},
            {"source_clips": [{"source_start": 0.0, "source_end": 5.0}]},
        ]
        result = deduplicate_groups(groups)
        assert len(result) == 1

    def test_empty_clips_pruned(self):
        groups = [
            {"source_clips": []},
            {"source_clips": [{"source_start": 0.0, "source_end": 5.0}]},
        ]
        result = deduplicate_groups(groups)
        assert len(result) == 1


class TestFinalizeEdit:
    """Test the full validation pipeline."""

    def test_valid_plan_passes(self, sample_reel_plan_dict):
        plan = finalize_edit(sample_reel_plan_dict, source_duration=60.0)
        assert isinstance(plan, ReelPlan)
        assert len(plan.reel_groups) > 0

    def test_empty_groups_raises(self):
        with pytest.raises(RuntimeError, match="No reel_groups"):
            finalize_edit({"reel_groups": []}, source_duration=60.0)

    def test_missing_groups_key_raises(self):
        with pytest.raises(RuntimeError, match="No reel_groups"):
            finalize_edit({}, source_duration=60.0)

    def test_preserves_ranked_segments(self, sample_reel_plan_dict):
        sample_reel_plan_dict["ranked_segments"] = [
            {"segment_id": 0, "score": 90, "reason": "Strong opener"}
        ]
        plan = finalize_edit(sample_reel_plan_dict, source_duration=60.0)
        assert len(plan.ranked_segments) == 1
        assert plan.ranked_segments[0].score == 90

    def test_preserves_explanations(self, sample_reel_plan_dict):
        sample_reel_plan_dict["explanations"] = ["Test explanation"]
        plan = finalize_edit(sample_reel_plan_dict, source_duration=60.0)
        assert len(plan.explanations) == 1
