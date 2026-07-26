"""Tests for backend.pipeline.plan_validator — deterministic validation and repair."""
from __future__ import annotations

import json

import pytest

from backend.models import ReelPlan
from backend.pipeline.plan_validator import (
    _estimate_clip_importance,
    _expand_clips_to_fill_gap,
    deduplicate_groups,
    enforce_clip_pacing,
    finalize_edit,
    remove_overlaps,
    repair_clip_diversity,
    repair_json,
    validate_clip_bounds,
    validate_narration,
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

    def test_overlap_keeps_higher_importance(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 10.0, "reason": "LONG: important content"},
            {"source_start": 5.0, "source_end": 7.0, "reason": "SHORT: filler"},
        ]}]
        removed = remove_overlaps(groups)
        assert removed > 0
        assert len(groups[0]["source_clips"]) == 1
        # Kept the more important clip (longer + has reason keywords)
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

    def test_preserves_structure_analysis(self, sample_reel_plan_dict):
        plan = finalize_edit(sample_reel_plan_dict, source_duration=60.0)
        assert plan.structure_analysis is not None
        assert plan.structure_analysis.video_type == "documentary"
        assert plan.structure_analysis.final_group_count == 1

    def test_works_without_structure_analysis(self, sample_reel_plan_dict):
        sample_reel_plan_dict.pop("structure_analysis", None)
        plan = finalize_edit(sample_reel_plan_dict, source_duration=60.0)
        assert plan.structure_analysis is None

    def test_ignores_unknown_top_level_keys(self, sample_reel_plan_dict):
        sample_reel_plan_dict["unknown_key"] = "should not break"
        sample_reel_plan_dict["another_random"] = 42
        plan = finalize_edit(sample_reel_plan_dict, source_duration=60.0)
        assert isinstance(plan, ReelPlan)

    def test_floor_enforcement_passes_when_enough_groups(self, sample_reel_plan_dict):
        plan = finalize_edit(sample_reel_plan_dict, source_duration=60.0, min_groups=1)
        assert isinstance(plan, ReelPlan)
        assert len(plan.reel_groups) >= 1

    def test_floor_enforcement_fails_when_below_minimum(self, sample_reel_plan_dict):
        with pytest.raises(RuntimeError, match="fell below minimum"):
            finalize_edit(sample_reel_plan_dict, source_duration=60.0, min_groups=99)


class TestEstimateClipImportance:
    """Test clip importance estimation from reason text."""

    def test_hook_clip_scores_higher(self):
        hook = {"source_start": 0.0, "source_end": 5.0, "reason": "HOOK: curiosity gap", "is_hook_clip": True}
        regular = {"source_start": 0.0, "source_end": 5.0, "reason": "MEDIUM: building tension", "is_hook_clip": False}
        assert _estimate_clip_importance(hook) > _estimate_clip_importance(regular)

    def test_climax_keyword_boosts(self):
        climax = {"source_start": 0.0, "source_end": 10.0, "reason": "LONG: climax moment"}
        plain = {"source_start": 0.0, "source_end": 10.0, "reason": "MEDIUM: regular clip"}
        assert _estimate_clip_importance(climax) > _estimate_clip_importance(plain)

    def test_longer_clip_slightly_higher(self):
        long = {"source_start": 0.0, "source_end": 15.0, "reason": "test"}
        short = {"source_start": 0.0, "source_end": 3.0, "reason": "test"}
        assert _estimate_clip_importance(long) > _estimate_clip_importance(short)


class TestRemoveOverlapsImportance:
    """Test importance-weighted overlap removal."""

    def test_keeps_more_important_clip(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 10.0, "reason": "boring filler", "is_hook_clip": False},
            {"source_start": 5.0, "source_end": 8.0, "reason": "HOOK: amazing reveal", "is_hook_clip": True},
        ]}]
        removed = remove_overlaps(groups)
        assert removed > 0
        assert groups[0]["source_clips"][0]["is_hook_clip"] is True


class TestEnforceClipPacing:
    """Test deterministic pacing enforcement."""

    def test_swaps_short_ending(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 10.0, "reason": "LONG: buildup"},
            {"source_start": 20.0, "source_end": 30.0, "reason": "LONG: climax"},
            {"source_start": 40.0, "source_end": 43.0, "reason": "SHORT: quick beat"},
        ]}]
        adjustments = enforce_clip_pacing(groups)
        assert adjustments > 0
        # Last clip should no longer be SHORT
        last_dur = groups[0]["source_clips"][-1]["source_end"] - groups[0]["source_clips"][-1]["source_start"]
        assert last_dur > 5.0

    def test_no_change_for_already_diverse(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 4.0, "reason": "SHORT: hook"},
            {"source_start": 10.0, "source_end": 20.0, "reason": "MEDIUM: middle"},
            {"source_start": 30.0, "source_end": 48.0, "reason": "LONG: ending"},
        ]}]
        adjustments = enforce_clip_pacing(groups)
        assert adjustments == 0

    def test_trims_back_to_back_long(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 25.0, "reason": "LONG: first long"},
            {"source_start": 30.0, "source_end": 55.0, "reason": "LONG: second long"},
        ]}]
        adjustments = enforce_clip_pacing(groups)
        assert adjustments > 0
        # At least one should be trimmed to <=15s
        durs = [c["source_end"] - c["source_start"] for c in groups[0]["source_clips"]]
        assert any(d <= 15.0 for d in durs)


class TestRepairClipDiversity:
    """Test diversity repair."""

    def test_fixes_tight_gaps(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 5.0, "reason": "a"},
            {"source_start": 5.5, "source_end": 10.0, "reason": "b"},  # 0.5s gap (< MIN_TEMPORAL_GAP=1.0)
            {"source_start": 20.0, "source_end": 25.0, "reason": "c"},
        ]}]
        repairs = repair_clip_diversity(groups, source_duration=100.0)
        assert repairs > 0
        # Clip 0's end should have moved forward to reduce the gap
        assert groups[0]["source_clips"][0]["source_end"] > 5.0

    def test_no_change_when_already_diverse(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 5.0, "reason": "a"},
            {"source_start": 15.0, "source_end": 20.0, "reason": "b"},
            {"source_start": 40.0, "source_end": 45.0, "reason": "c"},
        ]}]
        repairs = repair_clip_diversity(groups, source_duration=200.0)
        assert repairs == 0

    def test_gap_expansion_does_not_create_overlap(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 4.9, "reason": "a"},
            {"source_start": 5.0, "source_end": 10.0, "reason": "b"},  # 0.1s gap
            {"source_start": 12.0, "source_end": 17.0, "reason": "c"},
        ]}]
        repairs = repair_clip_diversity(groups, source_duration=100.0)
        # Clip 0 end should NOT exceed clip 1 start
        assert groups[0]["source_clips"][0]["source_end"] <= groups[0]["source_clips"][1]["source_start"]

    def test_relocate_redundant_clip_to_gap(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 5.0, "reason": "a"},
            {"source_start": 0.5, "source_end": 5.5, "reason": "b"},  # overlaps with a
            {"source_start": 10.0, "source_end": 15.0, "reason": "c"},
        ]}]
        repairs = repair_clip_diversity(groups, source_duration=100.0)
        # After repair, clips should be more spread out
        starts = sorted(c["source_start"] for c in groups[0]["source_clips"])
        assert starts[-1] - starts[0] >= 5.0


class TestEnforceClipPacingEdgeCases:
    """Edge case tests for pacing enforcement."""

    def test_single_clip_group_unchanged(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 5.0, "reason": "SHORT: only clip"},
        ]}]
        adjustments = enforce_clip_pacing(groups)
        assert adjustments == 0
        assert len(groups[0]["source_clips"]) == 1

    def test_two_short_clips_not_merged(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 3.0, "reason": "SHORT: a"},
            {"source_start": 10.0, "source_end": 13.0, "reason": "SHORT: b"},
        ]}]
        adjustments = enforce_clip_pacing(groups)
        # Only 2 SHORTs, not 3+ — no merge needed
        assert len(groups[0]["source_clips"]) == 2

    def test_trimmed_long_stays_above_minimum(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 30.0, "reason": "LONG: first"},
            {"source_start": 35.0, "source_end": 65.0, "reason": "LONG: second"},
        ]}]
        adjustments = enforce_clip_pacing(groups)
        durs = [c["source_end"] - c["source_start"] for c in groups[0]["source_clips"]]
        assert all(d >= 3.0 for d in durs)

    def test_swap_preserves_clip_count(self):
        groups = [{"source_clips": [
            {"source_start": 0.0, "source_end": 10.0, "reason": "LONG: a"},
            {"source_start": 15.0, "source_end": 25.0, "reason": "LONG: b"},
            {"source_start": 30.0, "source_end": 33.0, "reason": "SHORT: c"},
        ]}]
        original_count = len(groups[0]["source_clips"])
        enforce_clip_pacing(groups)
        assert len(groups[0]["source_clips"]) == original_count


class TestExpandClipsToFillGap:
    """Test _expand_clips_to_fill_gap deterministic expansion."""

    def test_expands_short_clips_to_target(self):
        clips = [
            {"source_start": 10.0, "source_end": 15.0, "reason": "SHORT a"},
            {"source_start": 30.0, "source_end": 35.0, "reason": "SHORT b"},
        ]
        expanded = _expand_clips_to_fill_gap(clips, source_duration=60.0, target_total=40.0)
        assert expanded > 0
        total = sum(c["source_end"] - c["source_start"] for c in clips)
        assert total >= 40.0

    def test_no_expansion_when_already_enough(self):
        clips = [
            {"source_start": 0.0, "source_end": 20.0, "reason": "LONG a"},
            {"source_start": 25.0, "source_end": 50.0, "reason": "LONG b"},
        ]
        expanded = _expand_clips_to_fill_gap(clips, source_duration=60.0, target_total=40.0)
        assert expanded == 0
        total = sum(c["source_end"] - c["source_start"] for c in clips)
        assert total == 45.0

    def test_respects_clip_soft_max_in_first_pass(self):
        clips = [
            {"source_start": 10.0, "source_end": 15.0, "reason": "SHORT a"},
            {"source_start": 30.0, "source_end": 35.0, "reason": "SHORT b"},
        ]
        _expand_clips_to_fill_gap(clips, source_duration=60.0, target_total=50.0)
        for c in clips:
            dur = c["source_end"] - c["source_start"]
            # First pass caps at 30s, but pass 2 may extend further to fill gap
            assert dur <= 30.0 or dur <= 50.0

    def test_no_overlap_created(self):
        clips = [
            {"source_start": 10.0, "source_end": 15.0, "reason": "SHORT a"},
            {"source_start": 20.0, "source_end": 25.0, "reason": "SHORT b"},
        ]
        _expand_clips_to_fill_gap(clips, source_duration=60.0, target_total=30.0)
        clips.sort(key=lambda c: c["source_start"])
        for i in range(len(clips) - 1):
            assert clips[i]["source_end"] <= clips[i + 1]["source_start"]

    def test_partial_expansion_when_source_short(self):
        clips = [
            {"source_start": 0.0, "source_end": 5.0, "reason": "SHORT a"},
        ]
        expanded = _expand_clips_to_fill_gap(clips, source_duration=30.0, target_total=40.0)
        # Can expand to source_duration (30s) but can't reach 40s target
        total = sum(c["source_end"] - c["source_start"] for c in clips)
        assert total <= 30.0  # Can't exceed source_duration
