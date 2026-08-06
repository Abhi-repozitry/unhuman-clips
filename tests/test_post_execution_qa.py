"""Tests for Phase 7 — Two-Tier QA (Python QA + LLM completeness critic)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.models import EntitySegment
from backend.pipeline.plan_executor import (
    post_execution_qa,
    _trim_to_duration,
    _trim_to_gap,
)
from backend.pipeline.analyzer import (
    _prompt_completeness_critic,
    _completeness_critic,
)


def _clip(start: float, end: float, beat: str = "escalation", entity_seg_id: str | None = None) -> dict:
    c = {
        "source_start": start,
        "source_end": end,
        "reason": f"{beat.upper()}: test clip",
        "_beat": beat,
        "is_hook_clip": beat == "hook",
    }
    if entity_seg_id:
        c["entity_segment_ids"] = [entity_seg_id]
    return c


def _group(idx: int, clips: list[dict], narration: list[dict] | None = None) -> dict:
    total = sum(c["source_end"] - c["source_start"] for c in clips)
    return {
        "group_index": idx,
        "group_reasoning": f"Group {idx}",
        "estimated_duration_seconds": round(total + 2.0, 1),
        "reel_summary": {"title": f"Group {idx}", "short_description": "", "source_understanding": "", "narrative_angle": "", "key_moment": ""},
        "source_clips": clips,
        "narration_events": narration or [],
    }


def _entity_seg(seg_id: str, start: float, end: float, block_ids: list[int] | None = None) -> EntitySegment:
    return EntitySegment(
        entity_segment_id=seg_id,
        entity_name=None,
        start=start,
        end=end,
        block_ids=block_ids or [],
        speaker_ids=[],
        evidence=["scene_cut"],
    )


# ---------------------------------------------------------------------------
# post_execution_qa — entity boundary trimming
# ---------------------------------------------------------------------------

class TestQAEntityBoundaryTrimming:
    def test_clip_trimmed_to_entity_end(self):
        seg = _entity_seg("e1", 0.0, 30.0)
        clips = [_clip(25.0, 35.0, "escalation", "e1")]
        groups = [_group(0, clips)]
        result = post_execution_qa(groups, 60.0, entity_segments=[seg])
        assert result[0]["source_clips"][0]["source_end"] == 30.0

    def test_clip_trimmed_to_entity_start(self):
        seg = _entity_seg("e1", 10.0, 40.0)
        clips = [_clip(5.0, 20.0, "escalation", "e1")]
        groups = [_group(0, clips)]
        result = post_execution_qa(groups, 60.0, entity_segments=[seg])
        assert result[0]["source_clips"][0]["source_start"] == 10.0

    def test_clip_within_entity_unchanged(self):
        seg = _entity_seg("e1", 0.0, 30.0)
        clips = [_clip(5.0, 25.0, "escalation", "e1")]
        groups = [_group(0, clips)]
        result = post_execution_qa(groups, 60.0, entity_segments=[seg])
        c = result[0]["source_clips"][0]
        assert c["source_start"] == 5.0
        assert c["source_end"] == 25.0


# ---------------------------------------------------------------------------
# post_execution_qa — payoff positioning
# ---------------------------------------------------------------------------

class TestQAPayoffPositioning:
    def test_payoff_has_latest_source_end(self):
        clips = [
            _clip(0.0, 5.0, "hook"),
            _clip(20.0, 30.0, "payoff"),
            _clip(10.0, 15.0, "escalation"),
        ]
        groups = [_group(0, clips)]
        result = post_execution_qa(groups, 60.0)
        final_clips = result[0]["source_clips"]
        payoff = [c for c in final_clips if c["_beat"] == "payoff"][0]
        latest_end = max(c["source_end"] for c in final_clips)
        assert payoff["source_end"] == latest_end

    def test_payoff_not_latest_moved_to_latest(self):
        clips = [
            _clip(0.0, 5.0, "hook"),
            _clip(10.0, 15.0, "payoff"),
            _clip(20.0, 35.0, "escalation"),
        ]
        groups = [_group(0, clips)]
        result = post_execution_qa(groups, 60.0)
        final_clips = result[0]["source_clips"]
        # After swap, the "payoff" metadata follows the timestamps — the clip
        # with payoff beat should now have the latest source range.
        payoff = [c for c in final_clips if c["_beat"] == "payoff"][0]
        latest_end = max(c["source_end"] for c in final_clips)
        assert payoff["source_end"] == latest_end


# ---------------------------------------------------------------------------
# post_execution_qa — duration bounds
# ---------------------------------------------------------------------------

class TestQADurationBounds:
    def test_over_max_trimmed(self):
        # 3 clips × 15s = 45s source + 2s pad = 47s > 40s max
        clips = [
            _clip(0.0, 15.0, "hook"),
            _clip(20.0, 35.0, "escalation"),
            _clip(40.0, 55.0, "payoff"),
        ]
        groups = [_group(0, clips)]
        result = post_execution_qa(groups, 60.0, reel_dur_min=30, reel_dur_max=40)
        total = sum(c["source_end"] - c["source_start"] for c in result[0]["source_clips"])
        # QA trims to g_max (not g_max-2) since executor already accounted for +2s pad
        assert total <= 40.5

    def test_under_min_kept(self):
        clips = [_clip(5.0, 8.0, "hook")]
        groups = [_group(0, clips)]
        result = post_execution_qa(groups, 60.0, reel_dur_min=30, reel_dur_max=50)
        assert len(result) == 1
        assert result[0]["source_clips"][0]["source_start"] == 5.0


# ---------------------------------------------------------------------------
# post_execution_qa — cross-group overlap
# ---------------------------------------------------------------------------

class TestQACrossGroupOverlap:
    def test_overlap_trimmed_in_lower_priority_group(self):
        groups = [
            _group(0, [_clip(10.0, 20.0, "hook")]),
            _group(1, [_clip(15.0, 25.0, "escalation")]),
        ]
        result = post_execution_qa(groups, 60.0)
        # Group 0 (higher priority) keeps its clip
        g0_clips = result[0]["source_clips"]
        assert g0_clips[0]["source_start"] == 10.0
        # Group 1 (lower priority) clip should be moved to a gap (0-10)
        g1_clips = result[1]["source_clips"]
        assert g1_clips[0]["source_end"] <= 10.1

    def test_no_overlap_unchanged(self):
        groups = [
            _group(0, [_clip(10.0, 20.0, "hook")]),
            _group(1, [_clip(30.0, 40.0, "escalation")]),
        ]
        result = post_execution_qa(groups, 60.0)
        assert result[0]["source_clips"][0]["source_start"] == 10.0
        assert result[1]["source_clips"][0]["source_start"] == 30.0


# ---------------------------------------------------------------------------
# post_execution_qa — unsalvageable groups dropped
# ---------------------------------------------------------------------------

class TestQAGroupDropping:
    def test_empty_group_dropped(self):
        groups = [
            _group(0, [_clip(10.0, 20.0, "hook")]),
            _group(1, []),
        ]
        result = post_execution_qa(groups, 60.0)
        assert len(result) == 1
        assert result[0]["group_index"] == 0


# ---------------------------------------------------------------------------
# post_execution_qa — estimate recalculation
# ---------------------------------------------------------------------------

class TestQAEstimateRecalc:
    def test_estimate_recalculated_after_trim(self):
        clips = [
            _clip(0.0, 15.0, "hook"),
            _clip(20.0, 35.0, "escalation"),
            _clip(40.0, 55.0, "payoff"),
        ]
        groups = [_group(0, clips)]
        # Over-max: estimate should be recalculated
        result = post_execution_qa(groups, 60.0, reel_dur_min=30, reel_dur_max=40)
        total = sum(c["source_end"] - c["source_start"] for c in result[0]["source_clips"])
        assert result[0]["estimated_duration_seconds"] == round(total + 2.0, 1)


# ---------------------------------------------------------------------------
# _trim_to_duration helper
# ---------------------------------------------------------------------------

class TestTrimToDuration:
    def test_trims_longest_non_hook_clip(self):
        clips = [
            _clip(0.0, 5.0, "hook"),
            _clip(10.0, 30.0, "escalation"),
            _clip(35.0, 45.0, "payoff"),
        ]
        _trim_to_duration(clips, 25.0)
        total = sum(c["source_end"] - c["source_start"] for c in clips)
        assert total <= 25.5

    def test_hook_not_trimmed(self):
        clips = [
            _clip(0.0, 5.0, "hook"),
            _clip(10.0, 30.0, "escalation"),
        ]
        _trim_to_duration(clips, 10.0)
        hook = clips[0]
        assert hook["source_end"] - hook["source_start"] >= 4.9


# ---------------------------------------------------------------------------
# _trim_to_gap helper
# ---------------------------------------------------------------------------

class TestTrimToGap:
    def test_fits_in_gap(self):
        clip = _clip(15.0, 25.0, "escalation")
        claimed = [(5.0, 15.0), (30.0, 40.0)]
        result = _trim_to_gap(clip, claimed, 60.0)
        assert result is not None
        # Should be placed in the gap (15, 30) — fits the full 10s clip
        assert result["source_start"] == 15.0
        assert result["source_end"] == 25.0

    def test_returns_none_when_no_gap(self):
        clip = _clip(5.0, 25.0, "escalation")
        claimed = [(0.0, 60.0)]
        result = _trim_to_gap(clip, claimed, 60.0)
        assert result is None


# ---------------------------------------------------------------------------
# Completeness critic prompt
# ---------------------------------------------------------------------------

class TestCompletenessCriticPrompt:
    def test_prompt_contains_group_indices(self):
        groups = [_group(0, [_clip(5.0, 10.0, "hook")])]
        prompt = _prompt_completeness_critic(groups)
        assert '"group_index": 0' in prompt

    def test_prompt_contains_arc_info(self):
        groups = [_group(0, [_clip(5.0, 10.0, "hook")])]
        prompt = _prompt_completeness_critic(groups)
        assert "hook" in prompt.lower()


# ---------------------------------------------------------------------------
# Completeness critic — LLM failure handling
# ---------------------------------------------------------------------------

class TestCompletenessCriticFailure:
    def test_llm_error_keeps_groups_unchanged(self):
        groups = [_group(0, [_clip(5.0, 10.0, "hook")])]
        with patch("backend.pipeline.analyzer._call_llm", side_effect=RuntimeError("API down")):
            result = _completeness_critic(groups)
        assert len(result) == 1
        assert result[0]["group_index"] == 0

    def test_malformed_output_keeps_groups_unchanged(self):
        groups = [_group(0, [_clip(5.0, 10.0, "hook")])]
        with patch("backend.pipeline.analyzer._call_llm", return_value="not json at all"):
            result = _completeness_critic(groups)
        assert len(result) == 1

    def test_fast_mode_skips_critique(self):
        groups = [_group(0, [_clip(5.0, 10.0, "hook")])]
        with patch("backend.config.FAST_MODE", True):
            result = _completeness_critic(groups)
        assert len(result) == 1
        assert "completeness_critic" not in result[0]
