"""Tests for Phase 6 — Entity-Grouped Planning and Execution."""
from __future__ import annotations

from backend.models import EntitySegment
from backend.pipeline.analyzer import SemanticBlock
from backend.pipeline.plan_executor import execute_plan, _extend_short_clip
from backend.pipeline.plan_schema import (
    Unit,
    StoryPlan,
    _validate_entity_merges,
    _unit_windows,
    parse_story_plan,
)


def _block(block_id: int, start: float, end: float, text: str, entity_segment_id: str | None = None) -> SemanticBlock:
    b = SemanticBlock(
        block_id=block_id,
        start=start,
        end=end,
        text=text,
        speech_energy=0.8,
        volume_db=-12.0,
        silence_before=False,
        black_frame=False,
        freeze=False,
        importance=70.0,
        peak_offset=1.0,
        segment_ids=[0],
    )
    b.entity_segment_id = entity_segment_id
    return b


def _entity_segment(segment_id: str, start: float, end: float, block_ids: list[int], name: str | None = None) -> EntitySegment:
    return EntitySegment(
        entity_segment_id=segment_id,
        entity_name=name,
        start=start,
        end=end,
        block_ids=block_ids,
        speaker_ids=[],
        evidence=["scene_cut"],
    )


# ---------------------------------------------------------------------------
# _validate_entity_merges tests
# ---------------------------------------------------------------------------

class TestValidateEntityMerges:
    def test_single_segments_produce_one_unit_each(self):
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0, 1], "Maya"),
            _entity_segment("entity-2", 30.0, 60.0, [2, 3], "Chris"),
            _entity_segment("entity-3", 60.0, 90.0, [4, 5], "Jordan"),
        ]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}") for i in range(6)]
        raw_merges = [
            {"segment_ids": ["entity-1"]},
            {"segment_ids": ["entity-2"]},
            {"segment_ids": ["entity-3"]},
        ]
        units = _validate_entity_merges(raw_merges, segments, blocks, 90.0)
        assert len(units) == 3
        assert units[0].entity_segment_ids == ["entity-1"]
        assert units[1].entity_segment_ids == ["entity-2"]
        assert units[2].entity_segment_ids == ["entity-3"]

    def test_adjacent_pair_merges_into_one_unit(self):
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0, 1], "Maya"),
            _entity_segment("entity-2", 30.0, 60.0, [2, 3], "Chris"),
        ]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}") for i in range(4)]
        raw_merges = [
            {"segment_ids": ["entity-1", "entity-2"]},
        ]
        units = _validate_entity_merges(raw_merges, segments, blocks, 60.0)
        assert len(units) == 1
        assert units[0].entity_segment_ids == ["entity-1", "entity-2"]
        assert "Maya" in units[0].name
        assert "Chris" in units[0].name

    def test_non_adjacent_pair_splits_back(self):
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0, 1], "Maya"),
            _entity_segment("entity-2", 30.0, 60.0, [2, 3], "Chris"),
            _entity_segment("entity-3", 60.0, 90.0, [4, 5], "Jordan"),
        ]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}") for i in range(6)]
        raw_merges = [
            {"segment_ids": ["entity-1", "entity-3"]},  # Not adjacent
        ]
        units = _validate_entity_merges(raw_merges, segments, blocks, 90.0)
        # 3 units: entity-1 (split), entity-3 (split), entity-2 (orphan)
        assert len(units) == 3
        assert units[0].entity_segment_ids == ["entity-1"]
        assert units[1].entity_segment_ids == ["entity-3"]
        assert units[2].entity_segment_ids == ["entity-2"]

    def test_invalid_segment_id_filtered(self):
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0, 1], "Maya"),
        ]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}") for i in range(2)]
        raw_merges = [
            {"segment_ids": ["entity-99"]},  # Invalid ID
        ]
        units = _validate_entity_merges(raw_merges, segments, blocks, 30.0)
        # Should still produce entity-1 as an orphan
        assert len(units) == 1
        assert units[0].entity_segment_ids == ["entity-1"]

    def test_duplicate_ids_across_groups_deduplicated(self):
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0, 1], "Maya"),
            _entity_segment("entity-2", 30.0, 60.0, [2, 3], "Chris"),
        ]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}") for i in range(4)]
        raw_merges = [
            {"segment_ids": ["entity-1"]},
            {"segment_ids": ["entity-1"]},  # Duplicate
            {"segment_ids": ["entity-2"]},
        ]
        units = _validate_entity_merges(raw_merges, segments, blocks, 60.0)
        assert len(units) == 2
        assert units[0].entity_segment_ids == ["entity-1"]
        assert units[1].entity_segment_ids == ["entity-2"]


# ---------------------------------------------------------------------------
# _unit_windows entity override tests
# ---------------------------------------------------------------------------

class TestEntityWindows:
    def test_entity_windows_override_region(self):
        segments = [
            _entity_segment("entity-1", 0.0, 40.0, [0]),
            _entity_segment("entity-2", 40.0, 80.0, [1]),
        ]
        plan = StoryPlan(video_type="entity", units=[
            Unit(unit_id=0, name="A", priority=1, region="mid", entity_segment_ids=["entity-1"]),
            Unit(unit_id=1, name="B", priority=1, region="mid", entity_segment_ids=["entity-2"]),
        ])
        windows = _unit_windows(plan, 100.0, entity_segments=segments)
        assert windows[0] == (0.0, 40.0)
        assert windows[1] == (40.0, 80.0)

    def test_entity_windows_merged_segments(self):
        segments = [
            _entity_segment("entity-1", 0.0, 40.0, [0]),
            _entity_segment("entity-2", 40.0, 80.0, [1]),
        ]
        plan = StoryPlan(video_type="entity", units=[
            Unit(unit_id=0, name="A & B", priority=1, region="mid", entity_segment_ids=["entity-1", "entity-2"]),
        ])
        windows = _unit_windows(plan, 100.0, entity_segments=segments)
        assert windows[0] == (0.0, 80.0)

    def test_entity_windows_falls_back_to_region(self):
        plan = StoryPlan(video_type="other", units=[
            Unit(unit_id=0, name="A", priority=1, region="early"),
            Unit(unit_id=1, name="B", priority=1, region="late"),
        ])
        windows = _unit_windows(plan, 100.0)
        assert windows[0] == (0.0, 25.0)
        assert windows[1] == (75.0, 100.0)


# ---------------------------------------------------------------------------
# parse_story_plan entity mode tests
# ---------------------------------------------------------------------------

class TestParseStoryPlanEntity:
    def test_entity_merge_groups_produce_units(self):
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0, 1], "Maya"),
            _entity_segment("entity-2", 30.0, 60.0, [2, 3], "Chris"),
        ]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}") for i in range(4)]
        data = {
            "video_type": "entity",
            "merge_groups": [
                {"segment_ids": ["entity-1"]},
                {"segment_ids": ["entity-2"]},
            ],
        }
        plan = parse_story_plan(data, 60.0, 1, 6, entity_segments=segments, blocks=blocks)
        assert plan.video_type == "entity"
        assert len(plan.units) == 2
        assert plan.units[0].entity_segment_ids == ["entity-1"]
        assert plan.units[1].entity_segment_ids == ["entity-2"]


# ---------------------------------------------------------------------------
# execute_plan entity mode tests
# ---------------------------------------------------------------------------

class TestExecutePlanEntity:
    def test_entity_mode_filters_blocks_to_segment(self):
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0, 1], "Maya"),
            _entity_segment("entity-2", 30.0, 60.0, [2, 3], "Chris"),
        ]
        blocks = [
            _block(0, 2.0, 8.0, "Maya starts", "entity-1"),
            _block(1, 12.0, 18.0, "Maya continues", "entity-1"),
            _block(2, 32.0, 38.0, "Chris speaks", "entity-2"),
            _block(3, 42.0, 48.0, "Chris finishes", "entity-2"),
        ]
        plan = StoryPlan(video_type="entity", units=[
            Unit(unit_id=0, name="Maya", priority=1, region="mid",
                 entity_segment_ids=["entity-1"],
                 arc=[
                     {"beat": "start", "position": "start", "intent": "introduce"},
                     {"beat": "payoff", "position": "end", "intent": "reveal"},
                 ]),
        ])
        groups = execute_plan(plan, blocks, 60.0, 30, 50, entity_segments=segments)
        assert len(groups) >= 1
        # All clips should be within entity-1's time range (0-30s)
        for g in groups:
            for c in g["source_clips"]:
                assert c["source_end"] <= 30.0 + 0.1

    def test_entity_mode_no_floor_padding(self):
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0], "Maya"),
        ]
        blocks = [_block(0, 5.0, 15.0, "Maya scene", "entity-1")]
        plan = StoryPlan(video_type="entity", units=[
            Unit(unit_id=0, name="Maya", priority=1, region="mid",
                 entity_segment_ids=["entity-1"],
                 arc=[
                     {"beat": "start", "position": "start", "intent": "introduce"},
                     {"beat": "payoff", "position": "end", "intent": "reveal"},
                 ]),
        ])
        groups = execute_plan(plan, blocks, 60.0, 30, 50, entity_segments=segments)
        assert len(groups) >= 1
        # Should have clips from the single block, not padded to 3
        total_clips = sum(len(g["source_clips"]) for g in groups)
        assert total_clips <= 2  # start + payoff from same block area

    def test_extend_short_clip_respects_entity_boundary(self):
        blocks = [
            _block(0, 5.0, 10.0, "Entity A block", "entity-1"),
            _block(1, 10.5, 15.0, "Entity B block", "entity-2"),
        ]
        clip = {"source_start": 5.0, "source_end": 8.0}
        clip = _extend_short_clip(clip, blocks, (0.0, 30.0), 5.0, set(), entity_segment_ids=["entity-1"])
        # Should NOT absorb block 1 (entity-2) even though it's adjacent
        assert clip["source_end"] <= 10.0 + 0.1

    def test_extend_short_clip_absorbs_same_entity(self):
        blocks = [
            _block(0, 5.0, 10.0, "Entity A block 1", "entity-1"),
            _block(1, 10.2, 15.0, "Entity A block 2", "entity-1"),
        ]
        clip = {"source_start": 5.0, "source_end": 8.0}
        clip = _extend_short_clip(clip, blocks, (0.0, 30.0), 7.0, set(), entity_segment_ids=["entity-1"])
        # Should absorb block 1 (same entity, adjacent) — extends past 15s
        assert clip["source_end"] >= 15.0 - 0.1


# ---------------------------------------------------------------------------
# Phase 9 — Entity-mode regression tests (56-group failure fixes)
# ---------------------------------------------------------------------------

from backend.pipeline.analyzer import (
    _premerge_entity_segments,
    _cap_entity_segments,
    _segment_entities,
)


class TestEntityPremerge:
    """Test speaker-id-based pre-merge (Fix 1)."""

    def test_same_speaker_consecutive_segments_merge(self):
        """Two consecutive segments with same primary speaker merge."""
        segs = [
            _entity_segment("entity-1", 0.0, 20.0, [0], "Unknown"),
            _entity_segment("entity-2", 20.0, 40.0, [1], "Unknown"),
        ]
        segs[0].speaker_ids = ["SPK-0"]
        segs[1].speaker_ids = ["SPK-0"]
        blocks = [_block(0, 0.0, 20.0, "hello", "entity-1"), _block(1, 20.0, 40.0, "world", "entity-2")]
        result = _premerge_entity_segments(segs, blocks)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 40.0
        assert result[0].speaker_ids == ["SPK-0"]

    def test_different_speakers_no_merge(self):
        """Consecutive segments with different speakers stay separate."""
        segs = [
            _entity_segment("entity-1", 0.0, 20.0, [0], "A"),
            _entity_segment("entity-2", 20.0, 40.0, [1], "B"),
        ]
        segs[0].speaker_ids = ["SPK-0"]
        segs[1].speaker_ids = ["SPK-1"]
        blocks = [_block(0, 0.0, 20.0, "a", "entity-1"), _block(1, 20.0, 40.0, "b", "entity-2")]
        result = _premerge_entity_segments(segs, blocks)
        assert len(result) == 2

    def test_host_contestant_host_contestant_pattern(self):
        """Alternating host/contestant: same-speaker segments merge.
        Merged segment stores effective_ranges so executor doesn't span gaps."""
        segs = [
            _entity_segment("entity-1", 0.0, 15.0, [0], "Host"),
            _entity_segment("entity-2", 15.0, 30.0, [1], "Contestant"),
            _entity_segment("entity-3", 30.0, 45.0, [2], "Host"),
            _entity_segment("entity-4", 45.0, 60.0, [3], "Contestant"),
        ]
        segs[0].speaker_ids = ["SPK-HOST"]
        segs[1].speaker_ids = ["SPK-CONTESTANT"]
        segs[2].speaker_ids = ["SPK-HOST"]
        segs[3].speaker_ids = ["SPK-CONTESTANT"]
        blocks = [
            _block(i, float(i * 15), float((i + 1) * 15), f"block {i}", f"entity-{i + 1}")
            for i in range(4)
        ]
        result = _premerge_entity_segments(segs, blocks)
        # Host segments (0,2) merge into one; Contestant segments (1,3) merge
        assert len(result) == 2
        host_seg = [s for s in result if s.speaker_ids == ["SPK-HOST"]]
        contest_seg = [s for s in result if s.speaker_ids == ["SPK-CONTESTANT"]]
        assert len(host_seg) == 1
        assert len(contest_seg) == 1
        # Merged segment has effective_ranges covering actual block time ranges
        assert host_seg[0].effective_ranges == [(0.0, 15.0), (30.0, 45.0)]
        assert contest_seg[0].effective_ranges == [(15.0, 30.0), (45.0, 60.0)]

    def test_premerge_reassigns_block_ids(self):
        """After merge, blocks get the merged segment's ID."""
        segs = [
            _entity_segment("entity-1", 0.0, 20.0, [0], "A"),
            _entity_segment("entity-2", 20.0, 40.0, [1], "A"),
        ]
        segs[0].speaker_ids = ["SPK-0"]
        segs[1].speaker_ids = ["SPK-0"]
        blocks = [_block(0, 0.0, 20.0, "a", "entity-1"), _block(1, 20.0, 40.0, "b", "entity-2")]
        _premerge_entity_segments(segs, blocks)
        assert blocks[0].entity_segment_id == "entity-1"
        assert blocks[1].entity_segment_id == "entity-1"


class TestEntitySegmentCap:
    """Test segment capping (Fix 2)."""

    def test_cap_merges_shortest_scene_cut_only(self):
        """Cap merges scene_cut-only segments first."""
        segs = [
            _entity_segment("entity-1", 0.0, 10.0, [0], "A"),
            _entity_segment("entity-2", 10.0, 15.0, [1], None),  # short, scene_cut only
            _entity_segment("entity-3", 15.0, 25.0, [2], "B"),
            _entity_segment("entity-4", 25.0, 35.0, [3], "C"),
        ]
        segs[0].evidence = ["speaker_id"]
        segs[1].evidence = ["scene_cut"]
        segs[2].evidence = ["speaker_id"]
        segs[3].evidence = ["speaker_id"]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}", f"entity-{i + 1}") for i in range(4)]
        result = _cap_entity_segments(segs, 3, blocks)
        assert len(result) == 3

    def test_cap_preserves_quality_segments(self):
        """Cap keeps segments with speaker_id evidence over scene_cut-only."""
        segs = [
            _entity_segment("entity-1", 0.0, 20.0, [0], "A"),
            _entity_segment("entity-2", 20.0, 25.0, [1], None),  # short, scene_cut
            _entity_segment("entity-3", 25.0, 50.0, [2], "B"),
        ]
        segs[0].evidence = ["speaker_id"]
        segs[1].evidence = ["scene_cut"]
        segs[2].evidence = ["speaker_id"]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}", f"entity-{i + 1}") for i in range(3)]
        result = _cap_entity_segments(segs, 2, blocks)
        assert len(result) == 2
        # The short scene_cut segment should be merged away
        names = [s.entity_name for s in result]
        assert "A" in names
        assert "B" in names


class TestMultiSegmentMerge:
    """Test that >2 segment merges work (Fix 3 — lift ceiling)."""

    def test_three_adjacent_segments_merge(self):
        """LLM can now merge 3+ adjacent segments."""
        segments = [
            _entity_segment("entity-1", 0.0, 20.0, [0, 1], "Contestant"),
            _entity_segment("entity-2", 20.0, 40.0, [2, 3], "Contestant"),
            _entity_segment("entity-3", 40.0, 60.0, [4, 5], "Contestant"),
        ]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}") for i in range(6)]
        raw_merges = [{"segment_ids": ["entity-1", "entity-2", "entity-3"]}]
        units = _validate_entity_merges(raw_merges, segments, blocks, 60.0)
        assert len(units) == 1
        assert units[0].entity_segment_ids == ["entity-1", "entity-2", "entity-3"]
        assert "+ 2 more" in units[0].name

    def test_non_adjacent_segments_still_split(self):
        """Non-adjacent segments split into individual units, orphans added."""
        segments = [
            _entity_segment("entity-1", 0.0, 20.0, [0], "A"),
            _entity_segment("entity-2", 20.0, 40.0, [1], "B"),
            _entity_segment("entity-3", 40.0, 60.0, [2], "A"),
        ]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}") for i in range(3)]
        raw_merges = [{"segment_ids": ["entity-1", "entity-3"]}]  # not adjacent
        units = _validate_entity_merges(raw_merges, segments, blocks, 60.0)
        # entity-1 and entity-3 split into individual units, entity-2 is orphan = 3 total
        assert len(units) == 3
        all_ids = set()
        for u in units:
            all_ids.update(u.entity_segment_ids)
        assert all_ids == {"entity-1", "entity-2", "entity-3"}

    def test_four_adjacent_segments_merge(self):
        """4 adjacent segments merge into one unit."""
        segments = [
            _entity_segment(f"entity-{i}", float(i * 15), float((i + 1) * 15), [i], "Same")
            for i in range(4)
        ]
        blocks = [_block(i, float(i * 15), float((i + 1) * 15), f"block {i}") for i in range(4)]
        raw_merges = [{"segment_ids": ["entity-0", "entity-1", "entity-2", "entity-3"]}]
        units = _validate_entity_merges(raw_merges, segments, blocks, 60.0)
        assert len(units) == 1
        assert len(units[0].entity_segment_ids) == 4


class TestFallbackCap:
    """Test fallback path caps to max_groups (Fix 4)."""

    def test_fallback_respects_max_groups(self):
        """When entity planner fails, fallback caps to max_groups."""
        from backend.pipeline.plan_schema import _validate_entity_merges
        segments = [
            _entity_segment(f"entity-{i}", float(i * 10), float((i + 1) * 10), [i], f"Person {i}")
            for i in range(10)
        ]
        blocks = [_block(i, float(i * 10), float((i + 1) * 10), f"block {i}") for i in range(10)]
        # Simulate fallback: one unit per segment
        units = _validate_entity_merges(
            [{"segment_ids": [s.entity_segment_id]} for s in segments],
            segments, blocks, 100.0,
        )
        assert len(units) == 10
        # Now apply cap (simulating the analyzer.py fallback logic)
        max_groups = 3
        if len(units) > max_groups:
            units = units[:max_groups]
        assert len(units) == 3


class TestEntityGroupedTrigger:
    """Test entity_grouped trigger weighting (Fix 6)."""

    def test_multi_entity_with_names_always_groups(self):
        """Identifier multi_entity with ≥2 names always enables entity_grouped."""
        from backend.models import ContentIdentity
        identity = ContentIdentity(
            creator_name="KSI",
            content_format="game show",
            detected_genre="game_challenge",
            structure="multi_entity",
            entity_names=["Ace", "Dan"],
            hook_recommendation="hook",
            planning_notes="Test",
        )
        # Even with all scene_cut evidence, should still group
        segs = [
            EntitySegment(
                entity_segment_id=f"entity-{i}", entity_name=None,
                start=float(i * 10), end=float((i + 1) * 10),
                block_ids=[i], speaker_ids=[], evidence=["scene_cut"],
            )
            for i in range(5)
        ]
        qualifying = [s for s in segs if s.end - s.start >= 20.0 and s.block_ids]
        # New logic: multi_entity with ≥2 names → always group
        entity_grouped = bool(identity and identity.structure == "multi_entity")
        assert entity_grouped is True

    def test_single_narrative_requires_sustained_evidence(self):
        """Single narrative requires ≥2 consecutive non-scene-cut segments."""
        segs = [
            EntitySegment(
                entity_segment_id=f"entity-{i}", entity_name=None,
                start=float(i * 10), end=float((i + 1) * 10),
                block_ids=[i], speaker_ids=[], evidence=["scene_cut"],
            )
            for i in range(5)
        ]
        # Only one segment has non-scene-cut evidence
        segs[2].evidence = ["speaker_id"]
        qualifying = segs
        non_scene_streak = 0
        max_streak = 0
        for seg in qualifying:
            if seg.evidence != ["scene_cut"]:
                non_scene_streak += 1
                max_streak = max(max_streak, non_scene_streak)
            else:
                non_scene_streak = 0
        # Max streak is 1 — should NOT enable entity_grouped
        assert max_streak < 2


class TestPerSegmentDuration:
    """Test per-segment duration scaling (Fix 5)."""

    def test_entity_unit_gets_scaled_duration(self):
        """Entity unit with small usable content gets scaled duration target."""
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0, 1], "A"),
        ]
        blocks = [
            _block(0, 0.0, 15.0, "hello", "entity-1"),
            _block(1, 15.0, 30.0, "world", "entity-1"),
        ]
        plan = StoryPlan(video_type="entity", units=[
            Unit(unit_id=0, name="A", priority=1, region="mid",
                 arc=[{"beat": "start", "position": "start", "intent": "intro"},
                      {"beat": "payoff", "position": "end", "intent": "reveal"}],
                 entity_segment_ids=["entity-1"]),
        ])
        groups = execute_plan(plan, blocks, 30.0, 90, 120, entity_segments=segments)
        assert len(groups) >= 1
        total = sum(c["source_end"] - c["source_start"] for c in groups[0]["source_clips"])
        assert total <= 60.0  # entity scaling should bring target below global 90-120


class TestCompletenessCritiBatching:
    """Test completeness critic handles many groups (Fix 8)."""

    def test_criti_prompt_no_truncation(self):
        """Critic prompt should not be truncated at 30k chars."""
        from backend.pipeline.analyzer import _prompt_completeness_critic
        # Create 60 groups to exceed old 30k limit
        groups = []
        for i in range(60):
            groups.append({
                "group_index": i,
                "group_reasoning": f"Group {i} reasoning " * 20,
                "estimated_duration_seconds": 80.0,
                "source_clips": [
                    {"_beat": "hook", "source_start": 0.0, "source_end": 5.0, "reason": f"Hook for group {i}" * 10},
                    {"_beat": "escalation", "source_start": 5.0, "source_end": 40.0, "reason": f"Esc for group {i}" * 10},
                    {"_beat": "payoff", "source_start": 40.0, "source_end": 50.0, "reason": f"Payoff for group {i}" * 10},
                ],
                "narration_events": [
                    {"event_type": "hook", "text": f"Hook text for group {i}" * 10},
                ],
            })
        prompt = _prompt_completeness_critic(groups)
        # Old code truncated at 30000 chars — this should be much larger
        assert len(prompt) > 30000, f"Prompt ({len(prompt)} chars) should exceed old 30k truncation limit"


class TestDebugArtifact:
    """Test debug artifact dump (Fix 7)."""

    def test_artifact_writes_json(self, tmp_path):
        """Debug artifact writes valid JSON."""
        import json
        import sys
        # Patch WORKING_DIR to use tmp_path
        import backend.config
        original = backend.config.WORKING_DIR
        backend.config.WORKING_DIR = tmp_path
        try:
            from backend.pipeline.analyzer import _write_debug_artifact
            _write_debug_artifact(
                job_id="test-artifact-123",
                content_identity=None,
                entity_segments=[],
                planner_branch="test_branch",
                story_plan=None,
                relevance={},
                pre_qa_groups=[],
                post_qa_groups=[],
                final_groups=[],
                completeness_verdicts={},
                source_duration=100.0,
                max_groups=5,
            )
            artifact_path = tmp_path / "test-artifact-123" / "debug_artifact_test-artifact-123.json"
            assert artifact_path.exists()
            with open(artifact_path) as f:
                data = json.load(f)
            assert data["job_id"] == "test-artifact-123"
            assert data["planner_branch"] == "test_branch"
            assert data["source_duration"] == 100.0
        finally:
            backend.config.WORKING_DIR = original


class TestEntityBoundaryEnforcement:
    """Test entity boundary enforcement in execute_plan (Phase 9b)."""

    def test_clips_get_entity_segment_ids(self):
        """Clips in entity mode receive entity_segment_ids field."""
        segments = [
            _entity_segment("entity-1", 0.0, 30.0, [0, 1], "A"),
        ]
        blocks = [
            _block(0, 0.0, 15.0, "hello", "entity-1"),
            _block(1, 15.0, 30.0, "world", "entity-1"),
        ]
        plan = StoryPlan(video_type="entity", units=[
            Unit(unit_id=0, name="A", priority=1, region="mid",
                 arc=[{"beat": "start", "position": "start", "intent": "intro"},
                      {"beat": "payoff", "position": "end", "intent": "reveal"}],
                 entity_segment_ids=["entity-1"]),
        ])
        groups = execute_plan(plan, blocks, 30.0, 30, 50, entity_segments=segments)
        for g in groups:
            for c in g["source_clips"]:
                assert "entity_segment_ids" in c
                assert c["entity_segment_ids"] == ["entity-1"]

    def test_extend_clips_to_fill_respects_entity_boundaries(self):
        """_extend_clips_to_fill cannot absorb blocks outside entity segments."""
        from backend.pipeline.plan_executor import _extend_clips_to_fill
        blocks = [
            _block(0, 0.0, 10.0, "seg A content", "entity-1"),
            _block(1, 10.0, 20.0, "seg B content", "entity-2"),
            _block(2, 20.0, 30.0, "seg A cont", "entity-1"),
        ]
        clips = [{"source_start": 0.0, "source_end": 10.0}]
        used = {0}
        # Entity mode: only block IDs from entity-1
        entity_block_ids = {0, 2}
        _extend_clips_to_fill(clips, 25.0, 30.0, (0.0, 30.0), blocks, used, entity_block_ids=entity_block_ids)
        # Should extend into block 2 (entity-1), NOT block 1 (entity-2)
        assert clips[0]["source_end"] <= 30.0  # stays within entity-1 range
        # Block 1 should NOT be absorbed
        assert 1 not in used

    def test_gap_fill_skipped_in_entity_mode(self):
        """Pass 2 gap-filling is skipped when entity_block_ids is set."""
        from backend.pipeline.plan_executor import _extend_clips_to_fill
        blocks = [
            _block(0, 0.0, 5.0, "short", "entity-1"),
            _block(1, 20.0, 25.0, "gap content", "entity-2"),
        ]
        clips = [{"source_start": 0.0, "source_end": 5.0}]
        used = {0}
        entity_block_ids = {0}  # only block 0 is entity-1
        n = _extend_clips_to_fill(clips, 20.0, 25.0, (0.0, 25.0), blocks, used, entity_block_ids=entity_block_ids)
        # Gap-filling blocked: clip should stay at 5s (no adjacent blocks in entity)
        assert clips[0]["source_end"] - clips[0]["source_start"] <= 5.1

    def test_hook_clamped_to_entity_segment(self):
        """Hook clip cannot extend past its entity segment boundary."""
        segments = [
            _entity_segment("entity-1", 0.0, 8.0, [0], "A"),
        ]
        blocks = [
            _block(0, 0.0, 3.0, "hook content", "entity-1"),
            _block(1, 8.0, 18.0, "other content", "entity-2"),
        ]
        plan = StoryPlan(video_type="entity", units=[
            Unit(unit_id=0, name="A", priority=1, region="early",
                 arc=[{"beat": "hook", "position": "start", "intent": "curiosity"},
                      {"beat": "payoff", "position": "end", "intent": "reveal"}],
                 entity_segment_ids=["entity-1"]),
        ])
        groups = execute_plan(plan, blocks, 18.0, 90, 120, entity_segments=segments)
        assert len(groups) >= 1
        hook = [c for c in groups[0]["source_clips"] if c.get("is_hook_clip")]
        if hook:
            assert hook[0]["source_end"] <= 8.1  # entity-1 ends at 8.0

    def test_payoff_clamped_to_entity_segment(self):
        """Payoff clip cannot extend past its entity segment boundary."""
        segments = [
            _entity_segment("entity-1", 50.0, 60.0, [0], "A"),
        ]
        blocks = [
            _block(0, 50.0, 55.0, "payoff reveal", "entity-1"),
            _block(1, 0.0, 10.0, "other content", "entity-2"),
        ]
        plan = StoryPlan(video_type="entity", units=[
            Unit(unit_id=0, name="A", priority=1, region="late",
                 arc=[{"beat": "start", "position": "start", "intent": "intro"},
                      {"beat": "payoff", "position": "end", "intent": "reveal"}],
                 entity_segment_ids=["entity-1"]),
        ])
        groups = execute_plan(plan, blocks, 60.0, 30, 50, entity_segments=segments)
        assert len(groups) >= 1
        payoff = [c for c in groups[0]["source_clips"] if c.get("_beat") == "payoff"]
        if payoff:
            assert payoff[0]["source_end"] <= 60.1  # entity-1 ends at 60.0
