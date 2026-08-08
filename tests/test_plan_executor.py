"""Tests for the plan executor — deterministic LLM-plan-to-clips execution."""
from __future__ import annotations

import itertools
import json

import pytest

from backend.pipeline.analyzer import SemanticBlock
from backend.pipeline.plan_executor import (
    estimate_speech_duration,
    execute_plan,
    place_narration_events,
)
from backend.pipeline.plan_schema import (
    StoryPlan,
    _unit_windows,
    heuristic_story_plan,
    parse_story_plan,
    plan_to_structure_analysis,
)

# ---------------------------------------------------------------------------
# Block factories
# ---------------------------------------------------------------------------

def mk_block(
    block_id: int,
    start: float,
    end: float,
    text: str = "Some spoken words happen here.",
    importance: float = 50.0,
    energy: float = 0.5,
    silence_before: bool = False,
    has_question: bool = False,
    has_exclamation: bool = False,
    has_stakes: bool = False,
    has_roast: bool = False,
    has_dating: bool = False,
    has_vulgarity: bool = False,
    peak_offset: float | None = None,
) -> SemanticBlock:
    return SemanticBlock(
        block_id=block_id,
        start=start,
        end=end,
        text=text,
        speech_energy=energy,
        volume_db=-18.0,
        silence_before=silence_before,
        black_frame=False,
        freeze=False,
        importance=importance,
        peak_offset=peak_offset if peak_offset is not None else (end - start) / 2.0,
        has_question=has_question,
        has_exclamation=has_exclamation,
        has_stakes=has_stakes,
        has_roast=has_roast,
        has_dating=has_dating,
        has_vulgarity=has_vulgarity,
    )


def spread_blocks(count: int, source_duration: float, **kwargs) -> list[SemanticBlock]:
    """Evenly spaced blocks across the whole source."""
    step = source_duration / count
    blocks = []
    for i in range(count):
        start = i * step
        end = min(start + step * 0.6, source_duration)
        blocks.append(mk_block(i, start, end, **kwargs))
    return blocks


def standard_plan() -> StoryPlan:
    """Two-unit plan: early hook story + late payoff story."""
    return StoryPlan(
        video_type="challenge",
        units=[
            {
                "unit_id": 0,
                "name": "The early stunt",
                "priority": 1,
                "region": "early",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": ["STAKES"], "intent": "the risk is announced"},
                    {"beat": "escalation", "position": "any", "flags": [], "intent": "tension builds"},
                    {"beat": "payoff", "position": "end", "flags": ["STAKES"], "intent": "the winner is revealed"},
                ],
            },
            {
                "unit_id": 1,
                "name": "The late reveal",
                "priority": 1,
                "region": "late",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": [], "intent": "a mystery opens"},
                    {"beat": "payoff", "position": "end", "flags": ["ROAST"], "intent": "the mic-drop reaction"},
                ],
            },
        ],
    )


# ---------------------------------------------------------------------------
# Determinism — the core guarantee
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        blocks = spread_blocks(40, 1200.0)
        plan = standard_plan()
        g1 = execute_plan(plan, blocks, 1200.0, 90, 100)
        g2 = execute_plan(plan, blocks, 1200.0, 90, 100)
        assert g1 == g2, "identical input must produce identical groups"

    def test_group_structure_valid(self):
        blocks = spread_blocks(40, 1200.0)
        groups = execute_plan(standard_plan(), blocks, 1200.0, 90, 100)
        assert len(groups) == 2
        for g in groups:
            assert g["source_clips"], "group must have clips"
            assert any(c["is_hook_clip"] for c in g["source_clips"]), "group must have a hook clip"
            assert g["reel_summary"]["title"], "group must have a title"
            assert g["estimated_duration_seconds"] > 0


# ---------------------------------------------------------------------------
# Beat selection quality
# ---------------------------------------------------------------------------

class TestBeatSelection:
    def test_hook_is_strongest_curiosity_block(self):
        """A mid-window block with question + silence + stakes must win the hook."""
        blocks = [
            mk_block(0, 0.0, 10.0, "Boring setup talking about the weather.", importance=60.0, silence_before=True),
            mk_block(1, 12.0, 18.0, "Why would anyone even try this?!", importance=45.0,
                     silence_before=True, has_question=True, has_exclamation=True, has_stakes=True),
            mk_block(2, 20.0, 26.0, "The winner is revealed and takes the cash.", importance=80.0, has_stakes=True),
            mk_block(3, 70.0, 90.0, "More mid content, nothing special.", importance=70.0),
        ]
        plan = StoryPlan(
            video_type="challenge",
            units=[{
                "unit_id": 0, "name": "Story", "priority": 1, "region": "early",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": ["STAKES"], "intent": "curiosity"},
                    {"beat": "payoff", "position": "end", "flags": ["STAKES"], "intent": "the result"},
                ],
            }],
        )
        groups = execute_plan(plan, blocks, 120.0, 45, 100)
        payoff = groups[0]["source_clips"][-1]
        assert 20.0 <= payoff["source_start"] < 26.0, "the stakes payoff block must be the payoff"
        hook = next(c for c in groups[0]["source_clips"] if c["is_hook_clip"])
        assert hook["source_start"] < 18.0 <= hook["source_end"] + 0.01

    def test_payoff_prefers_unit_flags(self):
        """Payoff must cover the STAKES block, not the higher-importance plain one."""
        blocks = [
            mk_block(0, 0.0, 20.0, "Loud exciting chatter with no stakes.", importance=90.0),
            mk_block(1, 60.0, 80.0, "The money is on the line now.", importance=70.0, has_stakes=True),
        ]
        plan = StoryPlan(
            video_type="challenge",
            units=[{
                "unit_id": 0, "name": "Story", "priority": 1, "region": "mid",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                    {"beat": "payoff", "position": "end", "flags": ["STAKES"], "intent": "prize won"},
                ],
            }],
        )
        groups = execute_plan(plan, blocks, 100.0, 45, 100)
        payoff = groups[0]["source_clips"][-1]
        assert payoff["source_start"] < 80.0 <= payoff["source_end"] + 0.01

    def test_payoff_keeps_reveal_end(self):
        """Payoff trimming preserves the tail of the payoff block.

        Note: _extend_clips_to_fill may extend clips to meet reel_dur_min,
        so the payoff end can exceed the original block end.
        """
        blocks = [
            mk_block(0, 0.0, 20.0, "Setup content.", importance=50.0),
            mk_block(1, 100.0, 140.0, "Long payoff sequence with the reveal at the end.",
                     importance=80.0, has_stakes=True),
        ]
        plan = StoryPlan(
            video_type="challenge",
            units=[{
                "unit_id": 0, "name": "Story", "priority": 1, "region": "late",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                    {"beat": "payoff", "position": "end", "flags": ["STAKES"], "intent": "reveal"},
                ],
            }],
        )
        groups = execute_plan(plan, blocks, 150.0, 45, 100)
        payoff = groups[0]["source_clips"][-1]
        # Payoff must cover the reveal (block end 140.0) — may be extended for duration
        assert payoff["source_end"] >= 140.0, "payoff must cover the block end"
        assert payoff["source_end"] <= 150.0, "payoff must not exceed source duration"

    def test_reel_order_hook_first_payoff_last(self):
        blocks = spread_blocks(40, 1200.0)
        groups = execute_plan(standard_plan(), blocks, 1200.0, 90, 100)
        for g in groups:
            clips = g["source_clips"]
            assert clips[0]["is_hook_clip"] is True
            assert "PAYOFF" in clips[-1]["reason"]


class TestEscalationPositioning:
    def test_escalations_spread_across_window(self):
        """Multiple escalations must come from different parts of the window."""
        blocks = spread_blocks(40, 1200.0)
        plan = StoryPlan(video_type="challenge", units=[{
            "unit_id": 0, "name": "Story", "priority": 1, "region": "mid",
            "arc": [
                {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                {"beat": "escalation", "position": "start", "flags": [], "intent": "early tension"},
                {"beat": "escalation", "position": "any", "flags": [], "intent": "middle tension"},
                {"beat": "escalation", "position": "end", "flags": [], "intent": "peak tension"},
                {"beat": "payoff", "position": "end", "flags": [], "intent": "result"},
            ],
        }])
        groups = execute_plan(plan, blocks, 1200.0, 90, 100)
        esc = [c["source_start"] for c in groups[0]["source_clips"] if "ESCALATION" in c["reason"]]
        assert len(esc) >= 3, f"expected >=3 escalation clips, got {len(esc)}"
        assert max(esc) - min(esc) >= 300.0, f"escalations clustered: {esc}"

    def test_escalation_end_hint_prefers_late_slot(self):
        blocks = spread_blocks(40, 1200.0)
        plan = StoryPlan(video_type="other", units=[{
            "unit_id": 0, "name": "Story", "priority": 1, "region": "mid",
            "arc": [
                {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                {"beat": "escalation", "position": "end", "flags": [], "intent": "tension"},
                {"beat": "payoff", "position": "end", "flags": [], "intent": "result"},
            ],
        }])
        groups = execute_plan(plan, blocks, 1200.0, 90, 100)
        esc = [c["source_start"] for c in groups[0]["source_clips"] if "ESCALATION" in c["reason"]]
        assert esc, "no escalation picked"
        assert max(esc) >= 1200.0 * 0.65, "the end-hinted escalation must come from the late window"

    def test_escalation_start_hint_prefers_early_slot(self):
        blocks = spread_blocks(40, 1200.0)
        plan = StoryPlan(video_type="other", units=[{
            "unit_id": 0, "name": "Story", "priority": 1, "region": "mid",
            "arc": [
                {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                {"beat": "escalation", "position": "start", "flags": [], "intent": "tension"},
                {"beat": "payoff", "position": "end", "flags": [], "intent": "result"},
            ],
        }])
        groups = execute_plan(plan, blocks, 1200.0, 90, 100)
        esc = [c["source_start"] for c in groups[0]["source_clips"] if "ESCALATION" in c["reason"]]
        assert esc, "no escalation picked"
        assert min(esc) <= 1200.0 * 0.45, "the start-hinted escalation must come from the early window"

    def test_escalation_picks_respect_budget(self):
        blocks = spread_blocks(40, 1200.0)
        plan = StoryPlan(video_type="other", units=[{
            "unit_id": 0, "name": "Story", "priority": 1, "region": "mid",
            "arc": [
                {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                {"beat": "escalation", "position": "any", "flags": [], "intent": "tension"},
                {"beat": "payoff", "position": "end", "flags": [], "intent": "result"},
            ],
        }])
        groups = execute_plan(plan, blocks, 1200.0, 90, 100)
        total = sum(c["source_end"] - c["source_start"] for c in groups[0]["source_clips"])
        assert total <= 100.0 + 0.01


# ---------------------------------------------------------------------------
# Beat relevance (semantic layer) — LLM ranks blocks, Python still picks
# ---------------------------------------------------------------------------

class TestRelevanceRanking:
    def test_relevance_bonus_tiers(self):
        from backend.pipeline.plan_executor import _relevance_bonus
        ranked = [10, 11, 12, 13, 14, 15]
        assert _relevance_bonus(ranked, 10) == 25.0
        assert _relevance_bonus(ranked, 11) == 18.0
        assert _relevance_bonus(ranked, 12) == 12.0
        assert _relevance_bonus(ranked, 13) == 8.0
        assert _relevance_bonus(ranked, 14) == 5.0
        assert _relevance_bonus(ranked, 15) == 0.0, "beyond tier list"
        assert _relevance_bonus(ranked, 99) == 0.0, "unranked block"
        assert _relevance_bonus([], 1) == 0.0, "empty rankings"

    def test_escalation_ranked_block_wins_inside_slot(self):
        """A lower-importance block that matches the intent beats loud noise."""
        from backend.pipeline.plan_executor import _pick_escalation_blocks
        from backend.pipeline.plan_schema import Beat

        b_loud = mk_block(0, 50.0, 65.0, "Loud roaring crowd noise over the pitch.", importance=95.0)
        b_match = mk_block(1, 66.0, 80.0, "He walks toward the pitch and the players come out.",
                           importance=40.0)
        beats = [Beat(beat="escalation", position="any", intent="tension builds")]
        window = (50.0, 100.0)

        no_rank = _pick_escalation_blocks([b_loud, b_match], None, None, 10.0, window, beats)
        assert [b.block_id for b in no_rank] == [0], "importance-only picks the loud block"

        ranked = _pick_escalation_blocks([b_loud, b_match], None, None, 10.0, window, beats, [1])
        assert [b.block_id for b in ranked] == [1], "ranked block must win the slot"

    def test_hook_ranked_block_preferred(self):
        blocks = [
            mk_block(0, 0.0, 8.0, "The final result lands.", importance=90.0, silence_before=True),
            mk_block(1, 9.0, 16.0, "What is that thing?! Who brought it here?", importance=50.0,
                     has_question=True, has_exclamation=True),
            mk_block(2, 17.0, 24.0, "There he is, right in front of me.", importance=30.0,
                     silence_before=True),
        ]
        plan = StoryPlan(video_type="other", units=[{
            "unit_id": 0, "name": "Story", "priority": 1, "region": "early",
            "arc": [
                {"beat": "hook", "position": "start", "flags": [], "intent": "the once-in-a-lifetime chance"},
                {"beat": "payoff", "position": "end", "flags": [], "intent": "result"},
            ],
        }])

        no_rank = execute_plan(plan, blocks, 100.0, 45, 100)
        hook_no = next(c for c in no_rank[0]["source_clips"] if c["is_hook_clip"])
        assert 9.0 <= hook_no["source_start"] < 16.0, "curiosity scoring picks the question block"

        ranked = execute_plan(plan, blocks, 100.0, 45, 100, {0: {"hook": [2]}})
        hook_rk = next(c for c in ranked[0]["source_clips"] if c["is_hook_clip"])
        assert 17.0 <= hook_rk["source_start"] < 24.0, "ranked block must win the hook"

    def test_payoff_ranked_block_preferred(self):
        blocks = [
            mk_block(0, 0.0, 40.0, "Loud excited shouting about anything.", importance=95.0),
            mk_block(1, 76.0, 88.0, "More excited shouting, still nothing specific.", importance=95.0),
            mk_block(2, 89.0, 99.0, "The near-miss, the picture stays imperfect.", importance=40.0),
        ]
        plan = StoryPlan(video_type="vlog", units=[{
            "unit_id": 0, "name": "Story", "priority": 1, "region": "late",
            "arc": [
                {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                {"beat": "payoff", "position": "end", "flags": [], "intent": "the picture is imperfect"},
            ],
        }])

        no_rank = execute_plan(plan, blocks, 100.0, 45, 100)
        payoff_no = no_rank[0]["source_clips"][-1]
        assert payoff_no["source_start"] < 89.0, "importance picks the loud late block"

        ranked = execute_plan(plan, blocks, 100.0, 45, 100, {0: {"payoff": [2]}})
        payoff_rk = ranked[0]["source_clips"][-1]
        assert 89.0 <= payoff_rk["source_start"] < 99.0, "ranked payoff block must be chosen"

    def test_unknown_block_ids_in_rankings_ignored(self):
        blocks = spread_blocks(10, 300.0)
        plan = StoryPlan(video_type="other", units=[{
            "unit_id": 0, "name": "Story", "priority": 1, "region": "mid",
            "arc": [
                {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                {"beat": "payoff", "position": "end", "flags": [], "intent": "result"},
            ],
        }])
        groups = execute_plan(plan, blocks, 300.0, 45, 100, {0: {"payoff": [999], "hook": [998]}})
        assert groups[0]["source_clips"], "rankings with unknown ids must not break picking"

    def test_deterministic_given_fixed_relevance(self):
        blocks = spread_blocks(40, 1200.0)
        plan = standard_plan()
        rel = {0: {"escalation": [5, 7]}, 1: {"payoff": [30]}}
        g1 = execute_plan(plan, blocks, 1200.0, 90, 100, rel)
        g2 = execute_plan(plan, blocks, 1200.0, 90, 100, rel)
        assert g1 == g2, "same plan + same rankings must give identical groups"


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------

class TestHardConstraints:
    def test_no_cross_group_overlap(self):
        blocks = spread_blocks(40, 1200.0)
        groups = execute_plan(standard_plan(), blocks, 1200.0, 90, 100)
        taken = []
        for g in groups:
            for c in g["source_clips"]:
                s, e = c["source_start"], c["source_end"]
                for s0, e0 in taken:
                    assert not (s < e0 - 0.05 and s0 < e - 0.05), f"overlap {s:.1f}-{e:.1f} vs {s0:.1f}-{e0:.1f}"
                taken.append((s, e))

    def test_duration_budget_respected(self):
        blocks = spread_blocks(40, 1200.0)
        groups = execute_plan(standard_plan(), blocks, 1200.0, 90, 100)
        for g in groups:
            total = sum(c["source_end"] - c["source_start"] for c in g["source_clips"])
            assert total <= 100.0 + 0.01, f"group over budget: {total:.1f}s"

    def test_clip_bounds_valid(self):
        blocks = spread_blocks(40, 1200.0)
        groups = execute_plan(standard_plan(), blocks, 1200.0, 90, 100)
        for g in groups:
            for c in g["source_clips"]:
                assert 0.0 <= c["source_start"] < c["source_end"] <= 1200.0
                assert c["source_end"] - c["source_start"] >= 3.0

    def test_black_freeze_blocks_excluded(self):
        good = [
            mk_block(0, 0.0, 20.0, "Good content.", importance=60.0),
            mk_block(1, 40.0, 60.0, "Great payoff.", importance=80.0, has_stakes=True),
        ]
        bad = SemanticBlock(
            block_id=2, start=20.0, end=40.0, text="Freeze frame.",
            speech_energy=0.5, volume_db=-18.0, silence_before=False,
            black_frame=False, freeze=True, importance=95.0, peak_offset=10.0,
        )
        black = SemanticBlock(
            block_id=3, start=60.0, end=80.0, text="Black frame.",
            speech_energy=0.5, volume_db=-18.0, silence_before=False,
            black_frame=True, freeze=False, importance=95.0, peak_offset=10.0,
        )
        plan = StoryPlan(
            video_type="other",
            units=[{
                "unit_id": 0, "name": "Story", "priority": 1, "region": "mid",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                    {"beat": "payoff", "position": "end", "flags": [], "intent": "result"},
                ],
            }],
        )
        groups = execute_plan(plan, [good[0], bad, good[1], black], 90.0, 45, 100)
        for c in groups[0]["source_clips"]:
            assert not (c["source_start"] < 40.0 < c["source_end"]), "freeze block used"
            assert not (c["source_start"] < 80.0 < c["source_end"]), "black block used"
        # The two usable blocks must have been picked (both inside mid region)
        picked_starts = [c["source_start"] for c in groups[0]["source_clips"]]
        assert any(40.0 <= s < 60.0 for s in picked_starts), "good mid block not picked"

    def test_empty_window_falls_back_to_whole_source(self):
        """Unit with an unusable window must still produce clips (min_groups floor)."""
        blocks = [
            mk_block(0, 400.0, 420.0, "Only usable content.", importance=60.0),
            mk_block(1, 500.0, 520.0, "Payoff content.", importance=80.0, has_stakes=True),
        ]
        plan = StoryPlan(
            video_type="other",
            units=[{
                "unit_id": 0, "name": "Story", "priority": 1, "region": "early",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                    {"beat": "payoff", "position": "end", "flags": ["STAKES"], "intent": "result"},
                ],
            }],
        )
        groups = execute_plan(plan, blocks, 600.0, 45, 100)
        assert len(groups) == 1
        assert groups[0]["source_clips"]


# ---------------------------------------------------------------------------
# Narration placement
# ---------------------------------------------------------------------------

class TestNarrationPlacement:
    def _group(self) -> dict:
        return {
            "group_index": 0,
            "estimated_duration_seconds": 92.0,
            "source_clips": [
                {"source_start": 10.0, "source_end": 16.0, "is_hook_clip": True},   # 6s hook
                {"source_start": 30.0, "source_end": 46.0, "is_hook_clip": False},  # 16s esc
                {"source_start": 80.0, "source_end": 95.0, "is_hook_clip": False},  # 15s payoff
            ],
        }

    def test_hook_starts_at_zero(self):
        group = self._group()
        events = [
            {"event_type": "hook", "text": "This is a hook line!", "persona": None},
            {"event_type": "commentary", "text": "A commentary line here.", "persona": "roast"},
        ]
        placed = place_narration_events(group, events)
        assert placed[0]["event_type"] == "hook"
        assert placed[0]["reel_start"] == 0.0

    def test_commentary_after_hook_clip_with_gap(self):
        group = self._group()
        events = [
            {"event_type": "hook", "text": "This is a hook line!", "persona": None},
            {"event_type": "commentary", "text": "A commentary line here.", "persona": "roast"},
        ]
        placed = place_narration_events(group, events)
        hook_end = placed[0]["reel_end"]
        c1 = placed[1]
        assert c1["reel_start"] >= hook_end + 0.8 - 0.01

    def test_two_commentaries_do_not_overlap(self):
        group = self._group()
        events = [
            {"event_type": "hook", "text": "This is a hook line!", "persona": None},
            {"event_type": "commentary", "text": "First commentary line.", "persona": "roast"},
            {"event_type": "commentary", "text": "Second commentary line.", "persona": "hype"},
        ]
        placed = place_narration_events(group, events)
        assert len(placed) == 3
        for a, b in itertools.pairwise(placed):
            assert b["reel_start"] >= a["reel_end"] + 0.8 - 0.01

    def test_last_three_seconds_free(self):
        group = self._group()
        events = [
            {"event_type": "hook", "text": "This is a hook line!", "persona": None},
            {"event_type": "commentary", "text": "First commentary line.", "persona": "roast"},
            {"event_type": "commentary", "text": "Second commentary line.", "persona": "hype"},
        ]
        placed = place_narration_events(group, events)
        est = group["estimated_duration_seconds"]
        for e in placed:
            assert e["reel_end"] <= est - 3.0 + 0.01

    def test_caps_at_three_events(self):
        group = self._group()
        events = (
            [{"event_type": "hook", "text": "Hook!", "persona": None}]
            + [{"event_type": "commentary", "text": f"Line {i} here please.", "persona": "roast"} for i in range(4)]
        )
        placed = place_narration_events(group, events)
        assert len(placed) == 3

    def test_unknown_event_types_dropped(self):
        group = self._group()
        events = [
            {"event_type": "hook", "text": "Hook line here!", "persona": None},
            {"event_type": "joke", "text": "Not a real type.", "persona": "roast"},
        ]
        placed = place_narration_events(group, events)
        assert all(e["event_type"] in ("hook", "commentary") for e in placed)
        assert len(placed) == 1


def test_estimate_speech_duration():
    assert estimate_speech_duration("five words only here") == pytest.approx(2.0)
    assert estimate_speech_duration("word " * 40) == pytest.approx(8.0)
    assert estimate_speech_duration("") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Plan schema validation / repair
# ---------------------------------------------------------------------------

class TestPlanSchema:
    def test_repairs_missing_payoff_and_duplicate_hook(self):
        data = {
            "video_type": "challenge",
            "units": [
                {
                    "unit_id": 0, "name": "Story A", "priority": 5, "region": "weird",
                    "arc": [
                        {"beat": "hook", "position": "start", "flags": ["STAKES"], "intent": "x"},
                        {"beat": "hook", "position": "start", "flags": [], "intent": "y"},
                        {"beat": "escalation", "position": "any", "flags": [], "intent": "z"},
                    ],
                }
            ],
        }
        plan = parse_story_plan(data, 600.0, 1, 6, hook_mode="required")
        unit = plan.units[0]
        assert unit.priority == 3, "priority clamped to 1-3"
        assert unit.region == "early", "top-priority unit must open the video (opening guarantee)"
        beats = [b.beat for b in unit.arc]
        assert beats.count("hook") == 1
        assert beats.count("payoff") == 1
        assert beats[0] == "hook"

    def test_duplicate_unit_ids_renumbered(self):
        data = {
            "video_type": "other",
            "units": [
                {"unit_id": 7, "name": "B", "priority": 1, "region": "mid",
                 "arc": [{"beat": "hook", "position": "start", "flags": [], "intent": ""}]},
                {"unit_id": 7, "name": "A", "priority": 1, "region": "mid",
                 "arc": [{"beat": "hook", "position": "start", "flags": [], "intent": ""}]},
                {"unit_id": 2, "name": "C", "priority": 1, "region": "mid",
                 "arc": [{"beat": "hook", "position": "start", "flags": [], "intent": ""}]},
            ],
        }
        plan = parse_story_plan(data, 600.0, 1, 6)
        ids = [u.unit_id for u in plan.units]
        assert ids == sorted(ids) and len(set(ids)) == len(ids), "unit_ids must be unique & ordered"
        windows = {u.unit_id: (0.0, 600.0) for u in plan.units}
        for u in plan.units:
            windows[u.unit_id] = _unit_windows(plan, 600.0)[u.unit_id]
        assert len(windows) == 3, "every unit must own its own window"

    def test_unit_count_clamped_to_ceiling(self):
        units = [
            {"unit_id": i, "name": f"Story {i}", "priority": 1, "region": "mid",
             "arc": [{"beat": "hook", "position": "start", "flags": [], "intent": ""}]}
            for i in range(10)
        ]
        plan = parse_story_plan({"video_type": "other", "units": units}, 1200.0, 1, 4)
        assert len(plan.units) == 4

    def test_unit_count_padded_to_floor(self):
        data = {
            "video_type": "other",
            "units": [
                {"unit_id": 0, "name": "Only story", "priority": 1, "region": "mid",
                 "arc": [{"beat": "hook", "position": "start", "flags": [], "intent": ""}]},
            ],
        }
        plan = parse_story_plan(data, 1200.0, 3, 6)
        assert len(plan.units) == 3

    def test_zero_units_falls_back_to_heuristic(self):
        plan = parse_story_plan({"video_type": "other", "units": []}, 1200.0, 2, 6)
        assert len(plan.units) >= 2

    def test_heuristic_plan_shape(self):
        plan = heuristic_story_plan(1800.0, 3, 6, hook_mode="required")
        assert 3 <= len(plan.units) <= 6
        for u in plan.units:
            beats = [b.beat for b in u.arc]
            assert "hook" in beats and "payoff" in beats

    def test_positions_normalized_to_beat_role(self):
        data = {
            "video_type": "other",
            "units": [{
                "unit_id": 0, "name": "S", "priority": 1, "region": "mid",
                "arc": [
                    {"beat": "hook", "position": "any", "flags": [], "intent": ""},
                    {"beat": "escalation", "position": "end", "flags": [], "intent": ""},
                    {"beat": "payoff", "position": "any", "flags": [], "intent": ""},
                ],
            }],
        }
        plan = parse_story_plan(data, 600.0, 1, 6, hook_mode="required")
        by_beat = {b.beat: b.position for b in plan.units[0].arc}
        assert by_beat["hook"] == "start", "hook is the opening trigger, never 'any'"
        assert by_beat["payoff"] == "end", "payoff is the closing reveal, never 'any'"
        assert by_beat["escalation"] == "end", "explicit escalation hint is kept"

    def test_missing_escalation_position_defaults_to_any(self):
        data = {
            "video_type": "other",
            "units": [{
                "unit_id": 0, "name": "S", "priority": 1, "region": "mid",
                "arc": [{"beat": "escalation", "flags": [], "intent": ""}],
            }],
        }
        plan = parse_story_plan(data, 600.0, 1, 6)
        esc = next(b for b in plan.units[0].arc if b.beat == "escalation")
        assert esc.position == "any"

    def test_plan_to_structure_analysis(self):
        plan = standard_plan()
        sa = plan_to_structure_analysis(plan, reasoning="test")
        assert sa["video_type"] == "challenge"
        assert sa["final_group_count"] == 2
        assert len(sa["identified_units"]) == 2


# ---------------------------------------------------------------------------
# Executor pipeline integration (monkeypatched LLM)
# ---------------------------------------------------------------------------

PLANNER_JSON = json.dumps({
    "video_type": "challenge",
    "units": [
        {
            "unit_id": 0, "name": "The early stunt", "priority": 1, "region": "early",
            "arc": [
                {"beat": "hook", "position": "start", "flags": ["STAKES"], "intent": "the risk is announced"},
                {"beat": "escalation", "position": "any", "flags": [], "intent": "tension builds"},
                {"beat": "payoff", "position": "end", "flags": ["STAKES"], "intent": "the winner is revealed"},
            ],
        },
        {
            "unit_id": 1, "name": "The late reveal", "priority": 1, "region": "late",
            "arc": [
                {"beat": "hook", "position": "start", "flags": [], "intent": "a mystery opens"},
                {"beat": "payoff", "position": "end", "flags": ["ROAST"], "intent": "the mic-drop reaction"},
            ],
        },
    ],
})

WRITER_JSON = json.dumps({
    "reel_groups": [
        {
            "group_index": 0,
            "reel_summary": {
                "title": "The Early Stunt",
                "short_description": "A risk taken at dawn",
                "source_understanding": "challenge unit",
                "narrative_angle": "stakes",
                "key_moment": "the winner reveal",
            },
            "narration_events": [
                {"event_type": "hook", "text": "This risk should have been illegal.", "persona": None},
                {"event_type": "commentary", "text": "He slipped and the crowd lost it.", "persona": "hype"},
                {"event_type": "commentary", "text": "The cash is his. All of it.", "persona": "deadpan"},
            ],
        },
        {
            "group_index": 1,
            "reel_summary": {
                "title": "The Late Reveal",
                "short_description": "A mystery resolved",
                "source_understanding": "reaction unit",
                "narrative_angle": "roast",
                "key_moment": "the final reaction",
            },
            "narration_events": [
                {"event_type": "hook", "text": "He has no idea what is coming.", "persona": None},
                {"event_type": "commentary", "text": "That face. That priceless face.", "persona": "roast"},
            ],
        },
        {
            "group_index": 2,
            "reel_summary": {
                "title": "The Mid Story",
                "short_description": "Tension in the middle",
                "source_understanding": "escalation unit",
                "narrative_angle": "stakes",
                "key_moment": "the near miss",
            },
            "narration_events": [
                {"event_type": "hook", "text": "Nobody expects the middle twist.", "persona": None},
                {"event_type": "commentary", "text": "That was way too close.", "persona": "sarcastic"},
            ],
        },
    ],
})


class TestExecutorPipeline:
    def test_executor_pipeline_full(self, monkeypatch, mock_reporter):
        from backend.pipeline import analyzer as analyzer_module

        def fake_call_llm(messages, progress_cb=None, reporter=None, interactions=None,
                          stage_name="", max_tokens=0, **kwargs):
            return PLANNER_JSON if stage_name == "story_planner" else WRITER_JSON

        monkeypatch.setattr(analyzer_module, "_call_llm", fake_call_llm)

        blocks = spread_blocks(40, 1200.0)

        reel_plan = analyzer_module._select_reel_plan_executor(
            video_title="Test video",
            video_description="A description",
            progress_cb=None,
            reporter=mock_reporter,
            interactions=None,
            rich_timeline=None,
            source_duration=1200.0,
            min_groups=3,
            max_groups=6,
            reel_dur_min=90,
            reel_dur_max=100,
            blocks=blocks,
            blocks_text="",
            usable_hints="",
        )

        assert len(reel_plan.reel_groups) >= 3, "floor enforced"
        for g in reel_plan.reel_groups:
            assert g.source_clips
            assert g.reel_summary.title, "writer summary applied"
            assert g.narration_events, "narration placed"
            assert g.narration_events[0].event_type == "hook"
            assert g.narration_events[0].reel_start == 0.0
            for e in g.narration_events:
                assert e.reel_end > e.reel_start >= 0.0

        # no cross-group overlap
        taken = []
        for g in reel_plan.reel_groups:
            for c in g.source_clips:
                s, e = c.source_start, c.source_end
                for s0, e0 in taken:
                    assert not (s < e0 - 0.05 and s0 < e - 0.05)
                taken.append((s, e))

    def test_executor_falls_back_to_heuristic_plan(self, monkeypatch, mock_reporter):
        from backend.pipeline import analyzer as analyzer_module

        calls = {"n": 0}

        def failing_llm(messages, progress_cb=None, reporter=None, interactions=None,
                        stage_name="", max_tokens=0, **kwargs):
            calls["n"] += 1
            if stage_name in ("story_planner", "genre_story_planner"):
                raise RuntimeError("API down")
            return WRITER_JSON

        monkeypatch.setattr(analyzer_module, "_call_llm", failing_llm)

        blocks = spread_blocks(40, 1200.0)
        reel_plan = analyzer_module._select_reel_plan_executor(
            video_title="Test video", video_description="", progress_cb=None,
            reporter=mock_reporter, interactions=None, rich_timeline=None,
            source_duration=1200.0, min_groups=3, max_groups=6,
            reel_dur_min=90, reel_dur_max=100, blocks=blocks,
            blocks_text="", usable_hints="",
        )
        assert calls["n"] == 5, "2 planner attempts + 1 ranker + 1 writer + 1 completeness_critic"
        assert len(reel_plan.reel_groups) >= 3


# ---------------------------------------------------------------------------
# Phase 5 — start BeatType tests
# ---------------------------------------------------------------------------

class TestStartBeatType:
    """Tests for the 'start' BeatType in plan_schema and plan_executor."""

    def test_heuristic_plan_skip_mode_has_start_no_hook(self):
        plan = heuristic_story_plan(1800.0, 3, 6, hook_mode="skip")
        for u in plan.units:
            beats = [b.beat for b in u.arc]
            assert "start" in beats, "skip mode must include start beat"
            assert "hook" not in beats, "skip mode must not include hook beat"
            assert "payoff" in beats

    def test_heuristic_plan_required_mode_has_hook_and_start(self):
        plan = heuristic_story_plan(1800.0, 3, 6, hook_mode="required")
        for u in plan.units:
            beats = [b.beat for b in u.arc]
            assert "hook" in beats, "required mode must include hook beat"
            assert "start" in beats, "required mode must include start beat"
            assert "payoff" in beats

    def test_heuristic_plan_skip_mode_has_start_no_hook(self):
        plan = heuristic_story_plan(1800.0, 3, 6, hook_mode="skip")
        for u in plan.units:
            beats = [b.beat for b in u.arc]
            assert "start" in beats, "skip mode must include start beat"
            assert "hook" not in beats, "skip mode must not include hook beat"
            assert "payoff" in beats

    def test_parse_story_plan_skip_mode_generates_start(self):
        data = {
            "video_type": "other",
            "units": [{
                "unit_id": 0, "name": "S", "priority": 1, "region": "mid",
                "arc": [
                    {"beat": "escalation", "position": "any", "flags": [], "intent": ""},
                    {"beat": "payoff", "position": "end", "flags": [], "intent": ""},
                ],
            }],
        }
        plan = parse_story_plan(data, 600.0, 1, 6, hook_mode="skip")
        beats = [b.beat for b in plan.units[0].arc]
        assert "start" in beats, "skip mode injects start beat"
        assert "hook" not in beats

    def test_parse_story_plan_required_mode_keeps_hook_and_start(self):
        data = {
            "video_type": "other",
            "units": [{
                "unit_id": 0, "name": "S", "priority": 1, "region": "mid",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": [], "intent": ""},
                    {"beat": "escalation", "position": "any", "flags": [], "intent": ""},
                    {"beat": "payoff", "position": "end", "flags": [], "intent": ""},
                ],
            }],
        }
        plan = parse_story_plan(data, 600.0, 1, 6, hook_mode="required")
        beats = [b.beat for b in plan.units[0].arc]
        assert "hook" in beats
        assert "start" in beats, "required mode always includes start"
        assert "payoff" in beats

    def test_reorder_reel_clips_starts_with_start_beat(self):
        from backend.pipeline.plan_executor import _reorder_reel_clips

        clips = [
            {"_beat": "escalation", "source_start": 10.0, "source_end": 20.0},
            {"_beat": "start", "source_start": 0.0, "source_end": 8.0},
            {"_beat": "payoff", "source_start": 40.0, "source_end": 50.0},
        ]
        _reorder_reel_clips(clips)
        assert clips[0]["_beat"] == "start"
        assert clips[-1]["_beat"] == "payoff"

    def test_place_narration_events_start_event_at_zero(self):
        group = {
            "estimated_duration_seconds": 30.0,
            "source_clips": [
                {"source_start": 0.0, "source_end": 8.0},
                {"source_start": 8.0, "source_end": 18.0},
                {"source_start": 18.0, "source_end": 30.0},
            ],
            "narration_events": [
                {"event_type": "start", "text": "Two players enter the arena.", "persona": None},
                {"event_type": "commentary", "text": "First move matters.", "persona": "neutral"},
                {"event_type": "commentary", "text": "The crowd goes quiet.", "persona": "neutral"},
            ],
        }
        placed = place_narration_events(group)
        start_ev = placed[0]
        assert start_ev["event_type"] == "start"
        assert start_ev["reel_start"] == 0.0
        assert start_ev["reel_end"] > 0.0

    def test_execute_plan_skip_mode_uses_start_as_opening(self, mock_reporter):
        plan = StoryPlan(
            video_type="other",
            units=[{
                "unit_id": 0, "name": "Test", "priority": 1, "region": "early",
                "arc": [
                    {"beat": "start", "position": "start", "flags": [], "intent": "introduce scene"},
                    {"beat": "escalation", "position": "any", "flags": [], "intent": "tension"},
                    {"beat": "payoff", "position": "end", "flags": [], "intent": "reveal"},
                ],
            }],
        )
        blocks = [mk_block(i, float(i * 10), float((i + 1) * 10), f"block {i}", 0) for i in range(5)]
        groups = execute_plan(plan, blocks, 60.0, 30, 50, {0: {"start": [0], "escalation": [1], "payoff": [4]}})
        assert len(groups) > 0
        clips = groups[0]["source_clips"]
        assert len(clips) > 0
        opening = clips[0]
        assert opening["is_hook_clip"] is False
        assert opening["beat"] == "start"


# ---------------------------------------------------------------------------
# hook_mode="skip" enforcement — regression tests
# ---------------------------------------------------------------------------

class TestHookModeSkipEnforcement:
    """Verify hook_mode='skip' deterministically strips hook beats from the plan.

    These are the regression tests that would have caught the original bug where
    'skip' was indistinguishable from 'auto' — the LLM could include a hook beat
    and no deterministic backend enforcement existed to strip it.
    """

    def test_repair_unit_strips_hook_beat_in_skip_mode(self):
        """_repair_unit with hook_mode='skip' must remove any hook beat from unit.arc."""
        from backend.pipeline.plan_schema import _repair_unit, Beat, Unit
        unit = Unit(
            unit_id=0, name="Test", priority=1, region="mid",
            arc=[
                Beat(beat="hook", position="start", intent="curiosity trigger"),
                Beat(beat="escalation", position="any", intent="tension"),
                Beat(beat="payoff", position="end", intent="reveal"),
            ],
        )
        repaired = _repair_unit(unit, hook_mode="skip")
        beats = [b.beat for b in repaired.arc]
        assert "hook" not in beats, "skip mode must strip hook beats"
        assert "start" in beats, "skip mode must insert start beat"
        assert "escalation" in beats
        assert "payoff" in beats
        assert beats[0] == "start", "start must be the opening beat"

    def test_repair_unit_keeps_hook_in_required_mode(self):
        """_repair_unit with hook_mode='required' must keep/add hook beat."""
        from backend.pipeline.plan_schema import _repair_unit, Beat, Unit
        unit = Unit(
            unit_id=0, name="Test", priority=1, region="mid",
            arc=[
                Beat(beat="hook", position="start", intent="curiosity trigger"),
                Beat(beat="escalation", position="any", intent="tension"),
                Beat(beat="payoff", position="end", intent="reveal"),
            ],
        )
        repaired = _repair_unit(unit, hook_mode="required")
        beats = [b.beat for b in repaired.arc]
        assert "hook" in beats, "required mode must keep hook beat"
        assert beats[0] == "hook", "hook must be first in required mode"

    def test_skip_mode_plan_has_zero_hook_clips(self):
        """End-to-end: hook_mode='skip' produces zero clips with _beat='hook'.

        This is THE regression test: if a permissive LLM returns a plan with
        hook beats and the backend doesn't enforce 'skip', this test catches it.

        Note: is_hook_clip=True is used for the opening clip (beat type "start")
        in skip mode — that's expected. The assertion checks that no clip is
        labeled with _beat='hook', which means no hook beat was executed.
        """
        # Simulate a permissive LLM that included hook beats despite skip mode
        plan_with_hook = StoryPlan(
            video_type="challenge",
            units=[{
                "unit_id": 0, "name": "Story", "priority": 1, "region": "early",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": ["STAKES"], "intent": "curiosity"},
                    {"beat": "escalation", "position": "any", "flags": [], "intent": "tension"},
                    {"beat": "payoff", "position": "end", "flags": ["STAKES"], "intent": "reveal"},
                ],
            }],
        )
        # Parse through repair (which enforces skip) then execute
        from backend.pipeline.plan_schema import parse_story_plan
        repaired_plan = parse_story_plan(
            plan_with_hook.model_dump(), 120.0, 1, 6, hook_mode="skip",
        )
        # Verify the plan itself has no hook beat
        for unit in repaired_plan.units:
            beats = [b.beat for b in unit.arc]
            assert "hook" not in beats, f"repaired plan must not have hook beat, got {beats}"

        blocks = spread_blocks(40, 120.0)
        groups = execute_plan(repaired_plan, blocks, 120.0, 45, 100)
        # No clip in any group should have _beat='hook' — the opening clip
        # uses _beat='start' in skip mode, not 'hook'
        for g in groups:
            for c in g["source_clips"]:
                assert c.get("_beat") != "hook", (
                    f"hook_mode='skip' produced a hook beat clip: {c.get('reason')}"
                )

    def test_heuristic_plan_skip_mode_has_no_hook(self):
        """heuristic_story_plan with hook_mode='skip' must produce no hook beats."""
        from backend.pipeline.plan_schema import heuristic_story_plan
        plan = heuristic_story_plan(600.0, 2, 4, hook_mode="skip")
        for unit in plan.units:
            beats = [b.beat for b in unit.arc]
            assert "hook" not in beats
            assert "start" in beats
            assert "payoff" in beats

    def test_skip_mode_start_clip_used_as_opening(self):
        """When hook_mode='skip', the opening clip should be labeled as 'start' beat."""
        plan = StoryPlan(
            video_type="other",
            units=[{
                "unit_id": 0, "name": "Test", "priority": 1, "region": "early",
                "arc": [
                    {"beat": "start", "position": "start", "flags": [], "intent": "introduce scene"},
                    {"beat": "escalation", "position": "any", "flags": [], "intent": "tension"},
                    {"beat": "payoff", "position": "end", "flags": [], "intent": "reveal"},
                ],
            }],
        )
        blocks = spread_blocks(20, 120.0)
        groups = execute_plan(plan, blocks, 120.0, 45, 100)
        assert len(groups) == 1
        opening = groups[0]["source_clips"][0]
        assert opening["is_hook_clip"] is False  # start beat is not a hook clip
        assert opening["beat"] == "start", "opening should be labeled as start beat"


# ---------------------------------------------------------------------------
# Scene-cut snapping tests
# ---------------------------------------------------------------------------

class TestSceneCutSnapping:
    """Tests for scene-cut boundary snapping in plan_executor."""

    def test_snap_boundary_near_scene_cut(self):
        """A clip boundary within tolerance of a scene cut should be snapped."""
        from backend.pipeline.plan_executor import _snap_to_scene_cut, SCENE_CUT_SNAP_TOLERANCE
        # Boundary at 10.2s, scene cut at 10.0s — within tolerance
        snapped = _snap_to_scene_cut(10.2, [10.0, 50.0, 100.0])
        assert snapped == 10.0, "should snap to the nearby scene cut"

    def test_snap_boundary_far_from_scene_cut(self):
        """A clip boundary far from any scene cut should remain unchanged."""
        from backend.pipeline.plan_executor import _snap_to_scene_cut
        # Boundary at 25.0s, nearest cut at 10.0s — far away
        snapped = _snap_to_scene_cut(25.0, [10.0, 50.0, 100.0])
        assert snapped == 25.0, "should not snap when no cut within tolerance"

    def test_snap_boundary_exact_on_cut(self):
        """A clip boundary exactly on a scene cut should stay there."""
        from backend.pipeline.plan_executor import _snap_to_scene_cut
        snapped = _snap_to_scene_cut(50.0, [10.0, 50.0, 100.0])
        assert snapped == 50.0

    def test_snap_boundary_at_tolerance_edge(self):
        """A clip boundary exactly at tolerance distance should snap."""
        from backend.pipeline.plan_executor import _snap_to_scene_cut, SCENE_CUT_SNAP_TOLERANCE
        cut = 10.0
        boundary = cut + SCENE_CUT_SNAP_TOLERANCE
        snapped = _snap_to_scene_cut(boundary, [cut])
        assert snapped == cut, "should snap at exactly the tolerance edge"

    def test_snap_boundary_just_beyond_tolerance(self):
        """A clip boundary just beyond tolerance should not snap."""
        from backend.pipeline.plan_executor import _snap_to_scene_cut, SCENE_CUT_SNAP_TOLERANCE
        cut = 10.0
        boundary = cut + SCENE_CUT_SNAP_TOLERANCE + 0.01
        snapped = _snap_to_scene_cut(boundary, [cut])
        assert snapped == boundary, "should not snap beyond tolerance"

    def test_snap_clip_boundaries_snaps_both_start_and_end(self):
        """_snap_clip_boundaries should snap both start and end of each clip."""
        from backend.pipeline.plan_executor import _snap_clip_boundaries
        clips = [
            {"source_start": 10.2, "source_end": 20.3},  # near cuts at 10.0 and 20.0
        ]
        _snap_clip_boundaries(clips, [10.0, 20.0, 50.0])
        assert clips[0]["source_start"] == 10.0
        assert clips[0]["source_end"] == 20.0

    def test_snap_clip_boundaries_no_snapping_without_cuts(self):
        """_snap_clip_boundaries with empty scene_cuts should be a no-op."""
        from backend.pipeline.plan_executor import _snap_clip_boundaries
        clips = [
            {"source_start": 10.2, "source_end": 20.3},
        ]
        _snap_clip_boundaries(clips, [])
        assert clips[0]["source_start"] == 10.2
        assert clips[0]["source_end"] == 20.3

    def test_multimodal_none_identical_output(self):
        """Passing scene_cut_at=None to execute_plan produces identical output.

        This is the most important regression test — it proves scene-cut
        snapping can't degrade quality for anyone (including OCR-skip users).
        """
        blocks = spread_blocks(40, 120.0)
        plan = standard_plan()
        g_no_mm = execute_plan(plan, blocks, 120.0, 90, 100, scene_cut_at=None)
        g_with_mm_none = execute_plan(plan, blocks, 120.0, 90, 100, scene_cut_at=None)
        assert g_no_mm == g_with_mm_none, "None scene_cut_at must produce identical groups"

    def test_multimodal_empty_list_identical_output(self):
        """Passing empty scene_cut_at list produces identical output to None."""
        blocks = spread_blocks(40, 120.0)
        plan = standard_plan()
        g_none = execute_plan(plan, blocks, 120.0, 90, 100, scene_cut_at=None)
        g_empty = execute_plan(plan, blocks, 120.0, 90, 100, scene_cut_at=[])
        assert g_none == g_empty, "empty scene_cut_at must produce identical groups to None"

    def test_scene_cut_snapping_does_not_change_block_selection(self):
        """Snapping only nudges boundaries — it never changes which blocks are picked."""
        blocks = [
            mk_block(0, 0.0, 8.0, "Hook content.", importance=70.0, silence_before=True),
            mk_block(1, 9.0, 20.0, "Escalation content.", importance=60.0),
            mk_block(2, 80.0, 95.0, "Payoff content.", importance=85.0, has_stakes=True),
        ]
        plan = StoryPlan(
            video_type="challenge",
            units=[{
                "unit_id": 0, "name": "Story", "priority": 1, "region": "mid",
                "arc": [
                    {"beat": "hook", "position": "start", "flags": [], "intent": "curiosity"},
                    {"beat": "escalation", "position": "any", "flags": [], "intent": "tension"},
                    {"beat": "payoff", "position": "end", "flags": ["STAKES"], "intent": "reveal"},
                ],
            }],
        )
        # Scene cuts aligned with block boundaries — snapping should be neutral
        g_no_snap = execute_plan(plan, blocks, 100.0, 45, 100, scene_cut_at=None)
        g_with_snap = execute_plan(plan, blocks, 100.0, 45, 100, scene_cut_at=[0.0, 8.0, 9.0, 20.0, 80.0, 95.0])
        # Same blocks selected (same clip start/end values) since cuts are on boundaries
        assert g_no_snap == g_with_snap, "cuts on block boundaries should not change selection"

    def test_snap_inversion_fallback_preserves_original(self):
        """When snapping would invert a clip's range, original boundaries are kept."""
        from backend.pipeline.plan_executor import _snap_clip_boundaries
        # A very short clip (10.2 to 10.4) with a scene cut at 10.0.
        # Both start and end would snap to 10.0 → inversion → originals kept.
        clips = [{"source_start": 10.2, "source_end": 10.4}]
        _snap_clip_boundaries(clips, [10.0])
        assert clips[0]["source_start"] == 10.2, "inverted snap should keep original start"
        assert clips[0]["source_end"] == 10.4, "inverted snap should keep original end"
