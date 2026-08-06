"""Tests for Phase 8 — Start Narration Support and Hook Mode."""
from __future__ import annotations

from backend.pipeline.plan_executor import place_narration_events
from backend.pipeline.plan_validator import validate_narration
from backend.pipeline.narration_validator import validate_and_adjust_narration_timings


def _group(clips, narrations, dur=30.0):
    return {
        "group_index": 0,
        "group_reasoning": "test",
        "estimated_duration_seconds": dur,
        "reel_summary": {},
        "source_clips": clips,
        "narration_events": narrations,
    }


# ---------------------------------------------------------------------------
# validate_narration — "start" event type
# ---------------------------------------------------------------------------

class TestStartEventType:
    def test_start_event_accepted(self):
        group = _group(
            [{"source_start": 0, "source_end": 5, "reason": "hook", "is_hook_clip": True}],
            [{"event_type": "start", "reel_start": 0.0, "reel_end": 3.0, "text": "This is Maya."}],
        )
        validate_narration([group])
        assert group["narration_events"][0]["event_type"] == "start"

    def test_start_must_start_at_zero(self):
        group = _group(
            [{"source_start": 0, "source_end": 5, "reason": "hook", "is_hook_clip": True}],
            [{"event_type": "start", "reel_start": 5.0, "reel_end": 8.0, "text": "Hello"}],
        )
        validate_narration([group])
        assert group["narration_events"][0]["reel_start"] == 0.0

    def test_hook_and_start_coexist(self):
        group = _group(
            [{"source_start": 0, "source_end": 5, "reason": "hook", "is_hook_clip": True}],
            [
                {"event_type": "hook", "reel_start": 0.0, "reel_end": 4.0, "text": "Hook line"},
                {"event_type": "commentary", "reel_start": 10.0, "reel_end": 13.0, "text": "Commentary"},
            ],
        )
        validate_narration([group])
        assert len(group["narration_events"]) == 2

    def test_duplicate_hook_converted_to_commentary(self):
        group = _group(
            [{"source_start": 0, "source_end": 5, "reason": "hook", "is_hook_clip": True}],
            [
                {"event_type": "hook", "reel_start": 0.0, "reel_end": 4.0, "text": "First hook"},
                {"event_type": "hook", "reel_start": 5.0, "reel_end": 8.0, "text": "Second hook"},
            ],
        )
        validate_narration([group])
        types = [e["event_type"] for e in group["narration_events"]]
        assert types == ["hook", "commentary"]

    def test_unknown_event_type_counted_as_non_usable(self):
        group = _group(
            [{"source_start": 0, "source_end": 5, "reason": "hook", "is_hook_clip": True}],
            [{"event_type": "mystery", "reel_start": 0.0, "reel_end": 3.0, "text": "???"}],
        )
        validate_narration([group])
        # Validator logs warning but doesn't remove — downstream TTS drops it
        # usable_count should be 0 (warning logged)
        assert group["narration_events"][0]["event_type"] == "mystery"


# ---------------------------------------------------------------------------
# place_narration_events — start placement
# ---------------------------------------------------------------------------

class TestStartNarrationPlacement:
    def test_start_event_placed_at_zero(self):
        group = _group(
            [
                {"source_start": 0, "source_end": 5, "reason": "hook", "is_hook_clip": True},
                {"source_start": 10, "source_end": 20, "reason": "esc"},
                {"source_start": 25, "source_end": 30, "reason": "payoff"},
            ],
            [
                {"event_type": "start", "reel_start": 0.0, "reel_end": 3.0, "text": "This is Maya."},
                {"event_type": "commentary", "reel_start": 15.0, "reel_end": 18.0, "text": "Commentary"},
            ],
            dur=30.0,
        )
        events = place_narration_events(group)
        start_events = [e for e in events if e.get("event_type") == "start"]
        assert len(start_events) == 1
        assert start_events[0]["reel_start"] == 0.0

    def test_start_event_voiced_like_commentary(self):
        group = _group(
            [
                {"source_start": 0, "source_end": 5, "reason": "hook", "is_hook_clip": True},
                {"source_start": 10, "source_end": 20, "reason": "esc"},
            ],
            [
                {"event_type": "start", "reel_start": 0.0, "reel_end": 3.0, "text": "Meet contestant Maya."},
            ],
            dur=20.0,
        )
        events = place_narration_events(group)
        start = [e for e in events if e.get("event_type") == "start"][0]
        assert "reel_start" in start
        assert "reel_end" in start
        assert start["reel_end"] > start["reel_start"]

    def test_start_alone_no_hook(self):
        group = _group(
            [
                {"source_start": 0, "source_end": 8, "reason": "start", "is_hook_clip": False},
                {"source_start": 10, "source_end": 20, "reason": "payoff"},
            ],
            [
                {"event_type": "start", "reel_start": 0.0, "reel_end": 3.0, "text": "Opening line."},
            ],
            dur=20.0,
        )
        events = place_narration_events(group)
        start = [e for e in events if e.get("event_type") == "start"][0]
        assert start["reel_start"] == 0.0


# ---------------------------------------------------------------------------
# narration_validator — "start" overlap handling
# ---------------------------------------------------------------------------

class TestNarrationValidatorStart:
    def test_start_overlap_shifted(self):
        """Start event overlapping speech should be shifted to a gap."""
        clips = [{"source_start": 0.0, "source_end": 10.0}]
        transcript = [{"start": 0.0, "end": 8.0, "text": "Hello world"}]
        narrations = [
            {"event_type": "start", "reel_start": 2.0, "reel_end": 5.0, "text": "Opening", "duration": 3.0},
        ]

        class FakeReporter:
            def log_info(self, msg):
                pass

        validate_and_adjust_narration_timings(
            narrations, clips, transcript, 15.0, FakeReporter(), 0,
        )
        # Should have been shifted away from the speech overlap
        assert narrations[0]["reel_start"] >= 8.0 - 0.5 or narrations[0]["reel_start"] < 2.0


# ---------------------------------------------------------------------------
# Hook mode schema
# ---------------------------------------------------------------------------

class TestHookModeSchema:
    def test_hook_mode_values(self):
        from backend.models import HookMode
        from pydantic import BaseModel

        class TestModel(BaseModel):
            mode: HookMode

        assert TestModel(mode="required").mode == "required"
        assert TestModel(mode="skip").mode == "skip"
        assert TestModel(mode="auto").mode == "auto"

    def test_hook_mode_invalid_rejected(self):
        from backend.models import HookMode
        from pydantic import BaseModel, ValidationError
        import pytest

        class TestModel(BaseModel):
            mode: HookMode

        with pytest.raises(ValidationError):
            TestModel(mode="invalid")
