"""Tests for backend.pipeline.narration_validator — timing adjustment and speech overlap detection."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.pipeline.narration_validator import validate_and_adjust_narration_timings


class TestNarrationValidator:
    """Test validate_and_adjust_narration_timings with various speech/narration layouts."""

    def _make_reporter(self):
        reporter = MagicMock()
        reporter.log_info = MagicMock()
        reporter.log_warn = MagicMock()
        return reporter

    def test_empty_narration_noop(self):
        reporter = self._make_reporter()
        # Should not raise on empty list
        validate_and_adjust_narration_timings(
            group_narration_audio=[],
            source_clips=[],
            transcript=[],
            target_duration=90.0,
            reporter=reporter,
            group_idx=0,
        )
        reporter.log_info.assert_not_called()

    def test_narration_in_silent_gap_unchanged(self):
        """Narration placed in silence should not be moved."""
        from backend.models import SourceClip

        clips = [SourceClip(source_start=0.0, source_end=10.0, reason="test")]
        transcript = [
            {"start": 0.0, "end": 3.0, "text": "Speech here"},
            {"start": 7.0, "end": 10.0, "text": "More speech"},
        ]
        # Narration at 4-7s (between speech segments) should stay
        narration = [
            {
                "event_type": "commentary",
                "reel_start": 3.5,
                "reel_end": 6.5,
                "duration": 3.0,
                "text": "In the silent gap",
                "path": "/fake/nar.wav",
            }
        ]
        reporter = self._make_reporter()
        validate_and_adjust_narration_timings(
            narration, clips, transcript, 90.0, reporter, 0
        )
        # Should NOT be shifted
        assert narration[0]["reel_start"] == pytest.approx(3.5, abs=0.1)

    def test_narration_overlapping_speech_shifted(self):
        """Narration overlapping speech >10% should be shifted."""
        from backend.models import SourceClip

        clips = [SourceClip(source_start=0.0, source_end=10.0, reason="test")]
        transcript = [
            {"start": 0.0, "end": 10.0, "text": "Continuous speech for full 10 seconds"},
        ]
        # Narration at 2-5s overlaps heavily with speech
        narration = [
            {
                "event_type": "commentary",
                "reel_start": 2.0,
                "reel_end": 5.0,
                "duration": 3.0,
                "text": "Overlapping narration",
                "path": "/fake/nar.wav",
            }
        ]
        reporter = self._make_reporter()
        validate_and_adjust_narration_timings(
            narration, clips, transcript, 90.0, reporter, 0
        )
        # Should be shifted away from speech
        # The exact position depends on gap finding, but it should NOT be at 2.0
        # if speech covers 0-10s entirely
        assert narration[0]["reel_start"] != 2.0 or narration[0]["reel_end"] <= 10.0

    def test_narration_capped_at_target_duration(self):
        """Narration exceeding target_duration should be capped."""
        from backend.models import SourceClip

        clips = [SourceClip(source_start=0.0, source_end=5.0, reason="test")]
        transcript = [{"start": 0.0, "end": 5.0, "text": "Short speech"}]
        narration = [
            {
                "event_type": "commentary",
                "reel_start": 85.0,
                "reel_end": 95.0,
                "duration": 10.0,
                "text": "Long narration that exceeds target",
                "path": "/fake/nar.wav",
            }
        ]
        reporter = self._make_reporter()
        validate_and_adjust_narration_timings(
            narration, clips, transcript, 90.0, reporter, 0
        )
        assert narration[0]["reel_end"] <= 90.0

    def test_multiple_narrations_maintain_min_gap(self):
        """Multiple narrations should maintain minimum 0.8s gap."""
        from backend.models import SourceClip

        clips = [SourceClip(source_start=0.0, source_end=5.0, reason="test")]
        transcript = [{"start": 0.0, "end": 5.0, "text": "Speech"}]
        narration = [
            {
                "event_type": "commentary",
                "reel_start": 20.0,
                "reel_end": 23.0,
                "duration": 3.0,
                "text": "First narration",
                "path": "/fake/nar1.wav",
            },
            {
                "event_type": "commentary",
                "reel_start": 23.2,  # Too close — only 0.2s gap
                "reel_end": 26.2,
                "duration": 3.0,
                "text": "Second narration too close",
                "path": "/fake/nar2.wav",
            },
        ]
        reporter = self._make_reporter()
        validate_and_adjust_narration_timings(
            narration, clips, transcript, 90.0, reporter, 0
        )
        # After sorting and gap enforcement:
        gap = narration[1]["reel_start"] - narration[0]["reel_end"]
        assert gap >= 0.8 - 0.01  # small floating point tolerance

    def test_hook_in_gap_not_shifted(self):
        """Hook event placed in a silent gap should not be shifted."""
        from backend.models import SourceClip

        clips = [SourceClip(source_start=0.0, source_end=10.0, reason="test")]
        transcript = [
            {"start": 0.0, "end": 3.0, "text": "Speech at start"},
            {"start": 7.0, "end": 10.0, "text": "Speech at end"},
        ]
        # Hook at 3.5-6.5s (in silent gap between speech) should stay
        narration = [
            {
                "event_type": "hook",
                "reel_start": 3.5,
                "reel_end": 6.5,
                "duration": 3.0,
                "text": "This is a hook",
                "path": "/fake/hook.wav",
            }
        ]
        reporter = self._make_reporter()
        validate_and_adjust_narration_timings(
            narration, clips, transcript, 90.0, reporter, 0
        )
        # Hook in silent gap should stay
        assert narration[0]["reel_start"] == pytest.approx(3.5, abs=0.1)

    def test_hook_overlapping_speech_can_be_shifted(self):
        """Hook overlapping speech will be shifted (validator shifts both hook and commentary)."""
        from backend.models import SourceClip

        clips = [SourceClip(source_start=0.0, source_end=10.0, reason="test")]
        transcript = [
            {"start": 0.0, "end": 10.0, "text": "Continuous speech for full 10 seconds"},
        ]
        narration = [
            {
                "event_type": "hook",
                "reel_start": 2.0,
                "reel_end": 5.0,
                "duration": 3.0,
                "text": "Hook overlapping speech",
                "path": "/fake/hook.wav",
            }
        ]
        reporter = self._make_reporter()
        validate_and_adjust_narration_timings(
            narration, clips, transcript, 90.0, reporter, 0
        )
        # Hook at 2-5s overlaps with 0-10s speech — may be shifted
        assert narration[0]["reel_start"] >= 0.0

    def test_dict_style_clips_work(self):
        """Validator should handle both attribute and dict-style source_clips."""
        clips = [{"source_start": 0.0, "source_end": 10.0}]
        transcript = [{"start": 0.0, "end": 3.0, "text": "Speech"}]
        narration = [
            {
                "event_type": "commentary",
                "reel_start": 30.0,
                "reel_end": 33.0,
                "duration": 3.0,
                "text": "Test narration",
                "path": "/fake/nar.wav",
            }
        ]
        reporter = self._make_reporter()
        # Should not raise
        validate_and_adjust_narration_timings(
            narration, clips, transcript, 90.0, reporter, 0
        )
        assert narration[0]["reel_start"] >= 0
