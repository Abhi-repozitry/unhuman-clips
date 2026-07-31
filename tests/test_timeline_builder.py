"""Tests for backend.pipeline.timeline_builder — Rich Timeline construction."""
from __future__ import annotations

import pytest

from backend.models import FFmpegMetrics, RichTimeline, RichTimelineSegment
from backend.pipeline.timeline_builder import (
    _check_silence_before,
    _compute_speech_energy,
)


class TestComputeSpeechEnergy:
    """Test _compute_speech_energy proportion calculation."""

    def test_empty_regions(self):
        assert _compute_speech_energy(0.0, 5.0, []) == 0.0

    def test_full_speech(self):
        regions = [{"start": 0.0, "end": 5.0}]
        assert _compute_speech_energy(0.0, 5.0, regions) == 1.0

    def test_no_speech(self):
        regions = [{"start": 10.0, "end": 15.0}]
        assert _compute_speech_energy(0.0, 5.0, regions) == 0.0

    def test_partial_speech(self):
        regions = [{"start": 0.0, "end": 2.5}]
        energy = _compute_speech_energy(0.0, 5.0, regions)
        assert abs(energy - 0.5) < 0.01

    def test_multiple_regions(self):
        regions = [
            {"start": 0.0, "end": 1.0},
            {"start": 3.0, "end": 5.0},
        ]
        energy = _compute_speech_energy(0.0, 5.0, regions)
        assert abs(energy - 0.6) < 0.01

    def test_capped_at_one(self):
        regions = [{"start": -1.0, "end": 10.0}]
        assert _compute_speech_energy(0.0, 5.0, regions) == 1.0

    def test_zero_duration(self):
        regions = [{"start": 0.0, "end": 5.0}]
        assert _compute_speech_energy(3.0, 3.0, regions) == 0.0


class TestCheckSilenceBefore:
    """Test _check_silence_before detection."""

    def test_no_regions(self):
        assert _check_silence_before(5.0, []) is False

    def test_speech_ends_right_before(self):
        regions = [{"start": 0.0, "end": 4.0}]
        assert _check_silence_before(5.0, regions, min_silence=0.3) is True

    def test_speech_ends_too_close(self):
        regions = [{"start": 0.0, "end": 4.8}]
        assert _check_silence_before(5.0, regions, min_silence=0.3) is False

    def test_speech_after_segment(self):
        regions = [{"start": 6.0, "end": 10.0}]
        assert _check_silence_before(5.0, regions, min_silence=0.3) is False


class TestRichTimelineModels:
    """Test Pydantic model construction and serialization."""

    def test_ffmpeg_metrics_defaults(self):
        m = FFmpegMetrics()
        assert m.volume_db is None
        assert m.black_frame is False

    def test_segment_construction(self):
        seg = RichTimelineSegment(
            segment_id=0, start=0.0, end=5.0, duration=5.0, speech="test"
        )
        assert seg.segment_id == 0
        assert seg.speech_energy == 0.0

    def test_timeline_construction(self):
        tl = RichTimeline(source_duration=60.0)
        assert tl.segments == []
        assert tl.source_duration == 60.0

    def test_serialization_roundtrip(self):
        seg = RichTimelineSegment(
            segment_id=0, start=0.0, end=5.0, duration=5.0, speech="test"
        )
        tl = RichTimeline(segments=[seg], source_duration=60.0)
        data = tl.model_dump()
        restored = RichTimeline.model_validate(data)
        assert len(restored.segments) == 1
        assert restored.segments[0].speech == "test"
