"""Tests for backend.pipeline.orchestrator — group orchestration flow with mocks."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.models import (
    NarrationEvent,
    ReelGroup,
    ReelSummary,
    SourceClip,
    VideoJob,
)
from backend.pipeline.orchestrator import GroupOrchestrator


@pytest.fixture
def sample_group():
    return ReelGroup(
        group_index=0,
        group_reasoning="Test group",
        estimated_duration_seconds=95.0,
        reel_summary=ReelSummary(
            title="Test Reel",
            short_description="A test reel",
            source_understanding="Test video",
            narrative_angle="Testing",
            key_moment="The test moment",
        ),
        source_clips=[
            SourceClip(source_start=0.0, source_end=5.0, reason="Opening"),
            SourceClip(source_start=12.0, source_end=24.0, reason="Key moment"),
        ],
        narration_events=[
            NarrationEvent(event_type="hook", reel_start=0.0, reel_end=3.0, text="Test hook"),
        ],
    )


@pytest.fixture
def sample_job():
    job = VideoJob(url="https://youtube.com/watch?v=test")
    job.transcript = [
        {"start": 0.0, "end": 5.0, "text": "Hello world"},
        {"start": 5.0, "end": 10.0, "text": "This is a test"},
    ]
    job.source_path = "/fake/source.mp4"
    return job


@pytest.fixture
def mock_broadcast():
    return AsyncMock()


@pytest.fixture
def mock_reporter():
    reporter = MagicMock()
    reporter.log_info = MagicMock()
    reporter.log_warn = MagicMock()
    reporter.progress_callback = MagicMock()
    reporter.update_stage = MagicMock()
    reporter.set_stage_data_key = MagicMock()
    return reporter


class TestGroupOrchestrator:
    """Test GroupOrchestrator pipeline stages with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_run_clipping_uses_checkpoint(self, sample_job, sample_group, mock_broadcast, mock_reporter):
        """If checkpoint exists, clipping should skip the actual cut."""
        orch = GroupOrchestrator(sample_job, mock_broadcast)

        # Mock checkpoint returning clip paths
        with patch.object(orch.ckpt, "load_stage", return_value={"clip_paths": ["/fake/clip1.mp4"]}):
            result = await orch.run_clipping(0, sample_group, mock_reporter, "/fake/source.mp4")
            assert result == ["/fake/clip1.mp4"]
            mock_reporter.log_info.assert_called()

    @pytest.mark.asyncio
    async def test_run_clipping_calls_cut_group_clips(self, sample_job, sample_group, mock_broadcast, mock_reporter):
        """Without checkpoint, clipping should call cut_group_clips."""
        orch = GroupOrchestrator(sample_job, mock_broadcast)

        with patch.object(orch.ckpt, "load_stage", return_value=None), \
             patch("backend.pipeline.orchestrator.cut_group_clips", return_value=["/fake/c1.mp4", "/fake/c2.mp4"]) as mock_cut:
            result = await orch.run_clipping(0, sample_group, mock_reporter, "/fake/source.mp4")
            assert result == ["/fake/c1.mp4", "/fake/c2.mp4"]
            mock_cut.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_clipping_saves_checkpoint(self, sample_job, sample_group, mock_broadcast, mock_reporter):
        """Clipping should save checkpoint after cutting."""
        orch = GroupOrchestrator(sample_job, mock_broadcast)

        with patch.object(orch.ckpt, "load_stage", return_value=None), \
             patch.object(orch.ckpt, "save_stage") as mock_save, \
             patch("backend.pipeline.orchestrator.cut_group_clips", return_value=["/c1.mp4"]):
            await orch.run_clipping(0, sample_group, mock_reporter, "/fake/source.mp4")
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_tts_uses_checkpoint(self, sample_job, sample_group, mock_broadcast, mock_reporter):
        """TTS stage should resume from checkpoint if available."""
        orch = GroupOrchestrator(sample_job, mock_broadcast)

        saved_narrations = [
            {"event_type": "hook", "reel_start": 0.0, "reel_end": 3.0, "text": "Hook", "path": "/n.wav", "duration": 3.0}
        ]
        with patch.object(orch.ckpt, "load_stage", return_value={"narration_audio": saved_narrations}):
            result = await orch.run_tts(0, sample_group, mock_reporter, Path("/fake/working"))
            # run_tts returns (narration_audio, narration_events)
            narration_audio, narration_events = result
            assert len(narration_audio) == 1
            assert narration_audio[0]["path"] == "/n.wav"

    @pytest.mark.asyncio
    async def test_run_tts_calls_synthesize(self, sample_job, sample_group, mock_broadcast, mock_reporter):
        """Without checkpoint, TTS should call synthesize_commentary."""
        orch = GroupOrchestrator(sample_job, mock_broadcast)

        mock_nar_result = {
            "event_type": "hook",
            "reel_start": 0.0,
            "reel_end": 3.0,
            "text": "Hook text",
            "path": "/fake/hook.wav",
            "duration": 3.0,
        }

        with patch.object(orch.ckpt, "load_stage", return_value=None), \
             patch("backend.pipeline.orchestrator.synthesize_commentary", return_value=3.0) as mock_tts:
            result = await orch.run_tts(0, sample_group, mock_reporter, Path("/fake/working"))
            narration_audio, narration_events = result
            assert len(narration_audio) >= 1
            mock_tts.assert_called()

    @pytest.mark.asyncio
    async def test_run_captioning_generates_files(self, sample_job, sample_group, mock_broadcast, mock_reporter):
        """Captioning should generate ASS files."""
        orch = GroupOrchestrator(sample_job, mock_broadcast)

        sample_group.narration_events = [
            NarrationEvent(event_type="hook", reel_start=0.0, reel_end=3.0, text="Hook"),
            NarrationEvent(event_type="commentary", reel_start=30.0, reel_end=33.0, text="Commentary"),
        ]
        narration_audio = [
            {"event_type": "hook", "reel_start": 0.0, "reel_end": 3.0, "text": "Hook", "path": "/n.wav", "duration": 3.0},
        ]

        with patch.object(orch.ckpt, "load_stage", return_value=None), \
             patch("backend.pipeline.orchestrator.generate_clip_ass", return_value="/fake/clip.ass") as mock_clip_ass, \
             patch("backend.pipeline.orchestrator.generate_commentary_ass", return_value="/fake/com.ass") as mock_com_ass:
            result = await orch.run_captioning(
                0, sample_group, mock_reporter, Path("/fake/working"), narration_audio
            )
            assert isinstance(result, tuple)
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_checkpoint_resume_skips_tts_for_empty_narration(self, sample_job, sample_group, mock_broadcast, mock_reporter):
        """TTS with no narration events should use fallback hook from reel_summary."""
        orch = GroupOrchestrator(sample_job, mock_broadcast)
        sample_group.narration_events = []

        with patch.object(orch.ckpt, "load_stage", return_value=None), \
             patch("backend.pipeline.orchestrator.synthesize_commentary", return_value=3.0):
            result = await orch.run_tts(0, sample_group, mock_reporter, Path("/fake/working"))
            narration_audio, narration_events = result
            # Orchestrator injects fallback hook from reel_summary.short_description
            assert len(narration_audio) == 1
            assert narration_audio[0]["event_type"] == "hook"
            assert "test reel" in narration_audio[0]["text"].lower()
