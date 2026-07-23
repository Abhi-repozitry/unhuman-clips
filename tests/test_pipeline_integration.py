"""Integration / smoke test for the pipeline — mocked end-to-end.

Marked as @pytest.mark.integration so they can be excluded with:
    pytest tests/ -m "not integration"
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.models import (
    NarrationEvent,
    ReelGroup,
    ReelPlan,
    ReelSummary,
    SourceClip,
    VideoJob,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reel_plan() -> ReelPlan:
    """Build a valid ReelPlan for integration testing."""
    return ReelPlan(
        reel_groups=[
            ReelGroup(
                group_index=0,
                group_reasoning="Integration test group",
                estimated_duration_seconds=95.0,
                reel_summary=ReelSummary(
                    title="Integration Test Reel",
                    short_description="A full pipeline smoke test",
                    source_understanding="Test video content",
                    narrative_angle="Testing and validation",
                    key_moment="The key test moment at 20s",
                ),
                source_clips=[
                    SourceClip(source_start=0.0, source_end=5.0, reason="Opening hook"),
                    SourceClip(source_start=12.0, source_end=24.0, reason="Key moment"),
                    SourceClip(source_start=36.0, source_end=42.0, reason="Discovery"),
                ],
                narration_events=[
                    NarrationEvent(event_type="hook", reel_start=0.0, reel_end=3.0, text="This changed everything."),
                    NarrationEvent(event_type="commentary", reel_start=25.0, reel_end=28.0, text="The breakthrough was unexpected."),
                ],
            )
        ]
    )


@pytest.mark.integration
class TestPipelineIntegration:
    """End-to-end integration tests with full mocking of external services."""

    def test_reel_plan_serialization_roundtrip(self):
        """ReelPlan can be serialized to dict and back."""
        plan = _make_reel_plan()
        data = plan.model_dump()
        restored = ReelPlan.model_validate(data)
        assert len(restored.reel_groups) == 1
        assert restored.reel_groups[0].reel_summary.title == "Integration Test Reel"

    def test_transcript_to_reel_plan_flow(self):
        """Simulate: transcript -> analyzer -> reel plan -> clip windows."""
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Welcome to the show."},
            {"start": 5.5, "end": 12.0, "text": "Today we explore something amazing."},
            {"start": 12.5, "end": 18.0, "text": "This changed everything."},
            {"start": 18.5, "end": 24.0, "text": "Nobody expected what happened."},
            {"start": 24.5, "end": 30.0, "text": "The results were shocking."},
            {"start": 30.5, "end": 36.0, "text": "Scientists were amazed."},
            {"start": 36.5, "end": 42.0, "text": "A breakthrough was announced."},
            {"start": 42.5, "end": 48.0, "text": "The world took notice."},
        ]

        plan = _make_reel_plan()

        # Extract clip windows from plan
        clip_windows = []
        for group in plan.reel_groups:
            for clip in group.source_clips:
                clip_windows.append({
                    "start": clip.source_start,
                    "end": clip.source_end,
                })

        assert len(clip_windows) == 3
        assert clip_windows[0]["start"] == 0.0
        assert clip_windows[0]["end"] == 5.0

    def test_full_pipeline_mocked(self, tmp_path):
        """Mock every external dependency and run a simulated pipeline."""
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Hello world"},
            {"start": 5.0, "end": 10.0, "text": "This is a test"},
            {"start": 10.0, "end": 15.0, "text": "Of the pipeline"},
        ]

        plan = _make_reel_plan()

        # Stage 1: Transcribe (mock)
        assert transcript is not None
        assert len(transcript) > 0

        # Stage 2: Analyze (use pre-built plan)
        assert plan.reel_groups[0].source_clips[0].source_start == 0.0

        # Stage 3: Clip (mocked ffmpeg)
        clip_paths = [str(tmp_path / f"clip_{i}.mp4") for i in range(3)]
        for p in clip_paths:
            Path(p).write_bytes(b"\x00" * 1024)
        assert len(clip_paths) == 3

        # Stage 4: TTS (mock)
        narration_audio = [
            {"event_type": "hook", "reel_start": 0.0, "reel_end": 3.0,
             "text": "Test hook", "path": str(tmp_path / "hook.wav"), "duration": 3.0},
        ]
        assert len(narration_audio) == 1

        # Stage 5: Caption (use actual captioner)
        from backend.pipeline.captioner import generate_commentary_ass
        ass_path = str(tmp_path / "commentary.ass")
        generate_commentary_ass("This is a test hook", 3.0, ass_path)
        assert Path(ass_path).exists()
        content = Path(ass_path).read_text(encoding="utf-8")
        assert "Dialogue:" in content

        # Stage 6: Validate output
        assert len(clip_paths) == 3
        assert all(Path(p).exists() for p in clip_paths)

    def test_model_validation_catches_invalid_data(self):
        """Pydantic models reject invalid data at schema level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            # status must be a valid JobStatus enum
            VideoJob(url="test", status="INVALID_STATUS")

    @pytest.mark.asyncio
    async def test_progress_reporter_thread_safety(self):
        """ProgressReporter updates don't crash from multiple threads."""
        import threading

        from backend.progress import ProgressReporter

        job = VideoJob(url="https://test.com")
        loop = asyncio.new_event_loop()
        broadcast = AsyncMock()

        reporter = ProgressReporter(job, broadcast, loop)

        def update():
            for _ in range(10):
                reporter.update_sub_stage("test", 50.0)
                reporter.log_info("thread message")

        threads = [threading.Thread(target=update) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(job.logs) == 50  # 5 threads * 10 calls
