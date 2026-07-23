"""Tests for backend.output_manager — finalization, duration probe, staging."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.output_manager import OutputManager


class TestOutputManager:
    """Test OutputManager finalization and probing logic."""

    @pytest.fixture
    def manager(self):
        return OutputManager()

    def test_probe_duration_sync_returns_float(self, manager, tmp_path):
        fake_file = tmp_path / "test.mp4"
        fake_file.write_bytes(b"\x00" * 100)

        mock_result = MagicMock(returncode=0, stdout="120.5\n")
        with patch("backend.output_manager.subprocess.run", return_value=mock_result):
            duration = manager._probe_duration_sync(str(fake_file))
            assert duration == pytest.approx(120.5)

    def test_probe_duration_sync_returns_zero_on_failure(self, manager, tmp_path):
        fake_file = tmp_path / "bad.mp4"
        fake_file.write_bytes(b"\x00" * 100)

        mock_result = MagicMock(returncode=1, stdout="")
        with patch("backend.output_manager.subprocess.run", return_value=mock_result):
            duration = manager._probe_duration_sync(str(fake_file))
            assert duration == 0.0

    def test_probe_duration_sync_returns_zero_on_non_numeric(self, manager, tmp_path):
        fake_file = tmp_path / "weird.mp4"
        fake_file.write_bytes(b"\x00" * 100)

        mock_result = MagicMock(returncode=0, stdout="not_a_number\n")
        with patch("backend.output_manager.subprocess.run", return_value=mock_result):
            duration = manager._probe_duration_sync(str(fake_file))
            assert duration == 0.0

    @pytest.mark.asyncio
    async def test_probe_duration_async(self, manager, tmp_path):
        fake_file = tmp_path / "async.mp4"
        fake_file.write_bytes(b"\x00" * 100)

        mock_result = MagicMock(returncode=0, stdout="90.0\n")
        with patch("backend.output_manager.subprocess.run", return_value=mock_result):
            duration = await manager.probe_duration(str(fake_file))
            assert duration == pytest.approx(90.0)

    def test_final_edit_group_copies_file(self, manager, tmp_path):
        # Setup
        input_path = tmp_path / "input.mp4"
        input_path.write_bytes(b"\x00" * 1024)

        from backend.models import ReelGroup, ReelSummary
        group = ReelGroup(
            group_index=0,
            group_reasoning="Test",
            estimated_duration_seconds=90.0,
            reel_summary=ReelSummary(
                title="Test Reel",
                short_description="A test",
                source_understanding="Test video",
                narrative_angle="Testing",
                key_moment="The test",
            ),
            source_clips=[],
            narration_events=[],
        )

        mock_probe = MagicMock(return_value=90.0)
        mock_ffmpeg = MagicMock(return_value="/usr/bin/ffmpeg")
        mock_ffprobe = MagicMock(return_value="/usr/bin/ffprobe")

        with patch.object(manager, "_probe_duration_sync", mock_probe), \
             patch("backend.output_manager.get_ffmpeg", mock_ffmpeg), \
             patch("backend.output_manager.get_ffprobe", mock_ffprobe), \
             patch("backend.config.OUTPUTS_DIR", tmp_path / "outputs"), \
             patch("backend.config.MAX_OUTPUT_DURATION", 180):
            result = manager.final_edit_group(
                str(input_path), group, tmp_path / "working", "testjob"
            )
            assert Path(result).exists()
            assert "testjob_reel_0" in result

    def test_final_edit_group_caps_duration(self, manager, tmp_path):
        input_path = tmp_path / "long_input.mp4"
        input_path.write_bytes(b"\x00" * 1024)

        from backend.models import ReelGroup, ReelSummary
        group = ReelGroup(
            group_index=1,
            group_reasoning="Test",
            estimated_duration_seconds=200.0,
            reel_summary=ReelSummary(
                title="Long Reel",
                short_description="Too long",
                source_understanding="Test",
                narrative_angle="Test",
                key_moment="Test",
            ),
            source_clips=[],
            narration_events=[],
        )

        mock_probe = MagicMock(return_value=200.0)
        mock_ffmpeg = MagicMock(return_value="/usr/bin/ffmpeg")
        mock_ffprobe = MagicMock(return_value="/usr/bin/ffprobe")
        mock_run = MagicMock(return_value=MagicMock(returncode=0))

        with patch.object(manager, "_probe_duration_sync", mock_probe), \
             patch("backend.output_manager.get_ffmpeg", mock_ffmpeg), \
             patch("backend.output_manager.get_ffprobe", mock_ffprobe), \
             patch("backend.output_manager.subprocess.run", mock_run), \
             patch("backend.config.OUTPUTS_DIR", tmp_path / "outputs"), \
             patch("backend.config.MAX_OUTPUT_DURATION", 180):
            result = manager.final_edit_group(
                str(input_path), group, tmp_path / "working", "testjob"
            )
            # Should have called ffmpeg to trim (not copy)
            mock_run.assert_called_once()
            assert "-t" in mock_run.call_args[0][0]
