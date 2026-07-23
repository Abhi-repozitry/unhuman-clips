"""Tests for backend.pipeline.clipper — parallel cutting, validation, error handling."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.pipeline.clipper import _validate_clip, cut_clips, cut_group_clips


class TestValidateClip:
    """Test _validate_clip for timestamp validation."""

    def test_valid_clip(self, tmp_path):
        video = tmp_path / "source.mp4"
        video.write_bytes(b"\x00" * 100)
        _validate_clip(str(video), 5.0, 15.0)  # No exception

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            _validate_clip(str(tmp_path / "missing.mp4"), 0.0, 10.0)

    def test_negative_start_raises(self, tmp_path):
        video = tmp_path / "source.mp4"
        video.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="negative"):
            _validate_clip(str(video), -1.0, 10.0)

    def test_end_before_start_raises(self, tmp_path):
        video = tmp_path / "source.mp4"
        video.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="after start"):
            _validate_clip(str(video), 10.0, 5.0)

    def test_equal_start_end_raises(self, tmp_path):
        video = tmp_path / "source.mp4"
        video.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="after start"):
            _validate_clip(str(video), 5.0, 5.0)

    def test_end_exceeds_duration_raises(self, tmp_path):
        video = tmp_path / "source.mp4"
        video.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="exceeds source duration"):
            _validate_clip(str(video), 5.0, 15.0, source_duration=10.0)

    def test_zero_duration_skips_bounds_check(self, tmp_path):
        video = tmp_path / "source.mp4"
        video.write_bytes(b"\x00" * 100)
        # source_duration=0 disables bounds check
        _validate_clip(str(video), 5.0, 15.0, source_duration=0.0)


class TestCutClips:
    """Test cut_clips with mocked ffmpeg subprocess."""

    def test_empty_windows_returns_empty(self):
        result = cut_clips("/fake/source.mp4", [], "job1")
        assert result == []

    @patch("backend.pipeline.clipper.subprocess.run")
    @patch("backend.pipeline.clipper.get_encoder", return_value="libx264")
    @patch("backend.pipeline.clipper.get_ffmpeg", return_value="/usr/bin/ffmpeg")
    def test_single_clip_calls_ffmpeg(self, mock_ffmpeg, mock_encoder, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        # Patch CLIPS_DIR
        with patch("backend.pipeline.clipper.CLIPS_DIR", tmp_path):
            windows = [{"start": 5.0, "end": 15.0}]
            result = cut_clips("/fake/source.mp4", windows, "testjob")
            assert len(result) == 1
            assert "testjob_clip_0.mp4" in result[0]
            mock_run.assert_called_once()

    @patch("backend.pipeline.clipper.subprocess.run")
    @patch("backend.pipeline.clipper.get_encoder", return_value="libx264")
    @patch("backend.pipeline.clipper.get_ffmpeg", return_value="/usr/bin/ffmpeg")
    def test_multiple_clips(self, mock_ffmpeg, mock_encoder, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("backend.pipeline.clipper.CLIPS_DIR", tmp_path):
            windows = [
                {"start": 0.0, "end": 10.0},
                {"start": 20.0, "end": 30.0},
                {"start": 40.0, "end": 50.0},
            ]
            result = cut_clips("/fake/source.mp4", windows, "testjob")
            assert len(result) == 3

    @patch("backend.pipeline.clipper.subprocess.run")
    @patch("backend.pipeline.clipper.get_encoder", return_value="libx264")
    @patch("backend.pipeline.clipper.get_ffmpeg", return_value="/usr/bin/ffmpeg")
    def test_ffmpeg_failure_raises(self, mock_ffmpeg, mock_encoder, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"error msg")
        with patch("backend.pipeline.clipper.CLIPS_DIR", tmp_path):
            windows = [{"start": 0.0, "end": 10.0}]
            with pytest.raises(RuntimeError, match="FFmpeg clip failed"):
                cut_clips("/fake/source.mp4", windows, "testjob")

    @patch("backend.pipeline.clipper.subprocess.run")
    @patch("backend.pipeline.clipper.get_encoder", return_value="libx264")
    @patch("backend.pipeline.clipper.get_ffmpeg", return_value="/usr/bin/ffmpeg")
    def test_progress_callback_called(self, mock_ffmpeg, mock_encoder, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("backend.pipeline.clipper.CLIPS_DIR", tmp_path):
            cb = MagicMock()
            windows = [{"start": 0.0, "end": 10.0}, {"start": 20.0, "end": 30.0}]
            cut_clips("/fake/source.mp4", windows, "testjob", progress_cb=cb)
            assert cb.call_count >= 1


class TestCutGroupClips:
    """Test cut_group_clips with mocked ffmpeg."""

    def test_empty_clips_returns_empty(self):
        result = cut_group_clips("/fake/source.mp4", [], "job1", 0)
        assert result == []

    @patch("backend.pipeline.clipper.subprocess.run")
    @patch("backend.pipeline.clipper.get_encoder", return_value="libx264")
    @patch("backend.pipeline.clipper.get_ffmpeg", return_value="/usr/bin/ffmpeg")
    def test_group_clips_naming(self, mock_ffmpeg, mock_encoder, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("backend.pipeline.clipper.CLIPS_DIR", tmp_path):
            clips = [
                {"source_start": 5.0, "source_end": 15.0},
                {"source_start": 25.0, "source_end": 35.0},
            ]
            result = cut_group_clips("/fake/source.mp4", clips, "job1", 2)
            assert len(result) == 2
            assert "job1_group2_clip_0.mp4" in result[0]
            assert "job1_group2_clip_1.mp4" in result[1]
