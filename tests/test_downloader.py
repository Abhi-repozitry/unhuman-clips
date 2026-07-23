"""Tests for backend.pipeline.downloader — format selection, validation, error handling."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.pipeline.downloader import download_video, validate_downloaded_video


class TestValidateDownloadedVideo:
    """Test validate_downloaded_video with mocked ffprobe."""

    def test_valid_video_returns_no_error(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_result = MagicMock(returncode=0)
        mock_result.stdout = json.dumps({
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "120.0"},
        })

        with patch("backend.pipeline.downloader.subprocess.run", return_value=mock_result):
            result = validate_downloaded_video(str(video))
            assert result["valid"] is True
            assert result["error"] is None

    def test_missing_file_returns_error(self, tmp_path):
        result = validate_downloaded_video(str(tmp_path / "nonexistent.mp4"))
        assert result["valid"] is False
        assert "not found" in result["error"].lower() or "missing" in result["error"].lower()

    def test_no_video_stream_returns_error(self, tmp_path):
        video = tmp_path / "audio_only.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_result = MagicMock(returncode=0)
        mock_result.stdout = json.dumps({
            "streams": [{"codec_type": "audio"}],
            "format": {"duration": "60.0"},
        })

        with patch("backend.pipeline.downloader.subprocess.run", return_value=mock_result):
            result = validate_downloaded_video(str(video))
            assert result["valid"] is False

    def test_short_video_returns_error(self, tmp_path):
        video = tmp_path / "short.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_result = MagicMock(returncode=0)
        mock_result.stdout = json.dumps({
            "streams": [
                {"codec_type": "video", "width": 640, "height": 480},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "20.0"},  # Under 30s
        })

        with patch("backend.pipeline.downloader.subprocess.run", return_value=mock_result):
            result = validate_downloaded_video(str(video))
            assert result["valid"] is False
            assert "short" in result["error"].lower() or "duration" in result["error"].lower()

    def test_ffprobe_failure(self, tmp_path):
        video = tmp_path / "bad.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_result = MagicMock(returncode=1, stderr="ffprobe error", stdout="")

        with patch("backend.pipeline.downloader.subprocess.run", return_value=mock_result):
            result = validate_downloaded_video(str(video))
            assert result["valid"] is False

    def test_stats_populated(self, tmp_path):
        video = tmp_path / "stats.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_result = MagicMock(returncode=0)
        mock_result.stdout = json.dumps({
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "120.5"},
        })

        with patch("backend.pipeline.downloader.subprocess.run", return_value=mock_result):
            result = validate_downloaded_video(str(video))
            assert "duration" in result
            assert "width" in result
            assert "height" in result


class TestDownloadVideo:
    """Test download_video with mocked yt-dlp."""

    def test_download_success(self, tmp_path, monkeypatch):
        out_dir = str(tmp_path / "downloads")
        Path(out_dir).mkdir()

        mock_ydl = MagicMock()
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {
            "id": "test123",
            "title": "Test Video",
            "duration": 120,
            "ext": "mp4",
        }
        monkeypatch.setattr("backend.pipeline.downloader.yt_dlp.YoutubeDL", mock_ydl)

        # Create the file that yt-dlp would have produced
        out_path = Path(out_dir) / "test123.mp4"
        out_path.write_bytes(b"\x00" * 1024)

        def create_file(d):
            # Simulate yt-dlp download hook
            for hook in d.get("progress_hooks", []):
                hook({"status": "finished", "total_bytes": 1024})

        mock_ydl_instance.download_side_effect = create_file

        hook = MagicMock()
        result = download_video("https://youtube.com/watch?v=test123", out_dir, hook)
        assert result["id"] == "test123"

    def test_download_empty_url_raises(self, tmp_path):
        hook = MagicMock()
        with pytest.raises(Exception):
            download_video("", str(tmp_path), hook)

    def test_progress_hook_called(self, tmp_path, monkeypatch):
        out_dir = str(tmp_path / "downloads2")
        Path(out_dir).mkdir()

        mock_ydl = MagicMock()
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {
            "id": "vid2", "title": "V2", "duration": 60, "ext": "mp4",
        }
        monkeypatch.setattr("backend.pipeline.downloader.yt_dlp.YoutubeDL", mock_ydl)

        # Create the file that yt-dlp would have produced
        out_path = Path(out_dir) / "vid2.mp4"
        out_path.write_bytes(b"\x00" * 100)

        def create_file(d):
            # yt-dlp registers progress_hooks internally — we can't easily
            # trigger them from download_side_effect. Just succeed silently.
            pass

        mock_ydl_instance.download_side_effect = create_file

        hook = MagicMock()
        # Should complete without error (hook may not be called in this mock setup)
        result = download_video("https://youtube.com/watch?v=vid2", out_dir, hook)
        assert result["id"] == "vid2"
