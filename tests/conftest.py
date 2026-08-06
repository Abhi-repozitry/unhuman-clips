"""Shared test fixtures for the unhuman-clips test suite.

Provides mock ffmpeg, a sample reel plan, a mock reporter, and
temporary file system fixtures used across all test modules.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Temp directory fixture (autouse for every test)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_tmp(tmp_path: Path):
    """Provide a fresh temp directory for each test (returned for use, not as CWD)."""
    return tmp_path


# ---------------------------------------------------------------------------
# Sample transcript and reel plan fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_transcript() -> list[dict]:
    """A small ordered transcript for analyzer formatting tests."""
    return [
        {"start": 0.0, "end": 5.0, "text": "Welcome to the show"},
        {"start": 5.0, "end": 10.0, "text": "Here is the challenge"},
        {"start": 10.0, "end": 15.0, "text": "The result surprises everyone"},
    ]


@pytest.fixture
def sample_reel_plan_dict() -> dict:
    """A valid reel plan dict matching the LLM JSON schema."""
    return {
        "structure_analysis": {
            "video_type": "documentary",
            "identified_units": [
                {"name": "Unit 1", "approx_start": 0.0, "approx_end": 60.0, "usable_seconds": 55, "kept": True},
            ],
            "final_group_count": 1,
            "reasoning": "Single continuous narrative.",
        },
        "reel_groups": [
            {
                "group_index": 0,
                "group_reasoning": "Short clips: 3, Medium clips: 3, Long clips: 1. Total 90s arc.",
                "estimated_duration_seconds": 95.0,
                "reel_summary": {
                    "title": "The Discovery That Changed Everything",
                    "short_description": "How one invention reshaped the world",
                    "source_understanding": "Documentary about a scientific breakthrough",
                    "narrative_angle": "Wonder and amazement",
                    "key_moment": "The breakthrough discovery at 36s",
                },
                "source_clips": [
                    {"source_start": 0.0, "source_end": 5.0, "reason": "SHORT: Punchy opening reaction"},
                    {"source_start": 12.0, "source_end": 24.0, "reason": "LONG: Key reveal moment"},
                    {"source_start": 30.0, "source_end": 36.0, "reason": "MEDIUM: Building tension"},
                    {"source_start": 42.0, "source_end": 48.0, "reason": "MEDIUM: Publishing findings"},
                    {"source_start": 54.0, "source_end": 58.0, "reason": "SHORT: Closing beat"},
                ],
                "narration_events": [
                    {"event_type": "hook", "reel_start": 0.0, "reel_end": 3.0, "text": "One discovery changed everything science knew."},
                    {"event_type": "commentary", "reel_start": 25.0, "reel_end": 28.0, "text": "The results shocked even veteran researchers."},
                    {"event_type": "commentary", "reel_start": 50.0, "reel_end": 53.0, "text": "Within weeks every lab in the world was racing to replicate it."},
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# Mock ffmpeg / ffprobe fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Patch get_ffmpeg() and get_ffprobe() to use a mock binary path."""
    fake_ffmpeg = str(tmp_path / "ffmpeg")
    fake_ffprobe = str(tmp_path / "ffprobe")

    # Create dummy executables so path checks pass
    Path(fake_ffmpeg).touch()
    Path(fake_ffprobe).touch()

    monkeypatch.setattr("backend.ffmpeg_utils.get_ffmpeg", lambda: fake_ffmpeg)
    monkeypatch.setattr("backend.ffmpeg_utils.get_ffprobe", lambda: fake_ffprobe)
    return fake_ffmpeg, fake_ffprobe


# ---------------------------------------------------------------------------
# Mock reporter fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_reporter():
    """A mock ProgressReporter that silently records all calls."""
    reporter = MagicMock()
    reporter.log_info = MagicMock()
    reporter.log_warn = MagicMock()
    reporter.log_error = MagicMock()
    reporter.log_debug = MagicMock()
    reporter.update_stage = MagicMock()
    reporter.update_sub_stage = MagicMock()
    reporter.progress_callback = MagicMock()
    reporter.set_stage_data_key = MagicMock()
    reporter.set_clip_details = MagicMock()
    reporter.update_clip_progress = MagicMock()
    return reporter
