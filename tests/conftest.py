"""Shared test fixtures for the unhuman-clips test suite.

Provides mock ffmpeg, mock LLM responses, sample transcripts, and
temporary file system fixtures used across all test modules.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Temp directory fixture (autouse for every test)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_tmp(tmp_path: Path):
    """Ensure every test runs with a fresh temp directory as CWD side-effect free."""
    return tmp_path


# ---------------------------------------------------------------------------
# Sample transcript fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_transcript() -> list[dict]:
    """10-segment transcript simulating a ~60s video."""
    return [
        {"start": 0.0, "end": 5.0, "text": "Welcome to the show everyone."},
        {"start": 5.5, "end": 12.0, "text": "Today we are going to look at something amazing."},
        {"start": 12.5, "end": 18.0, "text": "This invention changed the world forever."},
        {"start": 18.5, "end": 24.0, "text": "Nobody expected what happened next."},
        {"start": 24.5, "end": 30.0, "text": "The results were absolutely shocking."},
        {"start": 30.5, "end": 36.0, "text": "Scientists could not believe their eyes."},
        {"start": 36.5, "end": 42.0, "text": "It was a breakthrough discovery."},
        {"start": 42.5, "end": 48.0, "text": "They published the findings immediately."},
        {"start": 48.5, "end": 54.0, "text": "The world would never be the same."},
        {"start": 54.5, "end": 60.0, "text": "And that is the story of how it all began."},
    ]


@pytest.fixture
def short_transcript() -> list[dict]:
    """4-segment transcript for edge-case testing."""
    return [
        {"start": 0.0, "end": 3.0, "text": "Hello world."},
        {"start": 3.5, "end": 7.0, "text": "This is a test."},
        {"start": 7.5, "end": 11.0, "text": "Nothing more to say."},
        {"start": 11.5, "end": 15.0, "text": "Goodbye."},
    ]


@pytest.fixture
def sample_reel_plan_dict() -> dict:
    """A valid reel plan dict matching the LLM JSON schema."""
    return {
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
# LLM mock response fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_response(sample_reel_plan_dict: dict) -> str:
    """Return a valid JSON string the analyzer would parse."""
    return json.dumps(sample_reel_plan_dict)


@pytest.fixture
def mock_llm_response_with_fences(sample_reel_plan_dict: dict) -> str:
    """Return a fenced JSON string (common LLM output format)."""
    return f"```json\n{json.dumps(sample_reel_plan_dict, indent=2)}\n```"


@pytest.fixture
def mock_llm_truncated_response() -> str:
    """Return a truncated JSON to test repair logic."""
    return '{"reel_groups": [{"group_index": 0, "source_clips": [{"source_start": 0.0, "source_end": 5.0}], '


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


@pytest.fixture
def mock_subprocess_run(monkeypatch: pytest.MonkeyPatch):
    """Patch subprocess.run globally — returns a configurable MagicMock."""
    mock = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr="", text=True))
    monkeypatch.setattr(subprocess, "run", mock)
    return mock


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


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_source_clips():
    """Sample SourceClip objects matching sample_transcript."""
    from backend.models import SourceClip
    return [
        SourceClip(source_start=0.0, source_end=5.0, reason="Opening hook"),
        SourceClip(source_start=12.0, source_end=24.0, reason="Key reveal"),
        SourceClip(source_start=36.0, source_end=42.0, reason="Discovery moment"),
    ]


@pytest.fixture
def sample_narration_events():
    """Sample NarrationEvent objects for composition testing."""
    from backend.models import NarrationEvent
    return [
        NarrationEvent(event_type="hook", reel_start=0.0, reel_end=3.0, text="One discovery changed everything."),
        NarrationEvent(event_type="commentary", reel_start=30.0, reel_end=33.0, text="The results shocked researchers."),
    ]


@pytest.fixture
def sample_reel_group(sample_reel_plan_dict):
    """A ReelGroup model from the sample plan dict."""
    from backend.models import ReelGroup
    return ReelGroup(**sample_reel_plan_dict["reel_groups"][0])


@pytest.fixture
def sample_video_job():
    """A fresh VideoJob in QUEUED state."""
    from backend.models import VideoJob
    return VideoJob(url="https://www.youtube.com/watch?v=test123")
