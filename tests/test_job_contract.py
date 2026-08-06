"""Tests for job creation metadata and hook-mode contracts."""

from __future__ import annotations

import asyncio

import pytest

from backend import main
from backend.models import SourceMetadata
from backend.queue_manager import QueueManager


def test_queue_job_keeps_selected_hook_mode():
    loop = asyncio.new_event_loop()
    try:
        manager = QueueManager(loop)
        job = manager.add_job("https://youtube.com/watch?v=test", hook_mode="skip")
    finally:
        loop.close()

    assert job.hook_mode == "skip"


@pytest.mark.asyncio
async def test_preflight_returns_normalized_source_metadata(monkeypatch):
    monkeypatch.setattr(main, "_check_rate_limit", lambda: True)
    monkeypatch.setattr(
        main,
        "fetch_video_metadata",
        lambda url: {
            "video_id": "test",
            "title": "Test video",
            "description": "Video description",
            "channel_name": "Test channel",
        },
    )

    response = await main.preflight_job(main.PreflightJobRequest(url="https://youtube.com/watch?v=test"))

    assert response.suggested_hook_mode == "required"
    assert response.source_metadata == SourceMetadata(
        video_id="test",
        title="Test video",
        description="Video description",
        channel_name="Test channel",
    )


def test_create_job_request_validates_hook_mode():
    assert main.CreateJobRequest(url="https://youtube.com/watch?v=test").hook_mode == "auto"
    assert main.CreateJobRequest(url="https://youtube.com/watch?v=test", hook_mode="required").hook_mode == "required"
    with pytest.raises(ValueError):
        main.CreateJobRequest(url="https://youtube.com/watch?v=test", hook_mode="invalid")
