"""Pydantic models for video jobs, reel plans, and pipeline data structures."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pydantic import BaseModel, Field
import uuid


class OutputReel(BaseModel):
    output_index: int
    output_path: str | None = None
    output_url: str | None = None
    duration_seconds: float = 0.0
    group_reasoning: str = ""
    title: str = ""
    status: str = "pending"


class ReelSummary(BaseModel):
    title: str
    short_description: str = ""
    source_understanding: str
    narrative_angle: str
    key_moment: str


class SourceClip(BaseModel):
    source_start: float
    source_end: float
    reason: str


class NarrationEvent(BaseModel):
    event_type: str
    reel_start: float
    reel_end: float
    text: str
    voice_id: str | None = None


class ReelGroup(BaseModel):
    group_index: int
    group_reasoning: str
    estimated_duration_seconds: float
    reel_summary: ReelSummary
    source_clips: list[SourceClip]
    narration_events: list[NarrationEvent]


class ReelPlan(BaseModel):
    reel_groups: list[ReelGroup]
    is_fallback: bool = False


class LLMInteraction(BaseModel):
    """Structured log entry for an LLM interaction during pipeline processing.
    Collected in stage_data['llm_interactions'] and broadcast via WebSocket
    for live UI rendering."""
    timestamp: str = ""
    type: str = ""  # "prompt" | "response" | "error" | "retry"
    role: str = ""  # "system" | "user" | "assistant"
    content: str = ""  # preview (truncated for UI)
    full_content: str = ""  # full raw text (for expand modal)
    model: str = ""
    retry_count: int = 0
    error_type: str = ""  # "timeout" | "rate_limit" | "connection" | "json_parse" | "empty_content" | "unknown"
    token_count: str = ""  # e.g., "1500 out / 45000 in tokens" — for UI display of token usage, only populated on response type
    stage_name: str = ""  # e.g., "reel_plan", "reel_plan_retry" — the pipeline stage this interaction belongs to


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    DOWNLOADING = "DOWNLOADING"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING = "ANALYZING"
    CLIPPING = "CLIPPING"
    VOICING = "VOICING"
    CAPTIONING = "CAPTIONING"
    COMPOSITING = "COMPOSITING"
    EDITING = "EDITING"
    DONE = "DONE"
    ERROR = "ERROR"


class VideoJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    title: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_path: str | None = None
    transcript: list[dict] | None = None
    clip_windows: list[dict] | None = None
    commentary_lines: list[dict] | None = None
    clip_paths: list[str] | None = None
    commentary_audio: list[dict] | None = None
    caption_paths: list[str] | None = None
    narration_caption_path: str | None = None
    output_path: str | None = None
    outputs: list[OutputReel] = Field(default_factory=list)
    current_stage: str | None = None
    sub_stage: str | None = None
    sub_stage_progress: float = 0.0
    stage_index: int = 0
    total_stages: int = 8
    logs: list[str] = Field(default_factory=list)
    clip_details: list[dict] | None = None
    download_stats: dict | None = None
    stage_data: dict = Field(default_factory=dict)
    reel_plan: ReelPlan | None = None
    narration_audio: list[dict] | None = None
    has_existing_captions: list[bool] | None = None
    audio_download_stats: dict | None = None
    num_output_groups: int = 1
    current_group_index: int = 0