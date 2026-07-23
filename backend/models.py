from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional, List, Any
from pydantic import BaseModel, Field
import uuid


class OutputReel(BaseModel):
    output_index: int
    output_path: Optional[str] = None
    output_url: Optional[str] = None
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
    voice_id: Optional[str] = None


class ReelGroup(BaseModel):
    group_index: int
    group_reasoning: str
    estimated_duration_seconds: float
    reel_summary: ReelSummary
    source_clips: List[SourceClip]
    narration_events: List[NarrationEvent]


class ReelPlan(BaseModel):
    reel_groups: List[ReelGroup]
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
    title: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_path: Optional[str] = None
    transcript: Optional[List[dict]] = None
    clip_windows: Optional[List[dict]] = None
    commentary_lines: Optional[List[dict]] = None
    clip_paths: Optional[List[str]] = None
    commentary_audio: Optional[List[dict]] = None
    caption_paths: Optional[List[str]] = None
    narration_caption_path: Optional[str] = None
    output_path: Optional[str] = None
    outputs: List[OutputReel] = Field(default_factory=list)
    current_stage: Optional[str] = None
    sub_stage: Optional[str] = None
    sub_stage_progress: float = 0.0
    stage_index: int = 0
    total_stages: int = 8
    logs: List[str] = Field(default_factory=list)
    clip_details: Optional[List[dict]] = None
    download_stats: Optional[dict] = None
    stage_data: dict = Field(default_factory=dict)
    reel_plan: Optional[ReelPlan] = None
    narration_audio: Optional[List[dict]] = None
    has_existing_captions: Optional[List[bool]] = None
    audio_download_stats: Optional[dict] = None
    num_output_groups: int = 1
    current_group_index: int = 0