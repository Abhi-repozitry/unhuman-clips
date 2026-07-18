from datetime import datetime
from enum import StrEnum
from typing import Optional, List, Any
from pydantic import BaseModel, Field
import uuid


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    DOWNLOADING = "DOWNLOADING"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING = "ANALYZING"
    SCRIPTING = "SCRIPTING"
    CLIPPING = "CLIPPING"
    VOICING = "VOICING"
    CAPTIONING = "CAPTIONING"
    COMPOSITING = "COMPOSITING"
    DONE = "DONE"
    ERROR = "ERROR"


class VideoJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    title: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    source_path: Optional[str] = None
    transcript: Optional[List[dict]] = None
    clip_windows: Optional[List[dict]] = None
    commentary_lines: Optional[List[dict]] = None
    clip_paths: Optional[List[str]] = None
    commentary_audio: Optional[List[dict]] = None
    caption_paths: Optional[List[str]] = None
    output_path: Optional[str] = None
    current_stage: Optional[str] = None
    sub_stage: Optional[str] = None
    sub_stage_progress: float = 0.0
    stage_index: int = 0
    total_stages: int = 8
    logs: List[str] = Field(default_factory=list)
    clip_details: Optional[List[dict]] = None
    download_stats: Optional[dict] = None
    stage_data: dict = Field(default_factory=dict)
