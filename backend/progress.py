import asyncio
from datetime import datetime
from typing import Callable, Optional, Any
from backend.models import VideoJob


class ProgressReporter:
    def __init__(self, job: VideoJob, broadcast_fn: Callable, loop: asyncio.AbstractEventLoop):
        self.job = job
        self.broadcast_fn = broadcast_fn
        self.loop = loop

    def update_stage(self, stage: str, sub_stage: Optional[str] = None, progress: float = 0.0, stage_index: Optional[int] = None):
        self.job.current_stage = stage
        if sub_stage is not None:
            self.job.sub_stage = sub_stage
        self.job.sub_stage_progress = progress
        if stage_index is not None:
            self.job.stage_index = stage_index
        self.job.status = stage
        self._broadcast()

    def update_sub_stage(self, sub_stage: str, progress: float = 0.0):
        self.job.sub_stage = sub_stage
        self.job.sub_stage_progress = progress
        self._broadcast()

    def log(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"info": "📝", "warn": "⚠️", "error": "❌", "debug": "🔧"}.get(level, "📝")
        self.job.logs.append(f"[{timestamp}] {prefix} {message}")
        if len(self.job.logs) > 200:
            self.job.logs = self.job.logs[-200:]
        self._broadcast()

    def log_info(self, message: str):
        self.log(message, "info")

    def log_warn(self, message: str):
        self.log(message, "warn")

    def log_error(self, message: str):
        self.log(message, "error")

    def log_debug(self, message: str):
        self.log(message, "debug")

    def set_clip_details(self, clip_details: list):
        self.job.clip_details = clip_details
        self._broadcast()

    def update_clip_progress(self, clip_index: int, status: str, progress: float = 0.0):
        if self.job.clip_details is None:
            self.job.clip_details = []
        while len(self.job.clip_details) <= clip_index:
            self.job.clip_details.append({"index": len(self.job.clip_details), "status": "pending", "progress": 0.0})
        self.job.clip_details[clip_index]["status"] = status
        self.job.clip_details[clip_index]["progress"] = progress
        self._broadcast()

    def progress_callback(self, message: str, progress: float = 0.0):
        """Generic progress callback for pipeline functions (sync, for thread use)"""
        self.job.sub_stage = message
        self.job.sub_stage_progress = progress
        self._broadcast()

    async def async_progress_callback(self, message: str, progress: float = 0.0):
        """Async progress callback for direct use in async context (immediate broadcast)"""
        self.job.sub_stage = message
        self.job.sub_stage_progress = progress
        await self._async_broadcast()

    def _broadcast(self):
        """Schedule broadcast on event loop (for use from threads)"""
        asyncio.run_coroutine_threadsafe(self.broadcast_fn(self.job), self.loop)

    async def _async_broadcast(self):
        """Broadcast immediately (for use from async context)"""
        await self.broadcast_fn(self.job)
