"""Thread-safe progress reporter for pipeline job status updates.

Provides ProgressReporter for updating job state (stage, sub_stage, logs, clip_details)
from pipeline worker threads, with throttled queue-based broadcasting.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from backend.models import AnalyzerPhase, AnalyzerPhaseStatus, VideoJob

__all__ = ["ProgressReporter"]


class ProgressReporter:
    """Thread-safe reporter that updates job state and enqueues broadcasts.

    All update methods are safe to call from any thread. Broadcasting is
    done via a thread-safe queue that the event loop drains — no
    run_coroutine_threadsafe needed.
    """

    def __init__(self, job: VideoJob, enqueue_broadcast_fn: Callable[[VideoJob], Any]) -> None:
        self.job = job
        self.enqueue_broadcast = enqueue_broadcast_fn
        self._lock = threading.Lock()
        self._last_broadcast = 0.0
        self._broadcast_interval = 0.2  # 200ms min between broadcasts (~5/sec max)

    def update_stage(self, stage: str, sub_stage: str | None = None, progress: float = 0.0, stage_index: int | None = None) -> None:
        """Update the current pipeline stage, sub-stage description, and progress."""
        with self._lock:
            self.job.current_stage = stage
            if sub_stage is not None:
                self.job.sub_stage = sub_stage
            self.job.sub_stage_progress = progress
            if stage_index is not None:
                self.job.stage_index = stage_index
            self.job.status = stage
        self._broadcast()

    def update_sub_stage(self, sub_stage: str, progress: float = 0.0) -> None:
        """Update the sub-stage description and progress within the current stage."""
        with self._lock:
            self.job.sub_stage = sub_stage
            self.job.sub_stage_progress = progress
        self._broadcast()

    def set_analyzer_phase_plan(self, phases: list[dict[str, Any]]) -> None:
        """Replace the analyzer phase list with a fresh, ordered, all-pending plan.

        Call once the set of phases that will run is known (e.g. the fixed
        prefix at the start of ANALYZING, or again once branch/entity_grouped
        is resolved and the rest of the roadmap is known). Each dict needs
        at minimum {"id", "label", "kind"} — see analyzer.ANALYZER_PHASE_REGISTRY.
        """
        with self._lock:
            self.job.analyzer_phases = [
                AnalyzerPhase(id=p["id"], label=p["label"], kind=p["kind"])
                for p in phases
            ]
        self._broadcast()

    def append_analyzer_phases(self, phases: list[dict[str, Any]]) -> None:
        """Append additional pending phases once a branch decision extends the plan."""
        with self._lock:
            existing_ids = {p.id for p in self.job.analyzer_phases}
            for p in phases:
                if p["id"] in existing_ids:
                    continue
                self.job.analyzer_phases.append(
                    AnalyzerPhase(id=p["id"], label=p["label"], kind=p["kind"])
                )
        self._broadcast()

    def update_analyzer_phase(
        self,
        phase_id: str,
        status: str,
        progress: float = 0.0,
        detail: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update one analyzer phase by id (running/done/skipped/error) and broadcast.

        Defensive: if the phase wasn't in the plan (drift between the registry
        and what actually ran), it's appended rather than silently dropped —
        the UI should never show progress for a phase it never announced.
        """
        with self._lock:
            phase = next((p for p in self.job.analyzer_phases if p.id == phase_id), None)
            if phase is None:
                phase = AnalyzerPhase(id=phase_id, label=phase_id.replace("_", " ").title(), kind="python")
                self.job.analyzer_phases.append(phase)
            phase.status = AnalyzerPhaseStatus(status)
            phase.progress = progress
            if detail is not None:
                phase.detail = detail
            if error is not None:
                phase.error = error
            now = datetime.now()
            if phase.status == AnalyzerPhaseStatus.RUNNING and phase.started_at is None:
                phase.started_at = now
            if phase.status in (AnalyzerPhaseStatus.DONE, AnalyzerPhaseStatus.SKIPPED, AnalyzerPhaseStatus.ERROR):
                phase.ended_at = now
        self._broadcast()

    def log(self, message: str, level: str = "info") -> None:
        """Append a timestamped log message and broadcast."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"info": "📝", "warn": "⚠️", "error": "❌", "debug": "🔧"}.get(level, "📝")
        with self._lock:
            self.job.logs.append(f"[{timestamp}] {prefix} {message}")
            if len(self.job.logs) > 200:
                self.job.logs = self.job.logs[-200:]
        self._broadcast()

    def log_info(self, message: str) -> None:
        """Log an info-level message."""
        self.log(message, "info")

    def log_warn(self, message: str) -> None:
        """Log a warning-level message."""
        self.log(message, "warn")

    def log_error(self, message: str) -> None:
        """Log an error-level message."""
        self.log(message, "error")

    def log_debug(self, message: str) -> None:
        """Log a debug-level message."""
        self.log(message, "debug")

    def set_stage_data_key(self, key: str, value: Any) -> None:
        """Set a key on stage_data and broadcast immediately.

        Used for live-updating UI components (e.g., llm_interactions).
        """
        with self._lock:
            self.job.stage_data[key] = value
        self._broadcast()

    def set_clip_details(self, clip_details: list[dict[str, Any]]) -> None:
        """Replace the full clip details list and broadcast."""
        with self._lock:
            self.job.clip_details = clip_details
        self._broadcast()

    def update_clip_progress(self, clip_index: int, status: str, progress: float = 0.0) -> None:
        """Update the status and progress of a specific clip by index."""
        with self._lock:
            if self.job.clip_details is None:
                self.job.clip_details = []
            while len(self.job.clip_details) <= clip_index:
                self.job.clip_details.append({"index": len(self.job.clip_details), "status": "pending", "progress": 0.0})
            self.job.clip_details[clip_index]["status"] = status
            self.job.clip_details[clip_index]["progress"] = progress
        self._broadcast()

    def progress_callback(self, message: str, progress: float = 0.0) -> None:
        """Generic progress callback for pipeline functions (sync, safe for thread use)."""
        with self._lock:
            self.job.sub_stage = message
            self.job.sub_stage_progress = progress
        self._broadcast()

    def _broadcast(self):
        """Throttled enqueue of job state for broadcasting (thread-safe)."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_broadcast < self._broadcast_interval:
                return
            self._last_broadcast = now
        with contextlib.suppress(Exception):
            self.enqueue_broadcast(self.job)  # Queue full or closed — drop update
