"""Job queue manager — orchestrates the full video processing pipeline.

Coordinates downloading, transcription, analysis, and per-group rendering
through GroupOrchestrator, with checkpoint-based resumability.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import Any, Callable

from backend.config import GPU_SEMAPHORE_SIZE, get_job_working_dir
from backend.models import JobStatus, LLMInteraction, NarrationEvent, OutputReel, ReelGroup, ReelPlan, VideoJob
from backend.pipeline.checkpoint import PipelineCheckpoint
from backend.pipeline.downloader import download_video, validate_downloaded_video
from backend.pipeline.analyzer import select_reel_plan
from backend.pipeline.orchestrator import GroupOrchestrator
from backend.pipeline.timeline_builder import build_rich_timeline
from backend.pipeline.transcriber import transcribe_video
from backend.progress import ProgressReporter

__all__ = ["QueueManager", "format_bytes", "format_speed", "format_eta"]

logger = logging.getLogger(__name__)


def format_bytes(num_bytes: int | float) -> str:
    """Format a byte count into a human-readable string (e.g., '1.5 GB')."""
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024**3):.1f} GB"
    elif num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024**2):.1f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def format_speed(bytes_per_sec: int | float) -> str:
    """Format bytes/second into a human-readable speed string."""
    return f"{format_bytes(bytes_per_sec)}/s"


def format_eta(seconds: int | float | None) -> str:
    """Format an ETA in seconds into a human-readable string (e.g., '2m 15s')."""
    if seconds is None:
        return "..."
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    elif seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds)}s"


class QueueManager:
    """Manages a FIFO queue of VideoJob processing tasks.

    Provides methods to add, list, and delete jobs, and runs a background
    worker that processes jobs sequentially through the full pipeline.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.jobs: dict[str, VideoJob] = {}
        self.loop = loop
        self.gpu_semaphore = asyncio.Semaphore(GPU_SEMAPHORE_SIZE)
        # Thread-SAFE broadcast queue: worker threads push here, event loop polls
        self._broadcast_queue: queue.Queue[VideoJob] = queue.Queue(maxsize=200)

    def enqueue_broadcast(self, job: VideoJob) -> None:
        """Thread-SAFE: put a job update into the broadcast queue from ANY thread."""
        try:
            self._broadcast_queue.put_nowait(job)
        except queue.Full:
            pass  # Drop oldest under backpressure

    def add_job(self, url: str) -> VideoJob:
        """Create a new job, enqueue it, and return the job object."""
        job = VideoJob(url=url)
        self.jobs[job.id] = job
        self.queue.put_nowait(job.id)
        # Immediately enqueue a broadcast so the frontend sees the new job
        self.enqueue_broadcast(job)
        return job

    def get_jobs(self) -> list[VideoJob]:
        """Return all jobs sorted by creation time (oldest first)."""
        return sorted(self.jobs.values(), key=lambda j: j.created_at)

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID. Returns True if found and removed, False otherwise."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False

    async def broadcast_drain_loop(self, broadcast_fn: Callable[[VideoJob], Any]) -> None:
        """Event-loop coroutine: poll the thread-safe queue and send to WebSocket clients.

        Polls every 100ms. Deduplicates by job.id, sending only the latest state.
        """
        while True:
            await asyncio.sleep(0.1)  # Poll every 100ms
            # Drain all pending items (dedup: keep latest per job_id)
            pending: dict[str, VideoJob] = {}
            while True:
                try:
                    job = self._broadcast_queue.get_nowait()
                    pending[job.id] = job
                except queue.Empty:
                    break
            # Broadcast each unique job's latest state
            for latest_job in pending.values():
                try:
                    await broadcast_fn(latest_job)
                except Exception as e:
                    logger.debug("[QueueManager] Broadcast failed for %s: %s", latest_job.id, e)

    async def worker(self, broadcast_fn: Callable[[VideoJob], Any]) -> None:
        """Background worker that processes jobs one at a time (sequential queue)."""
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue

            try:
                await self._process_job(job, broadcast_fn)
            except Exception as e:
                job.status = JobStatus.ERROR
                job.error = str(e)
                self.enqueue_broadcast(job)
            finally:
                self.queue.task_done()

    async def _process_job(self, job: VideoJob, broadcast_fn: Callable[[VideoJob], Any]) -> None:
        """Execute the full pipeline for a single job.

        Stages: download → transcribe → analyze → per-group render (clip/TTS/caption/compose/edit).
        Supports checkpoint-based resumability and per-group retry.
        """
        reporter = ProgressReporter(job, self.enqueue_broadcast)
        ckpt = PipelineCheckpoint(get_job_working_dir(job.id))

        # Stage 1: DOWNLOADING
        job.stage_index = 1
        job.total_stages = 9
        job.stage_data = {
            "downloaded_bytes": 0, "total_bytes": 0,
            "speed": 0, "eta": 0, "status": "starting"
        }
        reporter.update_stage(JobStatus.DOWNLOADING, "Initializing download...", 0, 1)

        def sync_progress_hook(d: dict):
            downloaded = d.get("downloaded_bytes", 0) or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            speed = d.get("speed", 0) or 0
            eta = d.get("eta")
            status = d.get("status", "starting")

            stats = {
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "speed": speed,
                "eta": eta,
                "status": status,
            }
            job.download_stats = stats
            job.stage_data = stats

            if status == "downloading":
                if total > 0:
                    pct = min(99.0, (downloaded / total) * 100)
                    job.progress = pct
                    speed_str = format_speed(speed)
                    eta_str = format_eta(eta) if eta else "..."
                    downloaded_str = format_bytes(downloaded)
                    total_str = format_bytes(total)
                    reporter.progress_callback(
                        f"{downloaded_str} / {total_str} | {speed_str} | ETA {eta_str}", pct
                    )
                else:
                    reporter.progress_callback(f"Downloading... {format_bytes(downloaded)}", 0)
            elif status == "finished":
                job.progress = 100.0
                job.download_stats = {
                    "status": "finished",
                    "downloaded_bytes": downloaded,
                    "total_bytes": total,
                }
                job.stage_data = job.download_stats
                reporter.progress_callback("Download complete!", 100)
                reporter.log_info("Download complete, processing...")
            else:
                reporter.progress_callback(f"Status: {status}", 0)

        job_working_dir = get_job_working_dir(job.id)
        job_download_dir = job_working_dir / "downloads"

        # Check for download checkpoint
        download_ckpt = ckpt.load_stage("download")
        if download_ckpt and download_ckpt.get("source_path"):
            job.title = download_ckpt.get("title")
            job.source_path = download_ckpt["source_path"]
            reporter.log_info(f"Resuming from checkpoint: download already complete ({job.source_path})")
            result = download_ckpt  # Use checkpoint data for downstream fields
        else:
            result = await asyncio.to_thread(
                download_video, job.url, str(job_download_dir), sync_progress_hook
            )
            job.title = result.get("title")
            job.source_path = result.get("source_path")

            if not job.source_path:
                raise RuntimeError("Download succeeded but source_path was not produced by yt-dlp.")

            reporter.log_info(f"Downloaded: {job.title} -> {job.source_path}")
            ckpt.save_stage("download", {"title": job.title, "source_path": job.source_path, "description": result.get("description", "")})

        # Validate the downloaded video
        validation = await asyncio.to_thread(validate_downloaded_video, job.source_path)
        if not validation["valid"]:
            raise RuntimeError(
                f"Downloaded video failed validation: {validation['error']}"
            )
        for warning in validation["warnings"]:
            reporter.log_info(f"[WARN] {warning}")
        reporter.log_info(
            f"Video validated: {validation['duration']:.1f}s, "
            f"{validation['width']}x{validation['height']}"
        )

        # Stage 2: TRANSCRIBING
        job.stage_index = 2
        job.stage_data = {"total_segments": 0, "current_segment": 0}
        reporter.update_stage(JobStatus.TRANSCRIBING, "Loading Whisper model...", 0, 2)

        # Check for transcribe checkpoint
        transcribe_ckpt = ckpt.load_stage("transcribe")
        if transcribe_ckpt and transcribe_ckpt.get("transcript"):
            job.transcript = transcribe_ckpt["transcript"]
            reporter.log_info(f"Resuming from checkpoint: transcription already complete ({len(job.transcript)} segments)")
        else:
            def transcriber_progress(msg: str, prog: float):
                job.stage_data = {"total_segments": 0, "current_segment": 0, "message": msg, "progress": prog}
                reporter.progress_callback(msg, prog)

            job.transcript = await asyncio.to_thread(transcribe_video, job.source_path, transcriber_progress)
            if not job.transcript:
                raise RuntimeError(
                    "Transcription produced no segments. The downloaded file may be invalid/corrupted."
                )
            ckpt.save_stage("transcribe", {"transcript": job.transcript})

        job.stage_data = {"total_segments": len(job.transcript), "current_segment": len(job.transcript), "done": True}
        reporter.log_info(f"Transcribed {len(job.transcript)} segments")

        # Stage 3: BUILDING RICH TIMELINE
        job.stage_index = 3
        job.total_stages = 9
        job.stage_data = {"status": "building_timeline", "message": "Building Rich Timeline (VAD + OCR + FFmpeg metrics)..."}
        reporter.update_stage(JobStatus.BUILDING_TIMELINE, "Building Rich Timeline...", 0, 3)

        # Check for timeline checkpoint
        timeline_ckpt = ckpt.load_stage("rich_timeline")
        if timeline_ckpt and timeline_ckpt.get("segments"):
            from backend.models import RichTimeline
            job.rich_timeline = RichTimeline.model_validate(timeline_ckpt)
            reporter.log_info(f"Resuming from checkpoint: Rich Timeline already built ({len(job.rich_timeline.segments)} segments)")
        else:
            def timeline_progress(msg: str, prog: float):
                existing = job.stage_data if isinstance(job.stage_data, dict) else {}
                job.stage_data = {**existing, "status": "building_timeline", "message": msg, "progress": prog}
                reporter.progress_callback(msg, prog)

            job.rich_timeline = await asyncio.to_thread(
                build_rich_timeline, job.transcript, job.source_path,
                timeline_progress, reporter
            )
            ckpt.save_stage("rich_timeline", job.rich_timeline.model_dump())

        reporter.log_info(
            f"Rich Timeline: {len(job.rich_timeline.segments)} segments, "
            f"speech={job.rich_timeline.total_speech_duration:.1f}s, "
            f"VAD_regions={job.rich_timeline.speech_region_count}, "
            f"OCR_texts={job.rich_timeline.ocr_region_count}"
        )

        # Stage 4: ANALYZING
        job.stage_index = 4
        job.stage_data = {"status": "sending", "message": "Sending Rich Timeline to LLM..."}
        reporter.update_stage(JobStatus.ANALYZING, "Sending Rich Timeline to LLM...", 0, 4)

        llm_interactions: list[LLMInteraction] = []

        # Check for analyze checkpoint
        analyze_ckpt = ckpt.load_stage("analyze")
        if analyze_ckpt and analyze_ckpt.get("reel_plan"):
            reel_plan = ReelPlan.model_validate(analyze_ckpt["reel_plan"])
            reporter.log_info(f"Resuming from checkpoint: analysis already complete ({len(reel_plan.reel_groups)} groups)")
        else:
            def analyzer_progress(msg: str, prog: float):
                existing = job.stage_data if isinstance(job.stage_data, dict) else {}
                job.stage_data = {**existing, "status": "processing", "message": msg, "progress": prog}
                reporter.progress_callback(msg, prog)

            video_description = result.get("description", "")
            try:
                reel_plan = await asyncio.wait_for(
                    asyncio.to_thread(
                        select_reel_plan, job.transcript, job.title or "", video_description,
                        analyzer_progress, reporter, llm_interactions,
                        rich_timeline=job.rich_timeline,
                    ),
                    timeout=1200.0,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "ANALYZING exceeded 20-minute hard ceiling — LLM never responded "
                    "in time even after retries. Check NVIDIA API status."
                )
            ckpt.save_stage("analyze", {"reel_plan": reel_plan.model_dump()})

        if llm_interactions:
            job.stage_data["llm_interactions"] = [i.model_dump() for i in llm_interactions]
            reporter.set_stage_data_key("llm_interactions", [i.model_dump() for i in llm_interactions])
        else:
            job.stage_data["llm_interactions"] = []
            reporter.set_stage_data_key("llm_interactions", [])
        if getattr(reel_plan, "is_fallback", False):
            reporter.log_info("Using fallback reel plan (LLM was unavailable) — continuing with fallback.")
            existing = job.stage_data if isinstance(job.stage_data, dict) else {}
            job.stage_data = {**existing, "status": "fallback", "message": "LLM unavailable, using heuristic fallback plan."}
        job.reel_plan = reel_plan
        job.num_output_groups = len(reel_plan.reel_groups)
        job.current_group_index = 0

        # Initialize OutputReel objects for each group
        job.outputs = [
            OutputReel(
                output_index=i,
                group_reasoning=group.group_reasoning,
                title=group.reel_summary.title,
                status="pending",
            )
            for i, group in enumerate(reel_plan.reel_groups)
        ]

        total_clips = sum(len(g.source_clips) for g in reel_plan.reel_groups)
        existing = job.stage_data if isinstance(job.stage_data, dict) else {}
        job.stage_data = {**existing, "status": "done", "groups_found": job.num_output_groups, "total_source_clips": total_clips}
        reporter.log_info(f"Analyzed: {job.num_output_groups} output group(s), {total_clips} source clips")

        # Initialize clip details for tracking
        clip_details = []
        clip_idx = 0
        for group in reel_plan.reel_groups:
            for clip in group.source_clips:
                clip_details.append({
                    "index": clip_idx,
                    "group_index": group.group_index,
                    "start": clip.source_start,
                    "end": clip.source_end,
                    "status": "pending",
                    "progress": 0.0,
                })
                clip_idx += 1
        reporter.set_clip_details(clip_details)

        # Stages 4-8: Per-group pipeline loop
        orchestrator = GroupOrchestrator(job, self.enqueue_broadcast)

        for group_idx, group in enumerate(reel_plan.reel_groups):
            self.enqueue_broadcast(job)
            await orchestrator.run_group(group_idx, group, reporter, job.source_path)
            self.enqueue_broadcast(job)

        job.status = JobStatus.DONE
        job.progress = 100.0
        job.stage_data = {"status": "done"}
        reporter.update_stage(JobStatus.DONE, "All groups complete!", 100, 9)
        reporter.log_info(f"Job {job.id} complete with {job.num_output_groups} output(s)")
        self.enqueue_broadcast(job)
