import asyncio
from pathlib import Path
from typing import Optional, Callable, List
from backend.models import VideoJob, JobStatus, ReelPlan, ReelGroup, OutputReel
from backend.config import HOOK_SECONDS, get_job_working_dir, FFMPEG_PATH, FFPROBE_PATH
from backend.pipeline.downloader import download_video
from backend.pipeline.transcriber import transcribe_video
from backend.pipeline.analyzer import select_reel_plan, select_clips
from backend.pipeline.commentary import write_commentary
from backend.pipeline.clipper import cut_group_clips
from backend.pipeline.tts import synthesize_commentary
from backend.pipeline.captioner import generate_clip_ass, generate_commentary_ass
from backend.pipeline.compositor import compose_group
from backend.progress import ProgressReporter


def format_bytes(num_bytes):
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024**3):.1f} GB"
    elif num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024**2):.1f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def format_speed(bytes_per_sec):
    return f"{format_bytes(bytes_per_sec)}/s"


def format_eta(seconds):
    if seconds is None:
        return "..."
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    elif seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds)}s"


class QueueManager:
    def __init__(self, loop):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.jobs: dict[str, VideoJob] = {}
        self.loop = loop

    def add_job(self, url: str) -> VideoJob:
        self.jobs = {k: v for k, v in self.jobs.items() if v.status == "DONE"}
        job = VideoJob(url=url)
        self.jobs[job.id] = job
        self.queue.put_nowait(job.id)
        return job

    def get_jobs(self) -> list[VideoJob]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at)

    async def worker(self, broadcast_fn):
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
                await self._broadcast(broadcast_fn, job)
            finally:
                self.queue.task_done()

    async def _broadcast(self, broadcast_fn, job: VideoJob):
        coro = broadcast_fn(job)
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _process_job(self, job: VideoJob, broadcast_fn: Callable):
        reporter = ProgressReporter(job, broadcast_fn, self.loop)

        # Stage 1: DOWNLOADING
        job.stage_index = 1
        job.total_stages = 8
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

        result = await asyncio.to_thread(
            download_video, job.url, str(job_download_dir), sync_progress_hook
        )
        job.title = result.get("title")
        job.source_path = result.get("source_path")

        if not job.source_path:
            raise RuntimeError("Download succeeded but source_path was not produced by yt-dlp.")

        reporter.log_info(f"Downloaded: {job.title} -> {job.source_path}")

        # Stage 2: TRANSCRIBING
        job.stage_index = 2
        job.stage_data = {"total_segments": 0, "current_segment": 0}
        reporter.update_stage(JobStatus.TRANSCRIBING, "Loading Whisper model...", 0, 2)

        def transcriber_progress(msg: str, prog: float):
            job.stage_data = {"total_segments": 0, "current_segment": 0, "message": msg, "progress": prog}
            reporter.progress_callback(msg, prog)

        job.transcript = await asyncio.to_thread(transcribe_video, job.source_path, transcriber_progress)
        if not job.transcript:
            raise RuntimeError(
                "Transcription produced no segments. The downloaded file may be invalid/corrupted."
            )
        job.stage_data = {"total_segments": len(job.transcript), "current_segment": len(job.transcript), "done": True}
        reporter.log_info(f"Transcribed {len(job.transcript)} segments")

        # Stage 3: ANALYZING - now produces reel_plan with reel_groups
        job.stage_index = 3
        job.stage_data = {"status": "sending", "message": "Sending transcript to LLM..."}
        reporter.update_stage(JobStatus.ANALYZING, "Sending transcript to LLM...", 0, 3)

        def analyzer_progress(msg: str, prog: float):
            job.stage_data = {"status": "processing", "message": msg, "progress": prog}
            reporter.progress_callback(msg, prog)

        video_description = result.get("description", "")
        reel_plan: ReelPlan = await asyncio.to_thread(
            select_reel_plan, job.transcript, job.title or "", video_description, analyzer_progress
        )
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
        job.stage_data = {"status": "done", "groups_found": job.num_output_groups, "total_source_clips": total_clips}
        reporter.log_info(f"Analyzed: {job.num_output_groups} output group(s), {total_clips} source clips")

        # Initialize clip details for tracking (flattened across groups)
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
        # Stage mapping:
        # 4 = CLIPPING (cut source_clips for this group)
        # 5 = VOICING (TTS for narration_events in this group)
        # 6 = CAPTIONING (ASS for clip captions + narration captions)
        # 7 = COMPOSITING (build continuous video with overlay + ducking)
        # 8 = EDITING (light final trim/pad)

        working_dir = get_job_working_dir(job.id)

        for group_idx, group in enumerate(reel_plan.reel_groups):
            job.current_group_index = group_idx
            job.outputs[group_idx].status = "processing"
            await self._broadcast(broadcast_fn, job)

            # --- CLIPPING for this group ---
            job.stage_index = 4
            job.stage_data = {
                "status": "cutting",
                "group_index": group_idx,
                "total_groups": job.num_output_groups,
                "total_clips": len(group.source_clips),
                "current_clip": 0,
            }
            reporter.update_stage(JobStatus.CLIPPING, f"Group {group_idx+1}/{job.num_output_groups}: Cutting clips...", 0, 4)

            def clipper_progress(msg: str, prog: float, gi=group_idx):
                job.stage_data = {
                    "status": "cutting",
                    "group_index": gi,
                    "total_groups": job.num_output_groups,
                    "total_clips": len(group.source_clips),
                    "current_clip": int(prog / 100 * len(group.source_clips)) if prog > 0 else 0,
                    "message": msg,
                    "progress": prog,
                }
                reporter.progress_callback(msg, prog)

            group_clip_paths = await asyncio.to_thread(
                cut_group_clips, job.source_path, [c.model_dump() for c in group.source_clips], job.id, group_idx, clipper_progress, reporter
            )
            job.stage_data = {"status": "done", "group_index": group_idx, "clips_cut": len(group_clip_paths)}
            reporter.log_info(f"Group {group_idx+1}: Cut {len(group_clip_paths)} clips")

            # --- VOICING for this group ---
            job.stage_index = 5
            group_narration_events = [e for e in group.narration_events if e.event_type in ("hook", "commentary")]
            total_narration = len(group_narration_events)
            job.stage_data = {
                "status": "voicing",
                "group_index": group_idx,
                "total_groups": job.num_output_groups,
                "total": total_narration,
                "current": 0,
            }
            reporter.update_stage(JobStatus.VOICING, f"Group {group_idx+1}/{job.num_output_groups}: Generating TTS...", 0, 5)

            group_narration_audio = []
            for i, event in enumerate(group_narration_events):
                reporter.update_sub_stage(
                    f"Group {group_idx+1}: TTS for {event.event_type} ({i+1}/{total_narration})",
                    (i / total_narration) * 100 if total_narration > 0 else 100,
                )
                job.stage_data = {
                    "status": "voicing",
                    "group_index": group_idx,
                    "total": total_narration,
                    "current": i + 1,
                    "message": f"{event.event_type}: \"{event.text[:50]}\"",
                }

                def tts_progress(msg: str, prog: float, idx=i):
                    reporter.progress_callback(f"TTS: {msg}", prog)

                out_path = working_dir / f"group_{group_idx}_narration_{i}.wav"
                duration = await asyncio.to_thread(
                    synthesize_commentary, event.text, str(out_path), tts_progress
                )
                actual_reel_end = event.reel_start + duration
                group_narration_audio.append({
                    "event_type": event.event_type,
                    "reel_start": event.reel_start,
                    "reel_end": actual_reel_end,
                    "text": event.text,
                    "path": str(out_path),
                    "duration": duration,
                })
            job.narration_audio = group_narration_audio
            job.stage_data = {"status": "done", "group_index": group_idx, "files_generated": len(group_narration_audio)}
            reporter.log_info(f"Group {group_idx+1}: Generated {len(group_narration_audio)} narration audio files")

            # --- CAPTIONING for this group ---
            job.stage_index = 6
            job.stage_data = {
                "status": "captioning",
                "group_index": group_idx,
                "total_groups": job.num_output_groups,
                "total_clips": len(group.source_clips),
                "current": 0,
            }
            reporter.update_stage(JobStatus.CAPTIONING, f"Group {group_idx+1}/{job.num_output_groups}: Generating captions...", 0, 6)

            group_clip_captions = []
            group_narration_captions = []

            # Clip captions (from transcript, per source_clip)
            for i, clip in enumerate(group.source_clips):
                reporter.update_clip_progress(i, "captioning", (i / len(group.source_clips)) * 100)
                job.stage_data = {
                    "status": "captioning",
                    "group_index": group_idx,
                    "sub": "clip",
                    "current": i + 1,
                    "total": len(group.source_clips),
                }

                clip_caption_path = working_dir / f"group_{group_idx}_clip_caption_{i}.ass"

                def caption_progress(msg: str, prog: float, idx=i):
                    reporter.progress_callback(f"Clip {idx+1} caption: {msg}", prog)

                await asyncio.to_thread(
                    generate_clip_ass,
                    job.transcript,
                    clip.source_start,
                    clip.source_end,
                    str(clip_caption_path),
                    caption_progress,
                )
                group_clip_captions.append(str(clip_caption_path))

            # Narration captions (per narration_event, reel-relative)
            for i, event in enumerate(group.narration_events):
                if event.event_type not in ("hook", "commentary"):
                    continue
                job.stage_data = {
                    "status": "captioning",
                    "group_index": group_idx,
                    "sub": "narration",
                    "current": i + 1,
                    "total": len(group.narration_events),
                }

                narr_caption_path = working_dir / f"group_{group_idx}_narr_caption_{i}.ass"

                def narr_caption_progress(msg: str, prog: float, idx=i):
                    reporter.progress_callback(f"Narration caption {idx+1}: {msg}", prog)

                await asyncio.to_thread(
                    generate_commentary_ass,
                    event.text,
                    event.reel_end - event.reel_start,
                    str(narr_caption_path),
                    narr_caption_progress,
                )
                group_narration_captions.append({
                    "event_type": event.event_type,
                    "reel_start": event.reel_start,
                    "reel_end": event.reel_end,
                    "path": str(narr_caption_path),
                })

            job.caption_paths = group_clip_captions
            job.stage_data = {"status": "done", "group_index": group_idx, "clip_captions": len(group_clip_captions), "narration_captions": len(group_narration_captions)}
            reporter.log_info(f"Group {group_idx+1}: Generated {len(group_clip_captions)} clip + {len(group_narration_captions)} narration captions")

            # --- COMPOSITING for this group ---
            job.stage_index = 7
            job.stage_data = {
                "status": "compositing",
                "group_index": group_idx,
                "total_groups": job.num_output_groups,
                "message": "Building continuous video with overlay + ducking...",
            }
            reporter.update_stage(JobStatus.COMPOSITING, f"Group {group_idx+1}/{job.num_output_groups}: Compositing...", 0, 7)

            def compositor_progress(msg: str, prog: float, gi=group_idx):
                job.stage_data = {
                    "status": "compositing",
                    "group_index": gi,
                    "message": msg,
                    "progress": prog,
                }
                reporter.progress_callback(msg, prog)

            group_output_path = await asyncio.to_thread(
                compose_group,
                job.id,
                group_idx,
                group_clip_paths,
                [c.model_dump() for c in group.source_clips],
                group_narration_audio,
                group_clip_captions,
                [c["path"] for c in group_narration_captions],
                job.source_path,
                working_dir,
                compositor_progress,
            )

            # --- EDITING for this group ---
            job.stage_index = 8
            job.stage_data = {
                "status": "editing",
                "group_index": group_idx,
                "total_groups": job.num_output_groups,
            }
            reporter.update_stage(JobStatus.EDITING, f"Group {group_idx+1}/{job.num_output_groups}: Final edit...", 0, 8)

            # Light final trim/pad - for now just probe and validate duration
            final_path = await asyncio.to_thread(
                self._final_edit_group,
                group_output_path,
                group,
                working_dir,
            )

            job.outputs[group_idx].output_path = final_path
            job.outputs[group_idx].output_url = f"/outputs/{Path(final_path).name}"
            job.outputs[group_idx].duration_seconds = await self._probe_duration(final_path)
            job.outputs[group_idx].status = "done"
            reporter.log_info(f"Group {group_idx+1} complete: {final_path} ({job.outputs[group_idx].duration_seconds:.1f}s)")

            await self._broadcast(broadcast_fn, job)

        job.status = JobStatus.DONE
        job.progress = 100.0
        job.stage_data = {"status": "done"}
        reporter.update_stage(JobStatus.DONE, "All groups complete!", 100, 8)
        reporter.log_info(f"Job {job.id} complete with {job.num_output_groups} output(s)")
        await self._broadcast(broadcast_fn, job)

    def _final_edit_group(self, input_path: str, group: ReelGroup, working_dir) -> str:
        """Light final validation - probe duration, ensure under 90s."""
        from backend.config import OUTPUTS_DIR
        import subprocess

        probe = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True
        )
        duration = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0

        if duration > 90:
            output_path = OUTPUTS_DIR / f"{group.group_index}_{group.reel_summary.title[:50]}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [FFMPEG_PATH, "-loglevel", "error", "-i", input_path, "-t", "90", "-c", "copy", "-y", str(output_path)],
                check=True
            )
            return str(output_path)

        return input_path

    async def _probe_duration(self, path: str) -> float:
        import subprocess
        probe = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True
        )
        if probe.returncode == 0:
            try:
                return float(probe.stdout.strip())
            except ValueError:
                pass
        return 0.0
