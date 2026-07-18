import asyncio
from typing import Optional, Callable
from backend.models import VideoJob, JobStatus
from backend.config import HOOK_SECONDS, get_job_working_dir
from backend.pipeline.downloader import download_video
from backend.pipeline.transcriber import transcribe_video
from backend.pipeline.analyzer import select_clips
from backend.pipeline.commentary import write_commentary
from backend.pipeline.clipper import cut_clips
from backend.pipeline.tts import synthesize_commentary
from backend.pipeline.captioner import generate_clip_ass, generate_commentary_ass
from backend.pipeline.compositor import build_final_video
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
        # Clean up old terminal jobs so the frontend doesn't show stale data
        # Keep only DONE jobs (completed successfully), remove ERROR and all in-progress states
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

        # Job-isolated download directory to prevent stale "old video data" reuse.
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

        # Stage 3: ANALYZING
        job.stage_index = 3
        job.stage_data = {"status": "sending", "message": "Sending transcript to LLM..."}
        reporter.update_stage(JobStatus.ANALYZING, "Sending transcript to LLM...", 0, 3)

        def analyzer_progress(msg: str, prog: float):
            job.stage_data = {"status": "processing", "message": msg, "progress": prog}
            reporter.progress_callback(msg, prog)

        video_description = result.get("description", "")
        job.clip_windows = await asyncio.to_thread(
            select_clips, job.transcript, job.title or "", video_description, analyzer_progress
        )
        if not job.clip_windows:
            raise RuntimeError(
                "Clip selection returned no clips. Transcript may be too short/empty or analysis failed."
            )
        job.clip_windows.sort(key=lambda w: w["start"])
        job.stage_data = {"status": "done", "clips_found": len(job.clip_windows)}
        reporter.log_info(f"Selected {len(job.clip_windows)} clips")

        # Initialize clip details for tracking
        clip_details = [
            {
                "index": i,
                "start": w["start"],
                "end": w["end"],
                "status": "pending",
                "progress": 0.0,
            }
            for i, w in enumerate(job.clip_windows)
        ]
        reporter.set_clip_details(clip_details)

        # Stage 4: SCRIPTING
        job.stage_index = 4
        job.stage_data = {"status": "generating", "message": "Generating commentary script..."}
        reporter.update_stage(JobStatus.SCRIPTING, "Generating commentary script...", 0, 4)

        def commentary_progress(msg: str, prog: float):
            job.stage_data = {"status": "generating", "message": msg, "progress": prog}
            reporter.progress_callback(msg, prog)

        job.commentary_lines = await asyncio.to_thread(
            write_commentary, job.clip_windows, job.title or "", job.transcript, commentary_progress
        )
        job.stage_data = {"status": "done", "lines_generated": len(job.commentary_lines)}
        reporter.log_info(f"Generated {len(job.commentary_lines)} commentary lines")

        # Stage 5: CLIPPING
        job.stage_index = 5
        job.stage_data = {"status": "cutting", "total_clips": len(job.clip_windows), "current_clip": 0}
        reporter.update_stage(JobStatus.CLIPPING, "Cutting video clips...", 0, 5)

        def clipper_progress(msg: str, prog: float):
            job.stage_data = {
                "status": "cutting",
                "total_clips": len(job.clip_windows),
                "current_clip": int(prog / 100 * len(job.clip_windows)) if prog > 0 else 0,
                "message": msg,
                "progress": prog,
            }
            reporter.progress_callback(msg, prog)

        job.clip_paths = await asyncio.to_thread(
            cut_clips, job.source_path, job.clip_windows, job.id, clipper_progress, reporter
        )
        job.stage_data = {"status": "done", "clips_cut": len(job.clip_paths)}
        reporter.log_info(f"Cut {len(job.clip_paths)} clips")

        # Stage 6: VOICING
        job.stage_index = 6
        total_comments = len(job.commentary_lines)
        job.stage_data = {
            "status": "voicing", "total": total_comments, "current": 0
        }
        reporter.update_stage(JobStatus.VOICING, "Generating TTS audio...", 0, 6)

        working_dir = get_job_working_dir(job.id)
        commentary_audio = []
        for i, comment in enumerate(job.commentary_lines):
            clip_idx = i
            reporter.update_sub_stage(
                f"Generating hook/insight TTS for clip {i+1}/{total_comments}",
                (i / total_comments) * 100,
            )
            reporter.update_clip_progress(i, "voicing", (i / total_comments) * 100)
            job.stage_data = {
                "status": "voicing",
                "total": total_comments,
                "current": i + 1,
                "message": f"Clip {i+1}: \"{comment['hook_text'][:50]}\"",
            }

            def tts_progress(msg: str, prog: float, idx=clip_idx):
                reporter.progress_callback(f"Clip {idx+1} TTS: {msg}", prog)

            hook_path = working_dir / f"hook_{i}.wav"
            hook_duration = await asyncio.to_thread(
                synthesize_commentary, comment["hook_text"], str(hook_path), tts_progress
            )

            insight_path = working_dir / f"insight_{i}.wav"
            insight_duration = await asyncio.to_thread(
                synthesize_commentary, comment["insight_text"], str(insight_path), tts_progress
            )

            commentary_audio.append({
                "hook": {"path": str(hook_path), "duration": hook_duration},
                "insight": {"path": str(insight_path), "duration": insight_duration},
            })
            reporter.update_clip_progress(i, "voicing_done", 100)
        job.commentary_audio = commentary_audio
        job.stage_data = {"status": "done", "files_generated": len(commentary_audio) * 2}
        reporter.log_info(f"Generated {len(commentary_audio) * 2} audio files")

        # Stage 7: CAPTIONING
        job.stage_index = 7
        total_windows = len(job.clip_windows)
        job.stage_data = {
            "status": "captioning", "total": total_windows, "current": 0
        }
        reporter.update_stage(JobStatus.CAPTIONING, "Generating captions...", 0, 7)

        caption_paths = []
        commentary_caption_paths = []
        for i, window in enumerate(job.clip_windows):
            clip_idx = i
            reporter.update_sub_stage(
                f"Generating captions for clip {i+1}/{total_windows}",
                (i / total_windows) * 100,
            )
            reporter.update_clip_progress(i, "captioning", (i / total_windows) * 100)
            job.stage_data = {
                "status": "captioning",
                "total": total_windows,
                "current": i + 1,
                "sub": "clip",
            }
            clip_caption_path = working_dir / f"clip_caption_{i}.ass"

            def caption_progress(msg: str, prog: float, idx=clip_idx):
                reporter.progress_callback(f"Clip {idx+1} caption: {msg}", prog)

            await asyncio.to_thread(
                generate_clip_ass,
                job.transcript,
                min(window["end"], window["start"] + HOOK_SECONDS),
                window["end"],
                str(clip_caption_path),
                caption_progress,
            )
            caption_paths.append(str(clip_caption_path))

            job.stage_data = {
                "status": "captioning",
                "total": total_windows,
                "current": i + 1,
                "sub": "hook/insight",
            }
            hook_caption_path = working_dir / f"hook_caption_{i}.ass"
            insight_caption_path = working_dir / f"insight_caption_{i}.ass"

            def comm_caption_progress(msg: str, prog: float, idx=clip_idx):
                reporter.progress_callback(f"Clip {idx+1} commentary caption: {msg}", prog)

            await asyncio.to_thread(
                generate_commentary_ass,
                job.commentary_lines[i]["hook_text"],
                job.commentary_audio[i]["hook"]["duration"],
                str(hook_caption_path),
                comm_caption_progress,
            )
            await asyncio.to_thread(
                generate_commentary_ass,
                job.commentary_lines[i]["insight_text"],
                job.commentary_audio[i]["insight"]["duration"],
                str(insight_caption_path),
                comm_caption_progress,
            )
            commentary_caption_paths.append({
                "hook": str(hook_caption_path),
                "insight": str(insight_caption_path),
            })
            reporter.update_clip_progress(i, "captioning_done", 100)
        job.caption_paths = caption_paths
        job.stage_data = {"status": "done", "files_generated": len(caption_paths) * 3}
        reporter.log_info(f"Generated {len(caption_paths) * 3} caption files")

        # Stage 8: COMPOSITING
        job.stage_index = 8
        job.stage_data = {
            "status": "compositing",
            "total_segments": len(job.clip_paths),
            "current_segment": 0,
        }
        reporter.update_stage(JobStatus.COMPOSITING, "Compositing final video...", 0, 8)

        def compositor_progress(msg: str, prog: float):
            total_segs = len(job.clip_paths)
            current_seg = int(prog / 100 * total_segs) if prog > 0 else 0
            job.stage_data = {
                "status": "compositing",
                "total_segments": total_segs,
                "current_segment": min(current_seg, total_segs),
                "message": msg,
                "progress": prog,
            }
            reporter.progress_callback(msg, prog)

        job.output_path = await asyncio.to_thread(
            build_final_video,
            job.id,
            job.clip_paths,
            job.clip_windows,
            job.commentary_audio,
            commentary_caption_paths,
            job.caption_paths,
            compositor_progress,
        )

        job.status = JobStatus.DONE
        job.progress = 100.0
        job.stage_data = {"status": "done"}
        reporter.update_stage(JobStatus.DONE, "Complete!", 100, 8)
        reporter.log_info(f"Video saved to {job.output_path}")
        await self._broadcast(broadcast_fn, job)
