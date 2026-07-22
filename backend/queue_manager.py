import asyncio
from pathlib import Path
from typing import Optional, Callable, List
from backend.models import VideoJob, JobStatus, ReelPlan, ReelGroup, OutputReel, NarrationEvent
from backend.config import HOOK_SECONDS, get_job_working_dir, FFMPEG_PATH, FFPROBE_PATH
from backend.pipeline.downloader import download_video
from backend.pipeline.transcriber import transcribe_video
from backend.pipeline.analyzer import select_reel_plan
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


def validate_and_adjust_narration_timings(
    group_narration_audio: List[dict],
    source_clips: list,
    transcript: list,
    target_duration: float,
    reporter,
    group_idx: int,
):
    """
    Map reel timestamps to source transcript speech intervals across source_clips.
    Detect commentary narration events that collide with active speech (>30% overlap)
    and automatically shift their reel_start to nearby speech gaps or the post-clip area.
    Ensures non-overlapping narration windows and caps all narrations within target_duration.
    """
    if not group_narration_audio:
        return

    # 1. Map transcript speech segments to reel-relative timeline
    reel_speech_intervals = []
    cumulative_offset = 0.0
    for clip in source_clips:
        c_start = clip.source_start if hasattr(clip, "source_start") else clip["source_start"]
        c_end = clip.source_end if hasattr(clip, "source_end") else clip["source_end"]
        clip_dur = c_end - c_start
        for seg in transcript:
            s_start = seg["start"]
            s_end = seg["end"]
            ov_s = max(c_start, s_start)
            ov_e = min(c_end, s_end)
            if ov_s < ov_e - 0.1:  # meaningful speech duration
                reel_s = cumulative_offset + (ov_s - c_start)
                reel_e = cumulative_offset + (ov_e - c_start)
                reel_speech_intervals.append((reel_s, reel_e, seg.get("text", "")))
        cumulative_offset += clip_dur

    def get_speech_overlap(r_start: float, r_end: float) -> tuple:
        total_overlap = 0.0
        texts = []
        for s_start, s_end, text in reel_speech_intervals:
            ov_s = max(r_start, s_start)
            ov_e = min(r_end, s_end)
            if ov_s < ov_e:
                total_overlap += (ov_e - ov_s)
                if text:
                    texts.append(text)
        return total_overlap, texts

    def find_speech_gap(duration: float, search_start: float) -> float:
        """Find a gap of at least `duration` seconds with minimal speech overlap.
        Enforces 0.4s minimum distance from any speech boundary."""
        candidate = search_start
        max_search = max(target_duration, cumulative_offset + 30.0)
        while candidate + duration <= max_search:
            overlap, _ = get_speech_overlap(candidate, candidate + duration)
            if overlap <= 0.05:  # Tighter: near-zero overlap required
                # Verify 0.4s gap from nearest speech boundaries
                gap_ok = True
                for s_start, s_end, _ in reel_speech_intervals:
                    # Check narration start isn't too close to speech end
                    if abs(candidate - s_end) < 0.4 and candidate >= s_end - 0.1:
                        gap_ok = False
                        break
                    # Check narration end isn't too close to speech start
                    if abs((candidate + duration) - s_start) < 0.4 and (candidate + duration) <= s_start + 0.1:
                        gap_ok = False
                        break
                if gap_ok:
                    return candidate
            next_step = candidate + 0.3  # Finer search granularity
            for s_start, s_end, _ in reel_speech_intervals:
                if s_start <= candidate < s_end:
                    next_step = max(next_step, s_end + 0.4)  # 0.4s gap after speech
            candidate = next_step
        return search_start

    # 2. Inspect and shift commentary narrations that collide with active dialogue
    for nar in group_narration_audio:
        event_type = nar.get("event_type", "commentary")
        duration = nar.get("duration", 0.0)
        if duration <= 0.1:
            continue

        reel_s = nar["reel_start"]
        reel_e = nar["reel_start"] + duration
        overlap, texts = get_speech_overlap(reel_s, reel_e)
        overlap_ratio = overlap / duration if duration > 0 else 0.0

        if event_type in ("commentary", "hook") and overlap_ratio > 0.15:  # Stricter: 15% threshold
            sample_text = texts[0][:60] + "..." if texts else ""
            reporter.log_info(
                f"[WARN] Group {group_idx+1}: {event_type.capitalize()} narration '{nar['text'][:40]}...' "
                f"at reel [{reel_s:.2f}s-{reel_e:.2f}s] overlaps {overlap_ratio*100:.0f}% with transcript speech "
                f"(\"{sample_text}\"). Auto-shifting to nearest silent gap..."
            )
            new_s = find_speech_gap(duration, reel_s)
            if new_s != reel_s and new_s + duration <= target_duration:
                old_start = nar['reel_start']
                nar["reel_start"] = round(new_s, 2)
                nar["reel_end"] = round(new_s + duration, 2)
                reporter.log_info(
                    f"[INFO] Group {group_idx+1}: Auto-shifted {event_type} from "
                    f"[{old_start:.2f}s] -> [{nar['reel_start']:.2f}s-{nar['reel_end']:.2f}s] (gap verified)"
                )
            elif overlap_ratio > 0.5:  # Severe overlap and no gap found
                reporter.log_info(
                    f"[WARN] Group {group_idx+1}: Could not find gap for '{nar['text'][:30]}...' "
                    f"({overlap_ratio*100:.0f}% overlap). Consider removing this narration."
                )

    # 3. Ensure narrations do not overlap with each other
    group_narration_audio.sort(key=lambda x: x["reel_start"])
    for i in range(1, len(group_narration_audio)):
        prev_end = group_narration_audio[i-1]["reel_end"]
        curr_start = group_narration_audio[i]["reel_start"]
        curr_dur = group_narration_audio[i]["duration"]
        min_gap = 0.4  # Increased from 0.2s to 0.4s for cleaner separation
        if curr_start < prev_end + min_gap:
            new_start = prev_end + min_gap
            # Re-check speech overlap in case shifting caused a new collision
            new_start = find_speech_gap(curr_dur, new_start)
            
            if new_start + curr_dur <= target_duration:
                old_start = group_narration_audio[i]["reel_start"]
                group_narration_audio[i]["reel_start"] = round(new_start, 2)
                group_narration_audio[i]["reel_end"] = round(new_start + curr_dur, 2)
                reporter.log_info(
                    f"[INFO] Group {group_idx+1}: Shifted narration '{group_narration_audio[i]['text'][:30]}...' "
                    f"from [{old_start:.2f}s] -> [{group_narration_audio[i]['reel_start']:.2f}s-{group_narration_audio[i]['reel_end']:.2f}s] "
                    f"(min gap {min_gap}s from prior narration, speech-gap verified)"
                )

    # 4. Cap narrations at target_duration
    for nar in group_narration_audio:
        original_dur = nar.get("duration", 0.0)
        if nar["reel_end"] > target_duration:
            if nar["reel_start"] < target_duration - 0.5:
                nar["reel_end"] = target_duration
                nar["duration"] = nar["reel_end"] - nar["reel_start"]
            else:
                nar["reel_start"] = max(0.0, target_duration - 0.5)
                nar["reel_end"] = target_duration
                nar["duration"] = 0.5
                
            if nar["duration"] < original_dur:
                try:
                    import subprocess, os
                    from backend.config import FFMPEG_PATH
                    tmp_path = nar["path"] + ".tmp.wav"
                    subprocess.run([
                        FFMPEG_PATH, "-loglevel", "error", "-y",
                        "-i", nar["path"],
                        "-t", str(nar["duration"]),
                        "-c", "copy", tmp_path
                    ], check=True)
                    os.replace(tmp_path, nar["path"])
                    reporter.log_info(f"[INFO] Group {group_idx+1}: Trimmed narration audio file to {nar['duration']:.2f}s to fit target duration.")
                except Exception as e:
                    reporter.log_info(f"[WARN] Failed to trim narration audio: {e}")


class QueueManager:
    def __init__(self, loop):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.jobs: dict[str, VideoJob] = {}
        self.loop = loop

    def add_job(self, url: str) -> VideoJob:
        # Prune completed/terminal jobs only — NEVER drop a job that's still
        # in-flight (QUEUED/DOWNLOADING/.../COMPOSITING/EDITING). The previous
        # version kept only status == "DONE", which silently deleted any job
        # still processing the moment a new job was submitted — it kept
        # running in the background (the worker holds a direct object
        # reference) but vanished from get_jobs()/lookup-by-id, making it
        # look like the job or its output disappeared.
        terminal = (JobStatus.DONE, JobStatus.ERROR)
        self.jobs = {k: v for k, v in self.jobs.items() if v.status not in terminal}
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
        try:
            reel_plan: ReelPlan = await asyncio.wait_for(
                asyncio.to_thread(
                    select_reel_plan, job.transcript, job.title or "", video_description, analyzer_progress
                ),
                timeout=1200.0,  # 20 min hard ceiling; covers 2-model x 2-attempt x 240s budget.
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "ANALYZING exceeded 20-minute hard ceiling — LLM never responded "
                "in time even after retries. Check NVIDIA API status."
            )
        if getattr(reel_plan, "is_fallback", False):
            reporter.log_info("Using fallback reel plan (LLM was unavailable) — continuing with fallback.")
            job.stage_data = {
                "status": "fallback",
                "message": "LLM unavailable, using heuristic fallback plan.",
            }
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

            # Check whether the analyzer under-selected clips relative to its own duration target.
            # This is the upstream cause of the video-freeze bug (compositor pads with a frozen frame
            # when clips are far shorter than estimated_duration_seconds or MIN_OUTPUT_DURATION).
            actual_clip_dur = sum(c.source_end - c.source_start for c in group.source_clips)
            if group.estimated_duration_seconds > 0 and actual_clip_dur < 0.7 * group.estimated_duration_seconds:
                reporter.log_info(
                    f"[WARN] Group {group_idx+1}: analyzer under-selected clips — "
                    f"actual clip duration {actual_clip_dur:.1f}s is less than 70% of "
                    f"estimated {group.estimated_duration_seconds:.1f}s "
                    f"({actual_clip_dur / group.estimated_duration_seconds * 100:.0f}%). "
                    f"This will trigger freeze-pad capping in the compositor."
                )


            job.stage_index = 5
            raw_narration_events = list(group.narration_events)
            group_narration_events = []
            dropped = []
            for e in raw_narration_events:
                if e.event_type.strip().lower() in ("hook", "commentary"):
                    group_narration_events.append(e)
                else:
                    dropped.append(e)
            if dropped:
                reporter.log_info(
                    f"[WARN] Group {group_idx+1}: dropped {len(dropped)} narration event(s) "
                    f"with unrecognized event_type: {[e.event_type for e in dropped]!r} "
                    f"(only 'hook'/'commentary' are voiced)."
                )

            if not group_narration_events:
                # Analyzer produced zero usable narration for this group (LLM skipped
                # narration entirely, e.g. because it found no silent gap under the old
                # prompt rules). Never ship a fully-silent reel — inject a minimal hook
                # line from the group's own summary so there's at least SOME narration.
                fallback_text = (
                    group.reel_summary.short_description or group.reel_summary.title or ""
                ).strip()
                if fallback_text:
                    fallback_text = fallback_text[:80]
                    group.narration_events.append(
                        NarrationEvent(
                            event_type="hook",
                            reel_start=0.0,
                            reel_end=3.0,
                            text=fallback_text,
                            voice_id=None,
                        )
                    )
                    group_narration_events = [group.narration_events[-1]]
                    reporter.log_info(
                        f"[WARN] Group {group_idx+1}: analyzer returned NO usable narration "
                        f"events (raw count from LLM: {len(raw_narration_events)}). Injected "
                        f"a fallback hook line so this reel isn't fully silent: "
                        f"\"{fallback_text}\""
                    )
                else:
                    reporter.log_info(
                        f"[WARN] Group {group_idx+1}: analyzer returned NO usable narration "
                        f"events and no reel_summary text was available for a fallback. "
                        f"This group's final video will have NO narration audio."
                    )
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
                # Clean TTS text: remove all special characters that break captions/TTS
                import re as _re
                clean_text = event.text
                # Remove all banned characters
                for ch in ['/', '\\', '|', '*', '#', '_', '<', '>', '[', ']', '{', '}']:
                    clean_text = clean_text.replace(ch, ' ')
                # Normalize dashes to commas for natural TTS pauses
                clean_text = clean_text.replace('--', ',').replace('\u2014', ',')
                # Collapse multiple spaces and normalize whitespace
                clean_text = _re.sub(r'\s+', ' ', clean_text).strip()
                # Collapse repeated punctuation (e.g., "..." stays, but ",," becomes ",")
                clean_text = _re.sub(r'([,!?;:]){2,}', r'\1', clean_text)
                # Strip leading/trailing punctuation artifacts (but keep sentence-final punctuation)
                clean_text = clean_text.strip(' ,-;:')
                if not clean_text:
                    clean_text = event.text or "Notice this key moment."
                event.text = clean_text

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

            # --- Narration timing validation & auto-adjustment ---
            from backend.config import MAX_OUTPUT_DURATION, MIN_OUTPUT_DURATION
            total_clip_dur = sum((c.source_end - c.source_start) for c in group.source_clips)
            max_nar_end = max((nar.get("reel_end", 0) for nar in group_narration_audio), default=0.0)
            target_dur = max(total_clip_dur, max_nar_end, group.estimated_duration_seconds, float(MIN_OUTPUT_DURATION))
            target_dur = min(target_dur, float(MAX_OUTPUT_DURATION))

            validate_and_adjust_narration_timings(
                group_narration_audio=group_narration_audio,
                source_clips=group.source_clips,
                transcript=job.transcript,
                target_duration=target_dur,
                reporter=reporter,
                group_idx=group_idx,
            )

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
            cumulative_offset = 0.0
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
                    cumulative_offset,
                )
                group_clip_captions.append(str(clip_caption_path))
                cumulative_offset += (clip.source_end - clip.source_start)

            # Narration captions (per synthesized audio item, reel-relative)
            for i, nar in enumerate(group_narration_audio):
                job.stage_data = {
                    "status": "captioning",
                    "group_index": group_idx,
                    "sub": "narration",
                    "current": i + 1,
                    "total": len(group_narration_audio),
                }

                narr_caption_path = working_dir / f"group_{group_idx}_narr_caption_{i}.ass"

                def narr_caption_progress(msg: str, prog: float, idx=i):
                    reporter.progress_callback(f"Narration caption {idx+1}: {msg}", prog)

                await asyncio.to_thread(
                    generate_commentary_ass,
                    nar["text"],
                    nar["duration"],
                    str(narr_caption_path),
                    narr_caption_progress,
                    nar["reel_start"],
                )
                group_narration_captions.append({
                    "event_type": nar["event_type"],
                    "reel_start": nar["reel_start"],
                    "reel_end": nar["reel_end"],
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
                group.estimated_duration_seconds,
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
                job.id,
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

    def _final_edit_group(self, input_path: str, group: ReelGroup, working_dir, job_id: str) -> str:
        """Light final validation - ensure output file exists inside OUTPUTS_DIR and duration is capped."""
        from backend.config import OUTPUTS_DIR, MAX_OUTPUT_DURATION
        import subprocess
        import shutil

        title_slug = "".join(c for c in (group.reel_summary.title or "reel") if c.isalnum() or c in (' ', '_', '-')).strip()
        title_slug = title_slug.replace(' ', '_')[:40]
        output_filename = f"{job_id}_reel_{group.group_index}_{title_slug}.mp4"
        output_path = OUTPUTS_DIR / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        probe = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True
        )
        duration = float(probe.stdout.strip()) if probe.returncode == 0 and probe.stdout.strip() else 0.0

        if duration > float(MAX_OUTPUT_DURATION):
            subprocess.run(
                [FFMPEG_PATH, "-loglevel", "error", "-i", input_path, "-t", str(MAX_OUTPUT_DURATION), "-c", "copy", "-y", str(output_path)],
                check=True
            )
        else:
            shutil.copy2(input_path, output_path)

        return str(output_path)

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
