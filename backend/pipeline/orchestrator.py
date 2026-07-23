"""Group pipeline orchestrator — runs the full clip -> TTS -> caption -> compose pipeline for one group."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from backend.config import (
    HOOK_SECONDS, MAX_GROUP_RETRIES, MAX_OUTPUT_DURATION,
    MIN_OUTPUT_DURATION, get_job_working_dir,
)
from backend.models import JobStatus, NarrationEvent, ReelGroup, VideoJob
from backend.output_manager import OutputManager
from backend.pipeline.captioner import generate_clip_ass, generate_commentary_ass
from backend.pipeline.checkpoint import PipelineCheckpoint
from backend.pipeline.clipper import cut_group_clips
from backend.pipeline.compositor import compose_group
from backend.pipeline.narration_validator import validate_and_adjust_narration_timings
from backend.pipeline.tts import synthesize_commentary

__all__ = ["GroupOrchestrator"]

logger = logging.getLogger(__name__)


class GroupOrchestrator:
    """Runs the full pipeline for a single output group.

    Encapsulates the clip → TTS → caption → compose → edit stages
    that were previously inlined in QueueManager._process_job.
    """

    def __init__(self, job: VideoJob, enqueue_broadcast: Callable):
        self.job = job
        self.enqueue_broadcast = enqueue_broadcast
        self.output_manager = OutputManager()
        self.ckpt = PipelineCheckpoint(get_job_working_dir(job.id))

    # ------------------------------------------------------------------
    # Stage: CLIPPING
    # ------------------------------------------------------------------
    async def run_clipping(
        self, group_idx: int, group: ReelGroup, reporter: Any, source_path: str
    ) -> list[str]:
        """Cut source clips for this group.

        Args:
            group_idx: Index of this group in the reel plan.
            group: ReelGroup with source_clips to cut.
            reporter: ProgressReporter for status updates.
            source_path: Path to the source video file.

        Returns:
            List of clip file paths.
        """
        ckpt_key = f"group_{group_idx}_clips"
        checkpoint = self.ckpt.load_stage(ckpt_key)

        if checkpoint and "clip_paths" in checkpoint:
            reporter.log_info(f"Group {group_idx+1}: Resuming from checkpoint (clips already cut)")
            return checkpoint["clip_paths"]

        self.job.stage_index = 5
        self.job.stage_data = {
            "status": "cutting",
            "group_index": group_idx,
            "total_groups": self.job.num_output_groups,
            "total_clips": len(group.source_clips),
            "current_clip": 0,
        }
        reporter.update_stage(JobStatus.CLIPPING, f"Group {group_idx+1}/{self.job.num_output_groups}: Cutting clips...", 0, 5)

        def clipper_progress(msg: str, prog: float):
            self.job.stage_data = {
                "status": "cutting",
                "group_index": group_idx,
                "total_groups": self.job.num_output_groups,
                "total_clips": len(group.source_clips),
                "current_clip": int(prog / 100 * len(group.source_clips)) if prog > 0 else 0,
                "message": msg,
                "progress": prog,
            }
            reporter.progress_callback(msg, prog)

        group_clip_paths = await asyncio.to_thread(
            cut_group_clips, source_path, [c.model_dump() for c in group.source_clips],
            self.job.id, group_idx, clipper_progress, reporter
        )

        self.job.stage_data = {"status": "done", "group_index": group_idx, "clips_cut": len(group_clip_paths)}
        reporter.log_info(f"Group {group_idx+1}: Cut {len(group_clip_paths)} clips")

        # Warn if analyzer under-selected clips
        actual_clip_dur = sum(c.source_end - c.source_start for c in group.source_clips)
        if group.estimated_duration_seconds > 0 and actual_clip_dur < 0.7 * group.estimated_duration_seconds:
            reporter.log_info(
                f"[WARN] Group {group_idx+1}: analyzer under-selected clips — "
                f"actual clip duration {actual_clip_dur:.1f}s is less than 70% of "
                f"estimated {group.estimated_duration_seconds:.1f}s "
                f"({actual_clip_dur / group.estimated_duration_seconds * 100:.0f}%). "
                f"This will trigger freeze-pad capping in the compositor."
            )

        self.ckpt.save_stage(ckpt_key, {"clip_paths": group_clip_paths})
        return group_clip_paths

    # ------------------------------------------------------------------
    # Stage: TTS (VOICING)
    # ------------------------------------------------------------------
    async def run_tts(
        self, group_idx: int, group: ReelGroup, reporter: Any, working_dir: Path
    ) -> tuple[list[dict], list[NarrationEvent]]:
        """Generate TTS audio for narration events.

        Args:
            group_idx: Index of this group in the reel plan.
            group: ReelGroup with narration_events.
            reporter: ProgressReporter for status updates.
            working_dir: Working directory for temp files.

        Returns:
            Tuple of (narration_audio list, narration_events list).
        """
        ckpt_key = f"group_{group_idx}_tts"
        checkpoint = self.ckpt.load_stage(ckpt_key)

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
        self.job.stage_index = 6
        self.job.stage_data = {
            "status": "voicing",
            "group_index": group_idx,
            "total_groups": self.job.num_output_groups,
            "total": total_narration,
            "current": 0,
        }
        reporter.update_stage(JobStatus.VOICING, f"Group {group_idx+1}/{self.job.num_output_groups}: Generating TTS...", 0, 6)

        if checkpoint and "narration_audio" in checkpoint:
            reporter.log_info(f"Group {group_idx+1}: Resuming from checkpoint (TTS already generated)")
            return checkpoint["narration_audio"], group_narration_events

        group_narration_audio = []
        for i, event in enumerate(group_narration_events):
            reporter.update_sub_stage(
                f"Group {group_idx+1}: TTS for {event.event_type} ({i+1}/{total_narration})",
                (i / total_narration) * 100 if total_narration > 0 else 100,
            )
            clean_text = event.text
            if not clean_text:
                clean_text = "Notice this key moment."
            event.text = clean_text

            self.job.stage_data = {
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

        self.job.narration_audio = group_narration_audio
        self.job.stage_data = {"status": "done", "group_index": group_idx, "files_generated": len(group_narration_audio)}
        reporter.log_info(f"Group {group_idx+1}: Generated {len(group_narration_audio)} narration audio files")

        # Narration timing validation
        total_clip_dur = sum((c.source_end - c.source_start) for c in group.source_clips)
        max_nar_end = max((nar.get("reel_end", 0) for nar in group_narration_audio), default=0.0)
        target_dur = max(total_clip_dur, max_nar_end, group.estimated_duration_seconds, float(MIN_OUTPUT_DURATION))
        target_dur = min(target_dur, float(MAX_OUTPUT_DURATION))

        await asyncio.to_thread(
            validate_and_adjust_narration_timings,
            group_narration_audio=group_narration_audio,
            source_clips=group.source_clips,
            transcript=self.job.transcript,
            target_duration=target_dur,
            reporter=reporter,
            group_idx=group_idx,
        )

        self.ckpt.save_stage(ckpt_key, {
            "narration_audio": group_narration_audio,
            "narration_events": [e.model_dump() for e in group_narration_events],
        })
        return group_narration_audio, group_narration_events

    # ------------------------------------------------------------------
    # Stage: CAPTIONING
    # ------------------------------------------------------------------
    async def run_captioning(
        self, group_idx: int, group: ReelGroup, reporter: Any, working_dir: Path,
        group_narration_audio: list[dict]
    ) -> tuple[list[str], list[dict]]:
        """Generate ASS captions for clips and narration.

        Args:
            group_idx: Index of this group in the reel plan.
            group: ReelGroup with source_clips.
            reporter: ProgressReporter for status updates.
            working_dir: Working directory for caption files.
            group_narration_audio: TTS audio metadata list.

        Returns:
            Tuple of (clip_captions paths, narration_captions dicts).
        """
        ckpt_key = f"group_{group_idx}_captions"
        checkpoint = self.ckpt.load_stage(ckpt_key)

        self.job.stage_index = 7
        self.job.stage_data = {
            "status": "captioning",
            "group_index": group_idx,
            "total_groups": self.job.num_output_groups,
            "total_clips": len(group.source_clips),
            "current": 0,
        }
        reporter.update_stage(JobStatus.CAPTIONING, f"Group {group_idx+1}/{self.job.num_output_groups}: Generating captions...", 0, 7)

        if checkpoint and "clip_captions" in checkpoint:
            reporter.log_info(f"Group {group_idx+1}: Resuming from checkpoint (captions already generated)")
            return checkpoint["clip_captions"], checkpoint["narration_captions"]

        group_clip_captions = []
        group_narration_captions = []

        # Clip captions
        cumulative_offset = 0.0
        for i, clip in enumerate(group.source_clips):
            reporter.update_clip_progress(i, "captioning", (i / len(group.source_clips)) * 100)
            self.job.stage_data = {
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
                self.job.transcript,
                clip.source_start,
                clip.source_end,
                str(clip_caption_path),
                caption_progress,
                cumulative_offset,
            )
            group_clip_captions.append(str(clip_caption_path))
            cumulative_offset += (clip.source_end - clip.source_start)

        # Narration captions
        for i, nar in enumerate(group_narration_audio):
            self.job.stage_data = {
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

        self.job.caption_paths = group_clip_captions
        self.job.stage_data = {"status": "done", "group_index": group_idx, "clip_captions": len(group_clip_captions), "narration_captions": len(group_narration_captions)}
        reporter.log_info(f"Group {group_idx+1}: Generated {len(group_clip_captions)} clip + {len(group_narration_captions)} narration captions")

        self.ckpt.save_stage(ckpt_key, {
            "clip_captions": group_clip_captions,
            "narration_captions": group_narration_captions,
        })
        return group_clip_captions, group_narration_captions

    # ------------------------------------------------------------------
    # Stage: COMPOSITING
    # ------------------------------------------------------------------
    async def run_compositing(
        self, group_idx: int, group: ReelGroup, reporter: Any, working_dir: Path,
        group_clip_paths: list[str], group_narration_audio: list[dict],
        group_clip_captions: list[str], group_narration_captions: list[dict],
        source_path: str
    ) -> str:
        """Composite the final video for this group.

        Args:
            group_idx: Index of this group in the reel plan.
            group: ReelGroup with configuration.
            reporter: ProgressReporter for status updates.
            working_dir: Working directory for intermediate files.
            group_clip_paths: Paths to pre-cut clip files.
            group_narration_audio: TTS audio metadata list.
            group_clip_captions: Paths to clip caption ASS files.
            group_narration_captions: Narration caption metadata list.
            source_path: Path to the source video file.

        Returns:
            Path to the composited output video.
        """
        ckpt_key = f"group_{group_idx}_composite"
        checkpoint = self.ckpt.load_stage(ckpt_key)

        self.job.stage_index = 8
        self.job.stage_data = {
            "status": "compositing",
            "group_index": group_idx,
            "total_groups": self.job.num_output_groups,
            "message": "Building continuous video with overlay + ducking...",
        }
        reporter.update_stage(JobStatus.COMPOSITING, f"Group {group_idx+1}/{self.job.num_output_groups}: Compositing...", 0, 8)

        if checkpoint and "output_path" in checkpoint:
            reporter.log_info(f"Group {group_idx+1}: Resuming from checkpoint (composite already done)")
            return checkpoint["output_path"]

        def compositor_progress(msg: str, prog: float):
            self.job.stage_data = {
                "status": "compositing",
                "group_index": group_idx,
                "message": msg,
                "progress": prog,
            }
            reporter.progress_callback(msg, prog)

        compose_result = await asyncio.to_thread(
            compose_group,
            self.job.id,
            group_idx,
            group_clip_paths,
            [c.model_dump() for c in group.source_clips],
            group_narration_audio,
            group_clip_captions,
            [c["path"] for c in group_narration_captions],
            source_path,
            working_dir,
            group.estimated_duration_seconds,
            compositor_progress,
        )

        if isinstance(compose_result, dict):
            group_output_path = compose_result["output_path"]
            vad_stats = compose_result.get("vad_stats", {"active": False})
            vad_analysis = compose_result.get("vad_analysis", [])
        else:
            group_output_path = compose_result
            vad_stats = {"active": False}
            vad_analysis = []

        # Store VAD stats in stage_data for frontend display
        existing_stage_data = self.job.stage_data if isinstance(self.job.stage_data, dict) else {}
        self.job.stage_data = {
            **existing_stage_data,
            "vad_stats": vad_stats,
            "vad_analysis": vad_analysis,
        }
        reporter.set_stage_data_key("vad_stats", vad_stats)
        reporter.set_stage_data_key("vad_analysis", vad_analysis)

        self.ckpt.save_stage(ckpt_key, {
            "output_path": group_output_path,
            "vad_stats": vad_stats,
        })
        return group_output_path

    # ------------------------------------------------------------------
    # Stage: EDITING (final output)
    # ------------------------------------------------------------------
    async def run_editing(
        self, group_idx: int, group: ReelGroup, reporter: Any, working_dir: Path,
        group_output_path: str
    ) -> tuple[str, float]:
        """Run final edit and validation on the composited video.

        Args:
            group_idx: Index of this group in the reel plan.
            group: ReelGroup with metadata.
            reporter: ProgressReporter for status updates.
            working_dir: Working directory.
            group_output_path: Path to the composited video.

        Returns:
            Tuple of (final_path, duration_seconds).
        """
        self.job.stage_index = 9
        self.job.stage_data = {
            "status": "editing",
            "group_index": group_idx,
            "total_groups": self.job.num_output_groups,
        }
        reporter.update_stage(JobStatus.EDITING, f"Group {group_idx+1}/{self.job.num_output_groups}: Final edit...", 0, 9)

        final_path = await asyncio.to_thread(
            self.output_manager.final_edit_group,
            group_output_path, group, working_dir, self.job.id,
        )
        duration = await self.output_manager.probe_duration(final_path)

        self.job.stage_data = {
            "status": "editing",
            "group_index": group_idx,
            "duration": duration,
        }
        return final_path, duration

    # ------------------------------------------------------------------
    # Run all stages for one group (with retry)
    # ------------------------------------------------------------------
    def _cleanup_group_artifacts(self, group_idx: int) -> None:
        """Remove partial artifacts for a failed group so a retry starts clean.

        Clears checkpoint files for this group and removes partial output files
        (clips, TTS audio, captions, composite) that may be corrupted.

        Args:
            group_idx: Index of the group whose artifacts to remove.
        """
        working_dir = get_job_working_dir(self.job.id)
        # Clear all checkpoints for this group
        for stage_suffix in ("clips", "tts", "captions", "composite"):
            self.ckpt.clear_stage(f"group_{group_idx}_{stage_suffix}")
        # Remove partial clip files
        for path in working_dir.glob(f"group_{group_idx}_clip_*.mp4"):
            try:
                path.unlink()
            except OSError:
                pass
        # Remove partial TTS audio
        for path in working_dir.glob(f"group_{group_idx}_narration_*.wav"):
            try:
                path.unlink()
            except OSError:
                pass
        # Remove partial caption files
        for path in working_dir.glob(f"group_{group_idx}_*_caption_*.ass"):
            try:
                path.unlink()
            except OSError:
                pass
        logger.info("Cleaned up partial artifacts for group %d", group_idx)

    async def run_group(
        self, group_idx: int, group: ReelGroup, reporter: Any, source_path: str
    ) -> str:
        """Run the full pipeline for one group with automatic retry.

        On failure, partial artifacts are cleaned up and the pipeline retries
        from the last successful checkpoint. Up to MAX_GROUP_RETRIES attempts
        are made.

        Args:
            group_idx: Index of this group in the reel plan.
            group: The ReelGroup to process.
            reporter: ProgressReporter for status updates.
            source_path: Path to the source video file.

        Returns:
            Path to the final output video.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        from backend.config import MAX_GROUP_RETRIES

        self.job.current_group_index = group_idx
        self.job.outputs[group_idx].status = "processing"

        last_error = None
        for attempt in range(1, MAX_GROUP_RETRIES + 1):
            try:
                if attempt > 1:
                    reporter.log_info(
                        f"Group {group_idx+1}: Retry attempt {attempt}/{MAX_GROUP_RETRIES}..."
                    )

                working_dir = get_job_working_dir(self.job.id)
                reporter.log_info(f"Group {group_idx+1}: Starting pipeline (attempt {attempt})...")

                group_clip_paths = await self.run_clipping(group_idx, group, reporter, source_path)
                group_narration_audio, _ = await self.run_tts(group_idx, group, reporter, working_dir)
                group_clip_captions, group_narration_captions = await self.run_captioning(
                    group_idx, group, reporter, working_dir, group_narration_audio
                )
                group_output_path = await self.run_compositing(
                    group_idx, group, reporter, working_dir,
                    group_clip_paths, group_narration_audio,
                    group_clip_captions, group_narration_captions, source_path
                )
                final_path, duration = await self.run_editing(
                    group_idx, group, reporter, working_dir, group_output_path
                )

                self.job.outputs[group_idx].output_path = final_path
                self.job.outputs[group_idx].output_url = f"/outputs/{Path(final_path).name}"
                self.job.outputs[group_idx].duration_seconds = duration
                self.job.outputs[group_idx].status = "done"
                reporter.log_info(f"Group {group_idx+1} complete: {final_path} ({duration:.1f}s)")
                return final_path

            except Exception as e:
                last_error = e
                logger.warning(
                    "Group %d failed on attempt %d/%d: %s",
                    group_idx + 1, attempt, MAX_GROUP_RETRIES, e,
                )
                reporter.log_info(
                    f"[WARN] Group {group_idx+1}: Attempt {attempt}/{MAX_GROUP_RETRIES} failed: {e}"
                )

                if attempt < MAX_GROUP_RETRIES:
                    reporter.log_info(f"Group {group_idx+1}: Cleaning up and retrying...")
                    self._cleanup_group_artifacts(group_idx)
                else:
                    self.job.outputs[group_idx].status = "error"
                    reporter.log_info(
                        f"[ERROR] Group {group_idx+1}: All {MAX_GROUP_RETRIES} attempts failed."
                    )

        raise RuntimeError(
            f"Group {group_idx+1} failed after {MAX_GROUP_RETRIES} attempts. "
            f"Last error: {last_error}"
        ) from last_error
