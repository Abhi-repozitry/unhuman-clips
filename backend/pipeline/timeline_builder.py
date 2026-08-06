"""Rich Timeline builder — merges Whisper, Silero VAD, and FFmpeg metrics.

Constructs a RichTimeline from multiple analysis sources, producing the single
source of truth consumed by the LLM and all downstream pipeline stages.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from typing import Any

from backend.config import VAD_THRESHOLD
from backend.ffmpeg_utils import get_ffmpeg
from backend.models import FFmpegMetrics, RichTimeline, RichTimelineSegment

__all__ = ["build_rich_timeline"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Silero VAD — speech regions for the source video
# ---------------------------------------------------------------------------

def _extract_audio_to_wav(video_path: str, output_wav: str) -> bool:
    """Extract audio from a video file to 16kHz mono WAV using FFmpeg.

    Silero VAD's read_audio uses torchaudio which cannot read video containers
    on Windows (sox not supported, soundfile only handles audio formats).
    FFmpeg handles all container formats reliably.
    """
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",                      # no video
        "-acodec", "pcm_s16le",    # 16-bit PCM
        "-ar", "16000",            # 16kHz sample rate
        "-ac", "1",                # mono
        str(output_wav),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"FFmpeg audio extraction failed (code {result.returncode}): {result.stderr}")
            return False
        if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
            logger.error("FFmpeg audio extraction produced empty or missing WAV file")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg audio extraction timed out after 60s")
        return False
    except Exception as e:
        logger.error(f"FFmpeg audio extraction failed: {type(e).__name__}: {e}")
        return False


def _run_vad_on_source(
    video_path: str,
    threshold: float = VAD_THRESHOLD,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
) -> list[dict[str, float]]:
    """Run Silero VAD on the source video to detect speech regions.

    Extracts audio to a temporary WAV file first, since torchaudio's soundfile
    backend cannot read video containers on Windows.

    Returns list of {"start": float, "end": float} for each detected speech segment.
    """
    try:
        import soundfile as sf
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as e:
        logger.error(
            f"Silero VAD imports failed: {type(e).__name__}: {e} — "
            "install silero-vad, torch, and soundfile. Returning empty speech regions."
        )
        return []

    logger.info(f"Silero VAD: extracting audio from {os.path.basename(video_path)}")

    # Extract audio to a temporary WAV file, then load with soundfile.
    # We do NOT use silero_vad.read_audio() because it wraps torchaudio.load()
    # which fails on torchaudio >=2.9 without torchcodec. Since FFmpeg already
    # produces a valid 16kHz mono WAV, soundfile is the most reliable loader.
    tmp_wav = None
    try:
        tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="vad_audio_")
        os.close(tmp_fd)

        if not _extract_audio_to_wav(video_path, tmp_wav):
            logger.error("Silero VAD: audio extraction failed, returning empty speech regions")
            return []

        wav_size = os.path.getsize(tmp_wav)
        logger.info(f"Silero VAD: extracted audio to WAV ({wav_size} bytes)")

        # Load WAV directly with soundfile — no torchaudio dependency
        sampling_rate = 16000
        wav_np, file_sr = sf.read(tmp_wav, dtype='float32')
        wav = torch.from_numpy(wav_np)
        if len(wav) == 0:
            logger.warning("Silero VAD: soundfile loaded empty audio")
            return []

        if file_sr != sampling_rate:
            logger.warning(f"Silero VAD: expected {sampling_rate}Hz, got {file_sr}Hz from FFmpeg extraction")

        logger.info(f"Silero VAD: audio loaded, {len(wav)} samples at {file_sr}Hz ({len(wav)/file_sr:.1f}s)")

        # Load the Silero VAD model (required in silero_vad v6+)
        model = load_silero_vad()
        logger.info("Silero VAD: model loaded")

        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=True,
        )

        total_speech = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
        logger.info(
            f"Silero VAD: detected {len(speech_timestamps)} speech regions, "
            f"{total_speech:.1f}s of speech"
        )
        return [{"start": ts["start"], "end": ts["end"]} for ts in speech_timestamps]
    except Exception as e:
        logger.exception(f"Silero VAD failed on source video: {type(e).__name__}: {e}")
        return []
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            with contextlib.suppress(OSError):
                os.unlink(tmp_wav)


# ---------------------------------------------------------------------------
# FFmpeg metrics — volume, black frame, freeze detection
# ---------------------------------------------------------------------------

def _compute_ffmpeg_metrics(
    video_path: str,
    start: float,
    end: float,
) -> FFmpegMetrics:
    """Compute FFmpeg-derived metrics for a time range in the source video.

    Metrics: average volume (dB), peak volume (dB), black frame detection,
    freeze detection.

    Uses 2 FFmpeg invocations:
      1. volumedetect (audio filter)
      2. blackdetect + freezedetect combined (video filters)
    """
    duration = end - start
    if duration <= 0:
        return FFmpegMetrics()

    ffmpeg = get_ffmpeg()

    metrics = FFmpegMetrics()

    # --- 1. Volume metrics via volumedetect ---
    try:
        cmd = [
            ffmpeg, "-loglevel", "info",
            "-ss", str(start), "-t", str(duration),
            "-i", str(video_path),
            "-af", "volumedetect",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = result.stderr

        mean_match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
        if mean_match:
            metrics.volume_db = float(mean_match.group(1))

        peak_match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr)
        if peak_match:
            metrics.peak_db = float(peak_match.group(1))
    except Exception as e:
        logger.debug(f"Volume detection failed for [{start:.1f}-{end:.1f}]: {e}")

    # --- 2. Black frame + freeze detection (combined filter chain) ---
    try:
        cmd = [
            ffmpeg, "-loglevel", "info",
            "-ss", str(start), "-t", str(min(duration, 5.0)),
            "-i", str(video_path),
            "-vf", "blackdetect=d=0.5:pix_th=0.10,freezedetect=n=-60dB:d=1.0",
            "-an", "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if "black_start" in result.stderr:
            metrics.black_frame = True
        if "freeze_start" in result.stderr:
            metrics.freeze_detected = True
    except Exception as e:
        logger.debug(f"Black/freeze detection failed for [{start:.1f}-{end:.1f}]: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Speech energy — proportion of VAD speech within each segment
# ---------------------------------------------------------------------------

def _compute_speech_energy(
    segment_start: float,
    segment_end: float,
    speech_regions: list[dict[str, float]],
) -> float:
    """Compute speech energy (0.0-1.0) as proportion of segment covered by speech."""
    segment_duration = segment_end - segment_start
    if segment_duration <= 0:
        return 0.0

    speech_duration = 0.0
    for region in speech_regions:
        overlap_start = max(segment_start, region["start"])
        overlap_end = min(segment_end, region["end"])
        if overlap_start < overlap_end:
            speech_duration += overlap_end - overlap_start

    return min(1.0, speech_duration / segment_duration)


def _check_silence_before(
    segment_start: float,
    speech_regions: list[dict[str, float]],
    min_silence: float = 0.3,
) -> bool:
    """Check if there is a silence gap of at least min_silence before this segment.

    Only checks the immediately preceding VAD region, not arbitrary ones.
    """
    if not speech_regions:
        return False
    latest_end = 0.0
    found_preceding = False
    for region in speech_regions:
        if region["end"] <= segment_start:
            latest_end = max(latest_end, region["end"])
            found_preceding = True
    if not found_preceding:
        return False
    return (segment_start - latest_end) >= min_silence


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_rich_timeline(
    transcript: list[dict],
    video_path: str,
    progress_cb: Callable[[str, float], None] | None = None,
    reporter: Any = None,
) -> RichTimeline:
    """Construct a RichTimeline by merging Whisper, VAD, and FFmpeg metrics.

    This is the SINGLE source of truth for every downstream stage.
    No downstream component should directly consume raw Whisper, VAD, or FFmpeg output.

    Args:
        transcript: Whisper transcript segments with start/end/text/words keys.
        video_path: Path to the source video file.
        progress_cb: Optional progress callback.
        reporter: Optional ProgressReporter.

    Returns:
        RichTimeline with merged segments.
    """
    if not transcript:
        logger.warning("build_rich_timeline: empty transcript, returning empty timeline")
        return RichTimeline()

    logger.info(
        f"Building Rich Timeline: {len(transcript)} Whisper segments, "
        f"source={os.path.basename(video_path)}"
    )
    if progress_cb:
        progress_cb("Building Rich Timeline: running VAD on source video...", 10)

    # 1. Run Silero VAD on source video
    speech_regions = _run_vad_on_source(video_path)
    total_speech_vad = sum(r["end"] - r["start"] for r in speech_regions)
    logger.info(
        f"Rich Timeline source: Silero VAD -> {len(speech_regions)} speech regions, "
        f"{total_speech_vad:.1f}s speech"
    )
    if reporter:
        reporter.log_info(f"Rich Timeline: VAD detected {len(speech_regions)} speech regions, {total_speech_vad:.1f}s speech")

    if progress_cb:
        progress_cb("Building Rich Timeline: computing FFmpeg metrics...", 30)

    # 2. Compute FFmpeg metrics per segment
    segment_metrics: dict[int, FFmpegMetrics] = {}
    for i, seg in enumerate(transcript):
        metrics = _compute_ffmpeg_metrics(video_path, seg["start"], seg["end"])
        segment_metrics[i] = metrics
        if progress_cb and i % 10 == 0:
            pct = 30 + (i / len(transcript)) * 30
            progress_cb(f"Building Rich Timeline: FFmpeg metrics {i+1}/{len(transcript)}...", pct)

    metrics_with_data = sum(1 for m in segment_metrics.values() if m.volume_db is not None)
    logger.info(f"Rich Timeline source: FFmpeg -> {metrics_with_data}/{len(segment_metrics)} segments with volume data")

    if progress_cb:
        progress_cb("Building Rich Timeline: merging all sources...", 85)

    # 3. Merge into RichTimelineSegment list
    segments = []
    total_speech = 0.0
    total_silence = 0.0
    segments_with_energy = 0

    for i, seg in enumerate(transcript):
        start = seg["start"]
        end = seg["end"]
        duration = end - start

        # Speech energy and silence detection
        energy = _compute_speech_energy(start, end, speech_regions)
        silence_before = _check_silence_before(start, speech_regions)

        if energy > 0.0:
            segments_with_energy += 1

        # Words from Whisper
        words = seg.get("words", [])

        # Speech confidence (use VAD presence as proxy)
        speech_confidence = min(1.0, energy * 1.2) if energy > 0 else 0.0

        # Speech regions overlapping this segment
        overlapping_regions = [
            r for r in speech_regions
            if r["end"] > start and r["start"] < end
        ]

        # Accumulate totals
        if energy > 0.5:
            total_speech += duration
        else:
            total_silence += duration

        metrics = segment_metrics.get(i, FFmpegMetrics())

        # Richer feature computation
        speech_text = seg.get("text", "").strip()
        word_count = len([w for w in speech_text.split() if w.strip()]) if speech_text else 0
        word_density = word_count / duration if duration > 0 else 0.0
        has_question = "?" in speech_text
        has_exclamation = "!" in speech_text
        has_emphasis = bool(re.search(r"\b[A-Z]{3,}\b", speech_text)) if speech_text else False

        segment = RichTimelineSegment(
            segment_id=i,
            start=round(start, 3),
            end=round(end, 3),
            duration=round(duration, 3),
            speech=seg.get("text", "").strip(),
            words=words,
            speech_confidence=round(speech_confidence, 3),
            speech_energy=round(energy, 3),
            speech_regions=overlapping_regions,
            silence_before=silence_before,
            metrics=metrics,
            word_density=round(word_density, 3),
            has_question=has_question,
            has_exclamation=has_exclamation,
            has_emphasis=has_emphasis,
            speaker_id=str(seg.get("speaker_id") or seg.get("speaker") or "").strip() or None,
        )
        segments.append(segment)

    source_duration = transcript[-1]["end"] if transcript else 0.0

    timeline = RichTimeline(
        segments=segments,
        source_duration=round(source_duration, 3),
        total_speech_duration=round(total_speech, 3),
        total_silence_duration=round(total_silence, 3),
        speech_region_count=len(speech_regions),
    )

    logger.info(
        f"Rich Timeline MERGED: "
        f"Whisper={len(segments)} segments | "
        f"Silero={len(speech_regions)} regions ({total_speech_vad:.1f}s) | "
        f"FFmpeg={metrics_with_data} metrics | "
        f"Segments with energy={segments_with_energy} | "
        f"Speech={total_speech:.1f}s, Silence={total_silence:.1f}s"
    )
    if reporter:
        reporter.log_info(
            f"Rich Timeline built: {len(segments)} segments, "
            f"VAD={len(speech_regions)} regions ({total_speech_vad:.1f}s speech)"
        )

    if progress_cb:
        progress_cb("Rich Timeline complete", 100)

    return timeline
