"""Rich Timeline builder — merges Whisper, Silero VAD, OCR, and FFmpeg metrics.

Constructs a RichTimeline from multiple analysis sources, producing the single
source of truth consumed by the LLM and all downstream pipeline stages.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Callable

from backend.config import VAD_THRESHOLD
from backend.ffmpeg_utils import get_ffmpeg, get_ffprobe
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
        import torch
        import soundfile as sf
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError:
        logger.error(
            "silero-vad/torch/soundfile not importable — "
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
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# FFmpeg metrics — volume, brightness, black frame, freeze detection
# ---------------------------------------------------------------------------

def _compute_ffmpeg_metrics(
    video_path: str,
    start: float,
    end: float,
) -> FFmpegMetrics:
    """Compute FFmpeg-derived metrics for a time range in the source video.

    Metrics: average volume (dB), peak volume (dB), brightness estimate,
    black frame detection, freeze detection.
    """
    duration = end - start
    if duration <= 0:
        return FFmpegMetrics()

    ffmpeg = get_ffmpeg()
    ffprobe = get_ffprobe()

    metrics = FFmpegMetrics()

    # --- Volume metrics via volumedetect ---
    try:
        cmd = [
            ffmpeg, "-loglevel", "error",
            "-ss", str(start), "-t", str(duration),
            "-i", str(video_path),
            "-af", "volumedetect",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = result.stderr

        import re
        mean_match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
        if mean_match:
            metrics.volume_db = float(mean_match.group(1))

        peak_match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr)
        if peak_match:
            metrics.peak_db = float(peak_match.group(1))
    except Exception as e:
        logger.debug(f"Volume detection failed for [{start:.1f}-{end:.1f}]: {e}")

    # --- Brightness via signalstats (sample first frame) ---
    try:
        cmd = [
            ffmpeg, "-loglevel", "error",
            "-ss", str(start + duration / 2),
            "-i", str(video_path),
            "-vf", "signalstats=stat=tout+vrep+brng",
            "-frames:v", "1",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        stderr = result.stderr

        import re
        yav_match = re.search(r"YAVG:(\d+\.?\d*)", stderr)
        if yav_match:
            # YAVG is 0-255, normalize to 0.0-1.0
            metrics.brightness = float(yav_match.group(1)) / 255.0
    except Exception as e:
        logger.debug(f"Brightness detection failed for [{start:.1f}-{end:.1f}]: {e}")

    # --- Black frame detection ---
    try:
        cmd = [
            ffmpeg, "-loglevel", "error",
            "-ss", str(start), "-t", str(min(duration, 5.0)),
            "-i", str(video_path),
            "-vf", "blackdetect=d=0.5:pix_th=0.10",
            "-an", "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if "black_start" in result.stderr:
            metrics.black_frame = True
    except Exception as e:
        logger.debug(f"Black frame detection failed for [{start:.1f}-{end:.1f}]: {e}")

    # --- Freeze detection ---
    try:
        cmd = [
            ffmpeg, "-loglevel", "error",
            "-ss", str(start), "-t", str(min(duration, 10.0)),
            "-i", str(video_path),
            "-vf", "freezedetect=n=-60dB:d=1.0",
            "-an", "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if "freeze_start" in result.stderr:
            metrics.freeze_detected = True
    except Exception as e:
        logger.debug(f"Freeze detection failed for [{start:.1f}-{end:.1f}]: {e}")

    return metrics


# ---------------------------------------------------------------------------
# OCR — text detection on sampled frames
# ---------------------------------------------------------------------------

def _run_ocr_on_source(
    video_path: str,
    segments: list[dict],
    sample_interval: float = 5.0,
    max_frames: int = 30,
) -> dict[int, dict]:
    """Sample frames from the source video and run OCR.

    Returns dict mapping segment_id -> {"texts": [...], "confidence": float}.
    """
    try:
        from backend.pipeline.ocr import _try_ocr_engine, _extract_frame
    except ImportError as e:
        logger.error(f"OCR module not importable: {e} — skipping OCR analysis")
        return {}

    # Sample timestamps across the video
    if not segments:
        logger.info("OCR: no segments to sample from")
        return {}

    total_duration = segments[-1]["end"] if segments else 0.0
    sample_times = []
    t = 0.0
    while t < total_duration and len(sample_times) < max_frames:
        sample_times.append(t)
        t += sample_interval

    logger.info(f"OCR: will analyze {len(sample_times)} sampled frames from {total_duration:.1f}s video")

    ocr_results: dict[int, dict] = {}
    frames_extracted = 0
    frames_with_text = 0
    extraction_failures = 0

    for sample_t in sample_times:
        # Find which segment this timestamp belongs to
        seg_id = -1
        for seg in segments:
            if seg["start"] <= sample_t < seg["end"]:
                seg_id = seg.get("segment_id", segments.index(seg))
                break

        if seg_id < 0:
            continue

        # Extract frame and run OCR
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="ocr_frame_")
        os.close(tmp_fd)

        try:
            if _extract_frame(video_path, sample_t, tmp_path):
                frames_extracted += 1
                results = _try_ocr_engine(tmp_path)
                if results:
                    texts = [r["text"] for r in results if r.get("text")]
                    conf = max((r["confidence"] for r in results), default=0.0)
                    if texts:
                        frames_with_text += 1
                    if seg_id not in ocr_results:
                        ocr_results[seg_id] = {"texts": [], "confidence": 0.0}
                    ocr_results[seg_id]["texts"].extend(texts)
                    ocr_results[seg_id]["confidence"] = max(
                        ocr_results[seg_id]["confidence"], conf
                    )
            else:
                extraction_failures += 1
        except Exception as e:
            logger.warning(f"OCR failed at {sample_t:.1f}s: {type(e).__name__}: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    logger.info(
        f"OCR complete: {frames_extracted} frames extracted, "
        f"{frames_with_text} contained text, "
        f"{len(ocr_results)} segments with OCR data, "
        f"{extraction_failures} frame extraction failures"
    )
    return ocr_results


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
    """Check if there is a silence gap of at least min_silence before this segment."""
    for region in speech_regions:
        if region["end"] <= segment_start and (segment_start - region["end"]) >= min_silence:
            return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_rich_timeline(
    transcript: list[dict],
    video_path: str,
    progress_cb: Callable[[str, float], None] | None = None,
    reporter: Any = None,
) -> RichTimeline:
    """Construct a RichTimeline by merging Whisper, VAD, OCR, and FFmpeg metrics.

    This is the SINGLE source of truth for every downstream stage.
    No downstream component should directly consume raw Whisper, OCR, VAD, or FFmpeg output.

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
        progress_cb("Building Rich Timeline: running OCR on sampled frames...", 65)

    # 3. Run OCR on sampled frames
    ocr_data = _run_ocr_on_source(video_path, transcript)
    logger.info(f"Rich Timeline source: OCR -> {len(ocr_data)} segments with text")
    if reporter:
        reporter.log_info(f"Rich Timeline: OCR found text in {len(ocr_data)} segments")

    if progress_cb:
        progress_cb("Building Rich Timeline: merging all sources...", 85)

    # 4. Merge into RichTimelineSegment list
    segments = []
    total_speech = 0.0
    total_silence = 0.0
    ocr_count = 0
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

        # OCR data
        ocr_texts = []
        ocr_confidence = 0.0
        if i in ocr_data:
            ocr_texts = ocr_data[i]["texts"]
            ocr_confidence = ocr_data[i]["confidence"]
            ocr_count += len(ocr_texts)

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
            ocr=ocr_texts,
            ocr_confidence=round(ocr_confidence, 3),
            metrics=metrics,
        )
        segments.append(segment)

    source_duration = transcript[-1]["end"] if transcript else 0.0

    timeline = RichTimeline(
        segments=segments,
        source_duration=round(source_duration, 3),
        total_speech_duration=round(total_speech, 3),
        total_silence_duration=round(total_silence, 3),
        speech_region_count=len(speech_regions),
        ocr_region_count=ocr_count,
    )

    logger.info(
        f"Rich Timeline MERGED: "
        f"Whisper={len(segments)} segments | "
        f"Silero={len(speech_regions)} regions ({total_speech_vad:.1f}s) | "
        f"OCR={ocr_count} texts | "
        f"FFmpeg={metrics_with_data} metrics | "
        f"Segments with energy={segments_with_energy} | "
        f"Speech={total_speech:.1f}s, Silence={total_silence:.1f}s"
    )
    if reporter:
        reporter.log_info(
            f"Rich Timeline built: {len(segments)} segments, "
            f"VAD={len(speech_regions)} regions ({total_speech_vad:.1f}s speech), "
            f"OCR={ocr_count} texts"
        )

    if progress_cb:
        progress_cb("Rich Timeline complete", 100)

    return timeline
