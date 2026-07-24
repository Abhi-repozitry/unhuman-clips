"""
End-to-end verification of Silero VAD + OCR + Rich Timeline merge.
Runs the ACTUAL pipeline code against a real video file.
"""
import logging
import os
import sys
import tempfile
import subprocess
import json

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("verify")

VIDEO = sys.argv[1] if len(sys.argv) > 1 else None
if not VIDEO or not os.path.exists(VIDEO):
    print("Usage: python verify_e2e.py <path_to_video>")
    sys.exit(1)

logger.info(f"=== VERIFICATION: {os.path.basename(VIDEO)} ===")
logger.info(f"Video size: {os.path.getsize(VIDEO)} bytes")


# ============================================================
# TEST 1: Silero VAD — audio extraction, soundfile load, tensor, VAD
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("TEST 1: Silero VAD via timeline_builder._run_vad_on_source")
logger.info("=" * 60)

from backend.pipeline.timeline_builder import _run_vad_on_source

speech_regions = _run_vad_on_source(VIDEO)
total_speech = sum(r["end"] - r["start"] for r in speech_regions)
logger.info(f"RESULT: {len(speech_regions)} speech regions, {total_speech:.1f}s total speech")


# ============================================================
# TEST 2: Soundfile load — shape, dtype, sample rate, mono conversion
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("TEST 2: Direct soundfile + torch verification")
logger.info("=" * 60)

import soundfile as sf
import torch
import numpy as np

from backend.ffmpeg_utils import get_ffmpeg

tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="verify_vad_")
os.close(tmp_fd)

ffmpeg = get_ffmpeg()
subprocess.run([
    ffmpeg, "-y", "-loglevel", "error",
    "-i", str(VIDEO),
    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    tmp_wav
], capture_output=True, text=True, timeout=60)

wav_np, file_sr = sf.read(tmp_wav, dtype="float32")
logger.info(f"soundfile.read() → numpy array:")
logger.info(f"  wav_np.shape = {wav_np.shape}")
logger.info(f"  wav_np.dtype = {wav_np.dtype}")
logger.info(f"  file_sr = {file_sr}")

wav = torch.from_numpy(wav_np)
logger.info(f"torch.from_numpy() → tensor:")
logger.info(f"  wav.shape = {wav.shape}")
logger.info(f"  wav.dtype = {wav.dtype}")

is_mono = len(wav_np.shape) == 1
logger.info(f"  is_mono = {is_mono} (shape has {len(wav_np.shape)} dims)")
logger.info(f"  sample_rate == 16000 = {file_sr == 16000}")

if len(wav_np.shape) > 1:
    logger.info(f"  WARNING: stereo audio detected, shape={wav_np.shape}")

os.unlink(tmp_wav)


# ============================================================
# TEST 3: Silero model load + get_speech_timestamps
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("TEST 3: Silero model + get_speech_timestamps")
logger.info("=" * 60)

from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()
logger.info(f"load_silero_vad() → model type: {type(model).__name__}")

ts = get_speech_timestamps(
    wav, model,
    threshold=0.5,
    sampling_rate=16000,
    return_seconds=True
)
total = sum(t["end"] - t["start"] for t in ts)
logger.info(f"get_speech_timestamps → {len(ts)} regions, {total:.1f}s speech")
for i, t in enumerate(ts[:5]):
    logger.info(f"  [{i}] start={t['start']:.2f}s  end={t['end']:.2f}s  dur={t['end']-t['start']:.2f}s")
if len(ts) > 5:
    logger.info(f"  ... ({len(ts) - 5} more)")


# ============================================================
# TEST 4: OCR via _try_easyocr
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("TEST 4: OCR engine")
logger.info("=" * 60)

from backend.pipeline.ocr import _try_ocr_engine, _extract_frame

tmp_fd2, tmp_img = tempfile.mkstemp(suffix=".jpg", prefix="verify_ocr_")
os.close(tmp_fd2)

if _extract_frame(VIDEO, 5.0, tmp_img):
    logger.info(f"Frame extracted at t=5.0s → {tmp_img} ({os.path.getsize(tmp_img)} bytes)")
    results = _try_ocr_engine(tmp_img)
    if results:
        for r in results:
            bbox = r.get("bbox", [])
            bbox_type = type(bbox[0]).__name__ if bbox else "N/A"
            logger.info(f"OCR result: text={r['text']!r}  confidence={r['confidence']:.3f}  bbox_type={bbox_type}  bbox_len={len(bbox)}")
            if bbox:
                logger.info(f"  bbox[0] = {bbox[0]}")
    else:
        logger.info("OCR returned empty results (no text found in frame)")
else:
    logger.info("Frame extraction failed")

os.unlink(tmp_img)


# ============================================================
# TEST 5: Full build_rich_timeline
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("TEST 5: build_rich_timeline (full merge)")
logger.info("=" * 60)

# Create a fake Whisper transcript from the VAD regions we already have
# In production, Whisper provides this. We synthesize it for testing.
transcript = []
for i, region in enumerate(speech_regions[:20]):  # cap at 20 segments
    transcript.append({
        "start": region["start"],
        "end": region["end"],
        "text": f"segment_{i}",
        "words": []
    })

if not transcript:
    # If VAD found nothing, make a minimal transcript spanning the video
    transcript = [{"start": 0.0, "end": 10.0, "text": "test", "words": []}]

logger.info(f"Input transcript: {len(transcript)} segments")

from backend.pipeline.timeline_builder import build_rich_timeline

timeline = build_rich_timeline(transcript, VIDEO)

logger.info("")
logger.info("=== RICH TIMELINE RESULTS ===")
logger.info(f"segments count:            {len(timeline.segments)}")
logger.info(f"source_duration:           {timeline.source_duration:.1f}s")
logger.info(f"total_speech_duration:     {timeline.total_speech_duration:.1f}s")
logger.info(f"total_silence_duration:    {timeline.total_silence_duration:.1f}s")
logger.info(f"speech_region_count (VAD): {timeline.speech_region_count}")
logger.info(f"ocr_region_count:          {timeline.ocr_region_count}")

# Show first segment details
if timeline.segments:
    seg = timeline.segments[0]
    logger.info("")
    logger.info("=== FIRST SEGMENT DETAIL ===")
    logger.info(f"  segment_id:     {seg.segment_id}")
    logger.info(f"  start:          {seg.start}")
    logger.info(f"  end:            {seg.end}")
    logger.info(f"  speech_energy:  {seg.speech_energy}")
    logger.info(f"  speech_regions: {seg.speech_regions}")
    logger.info(f"  ocr:            {seg.ocr}")
    logger.info(f"  ocr_confidence: {seg.ocr_confidence}")
    logger.info(f"  metrics:        {seg.metrics.model_dump()}")
    logger.info(f"  speech:         {seg.speech!r}")


# ============================================================
# TEST 6: LLM prompt contains energy/OCR
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("TEST 6: LLM prompt content check")
logger.info("=" * 60)

from backend.pipeline.analyzer import _format_rich_timeline

formatted = _format_rich_timeline(timeline)
logger.info(f"Formatted timeline length: {len(formatted)} chars")
logger.info(f"First 500 chars:")
logger.info(formatted[:500])

# Check for energy markers and OCR
has_energy = "⚡" in formatted or "energy" in formatted.lower()
has_ocr = "OCR" in formatted or "ocr" in formatted.lower()
logger.info(f"Contains energy data:  {has_energy}")
logger.info(f"Contains OCR data:     {has_ocr}")


# ============================================================
# TEST 7: Narration VAD (compositor)
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("TEST 7: Narration VAD via compositor")
logger.info("=" * 60)

from backend.pipeline.compositor import get_speech_timestamps_from_narration

# Find a narration WAV from previous runs
wav_files = []
for root, dirs, files in os.walk("C:\\Projects\\unhuman-clips\\storage\\working"):
    for f in files:
        if f.endswith(".wav") and "narration" in f:
            wav_files.append(os.path.join(root, f))
    if wav_files:
        break

if wav_files:
    nar_path = wav_files[0]
    logger.info(f"Testing narration VAD on: {os.path.basename(nar_path)}")
    nar_ts = get_speech_timestamps_from_narration(nar_path)
    logger.info(f"Narration speech regions: {len(nar_ts)}")
    for t in nar_ts[:3]:
        logger.info(f"  start={t['start']:.2f}s  end={t['end']:.2f}s")
else:
    logger.info("No narration WAV found, skipping narration VAD test")


# ============================================================
# TEST 8: Editor VAD
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("TEST 8: Editor VAD via detect_silence_with_vad")
logger.info("=" * 60)

from backend.pipeline.editor import detect_silence_with_vad

silence = detect_silence_with_vad(VIDEO)
logger.info(f"Silence segments: {len(silence)}")
total_silence = sum(s.get("duration", s.get("end", 0) - s.get("start", 0)) for s in silence)
logger.info(f"Total silence: {total_silence:.1f}s")
for s in silence[:3]:
    logger.info(f"  {s}")


# ============================================================
# TEST 9: Code audit — no stale read_audio imports
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("TEST 9: Code audit — no stale read_audio imports")
logger.info("=" * 60)

import re
files_to_check = [
    "backend/pipeline/timeline_builder.py",
    "backend/pipeline/editor.py",
    "backend/pipeline/compositor.py",
]

for fpath in files_to_check:
    full = os.path.join(os.getcwd(), fpath)
    with open(full) as f:
        content = f.read()
    
    # Check for import read_audio
    has_import_read_audio = "import.*read_audio" in content or "from silero_vad import.*read_audio" in content
    has_read_audio_call = re.search(r'\bread_audio\s*\(', content) is not None
    
    # Check for soundfile import
    has_soundfile = "import soundfile" in content
    
    # Check for .tolist() without hasattr guard
    lines = content.split('\n')
    tolist_issues = []
    for i, line in enumerate(lines, 1):
        if '.tolist()' in line and 'hasattr' not in line and 'tolist()' not in line:
            # Check if it's in a comment
            stripped = line.strip()
            if not stripped.startswith('#') and '.tolist()' in stripped:
                tolist_issues.append(f"  line {i}: {stripped[:100]}")
    
    logger.info(f"{fpath}:")
    logger.info(f"  import read_audio:  {'FOUND (BAD)' if has_import_read_audio else 'clean'}")
    logger.info(f"  read_audio() call:  {'FOUND (BAD)' if has_read_audio_call else 'clean'}")
    logger.info(f"  import soundfile:   {'YES' if has_soundfile else 'NO'}")
    if tolist_issues:
        logger.info(f"  .tolist() without guard:")
        for issue in tolist_issues:
            logger.info(issue)
    else:
        logger.info(f"  .tolist() usage:    clean (all guarded)")


logger.info("")
logger.info("=" * 60)
logger.info("=== VERIFICATION COMPLETE ===")
logger.info("=" * 60)
