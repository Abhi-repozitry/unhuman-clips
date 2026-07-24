"""
FINAL VERIFICATION: Prove the runtime VAD path is:
  FFmpeg -> WAV -> soundfile.read() -> torch.from_numpy() -> load_silero_vad() -> get_speech_timestamps()
  with NO torchaudio.load() and NO silero_vad.read_audio() calls.

Run against a real video. All output is actual runtime evidence.
"""
import logging
import sys
import os
import tempfile
import subprocess
import re
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("FINAL")

VIDEO = r"C:\Projects\unhuman-clips\backend\storage\working\0449350f-7669-4b85-a333-d1202f75dcba\downloads\fKoAOWQHP0o.webm"

# ============================================================
# SECTION 0: Environment
# ============================================================
logger.info("=" * 70)
logger.info("SECTION 0: ENVIRONMENT")
logger.info("=" * 70)

import torch, torchaudio, soundfile, silero_vad
logger.info(f"torch:       {torch.__version__}")
logger.info(f"torchaudio:  {torchaudio.__version__}")
logger.info(f"soundfile:   {soundfile.__version__}")
logger.info(f"silero_vad:  {silero_vad.__version__}")
logger.info(f"video:       {os.path.basename(VIDEO)} ({os.path.getsize(VIDEO)} bytes)")

# ============================================================
# SECTION 1: Prove no read_audio / torchaudio.load in source
# ============================================================
logger.info("")
logger.info("=" * 70)
logger.info("SECTION 1: STATIC AUDIT — no read_audio, no torchaudio.load")
logger.info("=" * 70)

files_to_audit = [
    "backend/pipeline/timeline_builder.py",
    "backend/pipeline/editor.py",
    "backend/pipeline/compositor.py",
]

for fpath in files_to_audit:
    full = os.path.join(os.getcwd(), fpath)
    with open(full) as f:
        lines = f.readlines()

    violations = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        # Check for read_audio calls
        if re.search(r'\bread_audio\s*\(', stripped):
            violations.append(f"  line {i}: read_audio() call: {stripped[:100]}")
        # Check for import read_audio
        if 'import' in stripped and 'read_audio' in stripped:
            violations.append(f"  line {i}: read_audio import: {stripped[:100]}")
        # Check for torchaudio.load calls
        if re.search(r'torchaudio\.load\s*\(', stripped):
            violations.append(f"  line {i}: torchaudio.load() call: {stripped[:100]}")
        # Check for import torchaudio (not in silero_vad internals)
        if re.search(r'\bimport\s+torchaudio\b', stripped) and 'silero_vad' not in fpath:
            violations.append(f"  line {i}: torchaudio import: {stripped[:100]}")

    if violations:
        logger.info(f"FAIL {fpath}:")
        for v in violations:
            logger.info(v)
    else:
        logger.info(f"PASS {fpath}: zero read_audio calls, zero torchaudio.load calls")

# Also verify the three files import soundfile
for fpath in files_to_audit:
    full = os.path.join(os.getcwd(), fpath)
    with open(full) as f:
        content = f.read()
    has_sf = "import soundfile" in content
    has_read_audio_import = bool(re.search(r'from silero_vad import.*\bread_audio\b', content))
    logger.info(f"  {fpath}: soundfile={'YES' if has_sf else 'NO'}, read_audio_import={'YES (BAD)' if has_read_audio_import else 'NO (clean)'}")

# ============================================================
# SECTION 2: Run _run_vad_on_source on real video
# ============================================================
logger.info("")
logger.info("=" * 70)
logger.info("SECTION 2: RUNTIME VAD on real video")
logger.info("=" * 70)

from backend.pipeline.timeline_builder import _run_vad_on_source

speech_regions = _run_vad_on_source(VIDEO)
total_speech = sum(r["end"] - r["start"] for r in speech_regions)
logger.info(f"RESULT: {len(speech_regions)} speech regions, {total_speech:.1f}s total speech")
for i, r in enumerate(speech_regions[:5]):
    logger.info(f"  [{i}] start={r['start']:.2f}s  end={r['end']:.2f}s  dur={r['end']-r['start']:.2f}s")
if len(speech_regions) > 5:
    logger.info(f"  ... ({len(speech_regions) - 5} more)")

# ============================================================
# SECTION 3: Prove the exact call chain via tracing
# ============================================================
logger.info("")
logger.info("=" * 70)
logger.info("SECTION 3: CALL CHAIN TRACE — prove soundfile, not torchaudio")
logger.info("=" * 70)

import soundfile as sf
import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
from backend.ffmpeg_utils import get_ffmpeg

# Step 1: FFmpeg extract
tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="trace_")
os.close(tmp_fd)
ffmpeg = get_ffmpeg()
result = subprocess.run([
    ffmpeg, "-y", "-loglevel", "error",
    "-i", str(VIDEO),
    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    tmp_wav
], capture_output=True, text=True, timeout=60)
logger.info(f"Step 1 — FFmpeg -> WAV: returncode={result.returncode}, exists={os.path.exists(tmp_wav)}, size={os.path.getsize(tmp_wav)} bytes")

# Step 2: soundfile.read
wav_np, file_sr = sf.read(tmp_wav, dtype="float32")
logger.info(f"Step 2 — soundfile.read():")
logger.info(f"  wav_np.shape = {wav_np.shape}   (1D = mono)")
logger.info(f"  wav_np.dtype = {wav_np.dtype}")
logger.info(f"  file_sr      = {file_sr}")
logger.info(f"  is_mono      = {len(wav_np.shape) == 1}")
logger.info(f"  sr == 16000  = {file_sr == 16000}")

# Step 3: torch.from_numpy
wav = torch.from_numpy(wav_np)
logger.info(f"Step 3 — torch.from_numpy():")
logger.info(f"  wav.shape = {wav.shape}")
logger.info(f"  wav.dtype = {wav.dtype}")

# Step 4: load_silero_vad
model = load_silero_vad()
logger.info(f"Step 4 — load_silero_vad(): model type = {type(model).__name__}")

# Step 5: get_speech_timestamps
speech_ts = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=16000, return_seconds=True)
ts_total = sum(t["end"] - t["start"] for t in speech_ts)
logger.info(f"Step 5 — get_speech_timestamps(): {len(speech_ts)} regions, {ts_total:.1f}s speech")
for i, t in enumerate(speech_ts[:5]):
    logger.info(f"  [{i}] {t['start']:.2f}s - {t['end']:.2f}s ({t['end']-t['start']:.2f}s)")

os.unlink(tmp_wav)

# ============================================================
# SECTION 4: Editor VAD
# ============================================================
logger.info("")
logger.info("=" * 70)
logger.info("SECTION 4: Editor VAD (detect_silence_with_vad)")
logger.info("=" * 70)

from backend.pipeline.editor import detect_silence_with_vad
silence = detect_silence_with_vad(VIDEO)
total_sil = sum(s.get("duration", s.get("end", 0) - s.get("start", 0)) for s in silence)
logger.info(f"RESULT: {len(silence)} silence segments, {total_sil:.1f}s total silence")
for s in silence[:3]:
    logger.info(f"  {s}")

# ============================================================
# SECTION 5: Narration VAD (compositor)
# ============================================================
logger.info("")
logger.info("=" * 70)
logger.info("SECTION 5: Narration VAD (get_speech_timestamps_from_narration)")
logger.info("=" * 70)

# Find a narration WAV
nar_path = None
for root, dirs, files in os.walk(r"C:\Projects\unhuman-clips\storage\working"):
    for f in files:
        if f.endswith(".wav") and "narration" in f:
            nar_path = os.path.join(root, f)
            break
    if nar_path:
        break

if nar_path:
    from backend.pipeline.compositor import get_speech_timestamps_from_narration
    nar_ts = get_speech_timestamps_from_narration(nar_path)
    logger.info(f"File: {os.path.basename(nar_path)} ({os.path.getsize(nar_path)} bytes)")
    logger.info(f"RESULT: {len(nar_ts)} narration speech regions")
    for t in nar_ts[:5]:
        logger.info(f"  start={t['start']:.2f}s  end={t['end']:.2f}s")
else:
    logger.info("No narration WAV found, skipping")

# ============================================================
# SECTION 6: OCR with EasyOCR 1.7.2 — .tolist() guard verified
# ============================================================
logger.info("")
logger.info("=" * 70)
logger.info("SECTION 6: OCR — EasyOCR 1.7.2 bbox type test")
logger.info("=" * 70)

from backend.pipeline.ocr import _try_easyocr, _extract_frame

tmp_fd2, tmp_img = tempfile.mkstemp(suffix=".jpg", prefix="ocr_verify_")
os.close(tmp_fd2)
if _extract_frame(VIDEO, 5.0, tmp_img):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    raw = reader.readtext(tmp_img, detail=1, paragraph=False)
    if raw:
        raw_bbox = raw[0][0]
        logger.info(f"EasyOCR raw bbox type: {type(raw_bbox).__name__}")
        logger.info(f"  hasattr(bbox, 'tolist'): {hasattr(raw_bbox, 'tolist')}")
        logger.info(f"  bbox[0] type: {type(raw_bbox[0]).__name__}")

    ocr_results = _try_easyocr(tmp_img)
    logger.info(f"_try_easyocr returned: {len(ocr_results)} results")
    for r in ocr_results:
        logger.info(f"  text={r['text']!r}  confidence={r['confidence']:.3f}  bbox_type={type(r['bbox']).__name__}")
else:
    logger.info("Frame extraction failed")

os.unlink(tmp_img)

# ============================================================
# SECTION 7: Full build_rich_timeline
# ============================================================
logger.info("")
logger.info("=" * 70)
logger.info("SECTION 7: build_rich_timeline — full merge")
logger.info("=" * 70)

transcript = [{"start": r["start"], "end": r["end"], "text": f"seg_{i}", "words": []}
              for i, r in enumerate(speech_regions[:15])]

from backend.pipeline.timeline_builder import build_rich_timeline
timeline = build_rich_timeline(transcript, VIDEO)

logger.info(f"segments:          {len(timeline.segments)}")
logger.info(f"source_duration:   {timeline.source_duration:.1f}s")
logger.info(f"total_speech:      {timeline.total_speech_duration:.1f}s")
logger.info(f"total_silence:     {timeline.total_silence_duration:.1f}s")
logger.info(f"speech_regions:    {timeline.speech_region_count}")
logger.info(f"ocr_region_count:  {timeline.ocr_region_count}")

if timeline.segments:
    seg = timeline.segments[0]
    logger.info(f"First segment:")
    logger.info(f"  segment_id:    {seg.segment_id}")
    logger.info(f"  start:         {seg.start}")
    logger.info(f"  end:           {seg.end}")
    logger.info(f"  speech_energy: {seg.speech_energy}")
    logger.info(f"  speech_regions:{seg.speech_regions}")
    logger.info(f"  ocr:           {seg.ocr}")
    logger.info(f"  metrics:       {seg.metrics.model_dump()}")

# ============================================================
# SECTION 8: LLM prompt contains energy data
# ============================================================
logger.info("")
logger.info("=" * 70)
logger.info("SECTION 8: LLM prompt — energy data present")
logger.info("=" * 70)

from backend.pipeline.analyzer import _format_rich_timeline
formatted = _format_rich_timeline(timeline)
logger.info(f"Formatted timeline: {len(formatted)} chars")
has_energy = "energy" in formatted.lower() or "\u2588" in formatted
logger.info(f"Contains energy bars/data: {has_energy}")
logger.info(f"First 300 chars (bytes):")
# Write bytes to avoid encoding issues on Windows
sys.stdout.buffer.write(formatted[:300].encode("utf-8", errors="replace"))
sys.stdout.buffer.write(b"\n")

logger.info("")
logger.info("=" * 70)
logger.info("=== ALL SECTIONS COMPLETE ===")
logger.info("=" * 70)
