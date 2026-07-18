#!/usr/bin/env python3
"""Test script for download-to-compose pipeline"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # goes from test_pipeline.py/ -> project root
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from backend.config import DOWNLOADS_DIR, WORKING_DIR, OUTPUTS_DIR, get_job_working_dir
from backend.pipeline.downloader import download_video
from backend.pipeline.transcriber import transcribe_video
from backend.pipeline.analyzer import select_clips
from backend.pipeline.compositor import build_final_video
from backend.pipeline.clipper import cut_clips
from backend.pipeline.commentary import write_commentary
from backend.pipeline.tts import synthesize_commentary
from backend.pipeline.captioner import generate_clip_ass, generate_commentary_ass

TEST_URL = "https://youtu.be/mvcesPWvUIc?si=-4BUSTvNpfVZo5pw"
JOB_ID = "test-job-003"

import tempfile
import traceback

def progress_hook(d):
    status = d.get("status")
    if status == "downloading":
        downloaded = d.get("downloaded_bytes", 0)
        total = d.get("total_bytes", 0)
        speed = d.get("speed", 0)
        print(f"  Download: {downloaded}/{total} bytes @ {speed} bytes/s")
    elif status == "finished":
        print("  Download finished!")

try:
    print("=" * 60)
    print("STAGE 1: DOWNLOAD")
    print("=" * 60)
    job_dir = get_job_working_dir(JOB_ID)
    download_dir = job_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {TEST_URL} to {download_dir}")
    result = download_video(TEST_URL, str(download_dir), progress_hook)
    print(f"Download result: title={result.get('title')}, source_path={result.get('source_path')}")

    source_path = result.get("source_path")
    if not source_path or not os.path.exists(source_path):
        raise RuntimeError(f"Downloaded file not found at {source_path}")

    print("\n" + "=" * 60)
    print("STAGE 2: TRANSCRIBE")
    print("=" * 60)
    def transcribe_progress(msg, prog):
        print(f"  {msg} ({prog:.1f}%)")
    transcript = transcribe_video(source_path, transcribe_progress)
    print(f"Transcribed {len(transcript)} segments")
    if not transcript:
        raise RuntimeError("Transcript is empty")

    print("\n" + "=" * 60)
    print("STAGE 3: ANALYZE (select clips)")
    print("=" * 60)
    def analyze_progress(msg, prog):
        print(f"  {msg} ({prog:.1f}%)")
    video_title = result.get("title", "")
    video_desc = result.get("description", "")
    clip_windows = select_clips(transcript, video_title, video_desc, analyze_progress)
    print(f"Selected {len(clip_windows)} clips: {clip_windows}")

    if not clip_windows:
        raise RuntimeError("No clips selected")

    print("\n" + "=" * 60)
    print("STAGE 4: COMMENTARY (script)")
    print("=" * 60)
    def commentary_progress(msg, prog):
        print(f"  {msg} ({prog:.1f}%)")
    commentary_lines = write_commentary(clip_windows, video_title, commentary_progress)
    print(f"Generated {len(commentary_lines)} commentary lines")

    print("\n" + "=" * 60)
    print("STAGE 5: CUT CLIPS")
    print("=" * 60)
    def clipper_progress(msg, prog):
        print(f"  {msg} ({prog:.1f}%)")
    clip_paths = cut_clips(source_path, clip_windows, JOB_ID, clipper_progress)
    print(f"Cut {len(clip_paths)} clips: {clip_paths}")

    print("\n" + "=" * 60)
    print("STAGE 6: TTS (voice commentary)")
    print("=" * 60)
    commentary_audio = []
    for i, comment in enumerate(commentary_lines):
        out_path = job_dir / f"commentary_{i}.wav"
        print(f"  Generating TTS for clip {i+1}: {comment['text'][:50]}...")
        duration = synthesize_commentary(comment["text"], str(out_path))
        print(f"  Duration: {duration}s -> {out_path}")
        commentary_audio.append({"path": str(out_path), "duration": duration})

    print("\n" + "=" * 60)
    print("STAGE 7: CAPTIONS")
    print("=" * 60)
    caption_paths = []
    comm_caption_paths = []
    for i, window in enumerate(clip_windows):
        clip_caption_path = job_dir / f"clip_caption_{i}.ass"
        print(f"  Generating clip caption {i+1}...")
        generate_clip_ass(transcript, window["start"], window["end"], str(clip_caption_path))
        caption_paths.append(str(clip_caption_path))

        comm_caption_path = job_dir / f"commentary_caption_{i}.ass"
        print(f"  Generating commentary caption {i+1}...")
        generate_commentary_ass(commentary_lines[i]["text"], commentary_audio[i]["duration"], str(comm_caption_path))
        comm_caption_paths.append(str(comm_caption_path))

    print("\n" + "=" * 60)
    print("STAGE 8: COMPOSE FINAL VIDEO")
    print("=" * 60)
    def compositor_progress(msg, prog):
        print(f"  {msg} ({prog:.1f}%)")
    output_path = build_final_video(
        JOB_ID, clip_paths, clip_windows, commentary_audio,
        comm_caption_paths, caption_paths, compositor_progress
    )
    print(f"Final video: {output_path}")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)

except Exception as e:
    print(f"\nERROR: {e}")
    traceback.print_exc()
    sys.exit(1)