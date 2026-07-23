#!/usr/bin/env python3
"""Test the updated select_reel_plan with a real video."""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from backend.config import get_job_working_dir
from backend.pipeline.downloader import download_video
from backend.pipeline.transcriber import transcribe_video
from backend.pipeline.analyzer import select_reel_plan, _summarize_transcript_for_llm, _try_repair_truncated_json
from backend.models import ReelPlan
import json
import traceback

TEST_URL = "https://youtu.be/Ah_uuTwGOYU"
JOB_ID = "test-reelplan-001"

def progress_cb(msg, prog):
    print(f"  [{prog:.0f}%] {msg}")

try:
    print("=" * 70)
    print("TEST: Download + Analyze with improved reel plan")
    print("=" * 70)

    # Stage 1: Download
    print("\n--- STAGE 1: DOWNLOAD ---")
    job_dir = get_job_working_dir(JOB_ID)
    download_dir = job_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        status = d.get("status")
        if status == "downloading":
            downloaded = d.get("downloaded_bytes", 0)
            total = d.get("total_bytes", 0) or 1
            print(f"  Download: {downloaded//1024**2}MB/{total//1024**2}MB", end="\r")
        elif status == "finished":
            print("  Download finished!")

    result = download_video(TEST_URL, str(download_dir), hook)
    print(f"  Title: {result.get('title')}")
    source_path = result.get("source_path")
    if not source_path or not os.path.exists(source_path):
        raise RuntimeError(f"Downloaded file not found at {source_path}")

    # Stage 2: Transcribe
    print("\n--- STAGE 2: TRANSCRIBE ---")
    def trans_progress(msg, prog):
        print(f"  [{prog:.0f}%] {msg}")
    transcript = transcribe_video(source_path, trans_progress)
    print(f"  Transcript: {len(transcript)} segments")
    if not transcript:
        raise RuntimeError("Transcript is empty")

    total_dur = transcript[-1]["end"] - transcript[0]["start"]
    total_chars_raw = sum(len(e["text"]) for e in transcript)
    print(f"  Video duration: {total_dur:.0f}s, raw text: {total_chars_raw} chars")

    # Stage 3: Test summarization
    print("\n--- STAGE 3: TRANSCRIPT SUMMARIZATION ---")
    summarized = _summarize_transcript_for_llm(transcript, max_total_chars=10000)
    print(f"  Summarized length: {len(summarized)} chars")

    # Stage 4: Build reel plan
    print("\n--- STAGE 4: SELECT REEL PLAN ---")
    video_title = result.get("title", "")
    video_desc = result.get("description", "")

    try:
        reel_plan = select_reel_plan(transcript, video_title, video_desc, progress_cb)
    except Exception as e:
        print(f"\n[ERROR] select_reel_plan failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {type(reel_plan).__name__}")
    print(f"{'=' * 70}")

    if hasattr(reel_plan, 'is_fallback') and reel_plan.is_fallback:
        print("  ** FALLBACK plan (LLM was unavailable) **")
    
    groups = reel_plan.reel_groups if hasattr(reel_plan, 'reel_groups') else reel_plan.get('reel_groups', [])
    
    print(f"\n  Number of reel groups: {len(groups)}")
    
    total_clips = 0
    total_narrations = 0
    
    for i, group in enumerate(groups):
        g = group.model_dump() if hasattr(group, 'model_dump') else (group.dict() if hasattr(group, 'dict') else group)
            
        clips = g.get('source_clips', [])
        narrations = g.get('narration_events', [])
        duration = g.get('estimated_duration_seconds', 0)
        total_clips += len(clips)
        total_narrations += len(narrations)
        
        print(f"\n  Group {i}: {g.get('reel_summary', {}).get('title', 'Untitled')}")
        print(f"    Duration: {duration:.0f}s")
        print(f"    Clips: {len(clips)}")
        print(f"    Narrations: {len(narrations)}")
        print(f"    Reasoning: {g.get('group_reasoning', '')[:120]}")
        
        for j, clip in enumerate(clips):
            c = clip if isinstance(clip, dict) else clip.model_dump()
            print(f"      Clip {j}: {c.get('source_start', 0):.1f}s - {c.get('source_end', 0):.1f}s | {c.get('reason', '')[:80]}")
        
        for j, nar in enumerate(narrations):
            n = nar if isinstance(nar, dict) else nar.model_dump()
            print(f"      Narration {j}: [{n.get('event_type', '')}] {n.get('text', '')[:80]}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {len(groups)} groups, {total_clips} clips, {total_narrations} narrations")
    avg_dur = sum(g.get('estimated_duration_seconds', 0) if isinstance(g, dict) else g.estimated_duration_seconds for g in groups) / max(len(groups), 1)
    print(f"  Average duration: {avg_dur:.0f}s")
    
    # Target check - using model_dump() for Pydantic objects
    print(f"\nTARGET CHECK:")
    print(f"  Groups >= 4: {'✓' if len(groups) >= 4 else '✗'} ({len(groups)})")
    
    group_dicts = [g.model_dump() if hasattr(g, 'model_dump') else g for g in groups]
    all_over_30s = all(d.get('estimated_duration_seconds', 0) >= 30 for d in group_dicts)
    print(f"  All >= 30s duration: {'✓' if all_over_30s else '✗'}")
    all_multi_clip = all(len(d.get('source_clips', [])) >= 2 for d in group_dicts)
    print(f"  All multi-clip: {'✓' if all_multi_clip else '✗'}")

except Exception as e:
    print(f"\nFATAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nDone!")