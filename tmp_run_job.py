"""Create a job and poll it, capturing ffmpeg debug output for evidence."""
import requests
import time
import json
import sys

BASE = "http://127.0.0.1:8000"

url = sys.argv[1] if len(sys.argv) > 1 else "https://youtu.be/fKoAOWQHP0o?si=DQnLjASXQaKnOwD6"

r = requests.post(f"{BASE}/jobs", json={"url": url}, timeout=30)
r.raise_for_status()
job = r.json()
job_id = job.get("id")
print(f"JOB_ID: {job_id}")
print(f"Initial status: {job.get('status')}")

last_status = None
last_progress = -1
while True:
    r = requests.get(f"{BASE}/jobs", timeout=30)
    jobs = r.json()
    match = [j for j in jobs if j.get("id") == job_id]
    if not match:
        print("Job gone from queue")
        break
    j = match[0]
    status = j.get("status")
    progress = j.get("progress", 0)
    error = j.get("error")
    stage = j.get("stage_index", 0)
    stage_data = j.get("stage_data", {})
    
    if status != last_status or (progress != last_progress and progress % 10 < 5):
        print(f"[{status}] Stage {stage}: {progress:.0f}%  {stage_data.get('message', '')}")
        last_status = status
        last_progress = progress
        
    if status == "DONE":
        print(f"\n=== JOB DONE ===")
        output_path = j.get("output_path", "")
        final_dur = stage_data.get("final_duration", "?")
        print(f"Output: {output_path}")
        print(f"Final duration: {final_dur}s")
        clips = j.get("clip_windows", [])
        print(f"\n=== CLIP WINDOWS ({len(clips)}) ===")
        total_dur = 0
        for c in clips:
            d = c.get("end", 0) - c.get("start", 0)
            total_dur += d
            print(f"  [{c.get('start',0):.1f}-{c.get('end',0):.1f}] dur={d:.1f}s reason={c.get('reason','')[:60]}")
        print(f"  TOTAL selected duration: {total_dur:.1f}s")
        
        rp = j.get("reel_plan", {})
        events = rp.get("narration_events", [])
        print(f"\n=== NARRATION EVENTS ({len(events)}) ===")
        for e in events:
            print(f"  {e.get('event_id','?')}: [{e.get('reel_start',0):.1f}-{e.get('reel_end',0):.1f}] '{e.get('text','')[:60]}'")
        print(f"\n=== reel_plan JSON ===\n{json.dumps(rp, indent=2)}")
        break
    elif status == "ERROR":
        print(f"\n=== ERROR ===")
        print(f"Error: {error}")
        print(f"Stage data: {json.dumps(stage_data, indent=2)}")
        break
    
    time.sleep(3)