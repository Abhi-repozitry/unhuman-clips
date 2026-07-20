import requests
import time
import json
import sys

BASE = "http://127.0.0.1:9000"
url = "https://youtu.be/Z0PbIam6PdU?si=P3hmiR4o0NFC668A"

try:
    r = requests.post(f"{BASE}/jobs", json={"url": url}, timeout=30)
    r.raise_for_status()
    job = r.json()
    job_id = job.get("id")
    print(f"JOB_ID: {job_id}")
    print(f"Initial status: {job.get('status')}")

    last_status = None
    last_progress = -1
    while True:
        r = requests.get(f"{BASE}/jobs", timeout=60)
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
            msg = stage_data.get("message", "") or stage_data.get("status", "")
            print(f"[{status}] Stage {stage}: {progress:.0f}%  {msg}")
            last_status = status
            last_progress = progress

        if status == "DONE":
            print(f"\n=== JOB DONE ===")
            outputs = j.get("outputs", [])
            print(f"Outputs: {len(outputs)} group(s)")
            for i, o in enumerate(outputs):
                print(f"  Group {i}: {o.get('output_path')} - {o.get('duration_seconds', 0):.1f}s - {o.get('status')}")
                print(f"  output_url: {o.get('output_url')}")
            rp = j.get("reel_plan", {})
            groups = rp.get("reel_groups", [])
            print(f"Groups in plan: {len(groups)}")
            for g in groups:
                print(f"  Group {g['group_index']}: {len(g['source_clips'])} clips, {len(g['narration_events'])} narrations, est={g.get('estimated_duration_seconds')}s")
            break
        elif status == "ERROR":
            print(f"\n=== ERROR ===")
            print(f"Error: {error}")
            print(f"Stage data: {json.dumps(stage_data, indent=2)}")
            break

        time.sleep(15)
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()