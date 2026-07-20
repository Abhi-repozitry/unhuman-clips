import requests
import time
import json
import sys

BASE = "http://127.0.0.1:9000"
YT_URL = "https://youtu.be/Ah_uuTwGOYU?si=k9Ds1I1fbmxNn86D"

try:
    r = requests.post(f"{BASE}/jobs", json={"url": YT_URL}, timeout=30)
    r.raise_for_status()
    job = r.json()
    job_id = job.get("id")
    print(f"JOB_ID: {job_id}")
    print(f"Initial status: {job.get('status')}")
    sys.stdout.flush()

    last_status = None
    last_progress = -1
    last_stage = -1
    tick_count = 0

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
        sub_stage = j.get("sub_stage", "")
        current_stage = j.get("current_stage", "")
        logs = j.get("logs", [])
        outputs = j.get("outputs", [])
        source_path = j.get("source_path", "")

        if status != last_status or stage != last_stage or tick_count % 2 == 0:
            msg = stage_data.get("message", "") or stage_data.get("status", "")
            print(f"[{status}] Stage {stage}: {progress:.0f}%  sub: {sub_stage[:80]}  msg: {msg}")
            if logs:
                last_log = logs[-1]
                print(f"  last log: {last_log[:100]}")
            sys.stdout.flush()
            last_status = status
            last_progress = progress
            last_stage = stage

        if status == "DONE":
            print(f"\n=== JOB DONE ===")
            print(f"Outputs: {len(outputs)} group(s)")
            for i, o in enumerate(outputs):
                print(f"  Group {i}: {o.get('output_path')} - {o.get('duration_seconds', 0):.1f}s - {o.get('status')}")
            print(f"Source: {source_path}")
            break
        elif status == "ERROR":
            print(f"\n=== ERROR ===")
            print(f"Error: {error}")
            print(f"Stage data: {json.dumps(stage_data, indent=2)}")
            print(f"Last logs:")
            for log in logs[-5:]:
                print(f"  {log}")
            break

        tick_count += 1
        time.sleep(5)

    # Print full logs at end
    print("\n=== FULL LOGS ===")
    for log in logs:
        print(f"  {log}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()