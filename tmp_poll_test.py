"""Submit and poll a test job to verify the ducking and overlap-validation changes."""
import requests
import time
import json

BASE = "http://127.0.0.1:8080"
url = "https://youtu.be/Ah_uuTwGOYU?si=pAu07P3P1y88giFu"

# Create the job
r = requests.post(f"{BASE}/jobs", json={"url": url}, timeout=30)
r.raise_for_status()
job = r.json()
job_id = job.get("id")
print(f"JOB_ID: {job_id}")
print(f"Initial status: {job.get('status')}", flush=True)

last_status = None
last_progress = -1
logs_seen = set()
while True:
    try:
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

        if status != last_status or abs(progress - last_progress) >= 5:
            msg = stage_data.get("message", "")
            print(f"[{status}] Stage {stage}: {progress:.0f}%  {msg}", flush=True)
            last_status = status
            last_progress = progress

        # Print any new log entries with WARN or overlap
        logs = j.get("logs", [])
        for log in logs:
            if log not in logs_seen:
                logs_seen.add(log)
                if "WARN" in log or "overlap" in log.lower() or "Narration" in log:
                    print(f"  LOG: {log}", flush=True)

        if status == "DONE":
            print(f"\n=== JOB DONE ===", flush=True)
            outputs = j.get("outputs", [])
            print(f"Outputs: {json.dumps(outputs, indent=2)}", flush=True)
            # Print all logs
            print(f"\nFull logs ({len(logs)} entries):", flush=True)
            for log in logs:
                print(f"  {log}", flush=True)
            break
        elif status == "ERROR":
            print(f"\n=== ERROR ===", flush=True)
            print(f"Error: {error}", flush=True)
            print(f"Stage data: {json.dumps(stage_data, indent=2)}", flush=True)
            for log in logs:
                print(f"  {log}", flush=True)
            break
        time.sleep(5)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Poll error: {e}", flush=True)
        time.sleep(5)