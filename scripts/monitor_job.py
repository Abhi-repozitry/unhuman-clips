import requests
import time
import json
import sys

BASE = "http://127.0.0.1:9000"
url = sys.argv[1] if len(sys.argv) > 1 else "https://youtu.be/Z0PbIam6PdU?si=FtXbDySaQn0NazEJ"

r = requests.post(f"{BASE}/jobs", json={"url": url}, timeout=30)
r.raise_for_status()
job = r.json()
job_id = job.get("id")
print(f"JOB_ID: {job_id}")

analyze_start = None
analyze_end = None
last_line = ""

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
    stage = j.get("stage_index", 0)
    
    msg = ""
    if j.get("stage_data"):
        if j["stage_data"].get("message"):
            msg = j["stage_data"]["message"]
        elif j["stage_data"].get("status"):
            msg = j["stage_data"]["status"]
    
    if status == "ANALYZING" and analyze_start is None:
        analyze_start = time.time()
        print(f"ANALYZING_START: {time.ctime(analyze_start)}")
    
    if analyze_start and status != "ANALYZING" and analyze_end is None:
        analyze_end = time.time()
        duration = round(analyze_end - analyze_start, 1)
        print(f"ANALYZING_END: {time.ctime(analyze_end)}")
        print(f"ANALYZING_DURATION: {duration}s")
    
    line = f"[{status}] stage={stage} progress={progress:.1f}% {msg}"
    if line != last_line:
        print(line)
        last_line = line
    
    if status == "DONE":
        print(f"\n=== JOB DONE ===")
        outputs = j.get("outputs", [])
        print(f"Outputs: {len(outputs)} group(s)")
        for o in outputs:
            print(f"  Group {o.get('output_index')}: status={o.get('status')} duration={o.get('duration_seconds')}s url={o.get('output_url')}")
        rp = j.get("reel_plan", {})
        groups = rp.get("reel_groups", [])
        print(f"Groups in plan: {len(groups)}")
        for g in groups:
            print(f"  Group {g['group_index']}: {len(g['source_clips'])} clips, {len(g['narration_events'])} narrations, est={g.get('estimated_duration_seconds')}s")
        break
    elif status == "ERROR":
        print(f"\n=== ERROR ===")
        print(f"Error: {j.get('error')}")
        print(f"Stage data: {json.dumps(j.get('stage_data', {}), indent=2)}")
        break
    
    time.sleep(3)