import json, os
import requests

BASE = "http://127.0.0.1:9000"
res = requests.get(f"{BASE}/jobs", timeout=10)
res.raise_for_status()
jobs = res.json()

print("JOB_COUNT", len(jobs))
jobs_sorted = sorted(jobs, key=lambda j: j.get("created_at", ""))
j = jobs_sorted[-1] if jobs_sorted else None
if not j:
    print("MOST_RECENT_ID", None)
    raise SystemExit(0)

print("MOST_RECENT_ID", j.get("id"))
print("MOST_RECENT_STATUS", j.get("status"))
print("MOST_RECENT_STAGE", j.get("current_stage") or j.get("stage_index"))
print("MOST_RECENT_ERROR", j.get("error"))

logs = (j.get("logs") or [])[-10:]
print("MOST_RECENT_LOGS_TAIL", logs)
