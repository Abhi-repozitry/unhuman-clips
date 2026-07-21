import requests, time, json

API = "http://127.0.0.1:9000"
job_id = "c7453e6e-1dca-4d85-b599-438b0028133c"
deadline = time.time() + 1800
last_logs = 0

while time.time() < deadline:
    r = requests.get(f"{API}/jobs", timeout=30)
    r.raise_for_status()
    jobs = r.json()
    target = None
    for j in jobs:
        if j.get("id") == job_id:
            target = j
            break
    if target is None:
        print(f"  [t={int(deadline - time.time())}s] job not found")
        time.sleep(5)
        continue
    status = target.get("status")
    progress = target.get("progress")
    stage = target.get("current_stage") or target.get("stage_index")
    outputs = target.get("outputs", [])
    err = target.get("error")
    num_groups = target.get("num_output_groups", 0)
    logs = target.get("logs", [])
    print(f"  [t={int(deadline - time.time())}s] stage={stage} status={status} progress={progress}% groups={num_groups}")
    if logs and len(logs) != last_logs:
        for line in logs[last_logs:]:
            clean = line.encode("ascii", errors="replace").decode("ascii")
            print(f"    LOG: {clean[:200]}")
        last_logs = len(logs)
    if err:
        clean_err = err.encode("ascii", errors="replace").decode("ascii")
        print(f"  ERROR: {clean_err[:500]}")
    if status in ("DONE", "ERROR", "FAILED"):
        print("\n=== OUTPUT GROUPS ===")
        for o in outputs:
            dur = o.get("duration_seconds", 0)
            idx = o.get("output_index")
            path = o.get("output_path", "")
            os = o.get("status", "?")
            print(f"  Group {idx}: {dur:.1f}s status={os} path={path}")
        total = len(outputs)
        in_range = sum(1 for o in outputs if 90 <= o.get("duration_seconds", 0) <= 180)
        print(f"\nGroups in 90-180s range: {in_range}/{total}")
        break
    time.sleep(5)
else:
    print("TIMEOUT: job did not finish")