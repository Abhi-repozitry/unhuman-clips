"""
Quick test: POST job, poll every 5s, report duration of each output group.
"""
import os, json, time, sys, traceback, requests
from pathlib import Path

API = "http://127.0.0.1:9000"
TEST_URL = "https://youtu.be/Ah_uuTwGOYU?si=pAu07P3P1y88giFu"

REPO_ROOT = Path(r"C:\Projects\unhuman-clips")
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
REPORT_FILE = LOGS_DIR / "test_now_report.json"

def create_job():
    print(f"[INFO] POST /jobs url={TEST_URL}")
    r = requests.post(f"{API}/jobs", json={"url": TEST_URL}, timeout=60)
    r.raise_for_status()
    j = r.json()
    job_id = j.get("id")
    print(f"[OK] job_id={job_id} status={j.get('status')}")
    return job_id

def poll_all_jobs(job_id: str, timeout_s: int = 1800):
    deadline = time.time() + timeout_s
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
        err = target.get("error")
        stage = target.get("current_stage") or target.get("stage_index")
        num_groups = target.get("num_output_groups", 0)
        outputs = target.get("outputs", [])
        logs = target.get("logs", [])
        print(f"  [t={int(deadline - time.time())}s] stage={stage} status={status} progress={progress}% groups={num_groups} err={err}")
        if logs and len(logs) != last_logs:
            for line in logs[last_logs:]:
                print(f"    LOG: {line[:120]}")
            last_logs = len(logs)
        if status in ("DONE", "ERROR", "FAILED"):
            return target
        time.sleep(5)
    raise TimeoutError(f"Job {job_id} never reached terminal state in {timeout_s}s")

def probe_duration(path: Path) -> float:
    try:
        import subprocess
        ffprobe = Path(r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffprobe.exe")
        res = subprocess.run(
            [str(ffprobe), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15
        )
        return float(res.stdout.strip())
    except Exception:
        return -1.0

def main():
    try:
        job_id = create_job()
        final = poll_all_jobs(job_id)
        report = {"job_id": job_id, "final_status": final.get("status")}
        outputs = final.get("outputs", [])
        for o in outputs:
            path = o.get("output_path")
            idx = o.get("output_index")
            dur = o.get("duration_seconds", 0)
            if path:
                actual_dur = probe_duration(Path(path))
                print(f"  Group {idx}: reported={dur:.1f}s probed={actual_dur:.1f}s path={path}")
                report[f"group_{idx}"] = {"reported_dur": dur, "probed_dur": actual_dur, "path": path}
            else:
                print(f"  Group {idx}: reported={dur:.1f}s (no path)")
                report[f"group_{idx}"] = {"reported_dur": dur}
        total_groups = len(outputs)
        valid_groups = 0
        for o in outputs:
            dur = o.get("duration_seconds", 0)
            if 90 <= dur <= 180:
                valid_groups += 1
        print(f"\n=== RESULT ===")
        print(f"Total groups: {total_groups}")
        print(f"Groups in 90-180s range: {valid_groups}/{total_groups}")
        if total_groups > 0 and valid_groups < total_groups:
            print(f"[WARN] {total_groups - valid_groups} group(s) outside 90-180s target")
        REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"[REPORT] {REPORT_FILE}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()