"""
wait_then_run.py
================

1. Polls %APPDATA%/unhuman-clips/cookies.txt until it exists and is non-empty.
2. Creates a job against the test URL via POST /jobs.
3. Polls GET /jobs/{id} every 5s until status == DONE or ERROR.
4. On completion: prints final duration of edited_{id}.mp4, dumps ASS caption
   file contents, ffmpeg filter graph, and commentary text from the saved job.
5. Saves everything to logs/wait_then_run_report.json for later inspection.

Run: python wait_then_run.py
"""
import os
import json
import time
import sys
import traceback
from pathlib import Path

import requests

API = "http://127.0.0.1:9000"
TEST_URL = "https://youtu.be/fKoAOWQHP0o?si=DQnLjASXQaKnOwD6"

COOKIE_PATH = Path(os.environ["APPDATA"]) / "unhuman-clips" / "cookies.txt"
REPO_ROOT = Path(r"C:\Projects\unhuman-clips")
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

REPORT_FILE = LOGS_DIR / "wait_then_run_report.json"

def wait_for_cookies(timeout_s: int = 600) -> bool:
    deadline = time.time() + timeout_s
    print(f"[INFO] Waiting for cookies file at: {COOKIE_PATH}")
    while time.time() < deadline:
        try:
            if COOKIE_PATH.exists() and COOKIE_PATH.stat().st_size > 0:
                print(f"[OK] cookies.txt appeared, {COOKIE_PATH.stat().st_size} bytes")
                return True
        except OSError:
            pass
        time.sleep(2)
    print(f"[TIMEOUT] cookies.txt never appeared within {timeout_s}s")
    return False


def create_job() -> str:
    print(f"[INFO] POST /jobs url={TEST_URL}")
    r = requests.post(f"{API}/jobs", json={"url": TEST_URL}, timeout=60)
    r.raise_for_status()
    j = r.json()
    job_id = j.get("id") or j.get("job_id")
    print(f"[OK] created job_id={job_id} status={j.get('status')}")
    return job_id


def poll_job(job_id: str, timeout_s: int = 1800) -> dict:
    deadline = time.time() + timeout_s
    last_len = 0
    while time.time() < deadline:
        r = requests.get(f"{API}/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        j = r.json()
        stage = j.get("current_stage") or j.get("stage_index")
        status = j.get("status")
        progress = j.get("progress")
        err = j.get("error")
        logs = j.get("logs") or []
        print(f"  [t={int(deadline - time.time())}s] stage={stage} status={status} progress={progress}% err={err}")
        if logs and len(logs) != last_len:
            for line in logs[last_len:]:
                print(f"    LOG: {line}")
            last_len = len(logs)
        if status in ("DONE", "ERROR", "FAILED"):
            return j
        time.sleep(5)
    raise TimeoutError(f"Job {job_id} never reached terminal state in {timeout_s}s")


def collect_evidence(job_id: str) -> dict:
    """Gather final duration, ASS caption file, ffmpeg filter graph, commentary text."""
    out = {"job_id": job_id}
    edited = REPO_ROOT / "storage" / "outputs" / f"edited_{job_id}.mp4"
    if edited.exists():
        out["edited_path"] = str(edited)
        out["edited_size_mb"] = round(edited.stat().st_size / (1024 * 1024), 2)

    working = REPO_ROOT / "storage" / "working" / job_id
    # ffmpeg_filter_graph.txt written by compositor
    fg = working / "ffmpeg_filter_graph.txt"
    if fg.exists():
        out["ffmpeg_filter_graph_file"] = str(fg)
        out["ffmpeg_filter_graph"] = fg.read_text(encoding="utf-8")

    # ASS caption files — narration + commentary
    ass_files = list(working.rglob("*.ass"))
    out["ass_files"] = [str(p) for p in ass_files]
    ass_contents = {}
    for p in ass_files:
        try:
            ass_contents[str(p)] = p.read_text(encoding="utf-8")
        except Exception:
            pass
    out["ass_contents"] = ass_contents

    # Job JSON from queue_manager if present
    job_json = working / "job.json"
    if job_json.exists():
        try:
            data = json.loads(job_json.read_text(encoding="utf-8"))
            out["job_json"] = data
            # Pull narration events directly
            plan = data.get("reel_plan") or {}
            events = plan.get("narration_events") or []
            out["narration_events"] = events
            event_text = []
            for e in events:
                event_text.append({
                    "type": e.get("type"),
                    "clip_index": e.get("clip_index"),
                    "reel_start": e.get("reel_start"),
                    "reel_end": e.get("reel_end"),
                    "text": e.get("text"),
                    "ducking": e.get("ducking"),
                })
            out["commentary_text"] = event_text
            # clip_windows
            out["clip_windows"] = data.get("clip_windows")
        except Exception as e:
            out["job_json_error"] = str(e)

    return out


def probe_duration(path: Path) -> float:
    try:
        import subprocess
        ffmpeg = Path(r"C:\Users\starr\.vscode\extensions\kilocode.kilo-code-7.4.11-win32-x64\bin\ffmpeg.exe")
        if not ffmpeg.exists():
            ffmpeg = Path("ffmpeg")
        res = subprocess.run(
            [str(ffmpeg), "-i", str(path)],
            capture_output=True, text=True, timeout=30
        )
        import re
        m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr)
        if m:
            h, mi, s = m.groups()
            return int(h) * 3600 + int(mi) * 60 + float(s)
    except Exception:
        pass
    return -1.0


def main():
    try:
        if not wait_for_cookies():
            sys.exit(2)
        job_id = create_job()
        final_state = poll_job(job_id)
        report = collect_evidence(job_id)
        report["final_state"] = {
            "status": final_state.get("status"),
            "stage": final_state.get("current_stage"),
            "progress": final_state.get("progress"),
            "error": final_state.get("error"),
        }
        # Probe final duration
        if "edited_path" in report:
            report["edited_duration_seconds"] = probe_duration(Path(report["edited_path"]))

        # Find backend uvicorn log for filter graph + duck_expr lines (if not captured)
        uvicorn_log = None
        for candidate in (REPO_ROOT / "logs" / "uvicorn.log", REPO_ROOT / "uvicorn.log"):
            if candidate.exists():
                uvicorn_log = candidate
                break
        if uvicorn_log:
            content = uvicorn_log.read_text(encoding="utf-8", errors="ignore")
            tail = content[-20000:]
            for marker in ("[FFMPEG_DUCK_EXPR]", "[FFMPEG_FILTER_GRAPH]", "[FFMPEG_FULL_CMD]"):
                idx = tail.find(marker)
                if idx >= 0:
                    line_end = tail.find("\n", idx)
                    report.setdefault("uvicorn_markers", {})[marker] = tail[idx:line_end if line_end > idx else idx + 4000]

        REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\n[REPORT] {REPORT_FILE}")
        print(f"[DURATION] {report.get('edited_duration_seconds')}")
        print(f"[STATUS] {report['final_state']}")
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
