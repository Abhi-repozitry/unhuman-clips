import json, urllib.request

jobs = json.load(open(r"C:\Projects\unhuman-clips\job_status.json"))
job_id = jobs[0]["id"]

resp = urllib.request.urlopen(f"http://127.0.0.1:8081/jobs")
all_jobs = json.loads(resp.read())
job = next((j for j in all_jobs if j["id"] == job_id), None)

reel_plan = job.get("reel_plan") or {}
plan_groups = reel_plan.get("reel_groups", [])

print("=== ISSUE 1: Clip ordering within groups ===")
for g in plan_groups:
    gi = g.get("group_index", "?")
    clips = g.get("source_clips", [])
    starts = [c.get("source_start", 0) for c in clips]
    is_sorted = all(starts[i] <= starts[i+1] for i in range(len(starts)-1))
    print(f"Group {gi}: {'SORTED' if is_sorted else 'UNSORTED'} - starts: {[f'{s:.0f}' for s in starts]}")

print("\n=== ISSUE 2: Hook clips in skip mode ===")
for g in plan_groups:
    gi = g.get("group_index", "?")
    clips = g.get("source_clips", [])
    hook_clips = [c for c in clips if c.get("is_hook_clip")]
    print(f"Group {gi}: {len(hook_clips)} hook clips out of {len(clips)} total")
    for c in hook_clips:
        print(f"  beat={c.get('_beat', '?')} reason={c.get('reason', '')[:60]}")

print("\n=== ISSUE 3: Narration event types ===")
for g in plan_groups:
    gi = g.get("group_index", "?")
    narr = g.get("narration_events", [])
    event_types = [e.get("event_type", "?") for e in narr]
    print(f"Group {gi}: {event_types}")

print("\n=== ISSUE 4: Escalation clips not in window order ===")
for g in plan_groups:
    gi = g.get("group_index", "?")
    clips = g.get("source_clips", [])
    # Check if escalation clips are in source order
    esc_clips = [c for c in clips if "ESCALATION" in c.get("reason", "")]
    esc_starts = [c.get("source_start", 0) for c in esc_clips]
    is_esc_sorted = all(esc_starts[i] <= esc_starts[i+1] for i in range(len(esc_starts)-1))
    print(f"Group {gi}: escalations {'in order' if is_esc_sorted else 'OUT OF ORDER'} - {[f'{s:.0f}' for s in esc_starts]}")
