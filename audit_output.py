import json, urllib.request

jobs = json.load(open(r"C:\Projects\unhuman-clips\job_status.json"))
job_id = jobs[0]["id"]

resp = urllib.request.urlopen(f"http://127.0.0.1:8081/jobs")
all_jobs = json.loads(resp.read())
job = next((j for j in all_jobs if j["id"] == job_id), None)

if not job:
    print("Job not found")
    exit()

print("=== JOB STATUS ===")
print(f"Status: {job['status']}")
print(f"Hook mode: {job.get('hook_mode')}")
print(f"Groups: {job.get('num_output_groups')}")

ci = job.get("content_identity") or {}
print(f"\n=== ContentIdentity ===")
print(json.dumps(ci, indent=2)[:1000])

reel_plan = job.get("reel_plan") or {}
plan_groups = reel_plan.get("reel_groups", [])
print(f"\n=== Reel Plan: {len(plan_groups)} groups ===")

for g in plan_groups:
    gi = g.get("group_index", "?")
    clips = g.get("source_clips", [])
    dur = g.get("estimated_duration_seconds", 0)
    narr = g.get("narration_events", [])
    
    print(f"\n--- Group {gi}: {len(clips)} clips, {dur:.1f}s ---")
    
    for c in clips:
        beat = c.get("_beat", "MISSING")
        is_hook = c.get("is_hook_clip", False)
        reason = c.get("reason", "")[:100]
        start = c.get("source_start", 0)
        end = c.get("source_end", 0)
        print(f"  {start:.1f}-{end:.1f}s beat={beat} is_hook={is_hook} reason={reason}")
    
    print(f"  Narration ({len(narr)} events):")
    for e in narr:
        etype = e.get("event_type", "?")
        text = e.get("text", "")[:80]
        print(f"    [{etype}] {text}")

# Check outputs
outputs = job.get("outputs", [])
print(f"\n=== Outputs: {len(outputs)} files ===")
for o in outputs:
    print(f"  {o.get('path', '?')[-60:]}")
