import json, urllib.request

jobs = json.load(open(r"C:\Projects\unhuman-clips\job_status.json"))
job_id = jobs[0]["id"]

resp = urllib.request.urlopen(f"http://127.0.0.1:8081/jobs")
all_jobs = json.loads(resp.read())
job = next((j for j in all_jobs if j["id"] == job_id), None)

reel_plan = job.get("reel_plan") or {}
plan_groups = reel_plan.get("reel_groups", [])

print("=== Group 2 detailed clip analysis ===")
g = plan_groups[2]
clips = g.get("source_clips", [])
print(f"Clips count: {len(clips)}")
for i, c in enumerate(clips):
    start = c.get("source_start", 0)
    end = c.get("source_end", 0)
    beat = c.get("_beat", "?")
    is_hook = c.get("is_hook_clip", False)
    reason = c.get("reason", "")
    print(f"  [{i}] {start:.1f}-{end:.1f}s beat={beat} is_hook={is_hook}")
    print(f"       reason={reason[:80]}")

print(f"\nExpected order (by source_start):")
sorted_clips = sorted(clips, key=lambda c: c.get("source_start", 0))
for i, c in enumerate(sorted_clips):
    start = c.get("source_start", 0)
    end = c.get("source_end", 0)
    reason = c.get("reason", "")[:60]
    print(f"  [{i}] {start:.1f}-{end:.1f}s reason={reason}")

print(f"\n=== Narrative arc check ===")
# Check if clips tell a coherent story when played in source order
print("Group 2 clips in source order:")
for c in sorted_clips:
    start = c.get("source_start", 0)
    reason = c.get("reason", "")[:80]
    print(f"  {start:.0f}s: {reason}")
