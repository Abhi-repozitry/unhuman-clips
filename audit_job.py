import json, urllib.request

resp = urllib.request.urlopen("http://127.0.0.1:8081/jobs")
jobs = json.loads(resp.read())
job = jobs[-1]

print(f"=== JOB ===")
print(f"ID: {job['id']}")
print(f"Status: {job['status']}")
print(f"Title: {job.get('title')}")
print(f"Hook mode: {job.get('hook_mode')}")
print(f"Groups: {job.get('num_output_groups')}")
print(f"Multimodal: {job.get('multimodal_enabled', '?')}")

rp = job.get("reel_plan") or {}
groups = rp.get("reel_groups", [])

print(f"\n=== GROUPS: {len(groups)} ===")
for g in groups:
    gi = g.get("group_index", "?")
    clips = g.get("source_clips", [])
    est = g.get("estimated_duration_seconds", 0)
    rs = g.get("reel_summary", {})
    title = rs.get("title", "?")
    print(f"\n--- Group {gi}: {len(clips)} clips, {est:.1f}s ---")
    print(f"  Title: {title}")
    
    # Check clip ordering
    prev_end = -1
    order_ok = True
    for i, c in enumerate(clips):
        cs = c.get("source_start", 0)
        ce = c.get("source_end", 0)
        beat = c.get("beat") or c.get("_beat", "?")
        hook = c.get("is_hook_clip", False)
        reason = c.get("reason", "")[:60]
        
        if cs < prev_end:
            order_ok = False
        prev_end = ce
        
        marker = " [HOOK]" if hook else ""
        print(f"  [{i}] {cs:.1f}-{ce:.1f} beat={beat}{marker} reason={reason}")
    
    if not order_ok:
        print(f"  *** ORDER BUG: clips are NOT in chronological order! ***")
    else:
        print(f"  OK: clips are in chronological order")

# Check entity grouping
entity_grouped = rp.get("entity_grouped", False)
entity_segs = rp.get("entity_segments", [])
print(f"\n=== ENTITY ===")
print(f"entity_grouped: {entity_grouped}")
print(f"entity_segments: {len(entity_segs)}")

# Check content identity
ci = rp.get("content_identity") or {}
print(f"\n=== CONTENT IDENTITY ===")
print(f"creator: {ci.get('creator_name', '?')}")
print(f"format: {ci.get('content_format', '?')}")
print(f"genre: {ci.get('detected_genre', '?')}")
print(f"structure: {ci.get('structure', '?')}")
print(f"entities: {ci.get('entity_names', [])}")

# Check multimodal
ms = rp.get("multimodal_signals") or {}
print(f"\n=== MULTIMODAL ===")
print(f"scene_cuts: {len(ms.get('scene_cut_at', []))}")
print(f"ocr_texts: {len(ms.get('on_screen_text', []))}")

# Check narrations
print(f"\n=== NARRATION EVENTS ===")
for g in groups:
    gi = g.get("group_index", "?")
    events = g.get("narration_events", [])
    types = [e.get("event_type", "?") for e in events]
    print(f"  Group {gi}: {types}")

# Check for cross-group overlap
print(f"\n=== CROSS-GROUP OVERLAP CHECK ===")
all_ranges = []
for g in groups:
    gi = g.get("group_index", "?")
    for c in g.get("source_clips", []):
        cs = c.get("source_start", 0)
        ce = c.get("source_end", 0)
        all_ranges.append((cs, ce, gi))

all_ranges.sort()
for i in range(len(all_ranges)):
    for j in range(i+1, len(all_ranges)):
        r0, r1, g0 = all_ranges[i]
        s0, s1, g1 = all_ranges[j]
        overlap = max(0, min(r1, s1) - max(r0, s0))
        if overlap > 0.1 and g0 != g1:
            print(f"  OVERLAP: Group {g0} [{r0:.1f}-{r1:.1f}] vs Group {g1} [{s0:.1f}-{s1:.1f}] = {overlap:.1f}s")
