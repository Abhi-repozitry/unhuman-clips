import json

base = r"C:\Projects\unhuman-clips\backend\storage\working\94b3907b-d898-4cfc-90f3-f83828a64f0e"

with open(f"{base}/checkpoint_analyze.json") as f:
    data = json.load(f)

for g in data["reel_plan"]["reel_groups"]:
    clips = g["source_clips"]
    total = sum(c["source_end"] - c["source_start"] for c in clips)
    
    with open(f"{base}/checkpoint_group_{g['group_index']}_tts.json") as f:
        tts = json.load(f)
    max_nar = max((nar["reel_end"] for nar in tts.get("narration_audio", [])), default=0.0)
    
    status = "OK" if max_nar <= total else f"OVERFLOW by {max_nar - total:.1f}s"
    print(f"Group {g['group_index']}: source_clips={total:.1f}s narration_end={max_nar:.1f}s {status}")
