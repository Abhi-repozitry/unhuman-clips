import subprocess, glob, json

base = r"C:\Projects\unhuman-clips\backend\storage\working\94b3907b-d898-4cfc-90f3-f83828a64f0e"
clips_dir = r"C:\Projects\unhuman-clips\backend\storage\clips"
ffprobe = r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffprobe.exe"

for i in range(8):
    clip_files = sorted(glob.glob(f"{clips_dir}/94b3907b*_group{i}_clip_*.mp4"))
    total_dur = 0.0
    for f in clip_files:
        r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f], capture_output=True, text=True)
        dur = float(r.stdout.strip()) if r.stdout.strip() else 0.0
        total_dur += dur
    
    tts_path = f"{base}/checkpoint_group_{i}_tts.json"
    try:
        with open(tts_path) as f:
            tts = json.load(f)
        max_nar = max((nar["reel_end"] for nar in tts.get("narration_audio", [])), default=0.0)
        nar_count = len(tts.get("narration_audio", []))
    except:
        max_nar = 0
        nar_count = 0
    
    status = "OK" if max_nar <= total_dur else "OVERFLOW!"
    print(f"Group {i}: clips={total_dur:.1f}s narration_end={max_nar:.1f}s ({nar_count} events) {status}")
