import subprocess, json, glob

base = r"C:\Projects\unhuman-clips\backend\storage\working\160b024b-dd22-439d-868e-fff528dca198"
clips_dir = r"C:\Projects\unhuman-clips\backend\storage\clips"
ffprobe = r"C:\Projects\unhuman-clips\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffprobe.exe"

# Get all clip files for this job
clip_files = sorted(glob.glob(f"{clips_dir}/160b024b-dd22-439d-868e-fff528dca198_group*_clip_*.mp4"))

for f in clip_files:
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
        capture_output=True, text=True
    )
    dur = float(r.stdout.strip()) if r.stdout.strip() else 0
    name = f.split("\\")[-1]
    print(f"{name}: {dur:.1f}s")
