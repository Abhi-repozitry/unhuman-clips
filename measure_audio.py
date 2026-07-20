import subprocess
JID = "5fa05a65-3b1a-4669-aeb4-a3282b80867e"
W = f"backend/storage/working/{JID}/group_0_mixed_audio.wav"
NARR = [(3.0, 6.0), (7.0, 10.0)]
NO_NARR = [(0.0, 2.0), (11.0, 12.8)]
FFMPEG = 'C:\\Projects\\unhuman-clips\\ffmpeg\\ffmpeg-8.1.2-full_build\\bin\\ffmpeg.exe'
def measure(label, start, dur, wav):
    p = subprocess.run([FFMPEG,'-loglevel','info','-ss',str(start),'-i',wav,'-t',str(dur),'-af','volumedetect','-f','null','-'], capture_output=True, text=True)
    out = (p.stderr or '') + (p.stdout or '')
    print(f"=== {label} [{start}-{round(start+dur,1)}s] wav={wav.split('/')[-1]} ===")
    for ln in out.splitlines():
        if 'mean_volume' in ln or 'max_volume' in ln:
            print('  ', ln.strip())
for s,e in NARR:
    measure('NARR', s, e-s, W)
for s,e in NO_NARR:
    measure('NO_NARR', s, e-s, W)
WC = f"backend/storage/working/{JID}/group_0_clip_audio.wav"
for s,e in NARR:
    measure('NARR-ORIGCLIP', s, e-s, WC)
for s,e in NO_NARR:
    measure('NO_NARR-ORIGCLIP', s, e-s, WC)
