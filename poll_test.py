import requests
import time
import json

job_id = '3bd17c5d-c112-4458-8dad-2288ff97645a'
last_log = None

for i in range(200):
    r = requests.get('http://127.0.0.1:9000/jobs', timeout=30)
    jobs = r.json()
    match = [j for j in jobs if j.get('id') == job_id]
    if not match:
        print('Job gone from queue')
        break
    j = match[0]
    status = j.get('status')
    progress = j.get('progress', 0)
    stage = j.get('stage_index', 0)
    stage_data = j.get('stage_data', {})
    error = j.get('error')
    logs = j.get('logs', [])
    msg = stage_data.get('message', '')
    print(f'[{status}] Stage {stage}: {progress:.0f}%  {msg}')
    if logs and logs[-1] != last_log:
        log_msg = logs[-1].encode('ascii', 'replace').decode('ascii')
        print(f'  LOG: {log_msg}')
        last_log = logs[-1]
    if status == 'DONE':
        print('\n=== JOB DONE ===')
        print(f'Output: {j.get("output_path", "")}')
        print(f'Duration: {stage_data.get("final_duration", "?")}s')
        outputs = j.get('outputs', [])
        for o in outputs:
            print(f'  Group {o["output_index"]}: {o.get("output_path", "")} ({o.get("duration_seconds", 0):.1f}s)')
        # Count generated files
        import os
        from pathlib import Path
        working = Path('backend/storage/working') / job_id
        if working.exists():
            audio_files = list(working.glob('*.wav'))
            caption_files = list(working.glob('*.ass'))
            clip_files = list((working / 'clips').glob('*.mp4')) if (working / 'clips').exists() else []
            print(f'\n=== GENERATED FILE COUNTS ===')
            print(f'  Audio files (narration): {len(audio_files)}')
            print(f'  Caption files: {len(caption_files)}')
            print(f'  Cut clips: {len(clip_files)}')
        break
    elif status == 'ERROR':
        print('\n=== ERROR ===')
        print(f'Error: {error}')
        print(f'Stage data: {json.dumps(stage_data, indent=2)}')
        break
    time.sleep(3)