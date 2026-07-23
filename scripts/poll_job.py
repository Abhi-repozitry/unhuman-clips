import requests
import json
import time

job_id = '29043868-1aa2-4e29-bf98-99a1701ecd77'
for i in range(120):
    response = requests.get(f'http://127.0.0.1:9000/jobs/{job_id}')
    job = response.json()
    stage = job.get('current_stage')
    status = job.get('status')
    progress = job.get('progress')
    print(f'Stage: {stage}, Status: {status}, Progress: {progress}%')
    if job.get('logs'):
        for log in job['logs'][-3:]:
            print(f'  LOG: {log}')
    if job.get('stage_data'):
        print(f'  Stage data: {job["stage_data"]}')
    if job.get('error'):
        print(f'  ERROR: {job["error"]}')
        break
    if job.get('status') == 'DONE':
        print('JOB COMPLETE!')
        break
    time.sleep(5)