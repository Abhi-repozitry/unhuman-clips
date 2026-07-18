import requests
import time

BASE = 'http://127.0.0.1:8000'
r = requests.get(f'{BASE}/jobs', timeout=30)
jobs = r.json()
for j in jobs:
    print(f"Job {j.get('id')}: status={j.get('status')}, progress={j.get('progress')}, error={j.get('error')}")