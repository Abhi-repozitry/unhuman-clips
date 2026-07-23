import requests
r = requests.get('http://127.0.0.1:9000/jobs', timeout=10)
jobs = r.json()
print(len(jobs))
for j in jobs:
    print(j.get('id'), j.get('status'), j.get('progress'))