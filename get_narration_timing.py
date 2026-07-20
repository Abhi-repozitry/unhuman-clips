import requests
r = requests.get('http://127.0.0.1:9000/jobs', timeout=10)
for j in r.json():
    if j['id'] == '5fa05a65-3b1a-4669-aeb4-a3282b80867e':
        for g in j.get('reel_plan', {}).get('reel_groups', []):
            for ev in g.get('narration_events', []):
                if ev.get('event_type') in ('hook','commentary'):
                    print(f"  [{ev['reel_start']:.2f}-{ev['reel_end']:.2f}] {ev.get('event_type')}: {ev.get('text','')[:60]}")
