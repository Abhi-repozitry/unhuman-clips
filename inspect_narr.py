import requests, json
r = requests.get('http://127.0.0.1:9000/jobs', timeout=10).json()
for j in r:
    if j['id'] != '1fe173f5-dd52-4087-bd57-cd57394d31c2':
        continue
    for g in j.get('reel_plan', {}).get('reel_groups', []):
        print('GROUP', g['group_index'])
        print('  estimated_duration:', g['estimated_duration_seconds'])
        print('  key_moment:', g['reel_summary']['key_moment'])
        print('  narration_events:')
        for ev in g['narration_events']:
            if ev['event_type'] in ('hook', 'commentary'):
                print(f"    [{ev['reel_start']:.2f}-{ev['reel_end']:.2f}] {ev['event_type']}: {ev['text']}")
