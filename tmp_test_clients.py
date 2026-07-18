import yt_dlp
for client in ['ios', 'mweb', 'tv_embedded', 'android', 'web_safari', 'web_creator', 'mediaconnect']:
    try:
        opts = {
            'quiet': True,
            'no_warnings': True,
            'simulate': True,
            'extractor_args': {'youtube': {'player_client': [client]}},
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info('https://youtu.be/fKoAOWQHP0o?si=DQnLjASXQaKnOwD6', download=False)
            t = (info.get('title') or '')[:60]
            print(f'CLIENT={client}: OK title={t!r} duration={info.get("duration")}')
            break
    except Exception as e:
        msg = str(e)[:140]
        print(f'CLIENT={client}: FAIL {msg}')
