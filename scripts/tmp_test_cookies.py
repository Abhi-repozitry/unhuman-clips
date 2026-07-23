import yt_dlp
cf = r'C:\Users\starr\AppData\Roaming\unhuman-clips\cookies.txt'
opts = {
    'quiet': False,
    'no_warnings': False,
    'cookiefile': cf,
    'format': 'bestvideo[height<=1080][fps>=30]+bestaudio/best[height<=1080][fps>=30]/best',
}
for url in ['https://youtu.be/fKoAOWQHP0o', 'https://youtu.be/fKoAOWQHP0o?si=DQnLjASXQaKnOwD6']:
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            t = (info.get('title') or '')[:60]
            d = info.get('duration')
            print(f'URL={url} OK title={t!r} duration={d}')
            break
    except Exception as e:
        print(f'URL={url} FAIL {str(e)[:120]}')
