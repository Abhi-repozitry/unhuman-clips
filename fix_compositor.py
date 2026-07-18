with open(r'C:\Projects\unhuman-clips\backend\pipeline\compositor.py', 'r') as f:
    content = f.read()

# Replace all standalone "ffmpeg" command strings with FFMPEG_PATH
content = content.replace('"ffmpeg"', 'FFMPEG_PATH')
content = content.replace("'ffmpeg'", 'FFMPEG_PATH')
content = content.replace('ffmpeg",', 'FFMPEG_PATH,')
content = content.replace("ffmpeg',", "FFMPEG_PATH,")

with open(r'C:\Projects\unhuman-clips\backend\pipeline\compositor.py', 'w') as f:
    f.write(content)

print('Done')