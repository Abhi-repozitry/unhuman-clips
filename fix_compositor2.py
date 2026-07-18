with open(r'C:\Projects\unhuman-clips\backend\pipeline\compositor.py', 'r') as f:
    content = f.read()

# Fix line 7 - it should use the actual path, not FFMPEG_PATH variable
content = content.replace(
    'FFMPEG_PATH = str(Path(__file__).resolve().parent.parent.parent / FFMPEG_PATH / "ffmpeg-8.1.2-full_build" / "bin" / "ffmpeg.exe")',
    'FFMPEG_PATH = str(Path(__file__).resolve().parent.parent.parent / "ffmpeg" / "ffmpeg-8.1.2-full_build" / "bin" / "ffmpeg.exe")'
)

with open(r'C:\Projects\unhuman-clips\backend\pipeline\compositor.py', 'w') as f:
    f.write(content)

print('Fixed line 7')