import yt_dlp
import time
from pathlib import Path
import shutil
from backend.config import DOWNLOAD_MAX_HEIGHT


def download_video(url: str, out_dir: str, progress_hook) -> dict:
    """
    Download a video into a job-specific directory to avoid reusing stale files.
    Returns yt-dlp's extract_info result plus a `source_path` field pointing to the downloaded media.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Use yt-dlp's default filename template, but keep everything isolated to this out_dir.
    # Overwrite to avoid partially downloaded stale files.
    outtmpl = str(out_dir_path / "%(id)s.%(ext)s")

    result = {}
    last_progress_time = time.time()

    def yt_hook(d):
        nonlocal last_progress_time
        # Always push finished immediately
        if d['status'] == 'finished':
            progress_hook({
                "status": "finished",
                "downloaded_bytes": d.get("total_bytes", 0) or d.get("downloaded_bytes", 0),
                "total_bytes": d.get("total_bytes", 0),
                "speed": 0,
                "eta": 0,
            })
            return

        if d['status'] == 'downloading':
            downloaded = d.get('downloaded_bytes', 0) or 0
            total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            speed = d.get('speed', 0) or 0
            eta = d.get('eta')

            # Throttle to ~3 updates/sec so frontend isn't flooded
            now = time.time()
            if now - last_progress_time < 0.3 and speed > 0:
                return
            last_progress_time = now

            progress_hook({
                "status": "downloading",
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "speed": speed,
                "eta": eta,
            })

    ffmpeg_available = shutil.which("ffmpeg") is not None

    # IMPORTANT:
    # If ffmpeg isn't available, yt-dlp cannot mux separate video+audio streams.
    # In that case we try hard to download an already-muxed MP4. If yt-dlp still
    # selects split streams, we fail with a clear error instead of continuing
    # with stale/invalid artifacts.
    if ffmpeg_available:
        format_selector = f"bestvideo[height<={DOWNLOAD_MAX_HEIGHT}]+bestaudio/best[height<={DOWNLOAD_MAX_HEIGHT}]/best"
        postprocessors = None
    else:
        # Try a progressive MP4 that already contains both video+audio.
        # If none exists, yt-dlp may still fall back to split formats; we'll detect
        # that after download and raise a clear error.
        format_selector = f"best[ext=mp4][height<={DOWNLOAD_MAX_HEIGHT}][vcodec!=none][acodec!=none]/best[ext=mp4][height<={DOWNLOAD_MAX_HEIGHT}]/best[height<={DOWNLOAD_MAX_HEIGHT}]/best"
        postprocessors = []

    # Cookie file for YouTube auth (if exists)
    cookie_file = Path(__file__).resolve().parent.parent.parent / "cookies.txt"
    ydl_opts = {
        "format": format_selector,
        "outtmpl": outtmpl,
        "writeinfojson": True,
        "progress_hooks": [yt_hook],
        "no_color": True,
        # Avoid resuming/using stale partials when rerunning the same job URL
        "continue_dl": False,
        "overwrites": True,
        # Force extracting full info so we get title, description, etc.
        "extract_flat": False,
    }

    if cookie_file.exists():
        ydl_opts["cookiefile"] = str(cookie_file)

    if postprocessors is not None:
        ydl_opts["postprocessors"] = postprocessors

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
    except Exception as e:
        raise RuntimeError(str(e)) from e

    if not result:
        raise RuntimeError("yt-dlp returned no result")

    # Derive the actual downloaded file path.
    source_path = None
    filepath = result.get("requested_downloads", None)
    if isinstance(filepath, list) and filepath:
        last = filepath[-1]
        source_path = last.get("filepath") or last.get("filepath_unresolved")
    if not source_path:
        source_path = result.get("filepath") or result.get("_filename")

    if source_path:
        source_path = str(source_path)

    # Fallback: locate the only media file in out_dir matching the id/ext naming.
    if not source_path:
        video_id = result.get("id")
        ext = result.get("ext") or "mp4"
        candidate = out_dir_path / f"{video_id}.{ext}"
        if candidate.exists():
            source_path = str(candidate)

    if not source_path:
        raise RuntimeError(
            "Could not determine downloaded media file path from yt-dlp result."
        )

    # If ffmpeg is missing, ensure we actually got a single muxed media file.
    if not ffmpeg_available and not str(source_path).lower().endswith(".mp4"):
        raise RuntimeError(
            "ffmpeg is not installed/available, and yt-dlp did not produce a muxed MP4. "
            "Install ffmpeg (required to merge separate video/audio streams) or adjust the environment "
            "so yt-dlp can download a progressive MP4."
        )

    result["source_path"] = source_path
    return result
