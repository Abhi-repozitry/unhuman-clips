import yt_dlp
import time
import os
from pathlib import Path
import shutil
from backend.config import DOWNLOAD_MAX_HEIGHT, FFMPEG_PATH


def download_video(url: str, out_dir: str, progress_hook, max_retries: int = 4) -> dict:
    """
    Download a video into a job-specific directory to avoid reusing stale files.
    Returns yt-dlp's extract_info result plus a `source_path` field pointing to the downloaded media.

    Retries up to max_retries times with exponential backoff for transient network errors
    (timeouts, connection resets, HTTP 5xx, rate limiting).
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

    # Check ffmpeg availability — prefer config path, then system PATH
    ffmpeg_available = os.path.isfile(FFMPEG_PATH) or shutil.which("ffmpeg") is not None

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

    # Cookie file for YouTube auth — check multiple standard locations
    root_dir = Path(__file__).resolve().parent.parent.parent
    desktop_dir = Path.home() / "Desktop"
    cookie_candidates = [
        root_dir / "cookies.txt",
        desktop_dir / "cookies.txt",
        desktop_dir / "antigravity.google_cookies.txt",
        root_dir / "backend" / "storage" / "cookies.txt",
        root_dir / "backend" / "cookies.txt",
    ]
    found_cookie = next((p for p in cookie_candidates if p.exists() and p.stat().st_size > 0), None)

    ydl_opts = {
        "format": format_selector,
        "outtmpl": outtmpl,
        "writeinfojson": True,
        "progress_hooks": [yt_hook],
        "no_color": True,
        # Avoid resuming/using stale partials when rerunning the same job URL
        "continue_dl": False,
        "overwrites": True,
        # Enable Node.js EJS challenge solver for YouTube JS challenges
        "remote_components": ["ejs:github"],
        "js_runtimes": {"node": {}},
        # Network robustness: socket timeout and retry settings
        "socket_timeout": 30,
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 3,
        "extractor_retries": 3,
    }

    # Point yt-dlp to our bundled ffmpeg if available
    if os.path.isfile(FFMPEG_PATH):
        ffmpeg_dir = str(Path(FFMPEG_PATH).parent)
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    if found_cookie:
        print(f"[INFO] Using YouTube cookie file: {found_cookie}")
        ydl_opts["cookiefile"] = str(found_cookie)

    if postprocessors is not None:
        ydl_opts["postprocessors"] = postprocessors

    # Retry loop with exponential backoff for transient network errors
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Download attempt {attempt}/{max_retries} for {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(url, download=True)
            last_error = None
            break  # success
        except Exception as e:
            last_error = e
            err_str = str(e).lower()

            if "sign in to confirm you're not a bot" in err_str or "cookies for the authentication" in err_str:
                print(f"[WARN] YouTube bot check triggered on attempt {attempt}.")
                # Non-retryable without cookies
                break

            # Classify as retryable or fatal
            retryable_keywords = [
                "timed out", "timeout", "connection reset",
                "connection refused", "connection aborted",
                "temporary failure", "429", "too many requests",
                "503", "502", "500", "server error",
                "network", "urlopen error", "httpsconnectionpool",
                "read timed out", "incompleteread",
            ]
            is_retryable = any(kw in err_str for kw in retryable_keywords)

            if is_retryable and attempt < max_retries:
                wait = min(2 ** attempt, 30)  # 2, 4, 8, 16, max 30s
                print(f"[WARN] Download attempt {attempt} failed (retryable): {str(e)[:200]}")
                print(f"[INFO] Retrying in {wait}s...")
                time.sleep(wait)
            else:
                error_type = "retryable" if is_retryable else "fatal"
                print(f"[ERROR] Download attempt {attempt} failed ({error_type}): {str(e)[:300]}")
                if not is_retryable:
                    break  # non-retryable error, don't keep trying

    if last_error is not None:
        err_msg = str(last_error)
        if "sign in to confirm you're not a bot" in err_msg.lower():
            target_cookie_path = root_dir / "cookies.txt"
            raise RuntimeError(
                "YouTube requires authentication cookies for this video.\n"
                "FIX: Export your YouTube cookies from your browser (e.g. using the 'Get cookies.txt LOCALLY' Chrome/Edge extension) "
                f"and save the exported file as:\n  {target_cookie_path}"
            ) from last_error

        raise RuntimeError(
            f"Download failed after {max_retries} attempts. Last error: {last_error}"
        ) from last_error

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

    print(f"[INFO] Download complete: {source_path}")
    result["source_path"] = source_path
    return result
