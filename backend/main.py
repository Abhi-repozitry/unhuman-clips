"""FastAPI application — Unhuman Clips backend.

Provides REST endpoints for job management, WebSocket real-time updates,
and a static file server for the frontend and output videos.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import OUTPUTS_DIR
from backend.logging_config import setup_logging
from backend.models import HookMode, SourceMetadata, VideoJob
from backend.pipeline.downloader import fetch_video_metadata
from backend.queue_manager import QueueManager

__all__ = ["app"]

logger = logging.getLogger(__name__)

# Model provider: opencode only
SELECTED_MODEL_PROVIDER = "opencode"

# Caption generation: True = generate, False = skip
SELECTED_CAPTIONS = True

# Rate limiter state: timestamps of recent /jobs POST requests (global, in-memory)
_job_request_times: deque[float] = deque()
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = 10     # max requests per window


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug("[WS] Client connected (total: %d)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.debug("[WS] Client disconnected (total: %d)", len(self.active_connections))

    async def broadcast(self, job: VideoJob):
        message = job.model_dump_json()
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.debug("[WS] Failed to send to client: %s", e)
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

    async def heartbeat_loop(self):
        """Send periodic heartbeats to detect dead connections."""
        while True:
            await asyncio.sleep(15)
            if not self.active_connections:
                continue
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text('{"type":"heartbeat"}')
                except Exception:
                    disconnected.append(connection)
            for conn in disconnected:
                self.disconnect(conn)


connection_manager = ConnectionManager()
queue_manager: QueueManager | None = None
_worker_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — start worker on startup, cancel cleanly on shutdown."""
    global queue_manager, _worker_task

    setup_logging()
    logger.info("Starting Unhuman Clips backend...")
    loop = asyncio.get_event_loop()
    queue_manager = QueueManager(loop)

    # Start the broadcast drain loop (event-loop-driven, no thread-unsafe calls)
    _drain_task = asyncio.create_task(
        queue_manager.broadcast_drain_loop(connection_manager.broadcast),
        name="broadcast_drain",
    )
    logger.info("Broadcast drain loop started")

    # Start the worker as a background task
    _worker_task = asyncio.create_task(
        queue_manager.worker(connection_manager.broadcast),
        name="queue_worker",
    )
    logger.info("Worker started (task=%s)", _worker_task.get_name())

    # Start the heartbeat task for WebSocket dead-connection detection
    heartbeat_task = asyncio.create_task(
        connection_manager.heartbeat_loop(),
        name="ws_heartbeat",
    )
    logger.info("WebSocket heartbeat started")

    yield

    # Graceful shutdown
    logger.info("Shutting down gracefully...")
    heartbeat_task.cancel()
    _drain_task.cancel()
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await asyncio.wait_for(_worker_task, timeout=30.0)
        except asyncio.CancelledError:
            logger.info("Worker task cancelled successfully.")
        except TimeoutError:
            logger.warning("Worker task did not finish within 30s timeout.")

    logger.info("Shutdown complete.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "renderer"


class CreateJobRequest(BaseModel):
    url: str
    generate_captions: bool = True
    hook_mode: HookMode = "auto"


class PreflightJobRequest(BaseModel):
    url: str


class PreflightJobResponse(BaseModel):
    source_metadata: SourceMetadata
    suggested_hook_mode: HookMode = "required"


def _check_rate_limit() -> bool:
    """Check if the global rate limit has been exceeded.

    Returns True if the request is allowed, False if rate limited.
    """
    now = time.monotonic()
    # Prune timestamps outside the window
    while _job_request_times and _job_request_times[0] < now - _RATE_LIMIT_WINDOW:
        _job_request_times.popleft()
    if len(_job_request_times) >= _RATE_LIMIT_MAX:
        return False
    _job_request_times.append(now)
    return True


@app.post("/jobs")
async def create_job(body: CreateJobRequest):
    """Create a new processing job.

    Rate limited to {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW} seconds.
    """
    if queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Server starting up, try again shortly."})
    if not _check_rate_limit():
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Try again later."},
        )
    job = queue_manager.add_job(
        body.url,
        generate_captions=body.generate_captions,
        hook_mode=body.hook_mode,
    )
    return job


@app.post("/jobs/preflight")
async def preflight_job(body: PreflightJobRequest) -> PreflightJobResponse:
    """Fetch source metadata before a user commits a job to the queue."""
    if not _check_rate_limit():
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    try:
        metadata = await asyncio.to_thread(fetch_video_metadata, body.url)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PreflightJobResponse(source_metadata=SourceMetadata.model_validate(metadata))


@app.get("/jobs")
async def list_jobs():
    if queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Server starting up, try again shortly."})
    return queue_manager.get_jobs()


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Server starting up, try again shortly."})
    if queue_manager.delete_job(job_id):
        return {"ok": True}
    return JSONResponse(status_code=404, content={"error": "job not found"})


@app.get("/jobs/{job_id}/checkpoints")
async def list_checkpoints(job_id: str):
    """List completed checkpoint stages for a job."""
    if queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Server starting up, try again shortly."})
    job = queue_manager.jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    from backend.config import get_job_working_dir
    from backend.pipeline.checkpoint import PipelineCheckpoint
    ckpt = PipelineCheckpoint(get_job_working_dir(job_id))
    stages = ckpt.list_stages()
    return {"job_id": job_id, "checkpoints": stages}


class RetryRequest(BaseModel):
    from_stage: str | None = None  # None = full retry, str = retry from this stage


STAGE_ORDER = ["download", "transcribe", "rich_timeline", "multimodal", "analyze"]


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, body: RetryRequest = None):
    """Retry a failed job. Optionally delete checkpoints after a given stage to force re-processing."""
    if queue_manager is None:
        return JSONResponse(status_code=503, content={"error": "Server starting up, try again shortly."})
    job = queue_manager.jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    if job.status not in ("ERROR", "DONE"):
        return JSONResponse(status_code=400, content={"error": "can only retry ERROR or DONE jobs"})

    from backend.config import get_job_working_dir
    from backend.pipeline.checkpoint import PipelineCheckpoint
    ckpt = PipelineCheckpoint(get_job_working_dir(job_id))

    from_stage = body.from_stage if body else None
    if from_stage:
        # Delete this stage and all later stages to force re-processing
        stages_to_delete = []
        found = False
        for s in STAGE_ORDER:
            if s == from_stage:
                found = True
            if found:
                stages_to_delete.append(s)
        # Also delete per-group checkpoints if retrying from a group stage
        if from_stage.startswith("group_"):
            for s in ckpt.list_stages():
                if s.startswith(from_stage.split("_clips")[0]):
                    stages_to_delete.append(s)
        for s in stages_to_delete:
            ckpt.clear_stage(s)
        logger.info("Retry %s from stage '%s': deleted checkpoints %s", job_id, from_stage, stages_to_delete)
    else:
        # Full retry: clear all checkpoints
        count = ckpt.cleanup()
        logger.info("Full retry %s: cleared %d checkpoints", job_id, count)

    # Reset job status and re-enqueue
    job.status = "QUEUED"
    job.error = None
    job.progress = 0.0
    job.stage_index = 0
    job.stage_data = {}
    queue_manager.queue.put_nowait(job_id)
    queue_manager.enqueue_broadcast(job)
    return {"ok": True, "from_stage": from_stage}


@app.get("/health")
async def health_check():
    """Health check endpoint showing queue status and system info."""
    jobs = queue_manager.get_jobs() if queue_manager else []
    active_jobs = [j for j in jobs if j.status not in ("DONE", "ERROR", "QUEUED")]
    queued_jobs = [j for j in jobs if j.status == "QUEUED"]
    error_jobs = [j for j in jobs if j.status == "ERROR"]

    # Check ffmpeg availability
    ffmpeg_ok = False
    try:
        from backend.ffmpeg_utils import get_ffmpeg, get_ffprobe
        get_ffmpeg()
        get_ffprobe()
        ffmpeg_ok = True
    except RuntimeError:
        pass

    # Check API keys
    opencode_key_ok = False
    try:
        from backend.config import OPENCODE_API_KEY
        opencode_key_ok = bool(OPENCODE_API_KEY)
    except Exception:
        pass

    return {
        "status": "healthy",
        "queue": {
            "total_jobs": len(jobs),
            "queued": len(queued_jobs),
            "active": len(active_jobs),
            "completed": len([j for j in jobs if j.status == "DONE"]),
            "errors": len(error_jobs),
        },
        "system": {
            "ffmpeg_available": ffmpeg_ok,
            "opencode_api_key_configured": opencode_key_ok,
            "active_websocket_connections": len(connection_manager.active_connections),
        },
    }


# --- Mode config endpoints ---

class ModeRequest(BaseModel):
    mode: str  # "quality" or "fast"


@app.get("/config/mode")
async def get_mode():
    """Return current processing mode."""
    import backend.config as cfg
    return {"mode": "fast" if cfg.FAST_MODE else "quality"}


@app.put("/config/mode")
async def set_mode(body: ModeRequest):
    """Switch between quality and fast processing modes."""
    import backend.config as cfg
    if body.mode not in ("quality", "fast"):
        return JSONResponse(status_code=400, content={"error": "mode must be 'quality' or 'fast'"})
    cfg.FAST_MODE = body.mode == "fast"
    logger.info("Processing mode set to: %s", body.mode)
    return {"mode": body.mode, "fast_mode": cfg.FAST_MODE}


class CaptionRequest(BaseModel):
    generate: bool


@app.get("/config/caption")
async def get_caption():
    """Return current caption setting."""
    return {"generate": SELECTED_CAPTIONS}


@app.put("/config/caption")
async def set_caption(body: CaptionRequest):
    """Switch between generate and skip captions."""
    global SELECTED_CAPTIONS
    SELECTED_CAPTIONS = body.generate
    logger.info("Caption generation set to: %s", body.generate)
    return {"generate": body.generate}


class ProviderRequest(BaseModel):
    provider: str  # "mimo"


@app.get("/config/provider")
async def get_provider():
    """Return current AI provider setting."""
    import backend.config as cfg
    return {"provider": cfg.AI_PROVIDER, "model": cfg.OPENCODE_MODEL}


@app.put("/config/provider")
async def set_provider(body: ProviderRequest):
    """Set AI provider."""
    import backend.config as cfg
    if body.provider != "mimo":
        return JSONResponse(status_code=400, content={"error": "provider must be 'mimo'"})
    cfg.AI_PROVIDER = body.provider
    logger.info("AI provider set to: %s (model: %s)", body.provider, cfg.OPENCODE_MODEL)
    return {"provider": body.provider, "model": cfg.OPENCODE_MODEL}


class OcrRequest(BaseModel):
    mode: str  # "keep" or "skip"


@app.get("/config/ocr")
async def get_ocr():
    """Return current OCR mode setting."""
    import backend.config as cfg
    return {"mode": cfg.OCR_MODE}


@app.put("/config/ocr")
async def set_ocr(body: OcrRequest):
    """Switch between keep and skip OCR modes."""
    import backend.config as cfg
    if body.mode not in ("keep", "skip"):
        return JSONResponse(status_code=400, content={"error": "mode must be 'keep' or 'skip'"})
    cfg.OCR_MODE = body.mode
    logger.info("OCR mode set to: %s", body.mode)
    return {"mode": body.mode}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Respond to client pings to keep connection alive
            if data == "ping":
                await websocket.send_text("pong")
            # Ignore server heartbeats (client should not send these, but handle gracefully)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.debug("[WS] Connection error: %s", e)
        connection_manager.disconnect(websocket)


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    file_path = (FRONTEND_DIR / full_path).resolve()
    if not str(file_path).startswith(str(FRONTEND_DIR.resolve())):
        return JSONResponse(status_code=403, content={"error": "forbidden"})
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
