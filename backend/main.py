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
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import OUTPUTS_DIR
from backend.logging_config import setup_logging
from backend.models import VideoJob
from backend.queue_manager import QueueManager

__all__ = ["app"]

logger = logging.getLogger(__name__)

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
        except asyncio.TimeoutError:
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
    if not _check_rate_limit():
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Try again later."},
        )
    job = queue_manager.add_job(body.url)
    return job


@app.get("/jobs")
async def list_jobs():
    return queue_manager.get_jobs()


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if queue_manager.delete_job(job_id):
        return {"ok": True}
    return JSONResponse(status_code=404, content={"error": "job not found"})


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

    # Check NVIDIA API key
    nvidia_key_ok = False
    try:
        from backend.config import NVIDIA_API_KEY
        nvidia_key_ok = bool(NVIDIA_API_KEY)
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
            "nvidia_api_key_configured": nvidia_key_ok,
            "active_websocket_connections": len(connection_manager.active_connections),
        },
    }


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
    file_path = FRONTEND_DIR / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
