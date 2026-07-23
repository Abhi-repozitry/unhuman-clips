import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from backend.config import OUTPUTS_DIR
from backend.models import VideoJob
from backend.queue_manager import QueueManager
import os
from pathlib import Path


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, job: VideoJob):
        message = job.model_dump_json()
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


connection_manager = ConnectionManager()
queue_manager: QueueManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue_manager
    loop = asyncio.get_event_loop()
    queue_manager = QueueManager(loop)
    asyncio.create_task(queue_manager.worker(connection_manager.broadcast))
    yield


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


@app.post("/jobs")
async def create_job(body: CreateJobRequest):
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Respond to client pings to keep connection alive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception:
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
