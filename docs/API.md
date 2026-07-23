# API Reference

Base URL: `http://127.0.0.1:9000`

---

## Health Check

### `GET /health`

Returns system health status including disk space, memory, GPU, and dependency checks.

**Response** `200 OK`

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "checks": {
    "disk": "OK: 10.5 GB free",
    "memory": "OK: 8.2 GB available",
    "gpu": "OK: NVIDIA GeForce RTX 3060",
    "ffmpeg": "OK",
    "yt_dlp": "OK",
    "whisper": "OK",
    "torch_cuda": "OK: CUDA 12.1"
  }
}
```

---

## Jobs

### `POST /jobs`

Create a new video processing job.

**Request Body**

```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "options": {
    "voice": "en-US-ChristopherNeural",
    "target_groups": 3
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `string` | Yes | — | YouTube video URL |
| `options.voice` | `string` | No | `en-US-ChristopherNeural` | Edge-TTS voice for narration |
| `options.target_groups` | `int` | No | auto | Number of reels to generate (auto-scaled by duration) |

**Response** `202 Accepted`

```json
{
  "job_id": "a1b2c3d4",
  "status": "queued",
  "message": "Job queued for processing"
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| `400` | Invalid URL or missing required fields |
| `429` | Queue is full |
| `500` | Internal server error |

---

### `GET /jobs`

List all jobs with current status.

**Query Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | `int` | `20` | Max jobs to return |
| `offset` | `int` | `0` | Pagination offset |

**Response** `200 OK`

```json
{
  "jobs": [
    {
      "job_id": "a1b2c3d4",
      "url": "https://www.youtube.com/watch?v=...",
      "status": "processing",
      "progress": 65,
      "stage": "Clipping group 2/4",
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:01:30Z",
      "error": null,
      "duration_seconds": null,
      "output_files": []
    }
  ],
  "total": 15
}
```

**Status Values**

| Status | Description |
|--------|-------------|
| `queued` | Waiting in queue for processing |
| `downloading` | Downloading video with yt-dlp |
| `transcribing` | Whisper transcription in progress |
| `analyzing` | LLM reel plan generation |
| `processing` | Per-group pipeline (clip/TTS/caption/compose) |
| `completed` | All reels generated successfully |
| `failed` | Processing failed (see `error` field) |
| `cancelled` | User cancelled the job |

---

### `DELETE /jobs/{job_id}`

Cancel and remove a job. Stops processing if currently running.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | `string` | Job identifier |

**Response** `200 OK`

```json
{
  "message": "Job a1b2c3d4 removed",
  "status": "cancelled"
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| `404` | Job not found |

---

## WebSocket

### `WS /ws`

Real-time progress updates for all jobs. Connect with a WebSocket client.

**Protocol**

1. Server sends JSON messages as jobs progress
2. Client should send periodic pings to keep connection alive
3. Server responds to pings with `{"type": "pong"}`

**Message Types**

#### Job Created

```json
{
  "type": "job_created",
  "job_id": "a1b2c3d4",
  "url": "https://www.youtube.com/watch?v=...",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Progress Update

```json
{
  "type": "job_progress",
  "job_id": "a1b2c3d4",
  "progress": 65,
  "stage": "Clipping group 2/4",
  "message": "Cutting clip 3/5 for group 2",
  "timestamp": "2024-01-01T12:01:30Z"
}
```

#### Job Completed

```json
{
  "type": "job_completed",
  "job_id": "a1b2c3d4",
  "output_files": [
    "/storage/outputs/a1b2c3d4/group_0_final.mp4",
    "/storage/outputs/a1b2c3d4/group_1_final.mp4"
  ],
  "duration_seconds": 120,
  "timestamp": "2024-01-01T12:05:00Z"
}
```

#### Job Failed

```json
{
  "type": "job_failed",
  "job_id": "a1b2c3d4",
  "error": "Download failed: video unavailable",
  "timestamp": "2024-01-01T12:00:30Z"
}
```

#### LLM Interaction

```json
{
  "type": "llm_interaction",
  "job_id": "a1b2c3d4",
  "interaction": {
    "timestamp": "2024-01-01T12:01:00Z",
    "type": "response",
    "role": "assistant",
    "content": "Preview of LLM response...",
    "full_content": "Full response text...",
    "model": "openai/gpt-oss-120b",
    "token_count": "1500 out / 45000 in",
    "stage_name": "reel_plan"
  }
}
```

#### Pong

```json
{
  "type": "pong"
}
```

---

## Static Files

### `GET /*`

Serves the frontend Electron app static files. All unmatched routes fall through to the frontend router.

- `/` → `frontend/renderer/index.html`
- `/*.js`, `/*.css`, `/*.png` → Static assets from `frontend/renderer/`

---

## Notes

### Rate Limiting

- **Per-IP limit**: 30 requests per 60-second window
- Exceeding the limit returns `429 Too Many Requests`
- Rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### CORS

Default configuration allows all origins (`*`). For production, restrict to your domain.

### Job Timeout

Jobs are automatically cleaned up after `STALE_FILE_CLEANUP_HOURS` (default: 24 hours). Stale working directories and outputs are removed.
