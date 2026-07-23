# Unhuman Clips

**Automated vertical short-form video clip generator from long YouTube videos.**

Paste a YouTube URL. Get multiple 90–180 second vertical reels with AI-selected clips, narrated hooks, commentary overlays, and ASS captions — all composited with VAD-driven audio ducking.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server (:9000)                   │
│  REST API  ·  WebSocket (real-time progress)  ·  Frontend   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ QueueManager │  ← Job queue + worker loop
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
  ┌─────▼─────┐    ┌───────▼───────┐    ┌─────▼─────┐
  │ Download   │    │  Transcribe   │    │  Analyze   │
  │ (yt-dlp)   │    │ (Whisper GPU) │    │ (NVIDIA    │
  │            │    │               │    │  LLM)      │
  └─────┬─────┘    └───────┬───────┘    └─────┬─────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │ ReelPlan    │  → N groups, each with clips + narration
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
        │ Group 0    │ │Group 1│ │  Group N   │  (GroupOrchestrator)
        │            │ │       │ │            │
        │ clip→TTS→  │ │  ...  │ │    ...     │
        │ caption→   │ │       │ │            │
        │ compose    │ │       │ │            │
        └─────┬─────┘ └───┬───┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │  Output MP4  │  → storage/outputs/
                    └─────────────┘
```

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11+ | 3.12 |
| NVIDIA GPU | Any CUDA-capable | RTX 3060+ (12 GB VRAM) |
| CUDA / cuDNN | 11.8+ | 12.1+ |
| ffmpeg | 6.0+ | 8.1+ (with NVENC support) |
| RAM | 8 GB | 16 GB+ |
| Disk | 5 GB free | SSD recommended |

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> && cd unhuman-clips

# 2. Create virtual environment
python -m venv backend/venv
# Linux/Mac:
source backend/venv/bin/activate
# Windows:
backend\venv\Scripts\activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Configure environment
cp backend/.env.example backend/.env   # or create manually
# Edit backend/.env — set at minimum:
#   NVIDIA_API_KEY=nvapi-...
#   FFMPEG_PATH=/path/to/ffmpeg

# 5. Start the server
# Linux/Mac:
./start.sh
# Windows:
start.bat
# Or directly:
uvicorn backend.main:app --reload --host 127.0.0.1 --port 9000

# 6. Open the UI
open http://127.0.0.1:9000
```

## Configuration

All settings are configured via environment variables in `backend/.env`. See [docs/CONFIG.md](docs/CONFIG.md) for the full reference.

**Essential variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | *(required)* | NVIDIA API key for LLM analysis |
| `FFMPEG_PATH` | `C:\Projects\...\ffmpeg.exe` | Path to ffmpeg binary |
| `TTS_VOICE` | `en-US-ChristopherNeural` | Edge-TTS voice for narration |
| `MAX_OUTPUT_DURATION` | `180` | Maximum reel duration (seconds) |
| `MIN_OUTPUT_DURATION` | `90` | Minimum reel duration (seconds) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Create a new processing job |
| `GET` | `/jobs` | List all jobs with status |
| `DELETE` | `/jobs/{job_id}` | Cancel and remove a job |
| `GET` | `/health` | System health check |
| `WS` | `/ws` | WebSocket for real-time progress updates |
| `GET` | `/*` | Serve frontend static files |

See [docs/API.md](docs/API.md) for full endpoint documentation.

## Pipeline Stages

1. **Download** — yt-dlp with retry logic and format selection
2. **Transcribe** — Whisper large-v3-turbo (GPU, lazy-loaded)
3. **Analyze** — NVIDIA LLM generates structured ReelPlan (clips + narration)
4. **Clip** — Parallel ffmpeg cutting with NVENC acceleration
5. **TTS** — Edge-TTS narration generation (hook + commentary)
6. **Caption** — ASS subtitle generation with keyword highlighting
7. **Composite** — VAD-driven audio ducking, freeze-frame padding, caption burn-in
8. **Output** — Duration capping, file staging to `storage/outputs/`

Each stage supports **checkpoint-based resumability** — if the process crashes, it resumes from the last completed stage.

## Running Tests

```bash
# Fast unit tests (~5 seconds)
pytest tests/ -m "not integration"

# All tests including integration
pytest tests/

# Smoke test (no pytest needed)
python scripts/smoke_test.py
```

## Project Structure

```
unhuman-clips/
├── backend/
│   ├── main.py                 # FastAPI app, endpoints, WebSocket
│   ├── config.py               # Environment variables, validation
│   ├── models.py               # Pydantic models (VideoJob, ReelPlan, etc.)
│   ├── queue_manager.py        # Job queue, worker loop, pipeline stages
│   ├── progress.py             # Thread-safe progress reporter
│   ├── output_manager.py       # Final edit, duration probe, staging
│   ├── ffmpeg_utils.py         # ffmpeg/ffprobe path resolution, encoder detection
│   ├── logging_config.py       # Structured logging setup
│   ├── pipeline/
│   │   ├── downloader.py       # yt-dlp download with retry
│   │   ├── transcriber.py      # Whisper transcription (GPU)
│   │   ├── analyzer.py         # LLM reel plan generation
│   │   ├── clipper.py          # Parallel clip cutting
│   │   ├── tts.py              # Edge-TTS narration
│   │   ├── captioner.py        # ASS subtitle generation
│   │   ├── compositor.py       # VAD ducking, video composition
│   │   ├── orchestrator.py     # Per-group pipeline orchestration
│   │   ├── checkpoint.py       # Checkpoint persistence
│   │   ├── narration_validator.py  # Timing overlap detection
│   │   └── sanitize.py         # Text sanitization
│   └── providers/
│       └── llm.py              # NVIDIA LLM API client
├── tests/                      # pytest test suite (143 tests)
├── scripts/                    # Utility scripts
├── frontend/                   # Electron + renderer UI
├── docs/                       # Documentation
│   ├── API.md
│   ├── CONFIG.md
│   └── adr/                    # Architecture Decision Records
├── pyproject.toml              # Ruff + pytest configuration
├── .pre-commit-config.yaml     # Pre-commit hooks
├── start.sh                    # Linux/Mac startup script
└── start.bat                   # Windows startup script
```

## Documentation

- **[docs/API.md](docs/API.md)** — Full API reference
- **[docs/CONFIG.md](docs/CONFIG.md)** — Environment variable reference
- **[DEVELOPMENT.md](DEVELOPMENT.md)** — Developer guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines
- **[docs/adr/](docs/adr/)** — Architecture Decision Records

## License

Private — All rights reserved.
