# Development Guide

## Code Structure

```
backend/
├── main.py                 # FastAPI app, REST endpoints, WebSocket, lifespan
├── config.py               # All env vars, validation, directory constants
├── models.py               # Pydantic data models (VideoJob, ReelPlan, etc.)
├── queue_manager.py        # Job queue worker loop, pipeline stage coordination
├── progress.py             # Thread-safe ProgressReporter (WebSocket broadcasting)
├── output_manager.py       # Final edit, duration capping, file staging
├── ffmpeg_utils.py         # ffmpeg/ffprobe path resolution, encoder detection
├── logging_config.py       # Centralized logging setup
├── pipeline/
│   ├── downloader.py       # yt-dlp download with exponential backoff
│   ├── transcriber.py      # Whisper GPU transcription (lazy model loading)
│   ├── analyzer.py         # LLM-powered reel plan generation
│   ├── clipper.py          # Parallel clip cutting (ThreadPoolExecutor)
│   ├── tts.py              # Edge-TTS narration synthesis
│   ├── captioner.py        # ASS subtitle generation with keyword highlighting
│   ├── compositor.py       # VAD-driven ducking, freeze-frame padding, muxing
│   ├── orchestrator.py     # Per-group pipeline orchestration (clip→TTS→caption→compose)
│   ├── checkpoint.py       # JSON checkpoint persistence for resumability
│   ├── narration_validator.py  # Speech overlap detection, timing adjustment
│   └── sanitize.py         # Text sanitization for ASS/TTS/ffmpeg
├── providers/
│   └── llm.py              # NVIDIA LLM API client with retry + caching
└── requirements.txt        # Python dependencies
```

### Key Design Principles

- **Each pipeline stage is a pure function** — takes inputs, returns outputs, no global state.
- **GroupOrchestrator** encapsulates the per-group clip→TTS→caption→compose flow.
- **QueueManager** handles the outer loop: download→transcribe→analyze, then delegates to orchestrator.
- **ProgressReporter** is thread-safe — all pipeline stages call it from worker threads.
- **Checkpoints** are atomic JSON files — if the process crashes, the next run resumes from the last checkpoint.

## Adding a New Pipeline Stage

1. **Create the module** in `backend/pipeline/`:

```python
"""My new pipeline stage."""
from __future__ import annotations

import logging

__all__ = ["my_new_stage"]
logger = logging.getLogger(__name__)


def my_new_stage(input_data: list[dict], reporter: Any) -> list[dict]:
    """Process input and return results.

    Args:
        input_data: Input from previous stage.
        reporter: ProgressReporter for status updates.

    Returns:
        Processed output for next stage.
    """
    reporter.update_sub_stage("Running my stage...", 50)
    # ... processing ...
    reporter.update_sub_stage("Done", 100)
    return results
```

2. **Integrate into the orchestrator** (`backend/pipeline/orchestrator.py`):

```python
async def run_my_stage(self, group_idx, group, reporter, ...):
    """Run my new stage for this group."""
    ckpt_key = f"group_{group_idx}_my_stage"
    checkpoint = self.ckpt.load_stage(ckpt_key)
    if checkpoint:
        reporter.log_info(f"Group {group_idx+1}: Resuming from checkpoint")
        return checkpoint["results"]

    result = await asyncio.to_thread(
        my_new_stage, input_data, reporter
    )

    self.ckpt.save_stage(ckpt_key, {"results": result})
    return result
```

3. **Add tests** in `tests/test_my_stage.py`:

```python
"""Tests for backend.pipeline.my_new_stage."""
from backend.pipeline.my_new_stage import my_new_stage


class TestMyNewStage:
    def test_basic_processing(self, mock_reporter):
        result = my_new_stage([{"key": "value"}], mock_reporter)
        assert len(result) > 0

    def test_empty_input(self, mock_reporter):
        result = my_new_stage([], mock_reporter)
        assert result == []
```

4. **Run tests** to verify:

```bash
pytest tests/test_my_stage.py -v
```

## Debugging Tips

### LLM Debug Files

When the LLM is called, the raw response is saved to:
```
backend/storage/working/llm_debug_<timestamp>.txt
```
Check these files when the LLM returns unexpected JSON.

### Checkpoints

Each pipeline stage saves a checkpoint JSON in:
```
backend/storage/working/<job_id>/stage_<name>.json
```
Delete these files to force a stage to re-run from scratch.

### Logs

Logs are written to stderr and formatted with timestamps. To increase verbosity:

```bash
# In .env:
LOG_LEVEL=DEBUG
```

Or set the environment variable before starting:
```bash
LOG_LEVEL=DEBUG uvicorn backend.main:app --reload
```

### VAD Analysis

To debug VAD (Voice Activity Detection) ducking behavior, check the narration validator logs:

```bash
# Look for [WARN] and [INFO] messages from narration_validator
# These show when narration overlaps with speech and gets auto-shifted
```

### Whisper Transcription

If transcription fails, check:
1. CUDA/cuDNN is installed: `python -c "import torch; print(torch.cuda.is_available())"`
2. GPU memory is sufficient (large-v3-turbo needs ~4 GB VRAM)
3. The downloaded video has an audio stream

### Common Runtime Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ffmpeg not found` | Wrong path in `FFMPEG_PATH` | Set correct path in `.env` or add ffmpeg to PATH |
| `h264_nvenc not available` | No NVIDIA GPU or wrong drivers | Set `ALLOW_CPU_FFMPEG_FALLBACK=1` |
| `NVIDIA_API_KEY not set` | Missing API key | Get key from build.nvidia.com, add to `.env` |
| `Whisper CUDA model failed to load` | CUDA/cuDNN mismatch | Reinstall PyTorch with correct CUDA version |
| `edge-tts failed` | Network issue or invalid voice | Check network, verify `TTS_VOICE` value |

## Running Tests, Linting, and Pre-commit

### Tests

```bash
# Unit tests only (fast, ~5s)
pytest tests/ -m "not integration"

# All tests
pytest tests/

# Specific test file
pytest tests/test_analyzer.py -v

# With coverage
pytest tests/ --cov=backend --cov-report=term-missing

# Smoke test (no pytest)
python scripts/smoke_test.py
```

### Linting

```bash
# Check for lint errors
ruff check backend/ tests/

# Auto-fix
ruff check --fix backend/ tests/

# Format code
ruff format backend/ tests/
```

### Pre-commit

```bash
# Install pre-commit hooks (one-time)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## Common Issues & Solutions

### "No module named 'backend'" when running scripts

Run from the project root, or add it to PYTHONPATH:
```bash
# From project root:
python -m pytest tests/

# Or set PYTHONPATH:
PYTHONPATH=. python scripts/smoke_test.py
```

### Tests fail with "coroutine was never awaited"

This usually means you're mocking an async function with `MagicMock` instead of `AsyncMock`. Check if the function is called with `await` or `asyncio.to_thread`.

### ffmpeg hangs or times out

The clipper uses a 300-second timeout per clip. If ffmpeg hangs:
1. Check if the source file is valid: `ffprobe -v error source.mp4`
2. Check if NVENC is overloaded: reduce `MAX_WORKERS`
3. Try CPU fallback: set `ALLOW_CPU_FFMPEG_FALLBACK=1`

### Whisper runs out of GPU memory

- Use a smaller model: `WHISPER_MODEL_SIZE=base` in `.env`
- Process one job at a time: `MAX_WORKERS=1`
- Reduce video resolution: `DOWNLOAD_MAX_HEIGHT=720`

### WebSocket disconnects during long jobs

The server sends a "pong" response to client pings. Ensure your WebSocket client:
1. Sends periodic pings (every 30s recommended)
2. Handles reconnection gracefully
3. The `/ws` endpoint automatically cleans up disconnected clients
