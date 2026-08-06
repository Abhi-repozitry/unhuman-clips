# Configuration Reference

All settings are configured via environment variables in `backend/.env`. The server reads these at startup via `config.py`.

---

## Required Variables

| Variable | Type | Description |
|----------|------|-------------|
| `OPENCODE_API_KEY` | `string` | OpenCode API key for LLM analysis |
| `FFMPEG_PATH` | `string` (file path) | Path to ffmpeg binary. Examples: `/usr/bin/ffmpeg`, `C:\ffmpeg\bin\ffmpeg.exe` |

---

## LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCODE_BASE_URL` | `https://opencode.ai/zen/v1` | OpenCode API endpoint |
| `OPENCODE_MODEL` | `mimo-v2.5-free` | LLM model |
| `PLAN_MODE` | `executor` | Reel planning mode. `executor` = LLM plans story structure only, Python picks all clips deterministically (consistent output). `llm` = legacy multi-stage LLM that picks exact clip timestamps |

---

## Video Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `DOWNLOAD_MAX_HEIGHT` | `1080` | Max video resolution to download |
| `DOWNLOAD_FORMAT` | `bestvideo[height<=1080]+bestaudio/best[height<=1080]` | yt-dlp format selector |
| `MAX_OUTPUT_DURATION` | `100` | Maximum output reel duration (seconds) |
| `MIN_OUTPUT_DURATION` | `90` | Minimum output reel duration (seconds) |
| `TARGET_GROUP_DURATION` | `120` | Ideal group duration for LLM planning (seconds) |
| `ALLOW_CPU_FFMPEG_FALLBACK` | `0` | Set `1` to allow CPU encoding when NVENC fails |
| `CLIP_PADDING_FRAMES` | `15` | Extra frames to add around each clip for clean cuts |

### Multimodal Enrichment

Scene-cut detection runs on the CPU through OpenCV. OCR uses an OpenAI-compatible
vision endpoint and never allocates local GPU memory.

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIMODAL_ENABLED` | `1` | Enable scene detection and OCR enrichment. |
| `ENTITY_MIN_SEGMENT_SECONDS` | `20` | Minimum source duration for an entity segment after cut coalescing. |
| `SCENE_SAMPLE_FPS` | `2` | CPU frame sample rate for scene-cut detection. |
| `SCENE_CUT_THRESHOLD` | `0.45` | Normalized histogram/pixel-difference cut threshold. |
| `OCR_SAMPLE_INTERVAL_SECONDS` | `4` | Fallback OCR sampling interval. |
| `OCR_MAX_FRAMES` | `240` | Maximum OCR frames per source video. |
| `OCR_MAX_CONCURRENCY` | `20` | Maximum simultaneous hosted OCR requests. |
| `VISION_BASE_URL` | `http://localhost:20128/v1` | OpenAI-compatible OmniRoute endpoint. |
| `VISION_MODEL` | `mimo-v2.5` | Vision-capable model exposed by the endpoint. |
| `VISION_API_KEY` | `OPENCODE_API_KEY` | Optional endpoint key; falls back to the OpenCode key. |
| `VISION_OCR_ENABLED` | `1` | Disable only hosted OCR while retaining CPU scene cuts. |
| `VISION_TIMEOUT_SECONDS` | `30` | Timeout for one hosted OCR request. |

---

## Transcription

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Whisper model size. Options: `tiny`, `base`, `small`, `medium`, `large-v3-turbo` |
| `WHISPER_DEVICE` | `cuda` | Device for Whisper inference. Falls back to `cpu` if CUDA unavailable |

---

## TTS (Text-to-Speech)

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_VOICE` | `en-US-ChristopherNeural` | Edge-TTS voice name |
| `TTS_RATE` | `+0%` | Speech rate adjustment (e.g., `+10%`, `-5%`) |
| `TTS_VOLUME` | `+0%` | Volume adjustment |

---

## VAD (Voice Activity Detection)

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD_THRESHOLD` | `0.5` | Speech detection confidence (0.0–1.0) |
| `VAD_PRE_BUFFER_SECONDS` | `0.4` | Ducking start before speech (seconds) |
| `VAD_POST_BUFFER_SECONDS` | `0.25` | Ducking end after speech (seconds) |
| `VAD_DUCKING_DEPTH` | `0.97` | Volume reduction during speech (0.0=none, 1.0=mute) |
| `VAD_SCURVE_RAMP_SECONDS` | `0.15` | S-curve transition smoothing (seconds) |
| `VAD_SAMPLE_RATE` | `16000` | Audio sample rate for VAD analysis |

---

## Audio Ducking (Legacy/Simple)

These are used when VAD-based ducking is disabled or as fallback:

| Variable | Default | Description |
|----------|---------|-------------|
| `DUCKING_THRESHOLD_DB` | `-35` | dB threshold to trigger ducking |
| `DUCKING_REDUCTION_DB` | `-15` | dB reduction during ducking |
| `DUCKING_ATTACK_MS` | `200` | Ducking attack time (ms) |
| `DUCKING_RELEASE_MS` | `400` | Ducking release time (ms) |

---

## Queue & Workers

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_WORKERS` | `3` | Max concurrent jobs processed in parallel |
| `MAX_QUEUE_SIZE` | `50` | Max jobs in queue before rejecting |
| `MAX_GROUP_RETRIES` | `2` | Retries per group on failure |

---

## Cleanup & Retention

| Variable | Default | Description |
|----------|---------|-------------|
| `STALE_FILE_CLEANUP_HOURS` | `24` | Hours before stale files are cleaned up |
| `CLEANUP_ON_STARTUP` | `1` | Set `1` to clean stale files on server start |

---

## Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `9000` | Server port |
| `RELOAD` | `0` | Set `1` to enable auto-reload (development) |

---

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

---

## Example `.env`

```env
# Required
OPENCODE_API_KEY=oc-xxxxxxxxxxxxxxxxxxxxx
FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe

# LLM
OPENCODE_MODEL=mimo-v2.5-free

# Video
DOWNLOAD_MAX_HEIGHT=1080
MAX_OUTPUT_DURATION=120
MIN_OUTPUT_DURATION=90

# TTS
TTS_VOICE=en-US-ChristopherNeural

# Workers
MAX_WORKERS=3

# Logging
LOG_LEVEL=INFO
```

---

## Path Constants (Configured in `config.py`)

These are computed at startup and not configurable via environment variables:

| Constant | Value | Description |
|----------|-------|-------------|
| `BASE_DIR` | `backend/` | Backend directory |
| `STORAGE_DIR` | `backend/storage/` | Root storage directory |
| `WORKING_DIR` | `backend/storage/working/` | Per-job working files |
| `OUTPUT_DIR` | `backend/storage/outputs/` | Final output videos |
| `PROMPTS_DIR` | `backend/prompts/` | LLM prompt templates |

---

## FFmpeg Encoder Detection

The `get_encoder()` function in `ffmpeg_utils.py` automatically detects the best available encoder:

1. **NVENC** — `h264_nvenc` (NVIDIA GPU hardware encoding, fastest)
2. **AMF** — `h264_amf` (AMD GPU, Windows only)
3. **CPU fallback** — `libx264` (software encoding, always available)

Set `ALLOW_CPU_FFMPEG_FALLBACK=1` to allow CPU fallback when NVENC is unavailable. Without this flag, the server will raise an error if NVENC is not detected.
