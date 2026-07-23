# ADR-003: Orchestrator Architecture

## Status

Accepted

## Context

The pipeline has 8 stages (download → transcribe → analyze → clip → TTS → caption → compose → edit). Before the refactor, all stage logic was inlined in the 790-line `queue_manager.py`, making it difficult to test, debug, and extend.

The per-group stages (clip → TTS → caption → compose) needed to be:
1. Independently testable
2. Checkpoint-resumable
3. Retryable on failure
4. Cleanable on cancellation

## Decision

Extract a **GroupOrchestrator** class that encapsulates the per-group pipeline (clip → TTS → caption → compose → edit). The QueueManager handles the outer flow (download → transcribe → analyze) and delegates group processing to the orchestrator.

**Responsibilities:**

| Component | Responsibility |
|-----------|----------------|
| `QueueManager` | Job queue, worker loop, download, transcribe, analyze, group delegation |
| `GroupOrchestrator` | Per-group: clip → TTS → caption → compose → edit |
| `PipelineCheckpoint` | Atomic JSON persistence for stage results |
| `ProgressReporter` | Thread-safe status updates, WebSocket broadcasting |

**Flow:**

```
QueueManager._process_job()
├── Stage 1: download_video()
├── Stage 2: transcribe_video()
├── Stage 3: select_reel_plan()
└── For each ReelGroup:
    └── GroupOrchestrator.run_group()
        ├── run_clipping()     → checkpoint: group_N_clips
        ├── run_tts()          → checkpoint: group_N_tts
        ├── run_captioning()   → checkpoint: group_N_captions
        ├── run_compositing()  → checkpoint: group_N_composite
        └── run_editing()      → final output
```

**Checkpoint strategy:**
- Each stage saves results to `backend/storage/working/<job_id>/group_N_stage.json`
- On resume, completed stages are skipped
- Failed stages retry from scratch (up to `MAX_GROUP_RETRIES=2`)
- On cancellation, checkpoints are preserved for future resume

## Consequences

**Positive:**
- Each stage is independently testable with mocks
- Checkpoint resumability works across process restarts
- Clean separation of concerns (queue vs. group processing)
- Easy to add new stages (see [DEVELOPMENT.md](../../DEVELOPMENT.md))

**Negative:**
- More files to navigate
- Checkpoint serialization adds complexity for non-serializable data
- Group retry logic must handle partial failures gracefully

**Alternatives considered:**
- Continue inlining in queue_manager (rejected: unmaintainable at 790+ lines)
- Celery/RQ task queue (rejected: overkill for single-server deployment)
- State machine library (rejected: adds dependency, limited benefit)
