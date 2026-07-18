# Stack Decision Addendum v2 — unhuman-clips

Addendum to `.kilo/plans/1784109460412-llm-quota-cache-abstraction.md`. Original plan is still the source of truth; this file records deltas only — new tool decisions raised after Phase 1 was finalized, and why each was accepted, deferred, or rejected.

---

## Decision Table

| Tool | Verdict | Reasoning |
|---|---|---|
| yt-dlp | ✅ Keep | Already decided. Best downloader for the job, no change. |
| ffmpeg / ffprobe | ✅ Keep as render/probe engine | Was never the bottleneck. No swap needed for quality — codec/bitrate tuning (see below) covers the actual gap. |
| Faster-Whisper | ✅ Keep | Already in use, no change. |
| PySceneDetect | ✅ **Add** | Runs on the full source file to produce scene-boundary timestamps that define candidate clip window edges during window generation in `analyzer.py` — not just an extra score tacked onto already-fixed windows. CPU-only (OpenCV/PyAV decode), so it runs concurrently with Faster-Whisper (which uses the GPU) with zero resource contention — free wall-clock parallelization, not just a feature add. |
| MediaPipe | ✅ **Add** (scoped down, weak signal) | Face-presence-only (boolean/score per clip) feeding into ranking, NOT full gesture/reaction classification. Caveat: only meaningful for talking-head/gaming-facecam content — for wildlife narration or subject-less footage it'll read near-zero across every candidate, so treat it as one weak input, not a primary ranking factor. No keyframe-extraction reuse with OCR's Stage 7 keyframes — ranking runs on pre-Gate-1-edit candidate windows, OCR runs on post-edit final clips, so the timestamps don't line up. |
| Silero VAD | ✅ **Add** (replaces ffmpeg `silencedetect` in Stage 9) | `silencedetect` is a raw amplitude gate — it can't distinguish "no one's talking" from "quiet background music/narration bed under silence," which is common in downloaded source content. Silero VAD detects actual speech presence regardless of volume, so Stage 9's dead-air trim gets meaningfully more accurate. Low friction: CPU-only, tiny model, no CUDA install pain (unlike PaddleOCR). This swaps the detection method inside Stage 9 — it does not add a second detector running alongside the existing one. |
| Librosa | ⏸️ Defer to Phase 2 | Energy/tempo/loudness scoring is a legitimate ranking signal, but needs its own scoring-integration design (how it combines with LLM score + scene + face signals). Not a drop-in. |
| PaddleOCR | ⏸️ Skip for now | Unchanged from prior call: Windows+CUDA install friction, and `ocr_texts` isn't wired into anything downstream yet (compositor doesn't avoid caption/text overlap). Fix the wiring before swapping the engine — no point optimizing an unused signal. |
| YOLO11 | ❌ Reject (Phase 1) | New GPU model on a 4GB card already sequencing Whisper + OCR. This is a full new stage with its own inference wiring, not a config tweak. |
| Vision LLM scoring (Qwen2.5-VL etc.) | ❌ Reject (Phase 1) | Directly competes with the 35/min NIM budget the provider abstraction was just built to manage. Every visual-scoring call eats the same quota as ranking/commentary calls. Phase 2 candidate once real usage patterns show budget headroom. |
| OpenTimelineIO | ❌ Reject | See rationale below. |
| Custom JSON timeline + ffmpeg renderer | ✅ Already the plan | This is the Gate 1 edit payload (`clips[].cuts[]`, `speed`, `order`) already specified in the base plan. Not a new idea — just confirming the existing design is the right call. |
| DaVinci Resolve scripting API | ❌ Reject | Needs the GUI app alive and controlled interactively — wrong fit for an unattended, queued background job. Fragile reliability regression for a solo project. |
| Kdenlive | ❌ Reject | Same reasoning as Resolve. No headless automation story, no reason for an AI pipeline to drive a GUI editor. |

---

## Why not OTIO

OpenTimelineIO exists to let NLEs (Premiere, Resolve, Avid) exchange timelines with each other. That's not the problem here — the renderer only ever needs to read a JSON blob it produced itself. The lightweight edit-payload schema in the base plan already achieves OTIO's actual goal (a data-driven, engine-independent timeline) without the overhead of learning/maintaining a heavier spec built for cross-NLE interop nobody's doing.

If a real need shows up later (e.g. wanting to hand a clip to Resolve for a manual polish pass before posting), add an **OTIO exporter** that converts the existing JSON payload — don't make OTIO the internal representation now.

---

## Rendering quality note (separate from editing logic)

None of the above affects *render* quality — that's a codec/hardware question, not a tool-swap question. Correction from an earlier pass: NVENC isn't actually worth switching to here.
- At 3 clips/day, a single ~60s x264 CPU encode takes low tens of seconds — encode time isn't a real bottleneck at this volume. It's also not a clean quality win: the RTX 2050's Ampere-gen NVENC is close to x264 at matched bitrate but still a notch behind hand-tuned x264 presets — a wash, not an upgrade. Skip it, not worth the engineering time.
- Sequence GPU-heavy stages (Whisper → OCR) rather than parallelizing — 4GB VRAM won't hold multiple models comfortably at once. This still holds regardless of the NVENC call above.

---

## Updated Pipeline Order

Scene detection and face-presence scoring are folded into the existing ranking stage — **no stage count change, still 9 stages.**

```
Download → Transcribe → [ANALYZING: scene-detect + face-signal + LLM ranking] [Gate 1: approve+trim]
    → [SCRIPTING: commentary on post-cut transcript] [Gate 2]
    → Audio → Captions (+ OCR) → Compose → Edit (Stage 9: silence trim) → Metadata → Validate
```

---

## New / Updated Modules

| File | Change | Est. |
|---|---|---|
| `backend/pipeline/analyzer.py` | Add `detect_scenes()` (PySceneDetect wrapper) and `detect_face_presence()` (MediaPipe wrapper); fold both into ranking input before LLM call | ~80 lines |
| `backend/requirements.txt` | Add `scenedetect`, `mediapipe`, `silero-vad` (all CPU-only, no GPU conflict added) | trivial |
| `backend/queue_manager.py` | Stage 3 (ANALYZING) `stage_data` gains `scenes_detected`, `face_presence_scores` fields | ~15 lines |
| `backend/pipeline/editor.py` | Replace `detect_silence()`'s ffmpeg `silencedetect` implementation with `silero-vad` speech-probability windowing; `atrim`+`concat` trim logic unchanged | ~40 lines |
| `backend/pipeline/compositor.py` | (Follow-on, not blocking) consume `ocr_texts` to avoid caption/on-screen-text overlap | ~40 lines |

---

## Rejected/Deferred Log

| Item | Status | Revisit when |
|---|---|---|
| Librosa energy/tempo scoring | Deferred, Phase 2 | Ranking signal design is revisited post-launch |
| PaddleOCR | Skipped | `ocr_texts` is actually consumed downstream AND EasyOCR accuracy is a proven complaint |
| YOLO11 | Rejected, Phase 1 | VRAM budget allows a second concurrent model, or GPU is upgraded |
| Vision LLM scoring | Rejected, Phase 1 | Real usage data shows NIM quota headroom |
| OTIO (internal format) | Rejected | Never — only reconsider as an export target, not internal representation |
| DaVinci Resolve / Kdenlive | Rejected | A concrete need for manual human polish-pass emerges pre-publish |
