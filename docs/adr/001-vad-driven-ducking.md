# ADR-001: VAD-Driven Audio Ducking

## Status

Accepted

## Context

Traditional audio ducking lowers background music whenever any narration is present, even during silent pauses in the TTS audio. This creates an unnatural pumping effect where the background audio visibly drops during silence and rises between words.

The pipeline generates TTS narration (hook + commentary) that overlaps with the original video audio. We need the background to duck *only* when the TTS is actually speaking, not during silence or breath pauses within the narration.

## Decision

Use Silero VAD (Voice Activity Detection) to analyze the generated TTS audio and extract precise speech timestamps. The ffmpeg filter chain uses these timestamps to apply volume ducking only during actual speech segments.

**Implementation:**

1. After TTS generation, run Silero VAD on each narration WAV file
2. Extract speech segments: `[{start: 0.2, end: 2.8}, ...]`
3. Map these timestamps to the reel timeline (considering reel_start offset)
4. Build an ffmpeg volume filter chain with per-segment volume curves:
   - Pre-buffer: 0.4s before speech (smooth S-curve ramp-down)
   - During speech: duck to 3% volume (0.97 ducking depth)
   - Post-buffer: 0.25s after speech (smooth S-curve ramp-up)
5. Apply the filter chain to the original audio track

**Key parameters (configurable via env vars):**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `VAD_THRESHOLD` | 0.5 | Speech detection confidence threshold |
| `VAD_PRE_BUFFER_SECONDS` | 0.4 | Seconds of ducking before speech |
| `VAD_POST_BUFFER_SECONDS` | 0.25 | Seconds of ducking after speech |
| `VAD_DUCKING_DEPTH` | 0.97 | Volume reduction (0.0=none, 1.0=mute) |
| `VAD_SCURVE_RAMP_SECONDS` | 0.15 | S-curve transition smoothing |

## Consequences

**Positive:**
- Natural-sounding audio — background only ducks during actual speech
- No pumping effect during TTS silence/breaths
- Configurable parameters for fine-tuning per content type
- Works with any TTS voice or language

**Negative:**
- Adds processing time (~0.5s per narration segment)
- Requires Silero VAD + torch dependency (~2 GB)
- Edge cases: very short speech segments (<0.3s) may not trigger ducking

**Alternatives considered:**
- Simple threshold-based ducking (rejected: too crude, causes pumping)
- Sidechain compression in ffmpeg (rejected: compresses entire narration duration, not speech-only)
- External audio analysis tools (rejected: adds complexity, no benefit over Silero VAD)
