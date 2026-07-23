# ADR-002: Group-Based Reel Architecture

## Status

Accepted

## Context

YouTube videos range from 2 minutes to 2+ hours. A single output reel (90–180s) cannot capture the full content of longer videos. Users want multiple distinct reels from a single source, each telling a different story arc.

The LLM needs to understand the full transcript and select clips that form coherent narratives, not just extract random segments.

## Decision

The pipeline generates a **ReelPlan** containing multiple **ReelGroups**, each representing a distinct output reel. The LLM selects clips and narration for each group independently, with different narrative angles.

**Structure:**

```
ReelPlan
├── ReelGroup 0 (e.g., "The Discovery")
│   ├── reel_summary (title, description, key_moment)
│   ├── source_clips: [{source_start, source_end, reason}, ...]
│   └── narration_events: [{event_type, reel_start, reel_end, text}, ...]
├── ReelGroup 1 (e.g., "The Controversy")
│   └── ...
└── ReelGroup N
```

**Group count scaling:**

| Source Duration | Groups Generated |
|-----------------|------------------|
| < 5 minutes | 1–4 |
| 5–10 minutes | 3–6 |
| 10–20 minutes | 4–8 |
| 20+ minutes | 5–12 |

**Each group is processed independently** through the GroupOrchestrator:
1. Cut source clips for this group
2. Generate TTS narration for this group
3. Create ASS captions for this group
4. Composite the final reel for this group

## Consequences

**Positive:**
- Multiple distinct reels from a single long video
- Each reel tells a coherent story arc
- Independent processing enables parallelism and checkpointing
- Groups can have different narrative angles (educational, emotional, humorous)

**Negative:**
- LLM must understand full transcript context for good clip selection
- More complex than flat clip extraction
- Duplicate clips across groups (mitigated by REUSE RULE for short sources)

**Alternatives considered:**
- Flat clip list (rejected: no narrative structure, single output only)
- Fixed-duration segments (rejected: loses narrative coherence)
- Manual group specification (rejected: defeats automation purpose)
