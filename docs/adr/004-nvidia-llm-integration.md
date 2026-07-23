# ADR-004: NVIDIA LLM Integration

## Status

Accepted

## Context

The pipeline needs to analyze video transcripts and generate structured reel plans (clip selections + narration events). This requires:
1. Understanding long transcripts (up to 200k tokens for 2-hour videos)
2. Generating structured JSON output (ReelPlan schema)
3. Running at scale without rate limiting issues
4. Fast inference for good UX

Local LLMs are too slow and require significant VRAM. OpenAI/Claude APIs are expensive at scale. NVIDIA's API provides a good balance of speed, context window, and cost.

## Decision

Use NVIDIA's API (integrate.api.nvidia.com) as the primary LLM provider, with a fallback model for reliability.

**Configuration:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `NVIDIA_API_KEY` | *(required)* | API authentication |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | API endpoint |
| `NVIDIA_MODEL` | `openai/gpt-oss-120b` | Primary model |
| `NVIDIA_MODEL_FALLBACK` | `nvidia/llama-3.3-nemotron-super-49b-v1.5` | Fallback on primary failure |

**Retry strategy:**
1. Try primary model first
2. On failure, try fallback model
3. On both failures, use heuristic fallback plan (no LLM)
4. Exponential backoff between retries (handled by `call_llm_sync`)

**LLM Interaction Logging:**
Every LLM call is logged as a structured `LLMInteraction` record:
```python
LLMInteraction(
    timestamp="2024-01-01T12:00:00Z",
    type="response",          # prompt | response | error | retry
    role="assistant",
    content="preview...",     # truncated for UI
    full_content="...",       # full text for debug
    model="openai/gpt-oss-120b",
    token_count="1500 out / 45000 in",
    stage_name="reel_plan",
)
```

These are broadcast via WebSocket for live UI rendering and saved to debug files.

**Caching:**
The `cached_call_llm_sync()` function caches identical prompts for 5 minutes (TTL cache), reducing redundant API calls during development and retries.

**JSON Repair:**
The LLM sometimes returns truncated or malformed JSON. The `_try_repair_truncated_json()` function handles:
- Missing closing braces/brackets
- Trailing commas
- Unclosed string quotes
- Markdown code fences
- Nested JSON extraction

## Consequences

**Positive:**
- 256k token context window handles full transcripts without chunking
- Fast inference (~2-5s for reel plan generation)
- Structured output enables reliable JSON parsing
- Fallback model provides redundancy
- Interaction logging enables debugging and UI features

**Negative:**
- Requires NVIDIA API key (paid service)
- Network dependency — fails gracefully with heuristic fallback
- Model behavior can change between versions
- JSON output requires repair logic for edge cases

**Alternatives considered:**
- Local LLM (Ollama/llama.cpp): rejected — too slow, requires 16+ GB VRAM
- OpenAI API: rejected — expensive at scale, 128k context limit
- Claude API: rejected — expensive, no structured output guarantee
- Heuristic-only (no LLM): rejected — poor clip selection quality
