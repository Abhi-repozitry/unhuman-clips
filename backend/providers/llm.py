"""LLM provider — synchronous OpenAI-compatible API calls with retry logic.

Provides call_llm_sync() with exponential backoff, model fallback,
error classification, and structured LLMInteraction records for UI display.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

import openai

from backend.models import LLMInteraction

__all__ = ["call_llm", "call_llm_sync"]

logger = logging.getLogger(__name__)


def _now_timestamp() -> str:
    """Return a human-readable timestamp string for LLMInteraction records."""
    return datetime.now().strftime("%H:%M:%S.%f")[:12]


def _classify_llm_error(e: Exception) -> str:
    """Classify an LLM error into a canonical category for logging and retry logic."""
    if isinstance(e, openai.APITimeoutError):
        return "timeout"
    if isinstance(e, openai.RateLimitError):
        return "rate_limit"
    if isinstance(e, openai.APIConnectionError):
        return "connection"
    if isinstance(e, json.JSONDecodeError):
        return "json_parse"
    err_str = str(e).lower()
    if "504" in err_str or "timeout" in err_str or "gateway" in err_str:
        return "timeout"
    if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
        return "rate_limit"
    if "empty content" in err_str or "refusal" in err_str:
        return "empty_content"
    if "connection" in err_str or "econnrefused" in err_str or "econnreset" in err_str:
        return "connection"
    return "unknown"


def _truncate_preview(text: str, max_len: int = 300) -> str:
    """Return a short preview of the text for UI display."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def call_llm_sync(
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str = "https://opencode.ai/zen/v1",
    temperature: float = 0.0,
    max_tokens: int = 16384,
    timeout: float = 480.0,
    reporter: Any = None,
    interactions: list[LLMInteraction] | None = None,
    stage_name: str = "reel_plan",
    reasoning_effort: str = "high",
) -> str:
    """Synchronous LLM call with enhanced retry logic and exponential backoff.

    Features:
    - 5 total attempts with exponential backoff [1, 3, 6, 10]s
    - Classifies errors into categories (timeout, rate_limit, connection, etc.)
    - Collects structured LLMInteraction records for UI display
    - Detailed logging via reporter.log_info/log_warn
    - temperature=0.0 for determinism where possible
    - reasoning_effort controls model thinking depth ('low'/'medium'/'high')
    """
    models_to_try = [model]

    # Exponential backoff: 1s, 3s, 6s, 10s
    backoff_delays = [1, 3, 6, 10]
    # Total attempts per model: up to 4-5 retries
    max_attempts_per_model = 5

    last_error = None
    # Track prompt content for reporter logging (defined before if-interactions block for scope safety)
    prompt_content = ""

    # Capture the initial prompt as an interaction
    if interactions is not None:
        prompt_content = json.dumps(messages, indent=2) if isinstance(messages, list) else str(messages)
        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        interactions.append(LLMInteraction(
            timestamp=_now_timestamp(),
            type="prompt",
            role="user",
            content=_truncate_preview(user_msg.get("content", "") if user_msg else prompt_content),
            full_content=prompt_content,
            model=model,
            retry_count=0,
            stage_name=stage_name,
        ))
        if system_msg:
            interactions.append(LLMInteraction(
                timestamp=_now_timestamp(),
                type="prompt",
                role="system",
                content=_truncate_preview(system_msg.get("content", "")),
                full_content=system_msg.get("content", ""),
                model=model,
                retry_count=0,
                stage_name=stage_name,
            ))
    if reporter:
        # Compute prompt_content preview for logging even if interactions is None
        if not prompt_content:
            prompt_content = json.dumps(messages, indent=2) if isinstance(messages, list) else str(messages)
        reporter.log_info(f"[LLM] Prompt sent ({stage_name}) — {len(prompt_content)} chars")
        # Broadcast live interactions to UI during LLM processing (only if interactions exist)
        if interactions is not None:
            reporter.set_stage_data_key("llm_interactions", [i.model_dump() for i in interactions])

    for current_model in models_to_try:
        for attempt in range(max_attempts_per_model):
            try:
                if attempt > 0 and interactions is not None:
                    interactions.append(LLMInteraction(
                        timestamp=_now_timestamp(),
                        type="retry",
                        role="assistant",
                        content=f"Retrying {current_model} (attempt {attempt + 1}/{max_attempts_per_model})",
                        full_content=f"Retry #{attempt + 1} with {current_model} after {_classify_llm_error(last_error)} error",
                        model=current_model,
                        retry_count=attempt,
                        error_type=_classify_llm_error(last_error) if last_error else "unknown",
                        stage_name=stage_name,
                    ))
                    if reporter:
                        reporter.log_info(f"[LLM] Retry {attempt + 1}/{max_attempts_per_model} with {current_model} (reason: {_classify_llm_error(last_error)})")
                        reporter.set_stage_data_key("llm_interactions", [i.model_dump() for i in interactions])

                client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
                kwargs = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                    "seed": 42,
                    "reasoning_effort": reasoning_effort,
                }
                kwargs["response_format"] = {"type": "json_object"}
                kwargs["stream_options"] = {"include_usage": True}

                raw = None
                token_count = ""
                try:
                    kwargs["stream"] = True
                    response = client.chat.completions.create(**kwargs)
                    full_content = ""
                    reasoning_content = ""
                    chunk_count = 0
                    chunk = None
                    for chunk in response:
                        if chunk.choices:
                            delta = chunk.choices[0].delta if chunk.choices[0] else None
                            if delta and delta.content:
                                full_content += delta.content
                                chunk_count += 1
                                if chunk_count % 10 == 0 and reporter and interactions is not None:
                                    reporter.set_stage_data_key("llm_interactions", [i.model_dump() for i in interactions])
                            if delta and getattr(delta, "reasoning_content", None):
                                reasoning_content += delta.reasoning_content
                    if full_content.strip():
                        raw = full_content.strip()
                    elif reasoning_content.strip():
                        # Model returned only chain-of-thought / thinking, no
                        # usable output content — treat as empty so the retry
                        # logic in call_llm_sync kicks in.
                        logger.warning(
                            f"LLM returned only reasoning_content "
                            f"({len(reasoning_content)} chars), treating as empty"
                        )
                        raw = ""
                    else:
                        raw = ""
                    usage = getattr(chunk, 'usage', None) if chunk else None
                    if usage:
                        token_count = f" ({usage.completion_tokens} out / {usage.prompt_tokens} in tokens)"
                except Exception as stream_err:
                    # Fallback to non-streaming if streaming fails
                    logger.warning(f"Streaming failed, falling back to non-streaming: {stream_err}")
                    kwargs.pop("stream", None)
                    response = client.chat.completions.create(**kwargs)
                    if not response.choices:
                        raise RuntimeError("LLM API returned no choices in response.") from None
                    raw = response.choices[0].message.content
                    if raw is None:
                        finish_reason = response.choices[0].finish_reason
                        refusal = getattr(response.choices[0].message, 'refusal', None)
                        raise RuntimeError(
                            f"LLM API returned empty content. "
                            f"Finish reason: {finish_reason}. Refusal: {refusal}."
                        ) from None
                    raw = raw.strip()
                    usage = getattr(response, 'usage', None)
                    if usage:
                        token_count = f" ({usage.completion_tokens} out / {usage.prompt_tokens} in tokens)"

                if not raw:
                    raise RuntimeError(
                        "LLM returned empty content. "
                        "The model may have refused or hit an output limit."
                    )

                if interactions is not None:
                    interactions.append(LLMInteraction(
                        timestamp=_now_timestamp(),
                        type="response",
                        role="assistant",
                        content=_truncate_preview(raw),
                        full_content=raw,
                        model=current_model,
                        retry_count=attempt,
                        token_count=token_count.strip() if token_count else "",
                        stage_name=stage_name,
                    ))
                    if reporter:
                        reporter.log_info(f"[LLM] Response received{token_count} from {current_model}")
                        # Broadcast live interactions to UI immediately after response
                        reporter.set_stage_data_key("llm_interactions", [i.model_dump() for i in interactions])

                return raw

            except Exception as e:
                last_error = e
                error_type = _classify_llm_error(e)
                err_preview = _truncate_preview(str(e), 200)

                if interactions is not None:
                    interactions.append(LLMInteraction(
                        timestamp=_now_timestamp(),
                        type="error",
                        role="assistant",
                        content=f"[{error_type.upper()}] {err_preview}",
                        full_content=str(e),
                        model=current_model,
                        retry_count=attempt,
                        error_type=error_type,
                        stage_name=stage_name,
                    ))
                    if reporter:
                        reporter.log_warn(f"[LLM] Error with {current_model} (attempt {attempt + 1}): {error_type} — {err_preview[:100]}")

                # Determine if we should retry or move to next model
                if attempt < max_attempts_per_model - 1:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    if reporter:
                        reporter.log_info(f"[LLM] Backoff {delay}s before retry {attempt + 2} with {current_model}")
                    time.sleep(delay)
                else:
                    # Exhausted retries for this model
                    if reporter:
                        reporter.log_warn(f"[LLM] Model {current_model} exhausted all {max_attempts_per_model} retries")
                    break

    raise RuntimeError(
        f"LLM call failed after {max_attempts_per_model} retries. "
        f"Last error: {last_error}"
    )


def call_llm(
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str = "https://opencode.ai/zen/v1",
    temperature: float = 0.0,
    max_tokens: int = 16384,
    timeout: float = 480.0,
    reporter: Any = None,
    interactions: list[LLMInteraction] | None = None,
    stage_name: str = "reel_plan",
    reasoning_effort: str = "high",
) -> str:
    """Call LLM via OpenCode API."""
    return call_llm_sync(
        messages=messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        reporter=reporter,
        interactions=interactions,
        stage_name=stage_name,
        reasoning_effort=reasoning_effort,
    )
