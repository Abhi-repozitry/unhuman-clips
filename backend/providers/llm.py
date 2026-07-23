import json
import time
from typing import Any, Dict, List, Optional
import openai

from backend.models import LLMInteraction
from datetime import datetime


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
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    temperature: float = 0.0,
    max_tokens: int = 131072,
    timeout: float = 480.0,
    reporter: Optional[Any] = None,
    interactions: Optional[List[LLMInteraction]] = None,
    stage_name: str = "reel_plan",
) -> str:
    """Synchronous LLM call with enhanced retry logic and exponential backoff.

    Features:
    - 4-5 total attempts per model with exponential backoff [1, 3, 6, 10]s
    - Falls back to NVIDIA_MODEL_FALLBACK if primary model exhausts retries
    - Classifies errors into categories (timeout, rate_limit, connection, etc.)
    - Collects structured LLMInteraction records for UI display
    - Detailed logging via reporter.log_info/log_warn
    - temperature=0.0 for determinism where possible
    """
    from backend.config import NVIDIA_MODEL_FALLBACK

    models_to_try = [model]
    if NVIDIA_MODEL_FALLBACK and NVIDIA_MODEL_FALLBACK != model:
        models_to_try.append(NVIDIA_MODEL_FALLBACK)

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
        prompt_preview = _truncate_preview(prompt_content, 120)
        reporter.log_info(f"[LLM] Prompt sent ({stage_name}) — {len(prompt_content)} chars")
        # Broadcast live interactions to UI during LLM processing (only if interactions exist)
        if interactions is not None:
            reporter.set_stage_data_key("llm_interactions", [i.model_dump() for i in interactions])

    for m_idx, current_model in enumerate(models_to_try):
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
                }
                # Only add response_format if the model supports it
                try:
                    kwargs["response_format"] = {"type": "json_object"}
                except Exception:
                    pass

                raw = None
                token_count = ""
                try:
                    kwargs["stream"] = True
                    response = client.chat.completions.create(**kwargs)
                    full_content = ""
                    chunk_count = 0
                    for chunk in response:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta and delta.content:
                            full_content += delta.content
                            chunk_count += 1
                            if chunk_count % 10 == 0 and reporter and interactions is not None:
                                reporter.set_stage_data_key("llm_interactions", [i.model_dump() for i in interactions])
                    raw = full_content.strip()
                    usage = getattr(chunk, 'usage', None) if chunk else None
                    if usage:
                        token_count = f" ({usage.completion_tokens} out / {usage.prompt_tokens} in tokens)"
                except Exception as stream_err:
                    # Fallback to non-streaming if streaming fails
                    kwargs.pop("stream", None)
                    response = client.chat.completions.create(**kwargs)
                    raw = response.choices[0].message.content
                    if raw is None:
                        finish_reason = response.choices[0].finish_reason
                        refusal = getattr(response.choices[0].message, 'refusal', None)
                        raise RuntimeError(
                            f"NVIDIA API returned empty content. "
                            f"Finish reason: {finish_reason}. Refusal: {refusal}."
                        )
                    raw = raw.strip()
                    usage = getattr(response, 'usage', None)
                    if usage:
                        token_count = f" ({usage.completion_tokens} out / {usage.prompt_tokens} in tokens)"

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
                    # Exhausted retries for this model, try fallback
                    if reporter:
                        reporter.log_warn(f"[LLM] Model {current_model} exhausted all {max_attempts_per_model} retries, trying fallback")
                    break

    raise RuntimeError(
        f"All NVIDIA models failed after {max_attempts_per_model} retries each. "
        f"Last error: {last_error}"
    )
