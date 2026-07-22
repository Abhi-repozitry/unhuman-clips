import asyncio
import json
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import openai

from backend.providers.cache import get_cache, ContentCache
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
                token_count = ""
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


@dataclass
class ModelConfig:
    primary: str
    fallback: str
    base_url: str
    api_key_env: str


@dataclass
class RateLimitConfig:
    requests_per_minute: int
    window_seconds: int


@dataclass
class CacheConfig:
    database_path: str
    enabled: bool


@dataclass
class ProviderConfig:
    providers: Dict[str, Dict[str, str]]
    rate_limits: Dict[str, Dict[str, int]]
    cache: Dict[str, Any]


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def check_and_wait(self) -> bool:
        with self._lock:
            now = time.time()
            while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_requests:
                oldest = self._timestamps[0]
                wait_time = (oldest + self.window_seconds) - now
                return False

            self._timestamps.append(now)
            return True

    def get_wait_time(self) -> float:
        with self._lock:
            now = time.time()
            while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_requests:
                oldest = self._timestamps[0]
                return (oldest + self.window_seconds) - now
            return 0.0


class LLMProvider:
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self._client: Optional[openai.AsyncOpenAI] = None
        global_limits = self.config.rate_limits.get("global", {})
        self._rate_limiter = SlidingWindowRateLimiter(
            global_limits.get("requests_per_minute", 35),
            global_limits.get("window_seconds", 60)
        )
        self._per_job_limiter: Dict[str, SlidingWindowRateLimiter] = {}
        self._metrics_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._fallback_used = 0
        self._rate_limit_waits = 0
        self._total_wait_time = 0.0
        self._model_fallbacks = 0
        self._llm_calls = 0

    def _load_config(self, config_path: Optional[Path] = None) -> ProviderConfig:
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml"

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return ProviderConfig(
            providers=config_data.get("providers", {}),
            rate_limits=config_data.get("rate_limits", {}),
            cache=config_data.get("cache", {})
        )

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            nvidia_config = self.config.providers.get("nvidia_nim", {})
            self._client = openai.AsyncOpenAI(
                base_url=nvidia_config.get("base_url", "https://integrate.api.nvidia.com/v1"),
                api_key=__import__("os").environ.get(nvidia_config.get("api_key_env", "NVIDIA_API_KEY"))
            )
        return self._client

    def get_model_config(self, stage: str) -> Tuple[str, str]:
        nvidia_config = self.config.providers.get("nvidia_nim", {})
        primary = nvidia_config.get("primary", "stepfun-ai/step-3.7-flash")
        fallback = nvidia_config.get("fallback", "stepfun-ai/step-3.7-flash")
        return primary, fallback

    @staticmethod
    def generate_cache_key(
        stage: str,
        prompt: str,
        params: dict,
        model: str
    ) -> str:
        return ContentCache.generate_cache_key(stage, prompt, params, model)

    def _reset_metrics(self):
        with self._metrics_lock:
            self._cache_hits = 0
            self._cache_misses = 0
            self._fallback_used = 0
            self._rate_limit_waits = 0
            self._total_wait_time = 0.0
            self._model_fallbacks = 0
            self._llm_calls = 0

    def get_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            global_limit = self.config.rate_limits.get("global", {}).get("requests_per_minute", 35)
            return {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_enabled": self.config.cache.get("enabled", True),
                "fallback_used": self._fallback_used,
                "rate_limit_waits": self._rate_limit_waits,
                "total_wait_time": round(self._total_wait_time, 2),
                "model_fallbacks": self._model_fallbacks,
                "llm_calls": self._llm_calls,
                "quota_used": self._llm_calls,
                "quota_max": global_limit,
            }

    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        stage: str,
        job_id: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 65536,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        reporter: Optional[Any] = None,
        interactions: Optional[List[LLMInteraction]] = None,
    ) -> str:
        """Enhanced async LLM call with retry robustness and interaction collection."""
        primary, fallback = self.get_model_config(stage)
        models_to_try = [model or primary]
        if fallback and fallback != primary:
            models_to_try.append(fallback)

        params = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        prompt = messages[0]["content"] if messages else ""

        for m_idx, current_model in enumerate(models_to_try):
            cache_key = self.generate_cache_key(stage, prompt, params, current_model)

            if self.config.cache.get("enabled", True):
                cached_response = get_cache().get(cache_key)
                if cached_response is not None:
                    with self._metrics_lock:
                        self._cache_hits += 1
                    if progress_cb:
                        progress_cb(f"Cache hit for {current_model}", 100)
                    if reporter:
                        reporter.log_info(f"[LLM] Cache hit for {current_model}")
                    return cached_response
                with self._metrics_lock:
                    self._cache_misses += 1

            # Rate limiting
            wait_time = self._rate_limiter.get_wait_time()
            if wait_time > 0:
                with self._metrics_lock:
                    self._rate_limit_waits += 1
                    self._total_wait_time += wait_time
                if progress_cb:
                    progress_cb(f"Rate limited. Waiting {wait_time:.1f}s...", 50)
                if reporter:
                    reporter.log_info(f"[LLM] Rate-limited, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

            if job_id and job_id not in self._per_job_limiter:
                per_job_limits = self.config.rate_limits.get("per_job", {})
                self._per_job_limiter[job_id] = SlidingWindowRateLimiter(
                    per_job_limits.get("requests_per_minute", 10),
                    per_job_limits.get("window_seconds", 60)
                )

            if job_id:
                job_wait_time = self._per_job_limiter[job_id].get_wait_time()
                if job_wait_time > 0:
                    if progress_cb:
                        progress_cb(f"Job rate limited. Waiting {job_wait_time:.1f}s...", 50)
                    assert reporter is not None
                    if reporter:
                        reporter.log_info(f"[LLM] Job rate-limited, waiting {job_wait_time:.1f}s")
                    await asyncio.sleep(job_wait_time)

            # Log prompt interaction
            if interactions is not None:
                system_msg = next((m for m in messages if m.get("role") == "system"), None)
                user_msg = next((m for m in messages if m.get("role") == "user"), None)
                prompt_content = user_msg.get("content", "") if user_msg else json.dumps(messages)
                interactions.append(LLMInteraction(
                    timestamp=_now_timestamp(),
                    type="prompt",
                    role="user",
                    content=_truncate_preview(prompt_content),
                    full_content=prompt_content,
                    model=current_model,
                    retry_count=0,
                ))
                if system_msg:
                    interactions.append(LLMInteraction(
                        timestamp=_now_timestamp(),
                        type="prompt",
                        role="system",
                        content=_truncate_preview(system_msg.get("content", "")),
                        full_content=system_msg.get("content", ""),
                        model=current_model,
                        retry_count=0,
                    ))
                if reporter:
                    reporter.log_info(f"[LLM] Prompt sent ({stage}) — {len(prompt_content)} chars")

            # Enhanced retry loop (like call_llm_sync)
            backoff_delays = [1, 3, 6, 10]
            max_attempts = 5
            last_error = None

            for attempt in range(max_attempts):
                try:
                    with self._metrics_lock:
                        self._llm_calls += 1
                    self._rate_limiter.check_and_wait()
                    if job_id:
                        self._per_job_limiter[job_id].check_and_wait()

                    if attempt > 0 and interactions is not None:
                        interactions.append(LLMInteraction(
                            timestamp=_now_timestamp(),
                            type="retry",
                            role="assistant",
                            content=f"Retrying {current_model} (attempt {attempt + 1}/{max_attempts})",
                            full_content=f"Retry #{attempt + 1} with {current_model} after {_classify_llm_error(last_error)} error",
                            model=current_model,
                            retry_count=attempt,
                            error_type=_classify_llm_error(last_error) if last_error else "unknown",
                        ))
                        if reporter:
                            reporter.log_info(f"[LLM] Retry {attempt + 1}/{max_attempts} with {current_model} (reason: {_classify_llm_error(last_error)})")

                    response = await self.client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    raw_content = response.choices[0].message.content
                    if raw_content is None:
                        finish_reason = response.choices[0].finish_reason
                        refusal = getattr(response.choices[0].message, 'refusal', None)
                        raise RuntimeError(
                            f"NVIDIA API returned empty content. "
                            f"Finish reason: {finish_reason}. "
                            f"Refusal: {refusal}"
                        )

                    raw_content = raw_content.strip()
                    usage = getattr(response, 'usage', None)
                    token_count = ""
                    if usage:
                        token_count = f" ({usage.completion_tokens} out / {usage.prompt_tokens} in tokens)"

                    if self.config.cache.get("enabled", True):
                        get_cache().set(cache_key, raw_content, stage, current_model)

                    if interactions is not None:
                        interactions.append(LLMInteraction(
                            timestamp=_now_timestamp(),
                            type="response",
                            role="assistant",
                            content=_truncate_preview(raw_content),
                            full_content=raw_content,
                            model=current_model,
                            retry_count=attempt,
                        ))
                        if reporter:
                            reporter.log_info(f"[LLM] Response received{token_count} from {current_model}")

                    if progress_cb:
                        progress_cb(f"LLM response from {current_model}", 100)

                    return raw_content

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

                    if attempt < max_attempts - 1:
                        delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                        if reporter:
                            reporter.log_info(f"[LLM] Backoff {delay}s before retry {attempt + 2} with {current_model}")
                        await asyncio.sleep(delay)
                    else:
                        if reporter:
                            reporter.log_warn(f"[LLM] Model {current_model} exhausted all {max_attempts} retries")
                        with self._metrics_lock:
                            self._model_fallbacks += 1
                        break  # Try next model

        raise RuntimeError(f"All NIM models failed for stage {stage}. Last error: {last_error}")

    async def call_llm_with_fallback(
        self,
        messages: List[Dict[str, Any]],
        stage: str,
        job_id: Optional[str] = None,
        fallback_fn: Optional[Callable[[], str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 65536,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        reporter: Optional[Any] = None,
        interactions: Optional[List[LLMInteraction]] = None,
    ) -> str:
        try:
            return await self.call_llm(
                messages, stage, job_id, model, temperature, max_tokens, progress_cb,
                reporter=reporter, interactions=interactions,
            )
        except Exception as e:
            if fallback_fn:
                with self._metrics_lock:
                    self._fallback_used += 1
                if progress_cb:
                    progress_cb("Using local heuristic fallback...", 50)
                if reporter:
                    reporter.log_info("[LLM] All models failed, using local heuristic fallback")
                if interactions is not None:
                    interactions.append(LLMInteraction(
                        timestamp=_now_timestamp(),
                        type="retry",
                        role="system",
                        content="Falling back to local heuristic (all LLM models failed)",
                        full_content=str(e),
                        model="fallback",
                        retry_count=0,
                        error_type="fallback",
                    ))
                return fallback_fn()
            raise


provider = LLMProvider()