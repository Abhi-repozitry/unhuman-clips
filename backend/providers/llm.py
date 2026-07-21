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


def call_llm_sync(
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    temperature: float = 0.1,
    max_tokens: int = 1200,
    timeout: float = 480.0,
) -> str:
    """Synchronous LLM call with retry on failure and exponential backoff.
    Falls back to NVIDIA_MODEL_FALLBACK if primary model fails.
    """
    from backend.config import NVIDIA_MODEL_FALLBACK

    models_to_try = [model]
    if NVIDIA_MODEL_FALLBACK and NVIDIA_MODEL_FALLBACK != model:
        models_to_try.append(NVIDIA_MODEL_FALLBACK)

    backoff_delays = [3, 8, 15]
    last_error = None

    for m in models_to_try:
        for attempt in range(2):
            try:
                client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
                kwargs = {
                    "model": m,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                }
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
                return raw.strip()
            except Exception as e:
                last_error = e
                if attempt == 0:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    time.sleep(delay)
                    continue
                else:
                    break
    raise RuntimeError(f"All NVIDIA models failed. Last error: {last_error}")


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
        primary = nvidia_config.get("primary", "openai/gpt-oss-20b")
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
        max_tokens: int = 1200,
        progress_cb: Optional[Callable[[str, float], None]] = None
    ) -> str:
        primary, fallback = self.get_model_config(stage)
        models_to_try = [model or primary]
        if fallback and fallback != primary:
            models_to_try.append(fallback)

        params = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        prompt = messages[0]["content"] if messages else ""

        for model in models_to_try:
            cache_key = self.generate_cache_key(stage, prompt, params, model)

            if self.config.cache.get("enabled", True):
                cached_response = get_cache().get(cache_key)
                if cached_response is not None:
                    with self._metrics_lock:
                        self._cache_hits += 1
                    if progress_cb:
                        progress_cb(f"Cache hit for {model}", 100)
                    return cached_response
                with self._metrics_lock:
                    self._cache_misses += 1

            wait_time = self._rate_limiter.get_wait_time()
            if wait_time > 0:
                with self._metrics_lock:
                    self._rate_limit_waits += 1
                    self._total_wait_time += wait_time
                if progress_cb:
                    progress_cb(f"Rate limited. Waiting {wait_time:.1f}s...", 50)
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
                    await asyncio.sleep(job_wait_time)

            try:
                with self._metrics_lock:
                    self._llm_calls += 1
                self._rate_limiter.check_and_wait()
                if job_id:
                    self._per_job_limiter[job_id].check_and_wait()

                response = await self.client.chat.completions.create(
                    model=model,
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
                if self.config.cache.get("enabled", True):
                    get_cache().set(cache_key, raw_content.strip(), stage, model)
                if progress_cb:
                    progress_cb(f"LLM response from {model}", 100)
                return raw_content.strip()

            except Exception as e:
                if progress_cb:
                    progress_cb(f"Model {model} failed, trying fallback...", 30)
                print(f"[WARN] LLM call failed with model {model}: {e}")
                with self._metrics_lock:
                    self._model_fallbacks += 1
                continue

        raise RuntimeError(f"All NIM models failed for stage {stage}")

    async def call_llm_with_fallback(
        self,
        messages: List[Dict[str, Any]],
        stage: str,
        job_id: Optional[str] = None,
        fallback_fn: Optional[Callable[[], str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        progress_cb: Optional[Callable[[str, float], None]] = None
    ) -> str:
        try:
            return await self.call_llm(
                messages, stage, job_id, model, temperature, max_tokens, progress_cb
            )
        except Exception as e:
            if fallback_fn:
                with self._metrics_lock:
                    self._fallback_used += 1
                if progress_cb:
                    progress_cb("Using local heuristic fallback...", 50)
                return fallback_fn()
            raise


provider = LLMProvider()
