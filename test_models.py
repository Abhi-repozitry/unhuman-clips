"""Test both LLM models directly via OpenAI client."""
from __future__ import annotations

import sys
import time

import openai

from backend.config import (
    AVAILABLE_MODELS,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
)


def test_model(model_id: str, timeout: float = 60.0) -> tuple[bool, str, float]:
    """Test a single model. Returns (success, response, elapsed)."""
    client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, max_retries=0)
    messages = [{"role": "user", "content": "Say hello"}]

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=128,
            temperature=0.0,
            timeout=timeout,
        )
        elapsed = time.time() - start
        if response.choices:
            content = response.choices[0].message.content or ""
            finish = response.choices[0].finish_reason
            usage = getattr(response, 'usage', None)
            tokens = f"in={usage.prompt_tokens} out={usage.completion_tokens}" if usage else ""
            return True, f"[{finish}] {tokens} | {content.strip()[:100]}", elapsed
        return False, "No choices", elapsed
    except Exception as e:
        elapsed = time.time() - start
        return False, str(e)[:200], elapsed


def main():
    print("=" * 60)
    print("LLM MODEL TEST")
    print("=" * 60)
    print(f"API Key: {'SET' if NVIDIA_API_KEY else 'NOT SET'}")
    print(f"Base URL: {NVIDIA_BASE_URL}")

    models_to_try = [
        ("stepfun-ai/step-3.7-flash", 30.0),
        ("openai/gpt-oss-120b", 90.0),
    ]

    results = {}
    for model_id, timeout in models_to_try:
        print(f"\n--- {model_id} (timeout={timeout}s) ---")
        ok, resp, elapsed = test_model(model_id, timeout)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {elapsed:.1f}s")
        # Encode to ascii to avoid Windows console unicode errors
        print(f"  {resp.encode('ascii', 'replace').decode('ascii')}")
        results[model_id] = ok

    print(f"\n{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
