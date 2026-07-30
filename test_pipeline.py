import os
import sys

# Simulate env
os.environ["OPENCODE_API_KEY"] = "sk-J1sxOe4TNNrsRGzbETekNZovLl7CWwR7opsdXNzLHvD45kWLvNkZx2v43dMzX9Z0"

sys.path.insert(0, r"C:\Projects\unhuman-clips")

from backend.config import (
    OPENCODE_API_KEY, OPENCODE_BASE_URL, OPENCODE_MODEL, OPENCODE_MODEL_FALLBACK,
    AVAILABLE_MODELS, MODEL_FALLBACK_MAP,
)
from backend.providers.llm import call_llm_sync
from backend.models import LLMInteraction

print("=" * 50)
print("CONFIG CHECK")
print("=" * 50)
print(f"API Key:     {OPENCODE_API_KEY[:10]}...{OPENCODE_API_KEY[-6:]}")
print(f"Base URL:    {OPENCODE_BASE_URL}")
print(f"Model:       {OPENCODE_MODEL}")
print(f"Fallback:    {OPENCODE_MODEL_FALLBACK}")
print(f"Models:      {AVAILABLE_MODELS}")
print(f"Fallback Map:{MODEL_FALLBACK_MAP}")

print("\n" + "=" * 50)
print("LLM CALL TEST")
print("=" * 50)

interactions: list[LLMInteraction] = []
result = call_llm_sync(
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Respond concisely."},
        {"role": "user", "content": "Respond with exactly the word SUCCESS if you read this."},
    ],
    model=OPENCODE_MODEL,
    api_key=OPENCODE_API_KEY,
    base_url=OPENCODE_BASE_URL,
    max_tokens=50,
    temperature=0.0,
    interactions=interactions,
    stage_name="test",
)

print(f"Response:    {result}")
print(f"Interactions: {len(interactions)} recorded")

if "SUCCESS" in result.upper():
    print("\n[PASS] Model responded correctly!")
else:
    print(f"\n[WARN] Expected 'SUCCESS', got: {result}")

print("\nDone.")
