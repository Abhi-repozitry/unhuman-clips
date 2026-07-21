import os
from dotenv import load_dotenv
from backend.pipeline.analyzer import _call_llm, _build_reel_plan_prompt

load_dotenv()

print("=== NVIDIA LLM Connection Test ===")
print("API Key loaded:", bool(os.getenv("NVIDIA_API_KEY")))
print("Primary Model:", os.getenv("NVIDIA_MODEL"))
print("Fallback Model:", os.getenv("NVIDIA_MODEL_FALLBACK"))

# Very small test prompt
messages = [
    {"role": "system", "content": "You must respond with ONLY valid JSON. No other text."},
    {"role": "user", "content": "Return this exact JSON: {\"test\": \"hello world\", \"number\": 42}"}
]

try:
    response = _call_llm(messages)
    print("\n✅ LLM Response:")
    print(response)
except Exception as e:
    print("\n❌ LLM Call Failed:")
    print(type(e).__name__, str(e))