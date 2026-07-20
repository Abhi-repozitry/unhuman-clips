"""Test the NVIDIA API models."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from backend.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL, NVIDIA_MODEL_FALLBACK
import openai

client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, max_retries=0)

print(f"API Key present: {bool(NVIDIA_API_KEY)}")
print(f"Primary model: {NVIDIA_MODEL}")
print(f"Fallback model: {NVIDIA_MODEL_FALLBACK}")

models_to_test = [NVIDIA_MODEL]
if NVIDIA_MODEL_FALLBACK:
    models_to_test.append(NVIDIA_MODEL_FALLBACK)

for model in models_to_test:
    print(f"\nTesting model: {model}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You must respond with ONLY valid JSON."},
                {"role": "user", "content": 'Say {"test": "hello"} but in JSON only'}
            ],
            temperature=0.1,
            max_tokens=200,
            timeout=60.0
        )
        content = resp.choices[0].message.content
        print(f"  Response: {content!r}")
        print(f"  Finish reason: {resp.choices[0].finish_reason}")
        if resp.choices[0].message.refusal:
            print(f"  REFUSAL: {resp.choices[0].message.refusal}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Type: {type(e).__name__}")