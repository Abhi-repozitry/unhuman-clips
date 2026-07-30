import sys
from openai import OpenAI, AuthenticationError, PermissionDeniedError, APIConnectionError

# 1. Configuration
API_KEY = "sk-J1sxOe4TNNrsRGzbETekNZovLl7CWwR7opsdXNzLHvD45kWLvNkZx2v43dMzX9Z0"
BASE_URL = "https://opencode.ai/zen/v1"
MODEL_NAME = "mimo-v2.5-free"

print("🔄 Testing OpenCode connection...")

try:
    # 2. Initialize the OpenAI-compatible client
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # 3. Send a fast, low-token text completion request
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Respond with exactly the word 'SUCCESS' if you read this."}
        ],
        max_tokens=50,
        temperature=0.0
    )
    
    # 4. Extract and print response
    msg = response.choices[0].message
    output_text = (msg.content or msg.reasoning or "").strip()
    print("\n✅ CONNECTION SUCCESSFUL!")
    print(f"🤖 Model Response: {output_text}")

except AuthenticationError as e:
    print("\n❌ AUTHENTICATION ERROR:")
    # Try to extract the real error from the response body
    try:
        import json
        body = json.loads(e.response.text)
        msg = body.get("error", {}).get("message", e.response.text)
        print(msg)
    except Exception:
        print(e.response.text)

except PermissionDeniedError:
    print("\n❌ PERMISSION DENIED ERROR:")
    print("This usually means your account needs to complete Step 1 ('Enable billing') before this key becomes active.")

except APIConnectionError:
    print("\n❌ NETWORK ERROR:")
    print(f"Could not reach the server at {BASE_URL}. Check your internet connection or the URL path.")

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR:\n{str(e)}")
