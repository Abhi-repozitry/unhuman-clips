from openai import OpenAI

API_KEY = "sk-J1sxOe4TNNrsRGzbETekNZovLl7CWwR7opsdXNzLHvD45kWLvNkZx2v43dMzX9Z0"
BASE_URL = "https://opencode.ai/zen/v1"
MODEL_NAME = "mimo-v2.5-free"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"Chat with {MODEL_NAME} (type 'quit' to exit)\n")

messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit", "q"):
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )

    msg = response.choices[0].message
    reply = (msg.content or msg.reasoning or "").strip()
    messages.append({"role": "assistant", "content": reply})
    print(f"AI: {reply}\n")
