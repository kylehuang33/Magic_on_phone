import ollama

resp = ollama.chat(
    model="gemma3:1b",   # ⚠️ see note on model below
    messages=[
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
    stream=False,        # <--- IMPORTANT
)

print(resp["message"]["content"])
