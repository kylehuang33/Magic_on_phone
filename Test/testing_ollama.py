import ollama
response = ollama.chat(model='gemma3:1b', messages=[
  {
    'role': 'system',
    'content': 'You are an AI assistant.',
  },
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])