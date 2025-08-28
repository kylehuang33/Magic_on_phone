import ollama
client = ollama.Client(host='http://127.0.0.1:11434')

resp = client.chat(model='phi3:mini', messages=[
    {'role': 'user', 'content': 'Say hi in 5 words.'}
], stream=False)

print(resp['message']['content'])
