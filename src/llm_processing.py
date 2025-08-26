# llm_processing.py

import ollama
import json
from config import OLLAMA_MODEL

def query_ollama(text: str, system_prompt: str, task_id: str):
    """
    Sends the transcribed text and a system prompt to the Ollama model
    and prints the JSON response.
    """
    print(f"[{task_id}] Sending to Ollama: '{text}'")
    try:
        # Use both the system prompt and the user's transcribed text
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text}
            ],
            # Enforce JSON output for tool-calling
            options={'temperature': 0.0},
            format="json"
        )
        model_response = response['message']['content']
        print(f"--------------\n[{task_id}] OLLAMA RESPONSE (JSON Tool Call):\n{model_response}\n--------------")
        # In a real application, you would return this and call the MCP client
        return model_response

    except Exception as e:
        print(f"[{task_id}] Could not connect to Ollama server: {e}")
        print(f"[{task_id}] Please ensure the Ollama server is running with 'ollama serve'")
        return None