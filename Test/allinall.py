import asyncio
import websockets
import os
import sys
import webrtcvad
import wave
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor
import ollama
import json 

# ==================================================================================
# 2. ADDED: MCP Client imports.
# Note: You must have the 'mcp-client' library installed.
# You can typically install it via pip: pip install mcp-client
# ==================================================================================
from fastmcp import Client
from fastmcp.client.transports import SSETransport

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765

# --- Audio Storage Configuration ---
AUDIO_STORAGE_PATH = "/data/data/com.termux/files/home/audio_recordings"
os.makedirs(AUDIO_STORAGE_PATH, exist_ok=True)
print(f"Audio recordings will be saved in: {AUDIO_STORAGE_PATH}")

# --- Whisper.cpp Paths and Arguments ---
WHISPER_EXECUTABLE = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/models/ggml-base.en.bin"
WHISPER_ARGS = ["-m", MODEL_PATH, "-t", "4"]

# --- Ollama Configuration ---
OLLAMA_MODEL = "gemma3:1b"
print(f"Using Ollama model: {OLLAMA_MODEL}")

# ==================================================================================
# 3. ADDED: MCP Configuration
# ==================================================================================
MCP_SERVER_URL = "http://127.0.0.1:6001" # <-- Set this to your MCP server's address
print(f"MCP Server URL set to: {MCP_SERVER_URL}")


# --- VAD Configuration ---
VAD_AGGRESSIVENESS = 3
VAD_SAMPLE_RATE = 16000
VAD_FRAME_DURATION_MS = 30
BYTES_PER_SAMPLE = 2
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * (VAD_FRAME_DURATION_MS / 1000.0) * BYTES_PER_SAMPLE)

# --- Speech Buffering Configuration ---
SILENCE_FRAMES_THRESHOLD = 30
MAX_BUFFER_FRAMES = 300

# ==================================================================================
# 4. MODIFIED: The function to query Ollama now accepts a system_prompt.
# ==================================================================================
def query_ollama(text, system_prompt, task_id):
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
    except Exception as e:
        print(f"[{task_id}] Could not connect to Ollama server: {e}")
        print(f"[{task_id}] Please ensure the Ollama server is running with 'ollama serve'")

# ==================================================================================
# This is YOUR proven, working function to clean whisper output. NO CHANGES MADE HERE.
# ==================================================================================
def clean_whisper_output(text):
    if not text:
        return ""
    text = text.replace("[BLANK_AUDIO]", "").replace("[SOUND]", "")
    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            if line.startswith('[') and '->' in line and ']' in line:
                parts = line.split(']', 1)
                if len(parts) > 1:
                    line = parts[1].strip()
                else:
                    line = ""
            if line.endswith('.'):
                line = line[:-1]
            if line.endswith('...'):
                line = line[:-3]
            if line:
                cleaned_lines.append(line)
    cleaned_text = " ".join(cleaned_lines).strip()
    return cleaned_text

# ==================================================================================
# 5. MODIFIED: The transcription worker now accepts and passes the system_prompt.
# ==================================================================================
def transcription_worker(audio_buffer, system_prompt, task_id):
    if not audio_buffer:
        return

    print(f"[{task_id}] Worker starting. Processing {len(b''.join(audio_buffer))} bytes.")
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        wav_path = os.path.join(AUDIO_STORAGE_PATH, f"rec_{timestamp}.wav")

        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(VAD_SAMPLE_RATE)
            wf.writeframes(b''.join(audio_buffer))

        command = [WHISPER_EXECUTABLE] + WHISPER_ARGS + ["-f", wav_path]
        print(f"[{task_id}] Executing command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            raw_output = result.stdout
            transcription = clean_whisper_output(raw_output)
            
            if transcription:
                print(f"[{task_id}] Transcription: {transcription}")
                # --- THIS IS THE INTEGRATION POINT ---
                query_ollama(transcription, system_prompt, task_id)
                # -------------------------------------
            else:
                print(f"[{task_id}] Transcription: (No discernible speech detected)")
        else:
            print(f"[{task_id}] Whisper.cpp exited with error code: {result.returncode}")
            print(f"[{task_id}] Whisper.cpp stderr: {result.stderr.strip()}")
    except Exception as e:
        print(f"[{task_id}] An error occurred in the transcription worker: {e}")
    finally:
        print(f"[{task_id}] Worker finished.")

# ==================================================================================
# 6. MODIFIED: The async task wrapper now accepts and passes the system_prompt.
# ==================================================================================
async def transcribe_audio_task(audio_buffer, system_prompt, task_id):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, transcription_worker, audio_buffer, system_prompt, task_id)

# ==================================================================================
# 7. MODIFIED: The main audio handler now accepts the system_prompt.
# ==================================================================================
async def audio_handler(websocket, path, system_prompt):
    print(f"Client connected from {websocket.remote_address}")
    if not all(map(os.path.exists, [WHISPER_EXECUTABLE, MODEL_PATH])):
        print("FATAL ERROR: Whisper executable or model not found.")
        return
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    speech_buffer = []
    silent_frames_count = 0
    frame_accumulator = b''
    task_counter = 0
    try:
        async for audio_chunk in websocket:
            frame_accumulator += audio_chunk
            while len(frame_accumulator) >= VAD_FRAME_SIZE:
                frame = frame_accumulator[:VAD_FRAME_SIZE]
                frame_accumulator = frame_accumulator[VAD_FRAME_SIZE:]
                try:
                    is_speech = vad.is_speech(frame, VAD_SAMPLE_RATE)
                    if is_speech:
                        speech_buffer.append(frame)
                        silent_frames_count = 0
                    elif speech_buffer:
                        silent_frames_count += 1
                    should_transcribe = (speech_buffer and silent_frames_count >= SILENCE_FRAMES_THRESHOLD) or \
                                      (len(speech_buffer) >= MAX_BUFFER_FRAMES)
                    if should_transcribe:
                        print("Utterance detected. Handing off to executor...")
                        buffer_copy = list(speech_buffer)
                        task_id = f"Task-{task_counter}"
                        task_counter += 1
                        asyncio.create_task(transcribe_audio_task(buffer_copy, system_prompt, task_id))
                        speech_buffer.clear()
                        silent_frames_count = 0
                except Exception as e:
                    print(f"Error during VAD processing: {e}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
        if speech_buffer:
            print("Client disconnected. Processing final buffer...")
            task_id = f"Task-Final-{task_counter}"
            await transcribe_audio_task(speech_buffer, system_prompt, task_id)
    except Exception as e:
        print(f"An error occurred in audio_handler: {e}")
    finally:
        print("Client session finished.")

# ==================================================================================
# 8. ADDED: Function to fetch the tool list from the MCP server.
# ==================================================================================
async def get_mcp_tools():
    """Connects to the MCP server and fetches the list of available tools."""
    print(f"Attempting to connect to MCP server at {MCP_SERVER_URL} to get tools...")
    try:
        async with Client(transport=SSETransport(f"{MCP_SERVER_URL}")) as client:
            tool_list = await client.list_tools()
            print("Successfully fetched tool list from MCP server.")

            # Convert the list of Tool objects to a list of dictionaries
            tools_as_dicts = []
            for tool in tool_list:
                tool_dict = {
                    "name": getattr(tool, 'name', 'N/A'),
                    "description": getattr(tool, 'description', ''),
                    "parameters": getattr(tool, 'input_schema', {})
                }
                tools_as_dicts.append(tool_dict)

            # Pretty-print the tools as a JSON string for the prompt
            return json.dumps(tools_as_dicts, indent=2)
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to MCP server or fetch tools: {e}")
        print("Please ensure the MCP server is running and accessible.")
        return None

# ==================================================================================
# 9. MODIFIED: The main function now fetches tools and prepares the handler.
# ==================================================================================
async def main():
    # Fetch the MCP tool list on startup
    tool_list_str = await get_mcp_tools()
    if not tool_list_str:
        return # Exit if we can't get the tools

    # This is the system prompt you provided, now formatted with the fetched tool list
    system_prompt = f"""
You are a tool-calling assistant. Your only job is to convert a user's request into a JSON array of tool calls based on the tools provided.

**RULES:**
1.  **JSON ONLY:** Your entire response must be a valid JSON array. Do not add any other text, explanations, or greetings.
2.  **CORRECT FORMAT:** The array must contain one or more objects. Each object must have a "name" key (the tool name as a string) and an "arguments" key (an object with the parameters).
3.  **USE PROVIDED TOOLS:** Only use the exact tool names available in the `<tools>` list. Do not make up tools.
4.  **BE CONCISE:** If the user request is simple, only use the necessary tools.
<tools>
{tool_list_str}
</tools>
**EXAMPLE:**

User request: open the youtube APP.

Your response:
[
  {{"name": "mobile_use_default_device", "arguments": {{}}}},
  {{"name": "mobile_launch_app", "arguments": {{"packageName": "com.google.android.youtube"}}}}
]
"""
    # Create a handler that has the system_prompt "baked in"
    handler_with_prompt = lambda ws, path: audio_handler(ws, path, system_prompt=system_prompt)

    async with websockets.serve(handler_with_prompt, WEBSOCKET_HOST, WEBSOCKET_PORT):
        print(f"WebSocket server started at ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        print("Server is ready and waiting for a client connection...")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")