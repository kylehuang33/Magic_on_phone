import asyncio
import websockets
import os
import sys
import webrtcvad
import wave
import datetime
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
import ollama # <-- 1. IMPORT OLLAMA

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765
WHISPER_EXECUTABLE = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/models/ggml-base.en.bin"
WHISPER_ARGS = ["-m", MODEL_PATH, "-t", "4"]
OLLAMA_MODEL = "llama2-uncensored" # <-- 2. OLLAMA MODEL CONFIGURATION

# --- VAD Configuration (remains the same) ---
VAD_AGGRESSIVENESS = 3
VAD_SAMPLE_RATE = 16000
VAD_FRAME_DURATION_MS = 30
BYTES_PER_SAMPLE = 2
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * (VAD_FRAME_DURATION_MS / 1000.0) * BYTES_PER_SAMPLE)
SILENCE_FRAMES_THRESHOLD = 30
MAX_BUFFER_FRAMES = 300


# ==================================================================================
# NEW FUNCTION: Handles communication with the Ollama server
# ==================================================================================
def query_ollama(text, task_id):
    """
    Sends the transcribed text to the Ollama model and prints the response.
    """
    print(f"[{task_id}] Sending to Ollama: '{text}'")
    try:
        # The ollama.chat function communicates with the running Ollama server
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': text}]
        )
        # The model's response is in the 'content' of the message
        model_response = response['message']['content']
        print(f"[{task_id}] Ollama Response: {model_response}")

    except Exception as e:
        print(f"[{task_id}] Could not connect to Ollama server: {e}")
        print(f"[{task_id}] Please ensure the Ollama server is running with 'ollama serve'")


# ==================================================================================
# MODIFIED FUNCTION: Now calls the query_ollama function after transcription
# ==================================================================================
def transcription_worker(audio_buffer, task_id):
    """
    Saves audio to a temporary file, runs whisper-cli, and then sends
    the transcribed text to Ollama.
    """
    if not audio_buffer:
        return

    print(f"[{task_id}] Worker starting. Processing {len(b''.join(audio_buffer))} bytes.")

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            wav_path = tmpfile.name
            with wave.open(tmpfile, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(BYTES_PER_SAMPLE)
                wf.setframerate(VAD_SAMPLE_RATE)
                wf.writeframes(b''.join(audio_buffer))

            print(f"[{task_id}] Processing temporary file: {wav_path}")
            command = [WHISPER_EXECUTABLE] + WHISPER_ARGS + ["-f", wav_path]
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                transcription = result.stdout.strip().replace('[BLANK_AUDIO]', '').strip()
                if transcription and not transcription.startswith('['):
                    print(f"[{task_id}] Transcription: {transcription}")
                    # --- 4. INTEGRATION POINT ---
                    # Send the clean transcription to the Ollama model
                    query_ollama(transcription, task_id)
                    # ----------------------------
                else:
                    print(f"[{task_id}] Transcription: (No discernible speech detected)")
            else:
                print(f"[{task_id}] Whisper.cpp exited with error code: {result.returncode}")
                print(f"[{task_id}] Whisper.cpp stderr: {result.stderr.strip()}")

    except Exception as e:
        print(f"[{task_id}] An error occurred in the transcription worker: {e}")
    finally:
        print(f"[{task_id}] Worker finished.")


# --- No changes needed for the functions below this line ---

async def transcribe_audio_task(audio_buffer, task_id):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, transcription_worker, audio_buffer, task_id)

async def audio_handler(websocket, path):
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
                        asyncio.create_task(transcribe_audio_task(buffer_copy, task_id))
                        speech_buffer.clear()
                        silent_frames_count = 0
                except Exception as e:
                    print(f"Error during VAD processing: {e}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
        if speech_buffer:
            print("Client disconnected. Processing final buffer...")
            task_id = f"Task-Final-{task_counter}"
            await transcribe_audio_task(speech_buffer, task_id)
    except Exception as e:
        print(f"An error occurred in audio_handler: {e}")
    finally:
        print("Client session finished.")

async def main():
    async with websockets.serve(audio_handler, WEBSOCKET_HOST, WEBSOCKET_PORT):
        print(f"WebSocket server started at ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")