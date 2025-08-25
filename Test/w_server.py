import asyncio
import websockets
import os
import sys
import webrtcvad
import wave
import datetime
import subprocess # Use the standard subprocess module for the synchronous worker
from concurrent.futures import ThreadPoolExecutor

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
# NEW: Synchronous worker function for blocking tasks
# This function does the heavy lifting and will be run in a separate thread.
# ==================================================================================
def transcription_worker(audio_buffer, task_id):
    """
    This is a standard Python function (not async).
    It saves the audio file and runs the whisper-cli subprocess.
    """
    if not audio_buffer:
        return

    print(f"[{task_id}] Worker starting. Processing {len(b''.join(audio_buffer))} bytes.")

    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        wav_path = os.path.join(AUDIO_STORAGE_PATH, f"rec_{timestamp}.wav")

        # 1. Blocking I/O: Writing the file
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(VAD_SAMPLE_RATE)
            wf.writeframes(b''.join(audio_buffer))

        # 2. Blocking and CPU-intensive: Running the subprocess
        command = [WHISPER_EXECUTABLE] + WHISPER_ARGS + ["-f", wav_path]
        print(f"[{task_id}] Executing command: {' '.join(command)}")
        
        # Use the standard synchronous subprocess.run
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            transcription = result.stdout.strip()
            if transcription and not transcription.startswith('[') and ']' not in transcription:
                print(f"[{task_id}] Transcription: {transcription}")
        else:
            print(f"[{task_id}] Whisper.cpp exited with error code: {result.returncode}")
            print(f"[{task_id}] Whisper.cpp stderr: {result.stderr.strip()}")

    except Exception as e:
        print(f"[{task_id}] An error occurred in the transcription worker: {e}")
    finally:
        print(f"[{task_id}] Worker finished.")

# ==================================================================================
# MODIFIED: The async task now just handles handing off the work to the executor
# ==================================================================================
async def transcribe_audio_task(audio_buffer, task_id):
    """
    An async wrapper that runs the synchronous worker function in an executor.
    """
    loop = asyncio.get_running_loop()
    
    # loop.run_in_executor sends the blocking function (transcription_worker)
    # to a separate thread pool, so it doesn't block the main event loop.
    await loop.run_in_executor(
        None,  # Use the default ThreadPoolExecutor
        transcription_worker, # The function to run
        audio_buffer, # Arguments for the function
        task_id
    )

async def audio_handler(websocket, path):
    """
    Handles the WebSocket connection. This loop remains responsive.
    """
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
            # This print statement will now execute without interruption
            # print(f"Received {len(audio_chunk)} bytes...")
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
                        
                        # Create the async task that will run the worker in the background
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