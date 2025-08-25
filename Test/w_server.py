import asyncio
import websockets
import os
import sys
import webrtcvad
import wave
import datetime

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


async def transcribe_audio_buffer(audio_buffer):
    """
    This function now runs as an independent, background task.
    It saves the audio to a file and calls whisper-cli.
    """
    if not audio_buffer:
        return

    task_id = asyncio.current_task().get_name()
    print(f"[{task_id}] Starting transcription for {len(b''.join(audio_buffer))} bytes of audio.")

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

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE  # Capture stderr as well
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            transcription = stdout.decode('utf-8', errors='ignore').strip()
            if transcription and not transcription.startswith('[') and ']' not in transcription:
                print(f"[{task_id}] Transcription: {transcription}")
        else:
            # Log errors from whisper.cpp, which is crucial for debugging
            print(f"[{task_id}] Whisper.cpp exited with error code: {proc.returncode}")
            print(f"[{task_id}] Whisper.cpp stderr: {stderr.decode('utf-8', errors='ignore').strip()}")

    except Exception as e:
        print(f"[{task_id}] An error occurred during transcription: {e}")
    finally:
        print(f"[{task_id}] Transcription task finished.")


async def audio_handler(websocket, path):
    """
    Handles the WebSocket connection and incoming audio stream.
    """
    print(f"Client connected from {websocket.remote_address}")

    if not os.path.exists(WHISPER_EXECUTABLE) or not os.path.exists(MODEL_PATH):
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
                    else:
                        if speech_buffer:
                            silent_frames_count += 1
                    
                    # Check if we should trigger a transcription task
                    should_transcribe = (speech_buffer and silent_frames_count >= SILENCE_FRAMES_THRESHOLD) or \
                                      (len(speech_buffer) >= MAX_BUFFER_FRAMES)

                    if should_transcribe:
                        # ==========================================================
                        # THIS IS THE KEY FIX
                        # ==========================================================
                        # We have a complete utterance. Create a background task
                        # to process it. DO NOT await it here.
                        print("Utterance detected. Creating background task for transcription.")
                        
                        # Pass a COPY of the buffer to the task.
                        buffer_copy = list(speech_buffer)
                        
                        # Give the task a unique name for better logging
                        task_name = f"TranscriptionTask-{task_counter}"
                        task_counter += 1
                        
                        asyncio.create_task(transcribe_audio_buffer(buffer_copy), name=task_name)
                        
                        # Immediately reset the buffer and continue listening.
                        speech_buffer = []
                        silent_frames_count = 0
                        # ==========================================================

                except Exception as e:
                    print(f"Error during VAD processing: {e}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
        if speech_buffer:
            print("Client disconnected. Processing final audio buffer...")
            # Still process the last bit of audio, but no need to create a task
            # as the server loop for this client is ending anyway.
            await transcribe_audio_buffer(speech_buffer)
    except Exception as e:
        print(f"An error occurred in audio_handler: {e}")
    finally:
        print("Client session finished.")


async def main():
    """Starts the WebSocket server."""
    async with websockets.serve(audio_handler, WEBSOCKET_HOST, WEBSOCKET_PORT):
        print(f"WebSocket server started at ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")