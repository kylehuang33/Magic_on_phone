import asyncio
import websockets
import os
import sys
import webrcvad
import wave
import datetime  # Import datetime to create unique filenames

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"  # Listen on all available network interfaces
WEBSOCKET_PORT = 8765        # The port the server will listen on

# ==================================================================================
# NEW: Define a directory to store the recorded audio clips
# ==================================================================================
# You can change this to any path you have write access to.
AUDIO_STORAGE_PATH = "/data/data/com.termux/files/home/audio_recordings"

# --- Create the directory if it doesn't exist ---
os.makedirs(AUDIO_STORAGE_PATH, exist_ok=True)
print(f"Audio recordings will be saved in: {AUDIO_STORAGE_PATH}")
# ==================================================================================


# --- Path to your Whisper.cpp installation ---
WHISPER_EXECUTABLE = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/models/ggml-base.en.bin"

# --- Whisper.cpp Parameters ---
WHISPER_ARGS = [
    "-m", MODEL_PATH,
    "-t", "4",
]

# --- VAD Configuration ---
VAD_AGGRESSIVENESS = 3
VAD_SAMPLE_RATE = 16000
VAD_FRAME_DURATION_MS = 30
BYTES_PER_SAMPLE = 2 # 16-bit
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * (VAD_FRAME_DURATION_MS / 1000.0) * BYTES_PER_SAMPLE)

# --- Speech Buffering Configuration ---
SILENCE_FRAMES_THRESHOLD = 30
MAX_BUFFER_FRAMES = 300


# ==================================================================================
# MODIFIED: This function now saves the file permanently
# ==================================================================================
async def transcribe_audio_buffer(audio_buffer):
    """
    Saves the buffered audio to a permanent WAV file and calls whisper-cli
    to transcribe it.
    """
    if not audio_buffer:
        print("Transcription buffer is empty, skipping.")
        return

    print(f"Processing {len(b''.join(audio_buffer))} bytes of audio...")

    try:
        # Create a unique filename based on the current date and time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        wav_path = os.path.join(AUDIO_STORAGE_PATH, f"recording_{timestamp}.wav")

        print(f"Saving audio to: {wav_path}")

        # Use the wave module to write the raw audio bytes as a proper WAV file
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(BYTES_PER_SAMPLE)  # 16-bit
            wf.setframerate(VAD_SAMPLE_RATE)   # 16kHz
            wf.writeframes(b''.join(audio_buffer))

        # Now, call whisper-cli with the path to the newly saved WAV file
        command = [WHISPER_EXECUTABLE] + WHISPER_ARGS + ["-f", wav_path]

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode == 0 and stdout:
            transcription = stdout.decode('utf-8', errors='ignore').strip()
            if transcription and not transcription.startswith('[') and ']' not in transcription:
                print(f"Transcription: {transcription}")
        elif proc.returncode != 0:
            print(f"Whisper.cpp exited with error code: {proc.returncode}")

    except Exception as e:
        print(f"An error occurred during transcription: {e}")


async def audio_handler(websocket, path):
    """
    Handles a new WebSocket connection, receives audio, uses VAD to buffer speech,
    and then calls whisper.cpp for transcription.
    """
    print(f"Client connected from {websocket.remote_address}")

    if not os.path.exists(WHISPER_EXECUTABLE) or not os.path.exists(MODEL_PATH):
        print("FATAL ERROR: Whisper executable or model not found.")
        return

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    speech_buffer = []
    silent_frames_count = 0
    frame_accumulator = b''

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

                    if speech_buffer and silent_frames_count >= SILENCE_FRAMES_THRESHOLD:
                        await transcribe_audio_buffer(speech_buffer)
                        speech_buffer = []
                        silent_frames_count = 0

                    if len(speech_buffer) >= MAX_BUFFER_FRAMES:
                        print("Max buffer length reached, transcribing...")
                        await transcribe_audio_buffer(speech_buffer)
                        speech_buffer = []

                except Exception as e:
                    print(f"Error during VAD processing: {e}")


    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
        if speech_buffer:
            print("Processing remaining audio buffer after disconnection...")
            await transcribe_audio_buffer(speech_buffer)
    except Exception as e:
        print(f"An error occurred in audio_handler: {e}")
    finally:
        print("Client session finished.")


async def main():
    """Starts the WebSocket server."""
    async with websockets.serve(audio_handler, WEBSOCKET_HOST, WEBSOCKET_PORT):
        print(f"WebSocket server started at ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")