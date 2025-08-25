import asyncio
import websockets
import os
import sys
import webrtcvad

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"  # Listen on all available network interfaces
WEBSOCKET_PORT = 8765        # The port the server will listen on

# --- Path to your Whisper.cpp installation ---
# Using the confirmed absolute path to the executable and model
WHISPER_EXECUTABLE = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/models/ggml-base.en.bin"

# --- Whisper.cpp Parameters ---
# NOTE: We are no longer using --stream as we process buffered audio
WHISPER_ARGS = [
    "-m", MODEL_PATH,
    "-t", "4",          # Number of threads, adjust based on your phone's cores
    "-f", "-",          # Read audio from standard input (stdin)
]

# --- VAD Configuration ---
# NOTE: webrtcvad requires audio to be 16-bit 16kHz mono.
# The client sending the audio must be configured for this format.
VAD_AGGRESSIVENESS = 3  # 0 to 3, 3 is the most aggressive in filtering out non-speech
VAD_FRAME_DURATION_MS = 30  # Duration of a frame in milliseconds
VAD_SAMPLE_RATE = 16000
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * (VAD_FRAME_DURATION_MS / 1000.0)) # Bytes per frame

SILENCE_FRAMES_COUNT = 50 # Number of consecutive silent frames to trigger transcription

async def transcribe_audio_buffer(audio_buffer):
    """
    Calls the whisper-cli subprocess to transcribe the buffered audio data.
    """
    print(f"Processing {len(audio_buffer)} bytes of audio for transcription...")

    # --- Start the whisper.cpp subprocess for this audio chunk ---
    proc = await asyncio.create_subprocess_exec(
        WHISPER_EXECUTABLE,
        *WHISPER_ARGS,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr  # Show whisper.cpp errors in the console
    )

    # --- Write the buffered audio to whisper's stdin ---
    try:
        stdout, stderr = await proc.communicate(input=b''.join(audio_buffer))
        if stdout:
            transcription = stdout.decode('utf-8').strip()
            # Clean, simple filter for transcribed text
            if transcription and not transcription.startswith('[') and ']' not in transcription:
                print(f"Transcription: {transcription}")
    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()

async def audio_handler(websocket, path):
    """
    Handles a new WebSocket connection, receives audio, uses VAD to buffer speech,
    and then calls whisper.cpp for transcription.
    """
    print(f"Client connected from {websocket.remote_address}")

    # --- Verify that the whisper executable and model exist ---
    if not os.path.exists(WHISPER_EXECUTABLE):
        print(f"FATAL ERROR: Whisper executable not found at {WHISPER_EXECUTABLE}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Whisper model not found at {MODEL_PATH}")
        return

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    vad_buffer = []
    silent_frames = 0

    try:
        async for audio_chunk in websocket:
            # --- Ensure the audio chunk is the correct size for VAD ---
            # This example assumes the client sends audio in perfect frame sizes.
            # In a real-world application, you would need to buffer and segment
            # the incoming audio into VAD_FRAME_SIZE chunks.
            if len(audio_chunk) != VAD_FRAME_SIZE:
                # This part needs to be more robust in a production system
                # It should handle chunks that are smaller or larger than a frame.
                print(f"Warning: Received chunk of size {len(audio_chunk)}, expected {VAD_FRAME_SIZE}. Skipping.")
                continue

            is_speech = vad.is_speech(audio_chunk, VAD_SAMPLE_RATE)

            if is_speech:
                vad_buffer.append(audio_chunk)
                silent_frames = 0
            else:
                silent_frames += 1
                if vad_buffer and silent_frames > SILENCE_FRAMES_COUNT:
                    # We have a buffer of speech and have detected silence, so transcribe.
                    await transcribe_audio_buffer(vad_buffer)
                    vad_buffer = [] # Clear the buffer after processing
                    silent_frames = 0

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
        # If there's remaining audio in the buffer when the client disconnects,
        # process it.
        if vad_buffer:
            print("Processing remaining audio buffer after disconnection...")
            await transcribe_audio_buffer(vad_buffer)
    except Exception as e:
        print(f"An error occurred in audio_handler: {e}")
    finally:
        print("Client disconnected. Cleanup complete.")


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