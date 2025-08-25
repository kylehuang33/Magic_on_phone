import asyncio
import websockets
import os
import sys
import webrtcvad

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"  # Listen on all available network interfaces
WEBSOCKET_PORT = 8765        # The port the server will listen on

# --- Path to your Whisper.cpp installation ---
WHISPER_EXECUTABLE = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/models/ggml-base.en.bin"

# --- Whisper.cpp Parameters ---
WHISPER_ARGS = [
    "-m", MODEL_PATH,
    "-t", "4",          # Number of threads, adjust based on your phone's cores
    "-f", "-",          # Read audio from standard input (stdin)
]

# --- VAD Configuration ---
# NOTE: webrtcvad requires audio to be 16-bit 16kHz mono.
# The client sending the audio must be configured for this format.
VAD_AGGRESSIVENESS = 3      # 0 to 3, 3 is the most aggressive in filtering out non-speech
VAD_SAMPLE_RATE = 16000
# VAD requires specific frame durations: 10, 20, or 30 ms.
VAD_FRAME_DURATION_MS = 30  # Duration of a frame in milliseconds
BYTES_PER_SAMPLE = 2        # 16-bit audio means 2 bytes per sample
# This is the required size of a frame for the VAD
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * (VAD_FRAME_DURATION_MS / 1000.0) * BYTES_PER_SAMPLE)

# --- Speech Buffering Configuration ---
# How many consecutive silent frames trigger a transcription
SILENCE_FRAMES_THRESHOLD = 30
# If the buffer gets too long, transcribe it anyway to avoid excessive memory use
MAX_BUFFER_FRAMES = 300 # Approx 9 seconds of audio (300 frames * 30ms)

async def transcribe_audio_buffer(audio_buffer):
    """
    Calls the whisper-cli subprocess to transcribe the buffered audio data.
    """
    if not audio_buffer:
        print("Transcription buffer is empty, skipping.")
        return

    print(f"Processing {len(b''.join(audio_buffer))} bytes of audio for transcription...")

    # Start the whisper.cpp subprocess for this audio chunk
    proc = await asyncio.create_subprocess_exec(
        WHISPER_EXECUTABLE,
        *WHISPER_ARGS,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr  # Show whisper.cpp errors in the console
    )

    # Write the buffered audio to whisper's stdin and get the result
    try:
        stdout, stderr = await proc.communicate(input=b''.join(audio_buffer))
        if proc.returncode == 0 and stdout:
            transcription = stdout.decode('utf-8', errors='ignore').strip()
            # Clean, simple filter for transcribed text
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
    speech_buffer = []      # Buffer to hold frames of speech
    silent_frames_count = 0 # Counter for consecutive silent frames
    
    # *** THIS IS THE KEY FIX ***
    # Buffer to accumulate incoming audio chunks from the websocket
    frame_accumulator = b''

    try:
        async for audio_chunk in websocket:
            frame_accumulator += audio_chunk

            # Process the accumulator in VAD-sized frames
            while len(frame_accumulator) >= VAD_FRAME_SIZE:
                # Extract one frame for VAD processing
                frame = frame_accumulator[:VAD_FRAME_SIZE]
                # Remove the processed frame from the accumulator
                frame_accumulator = frame_accumulator[VAD_FRAME_SIZE:]

                try:
                    is_speech = vad.is_speech(frame, VAD_SAMPLE_RATE)
                    if is_speech:
                        speech_buffer.append(frame)
                        silent_frames_count = 0
                    else:
                        if speech_buffer: # If we have speech buffered and now detect silence
                            silent_frames_count += 1
                    
                    # Trigger transcription if we have a buffer and hit the silence threshold
                    if speech_buffer and silent_frames_count >= SILENCE_FRAMES_THRESHOLD:
                        await transcribe_audio_buffer(speech_buffer)
                        speech_buffer = [] # Reset buffer
                        silent_frames_count = 0

                    # Also trigger transcription if the buffer gets too long
                    if len(speech_buffer) >= MAX_BUFFER_FRAMES:
                        print("Max buffer length reached, transcribing...")
                        await transcribe_audio_buffer(speech_buffer)
                        speech_buffer = [] # Reset buffer

                except Exception as e:
                    # This can happen if the audio format is incorrect
                    print(f"Error during VAD processing: {e}")


    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
        # If there's remaining audio in the buffer when the client disconnects, process it.
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