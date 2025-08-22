import asyncio
import websockets
import os
import sys

# --- Configuration ---
SOURCE_WEBSOCKET_URL = "ws://127.0.0.1:8765"  # IMPORTANT: Change this

# --- Path to your Whisper.cpp installation ---
# Assumes you cloned whisper.cpp in the Termux home directory
# Adjust this path if you cloned it elsewhere
# HOME_DIR = "/data/data/com.termux/files/home/DOING_PROJECTS"
# WHISPER_CPP_DIR = os.path.join(HOME_DIR, "Magic_on_phone/modules/whisper.cpp")
# WHISPER_EXECUTABLE = os.path.join(WHISPER_CPP_DIR, "main")
WHISPER_EXECUTABLE = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/build/bin/main"

# You still need the model path, so make sure this line is correct too
MODEL_PATH = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/models/ggml-base.en.bin"

# --- Whisper.cpp Streaming Parameters ---
# These parameters are crucial for real-time performance.
# --step 4000: Process audio every 4000ms (4 seconds)
# --length 8000: Use an audio context of 8000ms (8 seconds)
# -t 4: Use 4 threads for processing (adjust based on your phone's cores)
# -f -: This is the magic flag that tells whisper.cpp to read audio from stdin
WHISPER_ARGS = [
    "-m", MODEL_PATH,
    "-t", "4",
    "--step", "4000",
    "--length", "8000",
    "-f", "-",
]

async def read_whisper_output(proc):
    """Asynchronously reads and prints the output from the whisper.cpp process."""
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        # The output from whisper.cpp includes formatting we can clean up
        # It prints transcribed text between [..] or directly.
        # This is a simple parser; you might want to make it more robust.
        text = line.decode('utf-8').strip()
        # Filter out non-transcription lines
        if text and not text.startswith('[') and ']' not in text:
             print(f"Transcription: {text}")


async def stream_audio_to_whisper():
    """Connects to WebSocket and pipes audio to a local whisper.cpp process."""
    print("Starting local transcription process...")

    # --- Check if whisper executable and model exist ---
    if not os.path.exists(WHISPER_EXECUTABLE):
        print(f"Error: Whisper executable not found at {WHISPER_EXECUTABLE}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Whisper model not found at {MODEL_PATH}")
        return

    # --- Start the whisper.cpp subprocess ---
    process = await asyncio.create_subprocess_exec(
        WHISPER_EXECUTABLE,
        *WHISPER_ARGS,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr  # Show errors from whisper.cpp in the console
    )
    print(f"Whisper.cpp process started with PID: {process.pid}")

    # --- Create a task to handle whisper's output ---
    output_task = asyncio.create_task(read_whisper_output(process))

    # --- Connect to the WebSocket and stream audio ---
    try:
        async with websockets.connect(SOURCE_WEBSOCKET_URL) as websocket:
            print(f"Connected to audio source at {SOURCE_WEBSOCKET_URL}")
            print("Receiving audio and piping to Whisper.cpp...")

            async for audio_chunk in websocket:
                # IMPORTANT: This assumes the audio chunk is in the correct format.
                # See the "Important Considerations" section below.
                if process.stdin.is_closing():
                    print("Subprocess stdin is closed. Cannot write more data.")
                    break

                process.stdin.write(audio_chunk)
                await process.stdin.drain() # Ensure the data is written

    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing whisper.cpp process...")
        if not process.stdin.is_closing():
            process.stdin.close()
            await process.wait()
        output_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(stream_audio_to_whisper())
    except KeyboardInterrupt:
        print("\nStream stopped by user.")