import asyncio
import websockets
import os
import sys

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"  # Listen on all available network interfaces
WEBSOCKET_PORT = 8765        # The port the server will listen on

# --- Path to your Whisper.cpp installation ---
# Using the confirmed absolute path to the executable and model
# WHISPER_EXECUTABLE = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/build/bin/main"
# --- Path to your Whisper.cpp installation ---
WHISPER_EXECUTABLE = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/models/ggml-base.en.bin"

# --- Whisper.cpp Streaming Parameters ---
WHISPER_ARGS = [
    "-m", MODEL_PATH,
    "-t", "4",          # Number of threads, adjust based on your phone's cores
    "--stream",         # <-- This is the crucial new flag for streaming
    "-f", "-",          # Read audio from standard input (stdin)
]

async def read_whisper_output(proc):
    """Asynchronously reads and prints the real-time transcription from whisper.cpp."""
    while True:
        line = await proc.stdout.readline()
        if not line:
            print("Whisper process stdout closed.")
            break
        text = line.decode('utf-8').strip()
        # Clean, simple filter for transcribed text
        if text and not text.startswith('[') and ']' not in text:
            print(f"Transcription: {text}")

async def audio_handler(websocket, path):
    """
    Handles a new WebSocket connection, receives audio, and pipes it to whisper.cpp.
    """
    print(f"Client connected from {websocket.remote_address}")

    # --- Verify that the whisper executable and model exist ---
    if not os.path.exists(WHISPER_EXECUTABLE):
        print(f"FATAL ERROR: Whisper executable not found at {WHISPER_EXECUTABLE}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Whisper model not found at {MODEL_PATH}")
        return

    print("Starting whisper.cpp subprocess...")
    # --- Start the whisper.cpp subprocess for this client connection ---
    proc = await asyncio.create_subprocess_exec(
        WHISPER_EXECUTABLE,
        *WHISPER_ARGS,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr  # Show whisper.cpp errors in the console
    )
    print(f"Whisper.cpp process started with PID: {proc.pid}")

    # --- Create a task to handle whisper's output ---
    output_task = asyncio.create_task(read_whisper_output(proc))

    try:
        # --- Receive audio from the client and forward it to whisper.cpp ---
        async for audio_chunk in websocket:
            if proc.stdin.is_closing():
                print("Whisper process stdin is closed. Cannot write more data.")
                break

            # The received 'audio_chunk' is bytes, which is exactly what we need.
            proc.stdin.write(audio_chunk)
            await proc.stdin.drain()

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
    except Exception as e:
        print(f"An error occurred in audio_handler: {e}")
    finally:
        print("Cleaning up whisper.cpp process...")
        # --- When the client disconnects, clean up the resources ---
        if proc.returncode is None: # Check if process is still running
            proc.stdin.close()
            await proc.wait()
        output_task.cancel()
        print("Cleanup complete.")

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