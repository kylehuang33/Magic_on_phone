import asyncio
import websockets
import os
import sys

# --- Configuration ---
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765

# --- Paths to your executables and models ---
# Ensure the whisper executable name is correct for your build.
# It might be 'main', 'stream', or 'whisper-cli'.
WHISPER_EXECUTABLE = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = "/data/data/com.termux/files/home/DOING_PROJECTS/Magic_on_phone/modules/whisper.cpp/models/ggml-base.en.bin"

# --- FFmpeg Command ---
# This command reads from stdin, assumes the input is webm/opus (common from browsers),
# and converts it to 16-bit PCM mono audio at 16kHz, which is what whisper.cpp expects.
FFMPEG_COMMAND = [
    "ffmpeg",
    "-i", "pipe:0",          # Read from standard input
    "-f", "s16le",           # Output format: signed 16-bit little-endian PCM
    "-ac", "1",              # Output audio channels: 1 (mono)
    "-ar", "16000",          # Output sample rate: 16kHz
    "pipe:1"                 # Write to standard output
]

# --- Whisper.cpp Command ---
# The `-f -` argument tells whisper to read audio data from stdin.
WHISPER_ARGS = [
    "-m", MODEL_PATH,
    "-t", "4",
    "-f", "-",
    # Add any other desired whisper.cpp flags here
    # "--no-timestamps",
]

async def read_whisper_output(proc, websocket):
    """Asynchronously reads transcription from whisper.cpp and sends it to the client."""
    while True:
        line = await proc.stdout.readline()
        if not line:
            print("Whisper process stdout closed.")
            break
        text = line.decode('utf-8', errors='ignore').strip()
        # Filter out progress/logging messages from whisper.cpp
        if text and not text.startswith('[') and ']' not in text:
            print(f"Transcription: {text}")
            try:
                # Send the clean transcription to the WebSocket client
                await websocket.send(text)
            except websockets.exceptions.ConnectionClosed:
                print("Client connection closed while sending transcription.")
                break

async def audio_handler(websocket, path):
    """
    Handles a WebSocket connection: starts FFmpeg and Whisper.cpp,
    pipes audio from the client to FFmpeg, then to Whisper, and
    sends transcription back to the client.
    """
    print(f"Client connected from {websocket.remote_address}")

    # --- Verify that executables and models exist ---
    if not os.path.exists(WHISPER_EXECUTABLE):
        print(f"FATAL ERROR: Whisper executable not found at {WHISPER_EXECUTABLE}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Whisper model not found at {MODEL_PATH}")
        return

    whisper_proc = None
    ffmpeg_proc = None
    output_task = None

    try:
        print("Starting FFmpeg subprocess...")
        ffmpeg_proc = await asyncio.create_subprocess_exec(
            *FFMPEG_COMMAND,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr  # Show FFmpeg errors in the console
        )

        print("Starting whisper.cpp subprocess...")
        whisper_proc = await asyncio.create_subprocess_exec(
            WHISPER_EXECUTABLE,
            *WHISPER_ARGS,
            stdin=ffmpeg_proc.stdout,  # Pipe ffmpeg's output to whisper's input
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr  # Show whisper.cpp errors in the console
        )
        print(f"Whisper.cpp process started with PID: {whisper_proc.pid}")

        # --- Create a task to handle whisper's output ---
        output_task = asyncio.create_task(read_whisper_output(whisper_proc, websocket))

        # --- Receive audio from the client and forward it to FFmpeg ---
        async for audio_chunk in websocket:
            if ffmpeg_proc.stdin.is_closing():
                print("FFmpeg process stdin is closed. Cannot write more data.")
                break
            try:
                ffmpeg_proc.stdin.write(audio_chunk)
                await ffmpeg_proc.stdin.drain()
            except BrokenPipeError:
                print("Broken pipe: FFmpeg process may have terminated unexpectedly.")
                break
            except Exception as e:
                print(f"Error writing to FFmpeg stdin: {e}")
                break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
    except Exception as e:
        print(f"An error occurred in audio_handler: {e}")
    finally:
        print("Cleaning up subprocesses...")
        if ffmpeg_proc and ffmpeg_proc.returncode is None:
            if not ffmpeg_proc.stdin.is_closing():
                ffmpeg_proc.stdin.close()
            await ffmpeg_proc.wait()
            print("FFmpeg process terminated.")

        if whisper_proc and whisper_proc.returncode is None:
            whisper_proc.terminate() # Ensure whisper process is terminated
            await whisper_proc.wait()
            print("Whisper.cpp process terminated.")

        if output_task:
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