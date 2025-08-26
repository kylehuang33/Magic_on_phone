import asyncio
import websockets
import webrtcvad
import os
import sys

# Import configurations and handlers
import config
from mcp_handler import get_mcp_tools_as_json, create_system_prompt
from audio_processing import transcribe_audio_task

async def audio_handler(websocket, path: str, system_prompt: str):
    """
    Handles a single client WebSocket connection.
    Listens for audio, performs voice activity detection (VAD), and buffers
    speech before dispatching it for transcription.
    """
    print(f"Client connected from {websocket.remote_address}")
    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
    speech_buffer = []
    silent_frames_count = 0
    frame_accumulator = b''
    task_counter = 0

    try:
        async for audio_chunk in websocket:
            frame_accumulator += audio_chunk
            # Process audio in VAD-compatible frame sizes
            while len(frame_accumulator) >= config.VAD_FRAME_SIZE:
                frame = frame_accumulator[:config.VAD_FRAME_SIZE]
                frame_accumulator = frame_accumulator[config.VAD_FRAME_SIZE:]

                try:
                    if vad.is_speech(frame, config.VAD_SAMPLE_RATE):
                        speech_buffer.append(frame)
                        silent_frames_count = 0
                    elif speech_buffer:
                        silent_frames_count += 1
                    
                    # Determine if an utterance has ended
                    is_utterance_end = (speech_buffer and silent_frames_count >= config.SILENCE_FRAMES_THRESHOLD)
                    is_buffer_full = (len(speech_buffer) >= config.MAX_BUFFER_FRAMES)
                    
                    if is_utterance_end or is_buffer_full:
                        print("Utterance detected. Handing off to executor...")
                        buffer_copy = list(speech_buffer)
                        task_id = f"Task-{websocket.id}-{task_counter}"
                        task_counter += 1
                        # Create a non-blocking task for transcription and LLM processing
                        asyncio.create_task(transcribe_audio_task(buffer_copy, system_prompt, task_id))
                        speech_buffer.clear()
                        silent_frames_count = 0
                except Exception as e:
                    print(f"Error during VAD processing: {e}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
        # Process any remaining audio in the buffer when the client disconnects
        if speech_buffer:
            print("Client disconnected. Processing final buffer...")
            task_id = f"Task-Final-{websocket.id}"
            await transcribe_audio_task(speech_buffer, system_prompt, task_id)
    except Exception as e:
        print(f"An unexpected error occurred in audio_handler: {e}")
    finally:
        print(f"Client session finished for {websocket.remote_address}")

async def start_server():
    """Initializes dependencies and starts the WebSocket server."""
    # 1. Verify paths and create directories
    os.makedirs(config.AUDIO_STORAGE_PATH, exist_ok=True)
    if not all(map(os.path.exists, [config.WHISPER_EXECUTABLE, config.MODEL_PATH])):
        print("FATAL ERROR: Whisper executable or model not found. Check paths in config.py")
        sys.exit(1)

    # 2. Fetch tools from MCP and build the system prompt
    tool_list_json = await get_mcp_tools_as_json()
    if not tool_list_json:
        print("Exiting: Could not fetch tools from MCP server.")
        sys.exit(1)

    system_prompt = create_system_prompt(tool_list_json)
    print("System prompt initialized successfully.")

    # 3. Create a partial handler with the system_prompt "baked in"
    handler_with_prompt = lambda ws, path: audio_handler(ws, path, system_prompt=system_prompt)

    # 4. Start the server
    async with websockets.serve(handler_with_prompt, config.WEBSOCKET_HOST, config.WEBSOCKET_PORT):
        print(f"WebSocket server started at ws://{config.WEBSOCKET_HOST}:{config.WEBSOCKET_PORT}")
        print("Server is ready and waiting for a client connection...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")