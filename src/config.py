import os

# --- WebSocket Server Configuration ---
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765

# --- Audio Storage Configuration ---
# Creates a subdirectory in the user's home directory for recordings
HOME_DIR = os.path.expanduser("~")
AUDIO_STORAGE_PATH = os.path.join(HOME_DIR, "audio_recordings")

# --- Whisper.cpp Configuration ---
# Assumes whisper.cpp is in a subdirectory of the user's home directory
WHISPER_EXECUTABLE = os.path.join(HOME_DIR, "DOING_PROJECTS", "Magic_on_phone", "modules", "whisper.cpp", "build", "bin", "whisper-cli")
MODEL_PATH = os.path.join(HOME_DIR, "DOING_PROJECTS", "Magic_on_phone", "modules", "whisper.cpp", "models", "ggml-base.en.bin")
WHISPER_ARGS = ["-m", MODEL_PATH, "-t", "4"] # 4 threads as an example

# --- Ollama Configuration ---
OLLAMA_MODEL = "gemma3:1b"

# --- MCP (Master Control Program) Configuration ---
MCP_SERVER_URL = "http://localhost:6001"

# --- VAD (Voice Activity Detection) Configuration ---
VAD_AGGRESSIVENESS = 3           # A value between 0 and 3. 3 is the most aggressive.
VAD_SAMPLE_RATE = 16000          # Must be 8000, 16000, 32000, or 48000
VAD_FRAME_DURATION_MS = 30       # Duration of a frame in milliseconds
BYTES_PER_SAMPLE = 2             # 16-bit audio
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * (VAD_FRAME_DURATION_MS / 1000.0) * BYTES_PER_SAMPLE)

# --- Speech Buffering Configuration ---
SILENCE_FRAMES_THRESHOLD = 30    # Number of consecutive silent frames to trigger transcription
MAX_BUFFER_FRAMES = 300          # Maximum number of frames to buffer before forcing transcription