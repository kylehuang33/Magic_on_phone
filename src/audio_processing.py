import asyncio
import wave
import datetime
import subprocess
import os
from config import (AUDIO_STORAGE_PATH, BYTES_PER_SAMPLE, VAD_SAMPLE_RATE,
                    WHISPER_EXECUTABLE, WHISPER_ARGS)
from utils import clean_whisper_output
from llm_processing import query_ollama

def transcription_worker(audio_buffer: list[bytes], system_prompt: str, task_id: str):
    """
    Saves an audio buffer to a .wav file and transcribes it using whisper.cpp.
    This function is designed to be run in a separate thread to avoid blocking.
    """
    if not audio_buffer:
        return

    print(f"[{task_id}] Worker starting. Processing {len(b''.join(audio_buffer))} bytes.")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    wav_path = os.path.join(AUDIO_STORAGE_PATH, f"rec_{timestamp}.wav")

    try:
        # Save audio buffer to a WAV file
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(VAD_SAMPLE_RATE)
            wf.writeframes(b''.join(audio_buffer))

        # Build and execute the whisper.cpp command
        command = [WHISPER_EXECUTABLE] + WHISPER_ARGS + ["-f", wav_path]
        print(f"[{task_id}] Executing command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            transcription = clean_whisper_output(result.stdout)
            if transcription:
                # If transcription is successful, send it to the LLM
                query_ollama(transcription, system_prompt, task_id)
            else:
                print(f"[{task_id}] Transcription: (No discernible speech detected)")
        else:
            print(f"[{task_id}] Whisper.cpp exited with error code: {result.returncode}")
            print(f"[{task_id}] Whisper.cpp stderr: {result.stderr.strip()}")
            
    except Exception as e:
        print(f"[{task_id}] An error occurred in the transcription worker: {e}")
    finally:
        print(f"[{task_id}] Worker finished.")
        # Optional: Uncomment to automatically delete the audio file after processing
        # if os.path.exists(wav_path):
        #     os.remove(wav_path)

async def transcribe_audio_task(audio_buffer: list[bytes], system_prompt: str, task_id: str):
    """Asynchronous wrapper to run the blocking transcription_worker in a thread pool."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, transcription_worker, audio_buffer, system_prompt, task_id)