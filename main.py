import openai
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
import pyaudio
import keyboard
import threading
import io
import time
import re
import queue
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")
OPENAI_HOST="https://api.openai.com/v1/"
LOCAL_API_KEY="not-needed"

# Configuration Constants
LOCAL_STT_CONFIG = {
    "BASE_URL": "http://10.1.1.24:8000/v1",
    "MODEL": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "API_KEY": LOCAL_API_KEY,
}

LOCAL_LLM_CONFIG = {
    "BASE_URL": "http://10.1.1.24:11434/v1",
    "MODEL": "gemma3:12b",
    "API_KEY": LOCAL_API_KEY,
}

LOCAL_TTS_CONFIG = {
    "BASE_URL": "http://10.1.1.24:8880/v1",
    "MODEL": "kokoro",
    "VOICE": "fable",
    "API_KEY": LOCAL_API_KEY,
}

OPENAI_STT_CONFIG = {
    "BASE_URL": OPENAI_HOST,
    "MODEL": "whisper-1",
    "API_KEY": OPENAI_API_KEY,
}

OPENAI_LLM_CONFIG = {
    "BASE_URL": OPENAI_HOST,
    "MODEL": "gpt-4o-mini",
    "API_KEY": OPENAI_API_KEY,
}

OPENAI_TTS_CONFIG = {
    "BASE_URL": OPENAI_HOST,
    "MODEL": "tts-1",
    "VOICE": "fable",
    "API_KEY": OPENAI_API_KEY,
}

STT_CONFIG=OPENAI_STT_CONFIG
LLM_CONFIG=OPENAI_LLM_CONFIG
TTS_CONFIG=OPENAI_TTS_CONFIG

conversation_history = [
    {
        "role": "system",
        "content": "You are a voice assistant. Respond concisely and avoid using emojis or markdown formatting.",
    }
]

# Initialize clients with respective configurations
stt_client = openai.OpenAI(
    base_url=STT_CONFIG["BASE_URL"], api_key=STT_CONFIG["API_KEY"]
)
llm_client = openai.OpenAI(
    base_url=LLM_CONFIG["BASE_URL"], api_key=LLM_CONFIG["API_KEY"]
)
tts_client = openai.OpenAI(
    base_url=TTS_CONFIG["BASE_URL"], api_key=TTS_CONFIG["API_KEY"]
)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16
RECORDING = False
audio_buffer = []

# Audio playback setup
audio = pyaudio.PyAudio()
audio_queue = queue.Queue()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=24000,
    output=True,
    start=False,
    frames_per_buffer=1024,
)


# Add playback thread
def playback_worker():
    while True:
        data = audio_queue.get()
        if data is None:
            break
        stream.write(data)
    stream.stop_stream()


playback_thread = threading.Thread(target=playback_worker, daemon=True)
playback_thread.start()


def record_callback(indata, frames, time, status):
    if RECORDING:
        audio_buffer.append(indata.copy())


def play_audio(audio_bytes):
    audio_queue.put(audio_bytes)


def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        transcription = stt_client.audio.transcriptions.create(
            model=STT_CONFIG["MODEL"], file=audio_file
        )
    return transcription.text


def generate_speech(text):
    response = tts_client.audio.speech.create(
        model=TTS_CONFIG["MODEL"], voice=TTS_CONFIG["VOICE"], input=text
    )
    return response.content


def on_press(key):
    global RECORDING
    if not RECORDING:
        print("Recording...")
        RECORDING = True


def on_release(key):
    global RECORDING, conversation_history
    if RECORDING:
        print("Processing...")
        RECORDING = False

        # Save recorded audio
        audio_data = np.concatenate(audio_buffer, axis=0)
        wav.write("input.wav", SAMPLE_RATE, audio_data)
        audio_buffer.clear()

        # Process audio
        transcript = transcribe_audio("input.wav")
        print(f"You: {transcript}")

        # Add user message to history
        conversation_history.append({"role": "user", "content": transcript})

        # Stream LLM response
        llm_stream = llm_client.chat.completions.create(
            model=LLM_CONFIG["MODEL"], messages=conversation_history, stream=True
        )

        buffer = ""
        full_response = ""
        stream.start_stream()
        for chunk in llm_stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                buffer += content
                full_response += content
                sentences = []
                while True:
                    match = re.search(r"[.!?](?:\s+|$)", buffer)
                    if match:
                        end_pos = match.end()
                        sentence = buffer[:end_pos].strip()
                        sentences.append(sentence)
                        buffer = buffer[end_pos:].lstrip()
                    else:
                        break

                # Process sentences
                for sentence in sentences:
                    print(f"Assistant: {sentence}")
                    speech = generate_speech(sentence)
                    audio = AudioSegment.from_mp3(io.BytesIO(speech))
                    audio = audio.set_frame_rate(24000).set_channels(1)
                    play_audio(audio.raw_data)

        # Process remaining buffer
        if buffer.strip():
            print(f"Assistant: {buffer}")
            speech = generate_speech(buffer)
            audio = AudioSegment.from_mp3(io.BytesIO(speech))
            audio = audio.set_frame_rate(24000).set_channels(1)
            play_audio(audio.raw_data)

        # Add assistant response to history
        conversation_history.append(
            {"role": "assistant", "content": full_response.strip()}
        )


def push_to_talk():
    keyboard.on_press_key("space", on_press)
    keyboard.on_release_key("space", on_release)
    keyboard.wait()


if __name__ == "__main__":
    # Start audio stream
    input_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=record_callback
    )
    input_stream.start()

    print("Press and hold SPACE to talk")
    push_to_talk_thread = threading.Thread(target=push_to_talk)
    push_to_talk_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        input_stream.stop()
        input_stream.close()
        audio_queue.put(None)  # Stop playback thread
        stream.close()
        audio.terminate()
