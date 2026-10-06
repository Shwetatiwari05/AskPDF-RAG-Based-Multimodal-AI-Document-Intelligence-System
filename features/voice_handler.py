"""
voice_handler.py — Voice Input for RAG Chatbot
Uses faster-whisper (M1 Mac compatible) for speech-to-text.

Flow:
    Record voice → faster-whisper transcribe → RAG query → Answer

Install:
    pip install faster-whisper sounddevice scipy
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    # No audio hardware on this machine (e.g. a cloud server), or the
    # sounddevice library itself failed to load for any reason — that's
    # fine, the API only needs transcription, not local recording.
    SOUNDDEVICE_AVAILABLE = False

import scipy.io.wavfile as wav
import numpy as np
import tempfile

# ─── LOAD FASTER-WHISPER MODEL ────────────────────────────────────────────────

MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")

_whisper_model = None

def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        print("Loading Whisper...")
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )

        print("Whisper ready!")

    return _whisper_model


# AUDIO RECORDING ─────────────────────────────────────────────────────────

def record_audio(duration: int = 5, sample_rate: int = 16000) -> str:
    """Record fixed duration audio, save to temp wav file. (Local testing only.)"""
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError(
            "No microphone hardware available on this machine. "
            "This function is for local testing only — on the deployed "
            "server, audio is recorded in the browser and uploaded to "
            "/voice/transcribe instead."
        )
    
    print(f"\n Recording for {duration} seconds... Speak now!")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    print("Recording complete!")

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(temp_file.name, sample_rate, audio_data)
    return temp_file.name


def record_until_silence(
    max_duration: int = 30,
    silence_threshold: float = 0.01,
    silence_duration: float = 2.0,
    sample_rate: int = 16000
) -> str:
    """Record audio and auto-stop after silence is detected."""
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError(
            "No microphone hardware available on this machine. "
            "This function is for local testing only — on the deployed "
            "server, audio is recorded in the browser and uploaded to "
            "/voice/transcribe instead."
        )

    print(f"\n Speak now... (auto-stops after {silence_duration}s of silence)")

    chunk_size = int(sample_rate * 0.1)   
    max_chunks = int(max_duration * sample_rate / chunk_size)
    silence_chunks = int(silence_duration * sample_rate / chunk_size)

    all_audio = []
    silent_count = 0
    started_speaking = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            all_audio.append(chunk.copy())

            amplitude = np.abs(chunk).mean() / 32768.0
            is_silent = amplitude < silence_threshold

            if not is_silent:
                started_speaking = True
                silent_count = 0
            elif started_speaking:
                silent_count += 1
                if silent_count >= silence_chunks:
                    print("Silence detected — stopping")
                    break

    print("Recording complete!")
    audio_data = np.concatenate(all_audio, axis=0)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(temp_file.name, sample_rate, audio_data)
    return temp_file.name


# ─── TRANSCRIPTION ────────────────────────────────────────────────────────────

def transcribe(audio_path: str, language: str = "en") -> str:
    """Transcribe audio file to text using faster-whisper."""

    model = get_whisper_model()

    print("Transcribing...")

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()

    if os.path.exists(audio_path):
        os.remove(audio_path)

    print(f'You said: "{text}"')
    return text


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────

def voice_to_query(
    mode: str = "auto",
    duration: int = 5,
    language: str = "en"
) -> str:
    """
    Record voice and return transcribed text query.

    Usage:
        from features.voice_handler import voice_to_query
        query = voice_to_query()
    """
    if mode == "auto":
        audio_path = record_until_silence()
    else:
        audio_path = record_audio(duration=duration)

    return transcribe(audio_path, language=language)


# ─── STANDALONE TEST ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Whisper Voice Test ===")
    print("Recording 5 seconds — speak your question!\n")
    query = voice_to_query(mode="fixed", duration=5)
    print(f"\n Transcribed: \"{query}\"")