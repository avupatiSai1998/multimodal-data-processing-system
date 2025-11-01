# audio_transcribe.py
import whisper
import os

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file to text using Whisper tiny model.
    Args:
        audio_path: path to a local audio file (.wav, .mp3, etc.)
    Returns:
        Transcribed text string
    """
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path)
        return result.get("text", "").strip()
    except Exception as e:
        return f"[Error transcribing audio: {e}]"

