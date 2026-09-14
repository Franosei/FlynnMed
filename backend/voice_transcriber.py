"""
Voice transcription using OpenAI Whisper.
Accepts raw audio bytes (webm/wav from the browser) and returns transcribed text.
"""
from __future__ import annotations
import io
import os

from dotenv import load_dotenv

load_dotenv()


class VoiceTranscriber:
    """Transcribes audio recordings via OpenAI Whisper (whisper-1)."""

    MODEL = "whisper-1"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        from openai import OpenAI
        from backend.clinical_llm_gateway import guard_clinical_client
        self.client = guard_clinical_client(OpenAI(api_key=api_key), purpose="clinical-voice-transcription")

    def transcribe(self, audio_bytes: bytes, filename: str = "recording.webm") -> str:
        """
        Transcribe raw audio bytes to text.
        Returns the transcription string, or "" on failure.
        """
        if not audio_bytes:
            return ""
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = filename
            result = self.client.audio.transcriptions.create(
                model=self.MODEL,
                file=audio_file,
                language="en",
            )
            return (result.text or "").strip()
        except Exception as exc:
            print(f"VoiceTranscriber: transcription failed -- {exc}")
            return ""
