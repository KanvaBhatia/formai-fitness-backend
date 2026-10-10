"""
Text-to-speech service using gTTS.

Converts coaching text to MP3 audio bytes for playback in the browser.
"""

from io import BytesIO
from gtts import gTTS


class TextToSpeech:
    """Converts text to speech audio (MP3 bytes) via Google TTS."""

    def speak(self, text: str, lang: str = "en") -> bytes | None:
        """Return MP3 audio bytes for the given text, or None if empty."""
        cleaned = (text or "").strip()
        if not cleaned:
            return None

        buffer = BytesIO()
        gTTS(text=cleaned, lang=lang).write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()
