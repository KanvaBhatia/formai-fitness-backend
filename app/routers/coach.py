"""
AI coaching routes — feedback generation and question answering.

Uses Groq LLM for text generation, Groq Whisper for speech-to-text,
and gTTS for text-to-speech audio playback.
"""

import base64
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File

from app.services.coaching.llm import LLMCoach
from app.services.coaching.tts import TextToSpeech
from app.services.coaching.voice_pipeline import VoicePipeline

router = APIRouter(prefix="/coach", tags=["coach"])

# These are injected at startup by main.py
_voice_pipeline: VoicePipeline | None = None
_llm_coach: LLMCoach | None = None
_tts: TextToSpeech | None = None
_groq_client = None


def init_coach(voice_pipeline: VoicePipeline, llm_coach: LLMCoach, tts: TextToSpeech, groq_client):
    """Called once during app startup to inject shared coach services."""
    global _voice_pipeline, _llm_coach, _tts, _groq_client
    _voice_pipeline = voice_pipeline
    _llm_coach = llm_coach
    _tts = tts
    _groq_client = groq_client


class FeedbackRequest(BaseModel):
    event: str
    exercise: str
    metrics: dict = {}


class FeedbackResponse(BaseModel):
    text: str
    audio_base64: str | None = None


class AskRequest(BaseModel):
    question: str
    context: str = ""


class AskResponse(BaseModel):
    text: str
    audio_base64: str | None = None


@router.post("/feedback", response_model=FeedbackResponse)
def coach_feedback(body: FeedbackRequest):
    """
    Generate coaching feedback for a workout event.

    Events: workout_started, set_completed, workout_completed,
            no_pose_detected, ongoing_form_check
    """
    if not _voice_pipeline:
        return FeedbackResponse(text="Coach not available", audio_base64=None)

    result = _voice_pipeline.process_event(body.event, body.exercise, body.metrics)

    if result is None:
        return FeedbackResponse(text="", audio_base64=None)

    audio_bytes, text = result
    audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
    return FeedbackResponse(text=text, audio_base64=audio_b64)


@router.post("/ask", response_model=AskResponse)
def coach_ask(body: AskRequest):
    """
    Answer a free-form user question about their workout.

    Optionally include current metrics as context for pose-aware answers.
    """
    if not _llm_coach or not _tts:
        return AskResponse(text="Coach not available", audio_base64=None)

    text = _llm_coach.answer_question(body.question, body.context)
    audio = _tts.speak(text)
    audio_b64 = base64.b64encode(audio).decode() if audio else None
    return AskResponse(text=text, audio_base64=audio_b64)


@router.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Transcribe user speech to text using Groq Whisper.

    Accepts an audio file upload (webm/wav/mp3) and returns the transcript.
    """
    if not _groq_client:
        return {"text": ""}

    audio_bytes = await audio.read()

    transcription = _groq_client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=(audio.filename or "audio.webm", audio_bytes),
    )

    return {"text": transcription.text}
