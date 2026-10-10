"""
FastAPI application entry point.

Initializes the database, LLM client (Kimi K3 via LiteLLM proxy),
Groq client (Whisper STT), voice coaching pipeline, and registers
all API + WebSocket routes.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.persistence.exercise_repository import init_db
from app.routers import auth, exercises, coach, websocket

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services on startup."""
    init_db()

    llm_api_key = os.getenv("LLM_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", "")
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    if llm_api_key and llm_base_url:
        from openai import OpenAI
        from app.services.coaching.llm import LLMCoach
        from app.services.coaching.tts import TextToSpeech
        from app.services.coaching.voice_pipeline import VoicePipeline

        # LLM client — OpenAI-compatible LiteLLM proxy (Kimi K3)
        openai_client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        llm_coach = LLMCoach(openai_client)
        tts = TextToSpeech()
        voice_pipeline = VoicePipeline(llm_coach, tts)

        # Groq client — only used for Whisper speech-to-text
        groq_client = None
        if groq_api_key:
            from groq import Groq
            groq_client = Groq(api_key=groq_api_key)

        coach.init_coach(voice_pipeline, llm_coach, tts, groq_client)
        print("✅ Voice coaching pipeline initialized (Kimi K3 via LiteLLM)")
        if not groq_client:
            print("⚠️  GROQ_API_KEY not set — speech-to-text disabled")
    else:
        print("⚠️  LLM_API_KEY / LLM_BASE_URL not set — voice coaching disabled")

    yield


app = FastAPI(
    title="AI Gym Trainer API",
    description="Backend for real-time pose detection, rep counting, and AI voice coaching.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow any origin (no credentials/cookies used, so wildcard is safe)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(exercises.router)
app.include_router(coach.router)
app.include_router(websocket.router)


@app.get("/health")
def health():
    return {"status": "ok"}
