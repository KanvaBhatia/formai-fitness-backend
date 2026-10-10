"""
FastAPI application entry point.

Initializes the database, Groq client, voice coaching pipeline,
and registers all API + WebSocket routes.
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

    # Initialize Groq client and coaching pipeline if API key is available
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key:
        from groq import Groq
        from app.services.coaching.llm import LLMCoach
        from app.services.coaching.tts import TextToSpeech
        from app.services.coaching.voice_pipeline import VoicePipeline

        groq_client = Groq(api_key=api_key)
        llm_coach = LLMCoach(groq_client)
        tts = TextToSpeech()
        voice_pipeline = VoicePipeline(llm_coach, tts)
        coach.init_coach(voice_pipeline, llm_coach, tts, groq_client)
        print("✅ Voice coaching pipeline initialized")
    else:
        print("⚠️  GROQ_API_KEY not set — voice coaching disabled")

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
