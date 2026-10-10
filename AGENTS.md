# Backend — AI Gym Trainer API

FastAPI backend for real-time pose detection, exercise tracking, and AI voice coaching.

## Running

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment

Create `.env` from `.env.example`:

- `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` — OpenAI-compatible LLM endpoint
  (LiteLLM proxy serving Kimi K3) for coaching feedback and Ask Coach.
- `GROQ_API_KEY` — Groq API key, used only for Whisper speech-to-text (`/coach/stt`).
- TTS uses gTTS (no key required).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/auth/login` | Login/register by username |
| GET | `/exercises/history/{user_id}` | Get user's workout history |
| POST | `/exercises/save` | Save completed exercise sets |
| POST | `/coach/feedback` | Get AI coaching feedback for an event |
| POST | `/coach/ask` | Ask the coach a free-form question |
| POST | `/coach/stt` | Speech-to-text via Groq Whisper |
| WS | `/ws/video` | Real-time pose detection WebSocket |

## WebSocket Protocol (`/ws/video`)

**Client sends:**
```json
{ "frame": "<base64_jpeg>", "exercise": "Squats" }
```

**Server responds:**
```json
{
  "frame": "<base64_jpeg_annotated>",
  "metrics": { "reps": 5, "pose_detected": true, ... }
}
```

## Adding a New Exercise

1. Create `app/detectors/my_exercise.py` extending `BaseExercise`
2. Implement `process(landmarks)` — return dict with `"reps"` key + exercise-specific metrics
3. Add to `EXERCISE_OPTIONS` and `METRICS_FIELDS` in `app/services/config/workout_config.py`
4. Register in `PoseProcessor._detectors` in `app/services/vision/pose_processor.py`
5. Add form issue mapping in `VoicePipeline._find_form_issue` in `app/services/coaching/voice_pipeline.py`
6. Add overlay drawing in `PoseProcessor._draw_overlays` in `app/services/vision/pose_processor.py`

## Key Files

- `app/main.py` — FastAPI app setup, CORS, lifespan
- `app/core/base_exercise.py` — abstract base for exercise detectors
- `app/services/vision/pose_processor.py` — MediaPipe + detector orchestration
- `app/services/coaching/voice_pipeline.py` — LLM + TTS pipeline
- `app/services/persistence/exercise_repository.py` — SQLite data layer
- `app/services/config/workout_config.py` — constants, prompt, metrics fields
