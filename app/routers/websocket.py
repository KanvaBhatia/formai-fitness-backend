"""
WebSocket endpoint for real-time pose detection.

Protocol:
  Client sends JSON: {"frame": "<base64_jpeg>", "exercise": "Squats"}
  Server responds JSON: {"frame": "<base64_jpeg_annotated>", "metrics": {...},
                         "speak"?: {"id": int, "text": str, "audio_base64": str|null}}

Form-issue cues bypass the LLM: when the detector flags a known issue, a
canned cue is TTS'd off the event loop and attached to a frame response
("speak") so feedback arrives within ~1s of detection.
"""

import asyncio
import base64
import json
import itertools

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.vision.pose_processor import PoseProcessor

router = APIRouter()

# Injected at startup by main.py
_voice_pipeline = None
_tts = None

# Monotonic ids for speak payloads so clients can dedupe.
_speak_ids = itertools.count(1)


def init_websocket(voice_pipeline, tts):
    """Called once during app startup to inject shared coach services."""
    global _voice_pipeline, _tts
    _voice_pipeline = voice_pipeline
    _tts = tts


@router.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    """
    Bidirectional WebSocket for live video pose analysis.

    Each message from the client contains a base64-encoded JPEG frame
    and the current exercise type. The server processes the frame through
    MediaPipe, draws skeleton overlays, runs the exercise detector, and
    returns the annotated frame plus metrics.
    """
    await websocket.accept()
    processor = PoseProcessor()

    # Pending speak payload waiting to be attached to the next response.
    pending_speak: dict | None = None
    tts_task: asyncio.Task | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            exercise = data.get("exercise", "Squats")
            processor.set_exercise(exercise)

            frame_b64 = data.get("frame", "")
            if not frame_b64:
                await websocket.send_json({"error": "empty frame"})
                continue

            # Decode base64 JPEG → numpy array
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "could not decode frame"})
                continue

            # Process through pose detection
            annotated, metrics = processor.process_frame(frame)

            # Instant form cue (no LLM): kick off TTS off the event loop.
            if (
                _voice_pipeline is not None
                and _tts is not None
                and tts_task is None
            ):
                cue = _voice_pipeline.cue_for_event("ongoing_form_check", exercise, metrics)
                if cue:
                    pending_speak = {"id": next(_speak_ids), "text": cue}
                    tts_task = asyncio.create_task(asyncio.to_thread(_tts.speak, cue))

            # Attach TTS audio once ready (usually within a frame or two).
            speak_payload = None
            if pending_speak is not None and tts_task is not None and tts_task.done():
                try:
                    audio = tts_task.result()
                except Exception:
                    audio = None
                speak_payload = {
                    **pending_speak,
                    "audio_base64": base64.b64encode(audio).decode() if audio else None,
                }
                pending_speak = None
                tts_task = None

            # Encode annotated frame back to base64 JPEG
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = base64.b64encode(buffer.tobytes()).decode()

            response = {
                "frame": annotated_b64,
                "metrics": metrics,
            }
            if speak_payload:
                response["speak"] = speak_payload

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
