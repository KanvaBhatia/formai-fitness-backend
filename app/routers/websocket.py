"""
WebSocket endpoint for real-time pose detection.

Protocol:
  Client sends JSON: {"frame": "<base64_jpeg>", "exercise": "Squats"}
  Server responds JSON: {"frame": "<base64_jpeg_annotated>", "metrics": {...}}
"""

import base64
import json

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.vision.pose_processor import PoseProcessor

router = APIRouter()


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

            # Encode annotated frame back to base64 JPEG
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = base64.b64encode(buffer.tobytes()).decode()

            await websocket.send_json({
                "frame": annotated_b64,
                "metrics": metrics,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
