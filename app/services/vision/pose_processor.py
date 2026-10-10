"""
Server-side pose detection processor.

Receives raw video frames (as numpy arrays), runs MediaPipe PoseLandmarker,
delegates to the correct exercise detector, and returns annotated frames
plus a metrics dict.
"""

import os
import threading

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.detectors.squat import SquatDetector
from app.detectors.pushup import PushUpDetector
from app.detectors.biceps_curl import BicepsCurlDetector
from app.detectors.shoulder_press import ShoulderPressDetector
from app.detectors.lunges import LungesDetector
from app.services.config.workout_config import POSE_CONNECTIONS

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "ml_models",
    "pose_landmarker_full.task",
)


class PoseProcessor:
    """
    Processes video frames through MediaPipe pose detection and exercise analysis.

    Thread-safe: uses a lock to protect shared state between the WebSocket
    handler (which sets the exercise type) and the frame-processing loop.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics: dict | None = None
        self._exercise_type = "Squats"

        base_option = python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

        self._frame_timestamps_ms = 0

    # ── Thread-safe accessors ──────────────────────────────────────────

    def set_latest_metrics(self, metrics: dict):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self) -> dict | None:
        with self._lock:
            return None if self._latest_metrics is None else self._latest_metrics.copy()

    def set_exercise(self, exercise_type: str):
        with self._lock:
            if exercise_type != self._exercise_type:
                self._exercise_type = exercise_type
                detector = self._detectors.get(exercise_type)
                if detector:
                    detector.reset()

    def get_exercise(self) -> str:
        with self._lock:
            return self._exercise_type

    def reset_detectors(self):
        """Reset all detector rep counts."""
        for d in self._detectors.values():
            d.reset()

    # ── Frame processing ───────────────────────────────────────────────

    def process_frame(self, image_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Process a single BGR frame.

        Returns (annotated_bgr, metrics_dict).
        """
        # Mirror the image for a natural user experience
        image = np.asarray(cv2.flip(image_bgr, 1), dtype=np.uint8)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )

        self._frame_timestamps_ms += 33  # ~30 fps
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamps_ms)

        metrics: dict = {"pose_detected": False}

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            self._draw_skeleton(image, landmarks)

            ex_type = self.get_exercise()
            detector = self._detectors.get(ex_type)

            if detector:
                metrics = detector.process(landmarks)
                metrics["pose_detected"] = True
                self._draw_overlays(image, metrics, ex_type)
                self.set_latest_metrics(metrics)
        else:
            self._draw_no_pose_warnings(image)
            with self._lock:
                if self._latest_metrics is not None:
                    self._latest_metrics["pose_detected"] = False
                else:
                    self._latest_metrics = {"pose_detected": False}
            metrics = self._latest_metrics.copy()

        return image, metrics

    # ── Drawing helpers ────────────────────────────────────────────────

    def _draw_skeleton(self, img: np.ndarray, landmarks):
        h, w = img.shape[:2]
        for start_idx, end_idx in POSE_CONNECTIONS:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]
            if p1.visibility > 0.7 and p2.visibility > 0.7:
                cv2.line(
                    img,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    (0, 255, 0),
                    8,
                )
        for lm in landmarks:
            if lm.visibility > 0.7:
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 8, (255, 0, 0), -1)

    def _draw_no_pose_warnings(self, img: np.ndarray):
        cv2.putText(img, "NO POSE DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "PLEASE FACE THE CAMERA", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def _draw_overlays(self, img: np.ndarray, metrics: dict, ex_type: str):
        h, _ = img.shape[:2]
        label = ""

        if ex_type == "Squats":
            label = f"DEPTH: {metrics.get('depth_status', '')}"
        elif ex_type == "Push-ups":
            label = f"BODY: {metrics.get('body_alignment', '')} | HIP: {metrics.get('hip_status', '')}"
        elif ex_type == "Biceps Curls (Dumbbell)":
            label = f"SWING: {metrics.get('swing_status', '')} | TEMPO: {metrics.get('tempo_status', '')}"
        elif ex_type == "Shoulder Press":
            label = f"EXT: {metrics.get('extension_status', '')} | BACK: {metrics.get('back_arch_status', '')}"
        elif ex_type == "Lunges":
            label = f"BALANCE: {metrics.get('balance_status', '')}"

        if label:
            cv2.putText(img, label, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
