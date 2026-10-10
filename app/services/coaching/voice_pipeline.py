"""
Voice coaching pipeline.

Form-issue cues bypass the LLM: cue_for_event() maps detected issues to
short canned lines instantly (used by the WebSocket handler, which attaches
TTS audio to frame responses).

Major events (workout_started, set_completed, workout_completed) still use
the LLM via process_event() — synchronous, called from POST /coach/feedback.
"""

import time


class VoicePipeline:
    """End-to-end pipeline: form issue detection -> LLM feedback -> TTS audio."""

    RATE_LIMIT_SECONDS = 5

    # Instant spoken cues for known form issues. Keyed by the issue strings
    # produced by _find_form_issue. These bypass the LLM entirely so form
    # feedback can be spoken within ~1s of detection.
    ISSUE_CUES: dict[str, str] = {
        "No pose detected! Please step into the camera frame.": "Step into the camera frame so I can see you.",
        # Squats
        "The user's squat is not deep enough — knees are not bending sufficiently.": "Go a bit deeper — bend those knees more.",
        "The user is leaning too far forward during the squat.": "Chest up — you're leaning too far forward.",
        # Push-ups
        "The user's body is not straight during the push-up.": "Keep your body in a straight line.",
        "The user's hips are sagging down during the push-up.": "Hips up — don't let them sag.",
        "The user's hips are too high — lower them to form a straight line.": "Lower your hips to form a straight line.",
        # Biceps curls
        "The user is performing reps too quickly — ask them to slow down to avoid injury.": "Slow down — control the weight.",
        "The user is swinging their torso during the curl — keep the body still.": "Keep your body still — no swinging.",
        "The user's elbow is drifting away from their side during the curl.": "Keep your elbows pinned to your sides.",
        # Shoulder press
        "The user is arching their lower back excessively during the press.": "Don't arch your back — brace your core.",
        "Slight back arch detected — encourage the user to brace their core.": "Slight back arch — brace your core.",
        # Lunges
        "The user is losing balance during the lunge — feet should be hip-width apart.": "Feet hip-width apart — find your balance.",
    }

    def __init__(self, llm, tts):
        self.llm = llm
        self.tts = tts
        self.last_spoken_at = 0.0
        self._last_cue_issue: str | None = None

    # ── Issue detection (unchanged rules) ────────────────────────────

    def _find_form_issue(self, exercise: str, metrics: dict) -> str | None:
        """Map raw exercise metrics to a human-readable form issue description."""
        if "issue" in metrics:
            return metrics["issue"]

        if exercise == "Squats":
            depth = metrics.get("depth_status", "")
            back_angle = metrics.get("back_angle", 180)
            if depth == "TOO HIGH":
                return "The user's squat is not deep enough — knees are not bending sufficiently."
            if isinstance(back_angle, (int, float)) and back_angle < 130:
                return "The user is leaning too far forward during the squat."

        elif exercise == "Push-ups":
            alignment = metrics.get("body_alignment", "")
            hip_status = metrics.get("hip_status", "")
            if alignment == "Poor Form":
                return "The user's body is not straight during the push-up."
            if hip_status == "SAGGING":
                return "The user's hips are sagging down during the push-up."
            if hip_status == "PIKED UP":
                return "The user's hips are too high — lower them to form a straight line."

        elif exercise == "Biceps Curls (Dumbbell)":
            tempo = metrics.get("tempo_status", "")
            swing = metrics.get("swing_status", "")
            shoulder = metrics.get("shoulder_status", "")
            if tempo == "SLOW DOWN":
                return "The user is performing reps too quickly — ask them to slow down to avoid injury."
            if swing == "SWINGING":
                return "The user is swinging their torso during the curl — keep the body still."
            if shoulder == "ELBOW DRIFTING":
                return "The user's elbow is drifting away from their side during the curl."

        elif exercise == "Shoulder Press":
            back_arch = metrics.get("back_arch_status", "")
            if back_arch == "Excessive Arch":
                return "The user is arching their lower back excessively during the press."
            if back_arch == "Slight Arch":
                return "Slight back arch detected — encourage the user to brace their core."

        elif exercise == "Lunges":
            balance = metrics.get("balance_status", "")
            if balance == "OFF BALANCE":
                return "The user is losing balance during the lunge — feet should be hip-width apart."

        return None

    # ── Instant cue path (WebSocket, no LLM) ─────────────────────────

    def cue_for_event(self, event: str, exercise: str, metrics: dict) -> str | None:
        """
        Return an instant canned cue for a minor event, or None.

        Used by the WebSocket handler to speak form feedback without an
        LLM round-trip. Rate-limited to one cue per RATE_LIMIT_SECONDS and
        only fires when the issue CHANGES (or recurs after it cleared), so
        a persistent issue doesn't retrigger every window.
        Major events return None — they still use the LLM path.
        """
        if event in ("workout_started", "set_completed", "workout_completed"):
            return None

        issue = self._find_form_issue(exercise, metrics)

        if issue is None:
            # Form is clean again — allow re-alerting if it recurs.
            self._last_cue_issue = None
            return None

        now = time.time()
        if issue == self._last_cue_issue:
            return None
        if now - self.last_spoken_at < self.RATE_LIMIT_SECONDS:
            return None

        self._last_cue_issue = issue
        self.last_spoken_at = now
        return self.ISSUE_CUES.get(issue, issue)

    # ── Event intake ─────────────────────────────────────────────────

    def process_event(self, event: str, exercise: str, metrics: dict) -> tuple[bytes, str] | None:
        """
        Process a coaching event synchronously and return (audio_bytes, text)
        or None. Used by POST /coach/feedback for MAJOR events
        (workout_started, set_completed, workout_completed).

        Minor form issues no longer come through here — they are handled
        instantly via cue_for_event() over the WebSocket.
        """
        issue = self._find_form_issue(exercise, metrics)

        is_major_event = event in ("workout_started", "set_completed", "workout_completed")

        if not is_major_event:
            # Minor events are served by the instant cue path; drop here.
            return None

        text = self.llm.give_feedback(event, issue)
        audio = self.tts.speak(text)
        return audio, text
