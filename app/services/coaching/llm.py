"""
LLM coaching service.

Wraps an OpenAI-compatible chat client (LiteLLM proxy serving Kimi K3)
to generate short, spoken coaching feedback based on workout events
and detected form issues.
"""

import os

from app.services.config.workout_config import PROMPT


class LLMCoach:
    """Generates coaching feedback via an OpenAI-compatible LLM endpoint."""

    def __init__(self, openai_client):
        self.client = openai_client
        self.model = os.getenv("LLM_MODEL", "global.moonshotai.kimi-k3")
        self.history: list[dict] = []
        self.system_prompt = PROMPT

    def give_feedback(self, event: str, issue: str | None = None) -> str:
        """Return a short coaching cue for the given event and optional form issue."""
        prompt = f"Event: {event}"
        if issue:
            prompt += f" Form Issue: {issue}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history[-10:],
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
        )

        text = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": text})
        return text

    def answer_question(self, question: str, context: str = "") -> str:
        """Answer a free-form user question with optional workout context."""
        user_msg = question
        if context:
            user_msg = f"Context: {context}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history[-10:],
            {"role": "user", "content": user_msg},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
        )

        text = response.choices[0].message.content.strip()
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": text})
        return text
