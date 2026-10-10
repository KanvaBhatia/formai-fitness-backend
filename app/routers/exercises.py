"""
Exercise history and persistence routes.
"""

from pydantic import BaseModel
from fastapi import APIRouter

from app.services.persistence.exercise_repository import (
    get_users_exercises,
    add_exercise,
    init_db,
)

router = APIRouter(prefix="/exercises", tags=["exercises"])


class ExerciseRecord(BaseModel):
    id: int
    exercise_name: str
    reps: int
    sets: int
    time: int
    created_at: str


class SaveExerciseRequest(BaseModel):
    user_id: int
    exercise_name: str
    reps: int
    sets: int
    time: int


@router.get("/history/{user_id}", response_model=list[ExerciseRecord])
def get_history(user_id: int):
    """Return all exercise records for a user, newest first."""
    init_db()
    rows = get_users_exercises(user_id)
    return [
        ExerciseRecord(
            id=r["id"],
            exercise_name=r["exercise_name"],
            reps=r["reps"],
            sets=r["sets"],
            time=r["time"],
            created_at=r["created_at"],
        )
        for r in reversed(rows)
    ]


@router.post("/save")
def save_exercise(body: SaveExerciseRequest):
    """Persist a completed set (or sets) for a user."""
    init_db()
    add_exercise(body.user_id, body.exercise_name, body.reps, body.sets, body.time)
    return {"ok": True}
