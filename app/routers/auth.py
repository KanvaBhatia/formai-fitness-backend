"""
Authentication routes.

Simple username-based login — creates a user record if one doesn't exist.
"""

from pydantic import BaseModel
from fastapi import APIRouter

from app.services.persistence.exercise_repository import get_or_create_user

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str


class LoginResponse(BaseModel):
    user_id: int
    username: str


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest):
    """Login or register by username. Returns user_id and username."""
    user = get_or_create_user(body.username.strip())
    return LoginResponse(user_id=user["id"], username=user["username"])
