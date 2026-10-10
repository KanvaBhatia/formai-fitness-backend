"""
SQLite persistence layer for users and exercise records.

Uses a single shared connection. The schema is identical to the original
Streamlit app so existing data can be migrated if needed.
"""

import sqlite3
from pathlib import Path

_DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "data.db")

_connection: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    """Return (and lazily create) the shared SQLite connection."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _connection.row_factory = sqlite3.Row
    return _connection


def init_db() -> None:
    """Create tables if they don't exist yet."""
    conn = _get_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exercises (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL REFERENCES users(id),
                exercise_name TEXT    NOT NULL,
                reps          INTEGER NOT NULL DEFAULT 0,
                sets          INTEGER NOT NULL DEFAULT 0,
                time          INTEGER NOT NULL DEFAULT 0,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_user(username: str) -> sqlite3.Row | None:
    conn = _get_connection()
    return conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()


def create_user(username: str) -> sqlite3.Row:
    conn = _get_connection()
    with conn:
        conn.execute("INSERT INTO users (username) VALUES (?)", (username,))
    return get_user(username)


def get_or_create_user(username: str) -> sqlite3.Row:
    user = get_user(username)
    if user is None:
        user = create_user(username)
    return user


def add_exercise(user_id: int, exercise_name: str, reps: int, sets: int, time: int) -> None:
    """Record completed exercise sets; merges into today's record if one exists."""
    conn = _get_connection()
    with conn:
        existing = conn.execute(
            """
            SELECT * FROM exercises
            WHERE user_id = ? AND exercise_name = ? AND Date('created_at') = Date('now')
            """,
            (user_id, exercise_name),
        ).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE exercises
                SET reps = reps + ?, sets = sets + ?, time = time + ?
                WHERE id = ?
                """,
                (reps, sets, time, existing["id"]),
            )
        else:
            conn.execute(
                """
                INSERT INTO exercises (user_id, exercise_name, sets, reps, time)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, exercise_name, sets, reps, time),
            )


def get_users_exercises(user_id: int) -> list[sqlite3.Row]:
    conn = _get_connection()
    return conn.execute(
        "SELECT * FROM exercises WHERE user_id = ?", (user_id,)
    ).fetchall()
