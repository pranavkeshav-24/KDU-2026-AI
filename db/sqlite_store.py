"""Lightweight SQLite interface for user profiles."""

import sqlite3
from datetime import datetime
from typing import Optional

from models.schemas import UserProfile
from config import settings

DATABASE_PATH = settings.DATABASE_URL.replace("sqlite:///", "")

def init_db() -> None:
    """Initialize the schema if it does not exist."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                timezone TEXT NOT NULL,
                unit_system TEXT NOT NULL,
                language TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


async def get_user_profile(user_id: str) -> Optional[UserProfile]:
    """Fetch a user profile by ID."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row:
            created_at = row["created_at"]
            if isinstance(created_at, str):
                # Simple parsing of SQLite default CURRENT_TIMESTAMP
                try:
                    created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    created_at = datetime.now()
            
            return UserProfile(
                user_id=row["user_id"],
                name=row["name"],
                city=row["city"],
                lat=row["lat"],
                lon=row["lon"],
                timezone=row["timezone"],
                unit_system=row["unit_system"],
                language=row["language"],
                created_at=created_at
            )
        return None


async def save_user_profile(profile: UserProfile) -> None:
    """Insert or update a user profile."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, name, city, lat, lon, timezone, unit_system, language, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name=excluded.name,
                city=excluded.city,
                lat=excluded.lat,
                lon=excluded.lon,
                timezone=excluded.timezone,
                unit_system=excluded.unit_system,
                language=excluded.language
            """,
            (
                profile.user_id,
                profile.name,
                profile.city,
                profile.lat,
                profile.lon,
                profile.timezone,
                profile.unit_system,
                profile.language,
                profile.created_at.strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        conn.commit()
