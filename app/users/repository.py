"""User database repository — all SQL queries live here."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.users.models import User


class UserRepository:
    """Encapsulates all User database operations.

    Business logic in ``service.py`` never writes raw SQL — it calls this class.
    """

    async def get_by_id(self, db: AsyncSession, user_id: str | uuid.UUID) -> User | None:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, db: AsyncSession, email: str) -> User | None:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def create(self, db: AsyncSession, data: dict) -> User:
        user = User(**data)
        db.add(user)
        await db.flush()   # gets generated id, timestamps
        await db.refresh(user)
        return user

    async def update(self, db: AsyncSession, user: User, data: dict) -> User:
        for field, value in data.items():
            setattr(user, field, value)
        db.add(user)
        await db.flush()
        await db.refresh(user)
        return user

    async def list_all(
        self, db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> list[User]:
        result = await db.execute(select(User).offset(skip).limit(limit))
        return list(result.scalars().all())


# Singleton used across the application
user_repository = UserRepository()
