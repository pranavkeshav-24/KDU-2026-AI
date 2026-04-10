"""Shared FastAPI dependency functions injected via Depends()."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import async_session
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import security_service

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")


# ── Database Session ──────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session, committing on success and rolling back on error."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Authentication ────────────────────────────────────────────────────────────

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """Validate the Bearer token and return the authenticated User ORM object.

    Import is deferred inside the function to avoid circular import issues
    (users module imports from core, core should not import from users at module level).
    """
    from app.users.repository import user_repository  # deferred to avoid circular

    try:
        payload = security_service.decode_token(token)
    except JWTError:
        raise AuthenticationError("Could not validate credentials")

    if payload.get("type") != "access":
        raise AuthenticationError("Invalid token type")

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Could not validate credentials")

    user = await user_repository.get_by_id(db, user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("User not found or inactive")

    return user


# ── Role-Based Access Control ─────────────────────────────────────────────────

class RoleChecker:
    """Callable dependency for role-based access control.

    Usage::

        @router.get("/admin")
        async def admin_only(user = Depends(RoleChecker(["admin"]))):
            ...
    """

    def __init__(self, allowed_roles: list[str]) -> None:
        self.allowed_roles = allowed_roles

    async def __call__(self, user=Depends(get_current_user)):
        if user.role.value not in self.allowed_roles:
            raise AuthorizationError("Insufficient permissions")
        return user


# Pre-built convenience instances
require_admin = RoleChecker(["admin"])
require_user = RoleChecker(["user", "admin"])  # admin inherits user-level access
