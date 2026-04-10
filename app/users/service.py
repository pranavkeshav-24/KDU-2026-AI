"""User business logic — orchestrates repository + security, free of HTTP concerns."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError, ConflictError, NotFoundError
from app.core.logging import get_logger
from app.core.security import security_service
from app.users.models import User
from app.users.repository import UserRepository, user_repository
from app.users.schemas import TokenRefresh, TokenResponse, UserCreate, UserLogin, UserResponse

logger = get_logger(__name__)


class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repo = repository

    # ── Registration ──────────────────────────────────────────────────────────

    async def register(self, db: AsyncSession, data: UserCreate) -> User:
        existing = await self._repo.get_by_email(db, data.email)
        if existing:
            raise ConflictError("Email already registered")

        hashed = security_service.hash_password(data.password)
        user = await self._repo.create(db, {
            "email": data.email,
            "password_hash": hashed,
            "full_name": data.full_name,
        })
        logger.info("user_registered", user_id=str(user.id), email=user.email)
        return user

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login(self, db: AsyncSession, data: UserLogin) -> TokenResponse:
        user = await self._repo.get_by_email(db, data.email)

        # Deliberate: same error for "user not found" and "wrong password"
        # to prevent user-enumeration attacks.
        if user is None or not security_service.verify_password(data.password, user.password_hash):
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            raise AuthenticationError("Account is deactivated")

        access_token = security_service.create_access_token(str(user.id), user.role.value)
        refresh_token = security_service.create_refresh_token(str(user.id))
        logger.info("user_logged_in", user_id=str(user.id))
        return TokenResponse(access_token=access_token, refresh_token=refresh_token)

    # ── Token Refresh ─────────────────────────────────────────────────────────

    async def refresh_tokens(self, db: AsyncSession, refresh_token: str) -> TokenResponse:
        from jose import JWTError

        try:
            payload = security_service.decode_token(refresh_token)
        except JWTError:
            raise AuthenticationError("Invalid or expired refresh token")

        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type — expected refresh token")

        user_id: str | None = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Invalid refresh token")

        user = await self._repo.get_by_id(db, user_id)
        if user is None or not user.is_active:
            raise AuthenticationError("User not found or inactive")

        new_access = security_service.create_access_token(str(user.id), user.role.value)
        new_refresh = security_service.create_refresh_token(str(user.id))
        logger.info("tokens_refreshed", user_id=user_id)
        return TokenResponse(access_token=new_access, refresh_token=new_refresh)

    # ── Profile ───────────────────────────────────────────────────────────────

    async def get_profile(self, user: User) -> UserResponse:
        return UserResponse.model_validate(user)

    # ── Admin: List Users ─────────────────────────────────────────────────────

    async def list_users(
        self, db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> list[UserResponse]:
        users = await self._repo.list_all(db, skip=skip, limit=limit)
        return [UserResponse.model_validate(u) for u in users]


# Singleton
user_service = UserService(user_repository)
