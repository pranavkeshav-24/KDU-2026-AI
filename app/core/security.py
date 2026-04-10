"""JWT token generation/validation and bcrypt password hashing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityService:
    """Encapsulates all security operations: password hashing and JWT management."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str,
        access_expire_minutes: int,
        refresh_expire_days: int,
    ) -> None:
        self._secret = secret_key
        self._algorithm = algorithm
        self._access_expire = timedelta(minutes=access_expire_minutes)
        self._refresh_expire = timedelta(days=refresh_expire_days)

    # ── Password ──────────────────────────────────────────────────────────────

    def hash_password(self, plain: str) -> str:
        return _pwd_context.hash(plain)

    def verify_password(self, plain: str, hashed: str) -> bool:
        return _pwd_context.verify(plain, hashed)

    # ── Token Creation ────────────────────────────────────────────────────────

    def create_access_token(self, user_id: str, role: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user_id),
            "role": role,
            "type": "access",
            "exp": now + self._access_expire,
            "iat": now,
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user_id),
            "type": "refresh",
            "exp": now + self._refresh_expire,
            "iat": now,
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    # ── Token Decoding ────────────────────────────────────────────────────────

    def decode_token(self, token: str) -> dict:
        """Decode and validate a JWT.

        Raises ``JWTError`` on invalid signature, expiry, or malformed token.
        """
        return jwt.decode(token, self._secret, algorithms=[self._algorithm])


# Singleton used across the application
security_service = SecurityService(
    secret_key=settings.JWT_SECRET_KEY,
    algorithm=settings.JWT_ALGORITHM,
    access_expire_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    refresh_expire_days=settings.REFRESH_TOKEN_EXPIRE_DAYS,
)
