"""Users and auth API endpoints.

Thin router layer — HTTP concerns only; business logic delegated to service.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import get_db, get_current_user, require_admin, require_user
from app.users.models import User
from app.users.schemas import (
    MessageResponse,
    TokenRefresh,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from app.users.service import user_service

limiter = Limiter(key_func=get_remote_address)

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
user_router = APIRouter(prefix="/users", tags=["Users"])


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@auth_router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    responses={
        409: {"description": "Email already registered"},
        422: {"description": "Validation error (weak password, bad email, etc.)"},
    },
)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def register(
    request,  # required by slowapi limiter
    data: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Create a new user account.

    - **email**: Valid email address (unique)
    - **password**: Min 8 chars, uppercase, lowercase, digit, special character
    - **full_name**: 2–150 characters
    """
    user = await user_service.register(db, data)
    return UserResponse.model_validate(user)


@auth_router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and obtain JWT tokens",
    responses={401: {"description": "Invalid credentials"}},
)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login(
    request,
    data: UserLogin,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate and receive an access + refresh token pair."""
    return await user_service.login(db, data)


@auth_router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh the access token using a refresh token",
    responses={401: {"description": "Invalid or expired refresh token"}},
)
async def refresh_tokens(
    data: TokenRefresh,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Exchange a refresh token for a new access + refresh token pair."""
    return await user_service.refresh_tokens(db, data.refresh_token)


# ══════════════════════════════════════════════════════════════════════════════
# PROTECTED ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@user_router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the current authenticated user's profile",
    responses={401: {"description": "Not authenticated"}},
)
async def get_me(current_user: User = Depends(require_user)) -> UserResponse:
    """Return the profile of the currently authenticated user."""
    return UserResponse.model_validate(current_user)


@user_router.get(
    "/",
    response_model=List[UserResponse],
    summary="List all users (admin only)",
    responses={
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions — admin required"},
    },
)
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
) -> List[UserResponse]:
    """Return a paginated list of all users. **Admin role required.**"""
    return await user_service.list_users(db, skip=skip, limit=limit)
