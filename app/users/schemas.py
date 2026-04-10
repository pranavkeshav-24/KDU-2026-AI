"""Pydantic request and response schemas for the users/auth feature."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator, model_validator

from app.users.models import UserRole

# ── Validators ────────────────────────────────────────────────────────────────

_SPECIAL_CHARS = re.compile(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?`~]")


def _validate_password_strength(v: str) -> str:
    errors = []
    if len(v) < 8:
        errors.append("at least 8 characters")
    if not re.search(r"[A-Z]", v):
        errors.append("at least one uppercase letter")
    if not re.search(r"[a-z]", v):
        errors.append("at least one lowercase letter")
    if not re.search(r"\d", v):
        errors.append("at least one digit")
    if not _SPECIAL_CHARS.search(v):
        errors.append("at least one special character (!@#$%^&*…)")
    if errors:
        raise ValueError("Password must contain: " + ", ".join(errors))
    return v


# ── Request Schemas ───────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)

    @field_validator("full_name")
    @classmethod
    def full_name_clean(cls, v: str) -> str:
        v = v.strip()
        if not 2 <= len(v) <= 150:
            raise ValueError("full_name must be between 2 and 150 characters")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenRefresh(BaseModel):
    refresh_token: str


# ── Response Schemas ──────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    email: str
    full_name: str
    is_active: bool
    role: UserRole
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str
