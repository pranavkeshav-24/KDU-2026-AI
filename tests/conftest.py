"""Pytest fixtures for async FastAPI testing.

Provides an isolated test database and HTTP client per test.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Callable

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.database import Base
from app.core.dependencies import get_db
from app.main import app
from app.users.schemas import UserCreate
from app.users.service import user_service

# Use a separate test database
TEST_DATABASE_URL = "postgresql+asyncpg://fastapi:secret@localhost:5432/fastapi_test"

# ── Global Test Engine ────────────────────────────────────────────────────────
test_engine = create_async_engine(TEST_DATABASE_URL, pool_pre_ping=True)


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Instruct pytest-asyncio to use asyncio."""
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
async def setup_test_db() -> AsyncGenerator[None, None]:
    """Create all tables before tests run, then drop them after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provides an isolated database session per test via nested transactions.

    Everything committed within the test is rolled back at the end, ensuring
    zero state leakage between tests.
    """
    connection = await test_engine.connect()
    transaction = await connection.begin()

    session = AsyncSession(
        bind=connection,
        expire_on_commit=False,
    )

    yield session

    await session.close()
    await transaction.rollback()
    await connection.close()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client with DB dependency override.

    Overrides `get_db` to return the test's isolated `db_session`.
    """
    app.dependency_overrides[get_db] = lambda: db_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def create_auth_headers(client: AsyncClient, db_session: AsyncSession) -> Callable:
    """Helper fixture to create a user and return auth headers."""

    async def _create(
        email: str = "test@example.com",
        password: str = "SecureP@ssw0rd1!",
        role: str = "user",
    ) -> dict[str, str]:
        # Pre-create user via service
        user = await user_service.register(
            db_session,
            UserCreate(email=email, password=password, full_name="Test User"),
        )
        if role == "admin":
            from app.users.models import UserRole
            user.role = UserRole.ADMIN
            db_session.add(user)
            await db_session.flush()

        # Login to get valid tokens
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    return _create
