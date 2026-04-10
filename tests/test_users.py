from typing import Callable

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_me_authenticated(client: AsyncClient, create_auth_headers: Callable):
    headers = await create_auth_headers(email="me@example.com", full_name="It is Me")
    response = await client.get("/api/v1/users/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "me@example.com"


@pytest.mark.asyncio
async def test_get_me_unauthenticated(client: AsyncClient):
    response = await client.get("/api/v1/users/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_users_as_admin(client: AsyncClient, create_auth_headers: Callable):
    # Create a normal user
    await create_auth_headers(email="user@example.com")
    
    # Create an admin user and get headers
    admin_headers = await create_auth_headers(email="admin@example.com", role="admin")
    
    response = await client.get("/api/v1/users/", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    emails = [u["email"] for u in data]
    assert "user@example.com" in emails
    assert "admin@example.com" in emails


@pytest.mark.asyncio
async def test_list_users_as_user_forbidden(client: AsyncClient, create_auth_headers: Callable):
    user_headers = await create_auth_headers(email="user2@example.com", role="user")
    response = await client.get("/api/v1/users/", headers=user_headers)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "AUTHORIZATION_ERROR"
