"""Integration tests for the FastAPI application."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "providers" in data


@pytest.mark.asyncio
async def test_chat_local_provider(client):
    payload = {
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }
    response = await client.post("/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] != ""
    assert "usage" in data
    assert data["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_chat_returns_cached_on_second_call(client):
    payload = {
        "messages": [{"role": "user", "content": "Cache test message 42"}],
    }
    r1 = await client.post("/v1/chat", json=payload)
    assert r1.status_code == 200
    assert r1.json()["cached"] is False

    r2 = await client.post("/v1/chat", json=payload)
    assert r2.status_code == 200
    assert r2.json()["cached"] is True


@pytest.mark.asyncio
async def test_embeddings_local_provider(client):
    payload = {"input": ["hello", "world"]}
    response = await client.post("/v1/embeddings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2
    assert len(data["data"][0]["embedding"]) > 0


@pytest.mark.asyncio
async def test_embeddings_cached_on_second_call(client):
    payload = {"input": ["embedding cache test"]}
    r1 = await client.post("/v1/embeddings", json=payload)
    assert r1.status_code == 200
    assert r1.json()["cached"] is False

    r2 = await client.post("/v1/embeddings", json=payload)
    assert r2.status_code == 200
    assert r2.json()["cached"] is True


@pytest.mark.asyncio
async def test_cost_analytics(client):
    # Make a request first so there's something to query
    await client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "cost test"}], "tenant_id": "test-tenant"},
    )
    response = await client.get("/v1/analytics/cost/test-tenant")
    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "test-tenant"
    assert "total_requests" in data


@pytest.mark.asyncio
async def test_rate_limit_exceeded(client):
    """Exhaust the per-user limit and verify 429 is returned."""
    from app.middleware import RateLimiter
    from app.main import get_rate_limiter

    tiny_limiter = RateLimiter(user_rpm=2, tenant_rpm=100)

    app.dependency_overrides[get_rate_limiter] = lambda: tiny_limiter

    try:
        payload = {"messages": [{"role": "user", "content": "rl test"}], "user_id": "rl_user"}
        r1 = await client.post("/v1/chat", json=payload)
        r2 = await client.post("/v1/chat", json=payload)
        r3 = await client.post("/v1/chat", json=payload)
        # At least one should be blocked
        statuses = {r1.status_code, r2.status_code, r3.status_code}
        assert 429 in statuses
    finally:
        app.dependency_overrides.pop(get_rate_limiter, None)
