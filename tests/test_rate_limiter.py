"""Tests for the rate limiter."""
from __future__ import annotations

import pytest

from app.middleware import RateLimiter


@pytest.fixture()
def limiter():
    return RateLimiter(user_rpm=5, tenant_rpm=10)


@pytest.mark.asyncio
async def test_allows_within_limit(limiter):
    for _ in range(5):
        assert await limiter.check("user1", "tenant1") is True


@pytest.mark.asyncio
async def test_blocks_after_user_limit_exceeded(limiter):
    limiter2 = RateLimiter(user_rpm=3, tenant_rpm=100)
    for _ in range(3):
        await limiter2.check("userX", "tenantX")
    result = await limiter2.check("userX", "tenantX")
    assert result is False


@pytest.mark.asyncio
async def test_blocks_after_tenant_limit_exceeded():
    limiter = RateLimiter(user_rpm=100, tenant_rpm=3)
    for _ in range(3):
        await limiter.check("user_a", "shared_tenant")
    result = await limiter.check("user_b", "shared_tenant")
    assert result is False


@pytest.mark.asyncio
async def test_different_users_independent(limiter):
    """Exhausting one user's limit should not affect another user."""
    limiter2 = RateLimiter(user_rpm=2, tenant_rpm=100)
    await limiter2.check("u1", "t1")
    await limiter2.check("u1", "t1")
    # u1 is exhausted; u2 should still be allowed
    assert await limiter2.check("u2", "t1") is True
