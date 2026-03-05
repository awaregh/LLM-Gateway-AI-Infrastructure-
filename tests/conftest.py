"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.cache import ResponseCache
from app.cost import CostTracker
from app.middleware import RateLimiter
from app.providers.local_provider import LocalProvider
from app.routing import ModelRouter, RoutingStrategy


@pytest.fixture(autouse=True)
async def _setup_app_state(tmp_path):
    """Initialise app.state so the lifespan doesn't need to run in tests."""
    providers = [LocalProvider()]
    app.state.router = ModelRouter(providers, strategy=RoutingStrategy.COST)
    app.state.cache = ResponseCache(ttl_seconds=60)
    app.state.rate_limiter = RateLimiter(user_rpm=60, tenant_rpm=300)
    tracker = CostTracker(database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await tracker.init_db()
    app.state.cost_tracker = tracker
    yield
    # Clean up overrides after each test
    app.dependency_overrides.clear()


@pytest.fixture()
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
