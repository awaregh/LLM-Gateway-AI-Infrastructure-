"""Tests for the cost tracker."""
from __future__ import annotations

import pytest

from app.cost import CostTracker


@pytest.fixture()
async def tracker(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/tracker_test.db"
    t = CostTracker(database_url=db_url)
    await t.init_db()
    return t


@pytest.mark.asyncio
async def test_record_and_summary(tracker):
    await tracker.record(
        tenant_id="acme",
        user_id="alice",
        provider="local",
        model="local-echo",
        endpoint="chat",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.0,
        latency_ms=5.0,
        cached=False,
    )
    summary = await tracker.get_tenant_summary("acme")
    assert summary["total_requests"] == 1
    assert summary["total_tokens"] == 150


@pytest.mark.asyncio
async def test_empty_summary(tracker):
    summary = await tracker.get_tenant_summary("nobody")
    assert summary["total_requests"] == 0
    assert summary["total_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_compute_cost_openai(tracker):
    cost = await tracker.compute_cost("openai", "gpt-4o-mini", 1000, 500)
    # 1000 * 0.00015/1000 + 500 * 0.0006/1000 = 0.00015 + 0.0003 = 0.00045
    assert abs(cost - 0.00045) < 1e-7


@pytest.mark.asyncio
async def test_compute_cost_local_is_zero(tracker):
    cost = await tracker.compute_cost("local", "local-echo", 1000, 1000)
    assert cost == 0.0


@pytest.mark.asyncio
async def test_multiple_records_aggregate(tracker):
    for i in range(3):
        await tracker.record(
            tenant_id="multi",
            user_id=f"user{i}",
            provider="local",
            model="local-echo",
            endpoint="chat",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01,
            latency_ms=5.0,
            cached=False,
        )
    summary = await tracker.get_tenant_summary("multi")
    assert summary["total_requests"] == 3
    assert abs(summary["total_cost_usd"] - 0.03) < 1e-7
