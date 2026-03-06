"""Tests for the response cache."""
from __future__ import annotations

import pytest

from app.cache import ResponseCache


@pytest.fixture()
def cache():
    return ResponseCache(ttl_seconds=60)


@pytest.mark.asyncio
async def test_miss_returns_none(cache):
    result = await cache.get("chat", {"messages": [{"role": "user", "content": "hi"}]})
    assert result is None


@pytest.mark.asyncio
async def test_set_then_get(cache):
    key_data = {"messages": [{"role": "user", "content": "hello"}], "model": "local-echo"}
    value = {"response": "world", "model": "local-echo"}
    await cache.set("chat", key_data, value)
    result = await cache.get("chat", key_data)
    assert result == value


@pytest.mark.asyncio
async def test_different_namespaces_are_isolated(cache):
    key_data = {"input": ["test"]}
    await cache.set("embeddings", key_data, {"embedding": [0.1, 0.2]})
    # Same key data under a different namespace should miss
    result = await cache.get("chat", key_data)
    assert result is None


@pytest.mark.asyncio
async def test_different_key_data_are_isolated(cache):
    await cache.set("chat", {"msg": "hello"}, {"reply": "hi"})
    result = await cache.get("chat", {"msg": "goodbye"})
    assert result is None


@pytest.mark.asyncio
async def test_overwrite(cache):
    key_data = {"msg": "same"}
    await cache.set("chat", key_data, {"v": 1})
    await cache.set("chat", key_data, {"v": 2})
    result = await cache.get("chat", key_data)
    assert result == {"v": 2}
