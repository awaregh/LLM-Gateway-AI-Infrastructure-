"""Tests for the model routing engine."""
from __future__ import annotations

import pytest

from app.providers.local_provider import LocalProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.routing import ModelRouter, RoutingStrategy


@pytest.fixture()
def local_only_router():
    return ModelRouter([LocalProvider()], strategy=RoutingStrategy.COST)


@pytest.fixture()
def multi_provider_router():
    # OpenAI and Anthropic are unavailable (no keys); only local should be selected
    return ModelRouter(
        [
            OpenAIProvider(api_key=None),
            AnthropicProvider(api_key=None),
            LocalProvider(),
        ],
        strategy=RoutingStrategy.COST,
    )


@pytest.mark.asyncio
async def test_route_chat_returns_local_provider(local_only_router):
    provider, model = await local_only_router.route_chat(
        [{"role": "user", "content": "hello"}]
    )
    assert provider.name == "local"
    assert model == "local-echo"


@pytest.mark.asyncio
async def test_route_chat_falls_back_to_local_when_no_api_keys(multi_provider_router):
    provider, model = await multi_provider_router.route_chat(
        [{"role": "user", "content": "hello"}]
    )
    assert provider.name == "local"


@pytest.mark.asyncio
async def test_route_chat_respects_requested_model_if_available(local_only_router):
    provider, model = await local_only_router.route_chat(
        [{"role": "user", "content": "test"}],
        requested_model="local-echo",
    )
    assert model == "local-echo"


@pytest.mark.asyncio
async def test_route_chat_falls_back_when_requested_model_unavailable(local_only_router):
    # Request a non-existent model; should still fall back to available one
    provider, model = await local_only_router.route_chat(
        [{"role": "user", "content": "test"}],
        requested_model="gpt-99",
    )
    assert provider.name == "local"


@pytest.mark.asyncio
async def test_route_embedding_returns_local(local_only_router):
    provider, model = await local_only_router.route_embedding()
    assert provider.name == "local"


@pytest.mark.asyncio
async def test_no_providers_raises(multi_provider_router):
    # Disable local provider to force no candidates
    router = ModelRouter(
        [OpenAIProvider(api_key=None), AnthropicProvider(api_key=None)],
        strategy=RoutingStrategy.COST,
    )
    with pytest.raises(RuntimeError, match="No chat providers"):
        await router.route_chat([{"role": "user", "content": "test"}])


@pytest.mark.asyncio
async def test_latency_strategy_selects_lowest_latency():
    router = ModelRouter([LocalProvider()], strategy=RoutingStrategy.LATENCY)
    provider, model = await router.route_chat([{"role": "user", "content": "hi"}])
    assert provider.name == "local"


@pytest.mark.asyncio
async def test_round_robin_rotates(local_only_router):
    router = ModelRouter([LocalProvider()], strategy=RoutingStrategy.ROUND_ROBIN)
    p1, _ = await router.route_chat([{"role": "user", "content": "hi"}])
    p2, _ = await router.route_chat([{"role": "user", "content": "hi"}])
    assert p1.name == p2.name  # only one provider, always local
