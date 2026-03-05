"""Tests for individual provider adapters."""
from __future__ import annotations

import pytest

from app.providers.local_provider import LocalProvider


@pytest.fixture()
def provider():
    return LocalProvider()


@pytest.mark.asyncio
async def test_local_is_available(provider):
    assert await provider.is_available() is True


@pytest.mark.asyncio
async def test_local_chat_echoes_last_message(provider):
    messages = [{"role": "user", "content": "ping"}]
    response = await provider.chat(messages=messages, model="local-echo")
    assert "ping" in response.content
    assert response.provider == "local"
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0


@pytest.mark.asyncio
async def test_local_embed_returns_4d_vector(provider):
    response = await provider.embed(texts=["test"], model="local-echo")
    assert len(response.embeddings) == 1
    assert len(response.embeddings[0]) == 4


@pytest.mark.asyncio
async def test_local_embed_multiple_texts(provider):
    response = await provider.embed(texts=["a", "b", "c"], model="local-echo")
    assert len(response.embeddings) == 3


@pytest.mark.asyncio
async def test_openai_unavailable_without_key():
    from app.providers.openai_provider import OpenAIProvider

    p = OpenAIProvider(api_key=None)
    assert await p.is_available() is False


@pytest.mark.asyncio
async def test_anthropic_unavailable_without_key():
    from app.providers.anthropic_provider import AnthropicProvider

    p = AnthropicProvider(api_key=None)
    assert await p.is_available() is False
