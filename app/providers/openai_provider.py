"""OpenAI provider adapter."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.providers import (
    BaseProvider,
    EmbeddingProviderResponse,
    LLMResponse,
    ProviderCapability,
)

logger = logging.getLogger(__name__)

_CHAT_MODELS: List[ProviderCapability] = [
    ProviderCapability(
        provider="openai",
        model="gpt-4o",
        supports_chat=True,
        supports_embeddings=False,
        max_context_tokens=128_000,
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        typical_latency_ms=800,
        priority=20,
    ),
    ProviderCapability(
        provider="openai",
        model="gpt-4o-mini",
        supports_chat=True,
        supports_embeddings=False,
        max_context_tokens=128_000,
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        typical_latency_ms=400,
        priority=10,
    ),
    ProviderCapability(
        provider="openai",
        model="text-embedding-3-small",
        supports_chat=False,
        supports_embeddings=True,
        max_context_tokens=8191,
        cost_per_1k_input_tokens=0.00002,
        cost_per_1k_output_tokens=0.0,
        typical_latency_ms=200,
        priority=5,
    ),
]

_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "openai"

    def capabilities(self) -> List[ProviderCapability]:
        return _CHAT_MODELS

    async def is_available(self) -> bool:
        return bool(self._api_key)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if not self._api_key:
            raise RuntimeError("OpenAI API key is not configured")

        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            provider=self.name,
            model=data.get("model", model),
            content=choice["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        **kwargs: Any,
    ) -> EmbeddingProviderResponse:
        if not self._api_key:
            raise RuntimeError("OpenAI API key is not configured")

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{_BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]
        usage = data.get("usage", {})
        return EmbeddingProviderResponse(
            provider=self.name,
            model=data.get("model", model),
            embeddings=embeddings,
            prompt_tokens=usage.get("prompt_tokens", 0),
        )
