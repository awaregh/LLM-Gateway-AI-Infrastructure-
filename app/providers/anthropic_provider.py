"""Anthropic provider adapter (Messages API)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.providers import BaseProvider, LLMResponse, ProviderCapability

logger = logging.getLogger(__name__)

_CHAT_MODELS: List[ProviderCapability] = [
    ProviderCapability(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        supports_chat=True,
        supports_embeddings=False,
        max_context_tokens=200_000,
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        typical_latency_ms=1200,
        priority=30,
    ),
    ProviderCapability(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        supports_chat=True,
        supports_embeddings=False,
        max_context_tokens=200_000,
        cost_per_1k_input_tokens=0.00025,
        cost_per_1k_output_tokens=0.00125,
        typical_latency_ms=600,
        priority=15,
    ),
]

_BASE_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "anthropic"

    def capabilities(self) -> List[ProviderCapability]:
        return _CHAT_MODELS

    async def is_available(self) -> bool:
        return bool(self._api_key)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-haiku-20240307",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if not self._api_key:
            raise RuntimeError("Anthropic API key is not configured")

        # Anthropic requires explicit max_tokens
        effective_max_tokens = max_tokens or 1024

        # Separate system messages from the conversation
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": effective_max_tokens,
            "messages": user_messages,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        if temperature is not None:
            payload["temperature"] = temperature

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{_BASE_URL}/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        content = "".join(
            block.get("text", "") for block in data.get("content", [])
        )
        usage = data.get("usage", {})
        return LLMResponse(
            provider=self.name,
            model=data.get("model", model),
            content=content,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "stop"),
        )
