"""Local / fallback provider.

Returns deterministic mock responses so the gateway works without any
real API keys in development or CI environments.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from app.providers import (
    BaseProvider,
    EmbeddingProviderResponse,
    LLMResponse,
    ProviderCapability,
)

_CAPABILITIES: List[ProviderCapability] = [
    ProviderCapability(
        provider="local",
        model="local-echo",
        supports_chat=True,
        supports_embeddings=True,
        max_context_tokens=8192,
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        typical_latency_ms=5,
        priority=100,  # lowest priority — only used as fallback
    ),
]


def _count_tokens(text: str) -> int:
    """Very rough token estimator (≈ 4 chars per token)."""
    return max(1, math.ceil(len(text) / 4))


class LocalProvider(BaseProvider):
    """Always-available fallback that echoes prompts back."""

    @property
    def name(self) -> str:
        return "local"

    def capabilities(self) -> List[ProviderCapability]:
        return _CAPABILITIES

    async def is_available(self) -> bool:
        return True

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "local-echo",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last = messages[-1]["content"] if messages else ""
        reply = f"[local-echo] {last}"
        prompt_tokens = sum(_count_tokens(m["content"]) for m in messages)
        return LLMResponse(
            provider=self.name,
            model=model,
            content=reply,
            prompt_tokens=prompt_tokens,
            completion_tokens=_count_tokens(reply),
        )

    async def embed(
        self,
        texts: List[str],
        model: str = "local-echo",
        **kwargs: Any,
    ) -> EmbeddingProviderResponse:
        # Produce a trivial 4-D embedding so the response is parseable
        embeddings = [
            [float(ord(c) % 256) / 255.0 for c in (t[:4].ljust(4))]
            for t in texts
        ]
        prompt_tokens = sum(_count_tokens(t) for t in texts)
        return EmbeddingProviderResponse(
            provider=self.name,
            model=model,
            embeddings=embeddings,
            prompt_tokens=prompt_tokens,
        )
