"""Abstract base class for all LLM providers."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProviderCapability:
    """Static description of what a provider/model can do."""
    provider: str
    model: str
    supports_chat: bool = True
    supports_embeddings: bool = False
    max_context_tokens: int = 4096
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    typical_latency_ms: float = 500.0
    priority: int = 10  # lower = higher priority in routing


@dataclass
class LLMResponse:
    """Normalised response returned by every provider."""
    provider: str
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class EmbeddingProviderResponse:
    provider: str
    model: str
    embeddings: List[List[float]]
    prompt_tokens: int
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseProvider(abc.ABC):
    """Common interface that every LLM provider adapter must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def capabilities(self) -> List[ProviderCapability]: ...

    @abc.abstractmethod
    async def is_available(self) -> bool: ...

    @abc.abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def embed(
        self,
        texts: List[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingProviderResponse:
        raise NotImplementedError(f"{self.name} does not support embeddings")
