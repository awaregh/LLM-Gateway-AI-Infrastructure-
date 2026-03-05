"""Model routing engine.

Selects the best provider/model for a given request based on:
  - cost
  - latency
  - model capability (context length)
  - availability
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, Tuple

from app.providers import BaseProvider, ProviderCapability

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    COST = "cost"          # prefer cheapest
    LATENCY = "latency"    # prefer fastest
    CAPABILITY = "capability"  # prefer highest context window
    ROUND_ROBIN = "round_robin"  # rotate providers evenly


def _estimate_input_tokens(messages: List[dict]) -> int:
    """Rough token count used during routing (no tokenizer dependency)."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, total_chars // 4)


def _score(cap: ProviderCapability, strategy: RoutingStrategy) -> float:
    """Lower score = better candidate."""
    if strategy == RoutingStrategy.COST:
        return cap.cost_per_1k_input_tokens + cap.cost_per_1k_output_tokens
    if strategy == RoutingStrategy.LATENCY:
        return cap.typical_latency_ms
    if strategy == RoutingStrategy.CAPABILITY:
        return -float(cap.max_context_tokens)  # negate: bigger is better
    # ROUND_ROBIN falls through to priority
    return float(cap.priority)


class ModelRouter:
    """Routes chat / embedding requests to the best available provider."""

    def __init__(
        self,
        providers: List[BaseProvider],
        strategy: RoutingStrategy = RoutingStrategy.COST,
    ) -> None:
        self._providers = providers
        self._strategy = strategy
        self._rr_index = 0  # round-robin counter

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_provider_by_name(self, name: str) -> Optional[BaseProvider]:
        for p in self._providers:
            if p.name == name:
                return p
        return None

    async def route_chat(
        self,
        messages: List[dict],
        requested_model: Optional[str] = None,
    ) -> Tuple[BaseProvider, str]:
        """Return (provider, model_name) for a chat request."""
        input_tokens = _estimate_input_tokens(messages)

        candidates = await self._get_chat_candidates(input_tokens)
        if not candidates:
            raise RuntimeError("No chat providers are currently available")

        # Honour explicit model selection if the provider is available
        if requested_model:
            for provider, cap in candidates:
                if cap.model == requested_model:
                    logger.info(
                        "Routing to requested model %s via %s",
                        requested_model,
                        provider.name,
                    )
                    return provider, cap.model
            logger.warning(
                "Requested model %s not available; falling back to routing",
                requested_model,
            )

        provider, cap = self._select(candidates)
        logger.info(
            "Routing chat → provider=%s model=%s strategy=%s",
            provider.name,
            cap.model,
            self._strategy,
        )
        return provider, cap.model

    async def route_embedding(
        self,
        requested_model: Optional[str] = None,
    ) -> Tuple[BaseProvider, str]:
        """Return (provider, model_name) for an embedding request."""
        candidates = await self._get_embedding_candidates()
        if not candidates:
            raise RuntimeError("No embedding providers are currently available")

        if requested_model:
            for provider, cap in candidates:
                if cap.model == requested_model:
                    return provider, cap.model

        provider, cap = self._select(candidates)
        logger.info(
            "Routing embedding → provider=%s model=%s",
            provider.name,
            cap.model,
        )
        return provider, cap.model

    def health(self) -> dict:
        return {
            p.name: {"registered": True} for p in self._providers
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_chat_candidates(
        self, input_tokens: int
    ) -> List[Tuple[BaseProvider, ProviderCapability]]:
        candidates = []
        for provider in self._providers:
            if not await provider.is_available():
                continue
            for cap in provider.capabilities():
                if not cap.supports_chat:
                    continue
                if cap.max_context_tokens < input_tokens:
                    continue
                candidates.append((provider, cap))
        return candidates

    async def _get_embedding_candidates(
        self,
    ) -> List[Tuple[BaseProvider, ProviderCapability]]:
        candidates = []
        for provider in self._providers:
            if not await provider.is_available():
                continue
            for cap in provider.capabilities():
                if cap.supports_embeddings:
                    candidates.append((provider, cap))
        return candidates

    def _select(
        self, candidates: List[Tuple[BaseProvider, ProviderCapability]]
    ) -> Tuple[BaseProvider, ProviderCapability]:
        if self._strategy == RoutingStrategy.ROUND_ROBIN:
            idx = self._rr_index % len(candidates)
            self._rr_index += 1
            return candidates[idx]

        return min(candidates, key=lambda pc: _score(pc[1], self._strategy))
