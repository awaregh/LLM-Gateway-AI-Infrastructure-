"""Rate limiting middleware using a sliding-window algorithm backed by Redis.

Falls back to an in-memory store when Redis is unavailable, so the gateway
remains functional in development / test environments.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory fallback store
# ---------------------------------------------------------------------------

class _InMemoryWindow:
    """Per-key sliding-window counter."""

    def __init__(self) -> None:
        self._windows: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        cutoff = now - window_seconds
        q = self._windows[key]
        # Prune old entries
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True


_fallback_store = _InMemoryWindow()


# ---------------------------------------------------------------------------
# Redis-backed store
# ---------------------------------------------------------------------------

async def _redis_is_allowed(
    redis,  # redis.asyncio.Redis
    key: str,
    limit: int,
    window_seconds: int,
) -> bool:
    """Sliding-window check using a sorted-set in Redis."""
    now = time.time()
    cutoff = now - window_seconds
    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, "-inf", cutoff)
    pipe.zadd(key, {str(now): now})
    pipe.zcard(key)
    pipe.expire(key, window_seconds + 1)
    results = await pipe.execute()
    count = results[2]
    return count <= limit


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------

class RateLimiter:
    """Check per-user and per-tenant sliding-window rate limits."""

    def __init__(
        self,
        user_rpm: int = 60,
        tenant_rpm: int = 300,
        redis=None,
    ) -> None:
        self._user_rpm = user_rpm
        self._tenant_rpm = tenant_rpm
        self._redis = redis

    async def check(self, user_id: str, tenant_id: str) -> bool:
        """Return True if the request is within limits, False otherwise."""
        user_key = f"rl:user:{user_id}"
        tenant_key = f"rl:tenant:{tenant_id}"

        if self._redis is not None:
            try:
                user_ok = await _redis_is_allowed(
                    self._redis, user_key, self._user_rpm, 60
                )
                tenant_ok = await _redis_is_allowed(
                    self._redis, tenant_key, self._tenant_rpm, 60
                )
                return user_ok and tenant_ok
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis rate-limit check failed, falling back: %s", exc)

        # In-memory fallback
        user_ok = _fallback_store.is_allowed(user_key, self._user_rpm, 60)
        tenant_ok = _fallback_store.is_allowed(tenant_key, self._tenant_rpm, 60)
        return user_ok and tenant_ok
