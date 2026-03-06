"""Response caching layer.

Caches chat and embedding responses keyed by (prompt, model, params).
Uses Redis when available, falls back to an in-process dict.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _make_key(namespace: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"cache:{namespace}:{digest}"


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------

class _InMemoryCache:
    def __init__(self) -> None:
        self._store: Dict[str, tuple] = {}  # key → (value, expires_at)

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at and time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl: int) -> None:
        expires_at = time.monotonic() + ttl if ttl else None
        self._store[key] = (value, expires_at)


_fallback_cache = _InMemoryCache()


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------

class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600, redis=None) -> None:
        self._ttl = ttl_seconds
        self._redis = redis

    # ------------------------------------------------------------------

    async def get(self, namespace: str, key_data: Dict[str, Any]) -> Optional[Any]:
        key = _make_key(namespace, key_data)
        raw: Optional[str] = None

        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis cache GET failed: %s", exc)

        if raw is None:
            raw = _fallback_cache.get(key)

        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set(
        self, namespace: str, key_data: Dict[str, Any], value: Any
    ) -> None:
        key = _make_key(namespace, key_data)
        raw = json.dumps(value)

        if self._redis is not None:
            try:
                await self._redis.setex(key, self._ttl, raw)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis cache SET failed: %s", exc)

        _fallback_cache.set(key, raw, self._ttl)
