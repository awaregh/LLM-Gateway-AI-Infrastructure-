"""Cost tracking — stores per-request records in a lightweight DB.

Uses SQLAlchemy async with SQLite by default; swap DATABASE_URL for Postgres.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class RequestRecord(Base):
    __tablename__ = "request_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    tenant_id = Column(String, index=True, nullable=False)
    user_id = Column(String, index=True, nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)  # "chat" | "embeddings"
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    cached = Column(Integer, default=0)  # 0 | 1


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CostTracker:
    def __init__(self, database_url: str) -> None:
        self._engine = create_async_engine(database_url, echo=False)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def record(
        self,
        *,
        tenant_id: str,
        user_id: str,
        provider: str,
        model: str,
        endpoint: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
        cached: bool,
    ) -> None:
        record = RequestRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            provider=provider,
            model=model,
            endpoint=endpoint,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            cached=int(cached),
        )
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()

    async def get_tenant_summary(
        self, tenant_id: str, period: Optional[str] = None
    ) -> dict:
        """Return aggregated stats for a tenant.

        *period* accepts 'today' or an ISO date string (YYYY-MM-DD).
        """
        async with self._session_factory() as session:
            stmt = select(
                func.count(RequestRecord.id).label("total_requests"),
                func.sum(RequestRecord.total_tokens).label("total_tokens"),
                func.sum(RequestRecord.cost_usd).label("total_cost_usd"),
            ).where(RequestRecord.tenant_id == tenant_id)

            if period:
                day = date.today() if period == "today" else date.fromisoformat(period)
                start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
                stmt = stmt.where(RequestRecord.created_at >= start)

            row = (await session.execute(stmt)).one()
            return {
                "tenant_id": tenant_id,
                "total_requests": row.total_requests or 0,
                "total_tokens": int(row.total_tokens or 0),
                "total_cost_usd": float(row.total_cost_usd or 0.0),
                "period": period or "all_time",
            }

    async def compute_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost in USD for a given usage."""
        # Look up pricing from provider capabilities
        from app.providers.openai_provider import _CHAT_MODELS as oai_models
        from app.providers.anthropic_provider import _CHAT_MODELS as ant_models
        from app.providers.local_provider import _CAPABILITIES as local_caps

        all_caps = oai_models + ant_models + local_caps
        for cap in all_caps:
            if cap.provider == provider and cap.model == model:
                cost = (
                    prompt_tokens / 1000 * cap.cost_per_1k_input_tokens
                    + completion_tokens / 1000 * cap.cost_per_1k_output_tokens
                )
                return round(cost, 8)
        return 0.0
