"""LLM Gateway — FastAPI application entry point."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status

from app.cache import ResponseCache
from app.config import settings
from app.cost import CostTracker
from app.middleware import RateLimiter
from app.models import (
    ChatRequest,
    ChatResponse,
    ChatChoice,
    CostSummary,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    Message,
    Usage,
)
from app.observability import log_request, setup_telemetry, timed
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.local_provider import LocalProvider
from app.providers.openai_provider import OpenAIProvider
from app.routing import ModelRouter, RoutingStrategy

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — initialise shared resources once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_telemetry(settings.otel_service_name)

    # Try to connect to Redis (optional)
    redis_client = None
    try:
        import redis.asyncio as aioredis  # type: ignore

        redis_client = aioredis.from_url(
            settings.redis_url, encoding="utf-8", decode_responses=True
        )
        await redis_client.ping()
        logger.info("Redis connected at %s", settings.redis_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis unavailable (%s), using in-memory fallbacks", exc)
        redis_client = None

    # Build providers
    providers = [
        OpenAIProvider(api_key=settings.openai_api_key),
        AnthropicProvider(api_key=settings.anthropic_api_key),
        LocalProvider(),
    ]

    app.state.router = ModelRouter(providers, strategy=RoutingStrategy.COST)
    app.state.cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds, redis=redis_client)
    app.state.rate_limiter = RateLimiter(
        user_rpm=settings.default_user_rpm,
        tenant_rpm=settings.default_tenant_rpm,
        redis=redis_client,
    )
    app.state.cost_tracker = CostTracker(database_url=settings.database_url)
    await app.state.cost_tracker.init_db()

    logger.info("LLM Gateway started (v%s)", settings.app_version)
    yield

    # Teardown
    if redis_client:
        await redis_client.aclose()
    logger.info("LLM Gateway shut down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Unified API gateway for routing requests across multiple LLM providers "
        "with rate limiting, caching, cost tracking and observability."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def get_router(request: Request) -> ModelRouter:
    return request.app.state.router


def get_cache(request: Request) -> ResponseCache:
    return request.app.state.cache


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def get_cost_tracker(request: Request) -> CostTracker:
    return request.app.state.cost_tracker


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health(router: ModelRouter = Depends(get_router)) -> HealthResponse:
    """Gateway health check — lists registered providers."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        providers=router.health(),
    )


@app.post("/v1/chat", response_model=ChatResponse, tags=["llm"])
async def chat(
    body: ChatRequest,
    router: ModelRouter = Depends(get_router),
    cache: ResponseCache = Depends(get_cache),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> ChatResponse:
    """Send a chat completion request through the gateway."""

    # Rate limit check
    allowed = await rate_limiter.check(body.user_id, body.tenant_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    messages_dicts = [m.model_dump() for m in body.messages]

    # Cache lookup
    cache_key = {
        "messages": messages_dicts,
        "model": body.model,
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
    }
    cached_data = await cache.get("chat", cache_key)
    if cached_data:
        cached_data["cached"] = True
        log_request(
            endpoint="chat",
            provider=cached_data.get("provider", "unknown"),
            model=cached_data.get("model", "unknown"),
            user_id=body.user_id,
            tenant_id=body.tenant_id,
            latency_ms=0,
            prompt_tokens=cached_data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=cached_data.get("usage", {}).get("completion_tokens", 0),
            cached=True,
        )
        return ChatResponse(**cached_data)

    # Route
    provider, model_name = await router.route_chat(messages_dicts, body.model)

    # Call provider
    latency: dict = {}
    error_msg = None
    try:
        async with timed() as latency:
            llm_response = await provider.chat(
                messages=messages_dicts,
                model=model_name,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            )
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        logger.error("Provider %s error: %s", provider.name, exc)
        log_request(
            endpoint="chat",
            provider=provider.name,
            model=model_name,
            user_id=body.user_id,
            tenant_id=body.tenant_id,
            latency_ms=latency.get("latency_ms", 0),
            prompt_tokens=0,
            completion_tokens=0,
            cached=False,
            error=error_msg,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Provider error: {exc}",
        ) from exc

    log_request(
        endpoint="chat",
        provider=provider.name,
        model=model_name,
        user_id=body.user_id,
        tenant_id=body.tenant_id,
        latency_ms=latency.get("latency_ms", 0),
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        cached=False,
    )

    cost = await cost_tracker.compute_cost(
        provider.name,
        model_name,
        llm_response.prompt_tokens,
        llm_response.completion_tokens,
    )

    await cost_tracker.record(
        tenant_id=body.tenant_id,
        user_id=body.user_id,
        provider=provider.name,
        model=model_name,
        endpoint="chat",
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        cost_usd=cost,
        latency_ms=latency.get("latency_ms", 0),
        cached=False,
    )

    response = ChatResponse(
        id=str(uuid.uuid4()),
        model=llm_response.model,
        provider=llm_response.provider,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=llm_response.content),
                finish_reason=llm_response.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
        ),
        cached=False,
        cost_usd=cost,
    )

    await cache.set("chat", cache_key, response.model_dump())
    return response


@app.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["llm"])
async def embeddings(
    body: EmbeddingRequest,
    router: ModelRouter = Depends(get_router),
    cache: ResponseCache = Depends(get_cache),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> EmbeddingResponse:
    """Generate embeddings through the gateway."""

    allowed = await rate_limiter.check(body.user_id, body.tenant_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    cache_key = {"input": body.input, "model": body.model}
    cached_data = await cache.get("embeddings", cache_key)
    if cached_data:
        cached_data["cached"] = True
        return EmbeddingResponse(**cached_data)

    provider, model_name = await router.route_embedding(body.model)

    latency: dict = {}
    async with timed() as latency:
        emb_response = await provider.embed(texts=body.input, model=model_name)

    cost = await cost_tracker.compute_cost(
        provider.name, model_name, emb_response.prompt_tokens, 0
    )

    await cost_tracker.record(
        tenant_id=body.tenant_id,
        user_id=body.user_id,
        provider=provider.name,
        model=model_name,
        endpoint="embeddings",
        prompt_tokens=emb_response.prompt_tokens,
        completion_tokens=0,
        cost_usd=cost,
        latency_ms=latency.get("latency_ms", 0),
        cached=False,
    )

    log_request(
        endpoint="embeddings",
        provider=provider.name,
        model=model_name,
        user_id=body.user_id,
        tenant_id=body.tenant_id,
        latency_ms=latency.get("latency_ms", 0),
        prompt_tokens=emb_response.prompt_tokens,
        completion_tokens=0,
        cached=False,
    )

    response = EmbeddingResponse(
        model=emb_response.model,
        provider=emb_response.provider,
        data=[
            EmbeddingData(index=i, embedding=emb)
            for i, emb in enumerate(emb_response.embeddings)
        ],
        usage=Usage(
            prompt_tokens=emb_response.prompt_tokens,
            completion_tokens=0,
            total_tokens=emb_response.prompt_tokens,
        ),
        cached=False,
        cost_usd=cost,
    )

    await cache.set("embeddings", cache_key, response.model_dump())
    return response


@app.get("/v1/analytics/cost/{tenant_id}", response_model=CostSummary, tags=["analytics"])
async def tenant_cost(
    tenant_id: str,
    period: str = "all_time",
    cost_tracker: CostTracker = Depends(get_cost_tracker),
) -> CostSummary:
    """Return cost analytics for a tenant."""
    effective_period = None if period == "all_time" else period
    data = await cost_tracker.get_tenant_summary(tenant_id, effective_period)
    return CostSummary(**data)
