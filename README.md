# LLM Gateway — AI Infrastructure

A **production-ready unified API gateway** for routing requests across multiple LLM providers (OpenAI, Anthropic, and local fallback) with built-in rate limiting, response caching, cost tracking, and OpenTelemetry observability.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI App                        │
│                                                         │
│  POST /v1/chat          POST /v1/embeddings             │
│  GET  /v1/analytics/cost/{tenant}   GET /health         │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Rate Limiter│  │ Response     │  │ Cost Tracker  │  │
│  │ (Redis /    │  │ Cache        │  │ (SQLite /     │  │
│  │  in-memory) │  │ (Redis /     │  │  Postgres)    │  │
│  │             │  │  in-memory)  │  │               │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Model Routing Engine                  │  │
│  │  Strategies: cost | latency | capability | rr      │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  OpenAI    │  │   Anthropic     │  │   Local     │  │
│  │  Provider  │  │   Provider      │  │  (fallback) │  │
│  └────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Implementation |
|---|---|
| **Unified API** | `POST /v1/chat`, `POST /v1/embeddings` |
| **Model routing** | Cost, latency, capability, round-robin strategies |
| **Rate limiting** | Sliding-window per user + per tenant (Redis or in-memory) |
| **Response caching** | SHA-256 keyed cache with configurable TTL (Redis or in-memory) |
| **Cost tracking** | Per-request token usage + USD cost stored in DB |
| **Cost analytics** | `GET /v1/analytics/cost/{tenant_id}` |
| **Observability** | Structured logging + OpenTelemetry metrics & traces |
| **Auto-docs** | Swagger UI at `/docs`, ReDoc at `/redoc` |
| **Graceful fallback** | Local echo provider when no API keys are present |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/awaregh/LLM-Gateway-AI-Infrastructure-.git
cd LLM-Gateway-AI-Infrastructure-
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Run with Docker Compose (recommended)

```bash
docker compose up
```

The gateway will start at `http://localhost:8000` with Redis and Postgres.

### 3. Run locally (development)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## API Reference

### POST /v1/chat

Send a chat completion request. The gateway routes to the optimal provider automatically.

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "user_id": "alice",
    "tenant_id": "acme"
  }'
```

**Request body:**

| Field | Type | Description |
|---|---|---|
| `messages` | `Message[]` | Conversation history |
| `model` | `string?` | Specific model (optional; router selects if omitted) |
| `max_tokens` | `int?` | Maximum tokens to generate |
| `temperature` | `float?` | Sampling temperature (0–2) |
| `user_id` | `string` | Caller user identifier (for rate limiting) |
| `tenant_id` | `string` | Caller tenant identifier (for rate limiting + billing) |

**Response:**

```json
{
  "id": "3f4a...",
  "object": "chat.completion",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
  "cached": false,
  "cost_usd": 0.0000027
}
```

---

### POST /v1/embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello world", "Goodbye world"], "tenant_id": "acme"}'
```

---

### GET /v1/analytics/cost/{tenant_id}

```bash
curl "http://localhost:8000/v1/analytics/cost/acme?period=today"
```

**Response:**

```json
{
  "tenant_id": "acme",
  "total_requests": 142,
  "total_tokens": 58300,
  "total_cost_usd": 0.0087,
  "period": "today"
}
```

`period` accepts `all_time` (default) or an ISO date (`2024-01-15`).

---

### GET /health

```bash
curl http://localhost:8000/health
```

---

## Model Routing

The routing engine selects the best provider/model for each request based on the configured strategy:

| Strategy | Description |
|---|---|
| `cost` (default) | Lowest combined input+output cost |
| `latency` | Lowest typical response latency |
| `capability` | Highest context window |
| `round_robin` | Rotate through available providers |

**Automatic fallback chain:**

1. Small/fast request → `gpt-4o-mini` (cheapest capable model)
2. Large context or high-quality → `claude-3-5-sonnet`
3. No API keys configured → `local-echo` (always available)

Change strategy via the `ROUTING_PREFER_COST` env var or extend `ModelRouter`.

---

## Rate Limiting

Sliding-window algorithm (60-second window by default):

- **Per-user:** `DEFAULT_USER_RPM` requests per minute (default: 60)
- **Per-tenant:** `DEFAULT_TENANT_RPM` requests per minute (default: 300)

When Redis is available, limits are enforced across all gateway instances. Otherwise an in-memory fallback is used per-process.

Exceeded limits return `HTTP 429 Too Many Requests`.

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `DATABASE_URL` | `sqlite+aiosqlite:///./llm_gateway.db` | Analytics DB URL |
| `CACHE_TTL_SECONDS` | `3600` | Cache TTL in seconds |
| `DEFAULT_USER_RPM` | `60` | Per-user rate limit |
| `DEFAULT_TENANT_RPM` | `300` | Per-tenant rate limit |
| `DEBUG` | `false` | Enable debug logging |
| `OTEL_SERVICE_NAME` | `llm-gateway` | OpenTelemetry service name |
| `OTEL_EXPORTER_ENDPOINT` | — | OTLP gRPC exporter endpoint |

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All tests run without external services (Redis/Postgres/LLM API keys) using in-memory fallbacks and a temporary SQLite database.

---

## Project Structure

```
app/
├── main.py              # FastAPI app, routes, dependency injection
├── config.py            # Pydantic Settings
├── models/              # Request/response schemas
├── providers/           # LLM provider adapters
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── local_provider.py
├── routing/             # Model routing engine
├── middleware/          # Rate limiting
├── cache/               # Response caching
├── cost/                # Cost tracking & analytics
└── observability/       # Logging + OpenTelemetry
tests/
├── conftest.py          # Shared fixtures
├── test_api.py          # Integration tests
├── test_routing.py      # Routing engine tests
├── test_rate_limiter.py # Rate limiter tests
├── test_cache.py        # Cache tests
├── test_cost_tracker.py # Cost tracker tests
└── test_providers.py    # Provider adapter tests
```