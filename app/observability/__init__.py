"""Observability — structured logging + OpenTelemetry metrics."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global OTel setup (call once at startup)
# ---------------------------------------------------------------------------

_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None
_request_counter: Any = None
_latency_histogram: Any = None
_token_counter: Any = None


def setup_telemetry(service_name: str = "llm-gateway") -> None:
    global _tracer, _meter, _request_counter, _latency_histogram, _token_counter

    # Tracing
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(service_name)

    # Metrics
    reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=30_000)
    meter_provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter(service_name)

    _request_counter = _meter.create_counter(
        "llm_gateway.requests",
        unit="1",
        description="Total LLM requests processed",
    )
    _latency_histogram = _meter.create_histogram(
        "llm_gateway.latency_ms",
        unit="ms",
        description="Request latency in milliseconds",
    )
    _token_counter = _meter.create_counter(
        "llm_gateway.tokens",
        unit="1",
        description="Total tokens consumed",
    )


# ---------------------------------------------------------------------------
# Per-request logging helper
# ---------------------------------------------------------------------------

def log_request(
    *,
    endpoint: str,
    provider: str,
    model: str,
    user_id: str,
    tenant_id: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    cached: bool,
    error: Optional[str] = None,
) -> None:
    attrs: Dict[str, Any] = {
        "endpoint": endpoint,
        "provider": provider,
        "model": model,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "cached": cached,
    }
    logger.info(
        "request endpoint=%s provider=%s model=%s user=%s tenant=%s "
        "latency_ms=%.1f prompt_tokens=%d completion_tokens=%d cached=%s error=%s",
        endpoint,
        provider,
        model,
        user_id,
        tenant_id,
        latency_ms,
        prompt_tokens,
        completion_tokens,
        cached,
        error,
    )
    if _request_counter:
        _request_counter.add(1, attrs)
    if _latency_histogram:
        _latency_histogram.record(latency_ms, attrs)
    if _token_counter:
        _token_counter.add(prompt_tokens + completion_tokens, attrs)


@asynccontextmanager
async def timed() -> AsyncIterator[Dict[str, float]]:
    """Context manager that measures wall-clock time."""
    result: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["latency_ms"] = (time.perf_counter() - start) * 1000
