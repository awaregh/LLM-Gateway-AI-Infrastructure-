"""Configuration for the LLM Gateway."""
from __future__ import annotations

from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_title: str = "LLM Gateway"
    app_version: str = "0.1.0"
    debug: bool = False

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # --- Database (SQLite default, switch to Postgres via env) ---
    database_url: str = "sqlite+aiosqlite:///./llm_gateway.db"

    # --- Provider API keys (optional — gateway still works without them) ---
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # --- Rate limiting ---
    default_user_rpm: int = 60       # requests per minute per user
    default_tenant_rpm: int = 300    # requests per minute per tenant

    # --- Routing weights ---
    routing_prefer_cost: bool = True  # prefer cheaper model when equal capability

    # --- OpenTelemetry ---
    otel_service_name: str = "llm-gateway"
    otel_exporter_endpoint: Optional[str] = None  # e.g. http://localhost:4317


settings = Settings()
