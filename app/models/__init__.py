"""Pydantic request / response models for the gateway API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str


# ---------------------------------------------------------------------------
# /v1/chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: Optional[str] = Field(
        None,
        description="Requested model. If omitted the router selects automatically.",
    )
    messages: List[Message]
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    stream: bool = False
    # Gateway meta-fields
    user_id: str = Field("anonymous", description="Caller user identifier")
    tenant_id: str = Field("default", description="Caller tenant identifier")


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    provider: str
    choices: List[ChatChoice]
    usage: Usage
    cached: bool = False
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: List[str]
    user_id: str = "anonymous"
    tenant_id: str = "default"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    provider: str
    data: List[EmbeddingData]
    usage: Usage
    cached: bool = False
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Analytics / cost
# ---------------------------------------------------------------------------

class CostSummary(BaseModel):
    tenant_id: str
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    period: str


class HealthResponse(BaseModel):
    status: str
    version: str
    providers: Dict[str, Any]
