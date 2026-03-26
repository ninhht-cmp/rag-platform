"""
app/models/domain.py
─────────────────────
Domain models (Pydantic v2).
These are the contracts between layers — never leak ORM models to API layer.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# ── Enums ─────────────────────────────────────────────────────────

class Role(StrEnum):
    ADMIN = "admin"
    USER = "user"
    SUPPORT_AGENT = "support_agent"
    SALES_REP = "sales_rep"
    ANALYST = "analyst"


class QueryStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ── Base ──────────────────────────────────────────────────────────

class BaseSchema(BaseModel):
    model_config = {"from_attributes": True, "populate_by_name": True}


# ── User ──────────────────────────────────────────────────────────

class User(BaseSchema):
    id: UUID = Field(default_factory=uuid4)
    email: str
    name: str
    roles: list[Role] = Field(default_factory=lambda: [Role.USER])
    department: str = ""
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TokenPayload(BaseSchema):
    sub: str   # user id
    roles: list[str]
    exp: int


# ── Document ──────────────────────────────────────────────────────

class DocumentChunk(BaseSchema):
    """A single chunk stored in Qdrant."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    chunk_index: int = 0
    score: float = 0.0          # retrieval score (populated post-search)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()


class Document(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    use_case_id: str
    user_id: str
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: datetime | None = None
    error_message: str | None = None


# ── Query / Response ──────────────────────────────────────────────

class QueryRequest(BaseSchema):
    query: str = Field(min_length=1, max_length=2000)
    use_case_id: str | None = None   # auto-routed if None
    session_id: str | None = None    # for conversation memory
    stream: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return v.strip()


class Citation(BaseSchema):
    document_id: str
    filename: str
    chunk_id: str
    content_preview: str   # first 200 chars
    score: float
    page_number: int | None = None
    url: str | None = None


class QueryResponse(BaseSchema):
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    answer: str
    use_case_id: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0         # 0-1
    status: QueryStatus = QueryStatus.COMPLETED
    escalated: bool = False
    escalation_reason: str | None = None
    session_id: str | None = None
    latency_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Ingestion ─────────────────────────────────────────────────────

class IngestionRequest(BaseSchema):
    use_case_id: str
    source_type: str              # "upload" | "confluence" | "gdrive" | "s3"
    source_config: dict[str, Any] = Field(default_factory=dict)
    chunk_size: int | None = None  # override default
    chunk_overlap: int | None = None


class IngestionResult(BaseSchema):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunks_created: int
    status: DocumentStatus
    processing_time_ms: int
    error_message: str | None = None


# ── Evaluation ────────────────────────────────────────────────────

class EvalMetrics(BaseSchema):
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    passed: bool
    use_case_id: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    sample_size: int = 0


# ── Health ────────────────────────────────────────────────────────

class HealthStatus(BaseSchema):
    status: str                          # "healthy" | "degraded" | "unhealthy"
    version: str
    environment: str
    components: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
