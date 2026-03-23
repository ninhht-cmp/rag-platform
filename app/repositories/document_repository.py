"""
app/repositories/document_repository.py
─────────────────────────────────────────
PostgreSQL repository for document metadata and query audit logs.
Uses raw asyncpg for performance — no ORM overhead on hot paths.
Pattern: Repository + Unit of Work (manual transactions).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.logging import get_logger
from app.models.domain import Document, DocumentStatus, EvalMetrics, QueryResponse

logger = get_logger(__name__)

# ── Async engine (singleton) ──────────────────────────────────────
_engine = create_async_engine(
    str(settings.DATABASE_URL),
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

AsyncSessionLocal = async_sessionmaker(
    bind=_engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session  # type: ignore[misc]


# ── Document Repository ───────────────────────────────────────────

class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def save(self, doc: Document) -> Document:
        await self._s.execute(
            text("""
                INSERT INTO documents
                    (id, use_case_id, user_id, filename, content_type,
                     size_bytes, status, chunk_count, metadata, created_at)
                VALUES
                    (:id, :use_case_id, :user_id, :filename, :content_type,
                     :size_bytes, :status, :chunk_count, :metadata::jsonb, :created_at)
                ON CONFLICT (id) DO UPDATE SET
                    status      = EXCLUDED.status,
                    chunk_count = EXCLUDED.chunk_count,
                    indexed_at  = EXCLUDED.indexed_at,
                    error_message = EXCLUDED.error_message
            """),
            {
                "id": str(doc.id),
                "use_case_id": doc.use_case_id,
                "user_id": doc.user_id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "size_bytes": doc.size_bytes,
                "status": doc.status.value,
                "chunk_count": doc.chunk_count,
                "metadata": __import__("orjson").dumps(doc.metadata).decode(),
                "created_at": doc.created_at,
            },
        )
        await self._s.commit()
        return doc

    async def update_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        chunk_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        await self._s.execute(
            text("""
                UPDATE documents
                SET status = :status,
                    chunk_count = :chunk_count,
                    indexed_at = CASE WHEN :status = 'indexed' THEN NOW() ELSE indexed_at END,
                    error_message = :error_message
                WHERE id = :id
            """),
            {
                "id": doc_id,
                "status": status.value,
                "chunk_count": chunk_count,
                "error_message": error_message,
            },
        )
        await self._s.commit()

    async def get_by_user(
        self,
        user_id: str,
        use_case_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM documents WHERE user_id = :uid"
        params: dict[str, Any] = {"uid": user_id}
        if use_case_id:
            q += " AND use_case_id = :uc"
            params["uc"] = use_case_id
        q += " ORDER BY created_at DESC LIMIT :limit"
        params["limit"] = limit

        result = await self._s.execute(text(q), params)
        return [dict(row._mapping) for row in result]

    async def delete(self, doc_id: str) -> None:
        await self._s.execute(
            text("DELETE FROM documents WHERE id = :id"),
            {"id": doc_id},
        )
        await self._s.commit()


# ── Query Log Repository ──────────────────────────────────────────

class QueryLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def log(self, response: QueryResponse, user_id: str) -> None:
        """Persist every query for audit and analytics."""
        import orjson
        await self._s.execute(
            text("""
                INSERT INTO query_logs
                    (user_id, use_case_id, query, answer, confidence,
                     escalated, latency_ms, token_usage, session_id, created_at)
                VALUES
                    (:user_id, :use_case_id, :query, :answer, :confidence,
                     :escalated, :latency_ms, :token_usage::jsonb, :session_id, NOW())
            """),
            {
                "user_id": user_id,
                "use_case_id": response.use_case_id,
                "query": response.query,
                "answer": response.answer,
                "confidence": response.confidence,
                "escalated": response.escalated,
                "latency_ms": response.latency_ms,
                "token_usage": orjson.dumps(response.token_usage).decode(),
                "session_id": response.session_id,
            },
        )
        await self._s.commit()

    async def get_stats(
        self,
        use_case_id: str | None = None,
        days: int = 7,
    ) -> dict[str, Any]:
        since = datetime.utcnow() - timedelta(days=days)
        params: dict[str, Any] = {"since": since}
        where = "WHERE created_at >= :since"
        if use_case_id:
            where += " AND use_case_id = :uc"
            params["uc"] = use_case_id

        result = await self._s.execute(
            text(f"""
                SELECT
                    use_case_id,
                    COUNT(*)                          AS total_queries,
                    AVG(confidence)                   AS avg_confidence,
                    AVG(latency_ms)                   AS avg_latency_ms,
                    SUM(CASE WHEN escalated THEN 1 ELSE 0 END) AS escalated_count,
                    SUM((token_usage->>'input_tokens')::int)   AS total_input_tokens,
                    SUM((token_usage->>'output_tokens')::int)  AS total_output_tokens
                FROM query_logs
                {where}
                GROUP BY use_case_id
                ORDER BY total_queries DESC
            """),
            params,
        )
        rows = [dict(r._mapping) for r in result]
        return {"period_days": days, "use_cases": rows}


# ── Token Usage Repository ────────────────────────────────────────

class TokenUsageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def record(
        self,
        use_case_id: str,
        user_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
    ) -> None:
        await self._s.execute(
            text("""
                INSERT INTO token_usage
                    (use_case_id, user_id, input_tokens, output_tokens, cost_usd)
                VALUES (:uc, :uid, :inp, :out, :cost)
            """),
            {
                "uc": use_case_id,
                "uid": user_id,
                "inp": input_tokens,
                "out": output_tokens,
                "cost": cost_usd,
            },
        )
        await self._s.commit()

    async def daily_cost(self, use_case_id: str) -> float:
        """Total cost today for a use case — used for budget guard."""
        result = await self._s.execute(
            text("""
                SELECT COALESCE(SUM(cost_usd), 0.0) AS total
                FROM token_usage
                WHERE use_case_id = :uc
                  AND DATE(recorded_at) = CURRENT_DATE
            """),
            {"uc": use_case_id},
        )
        row = result.fetchone()
        return float(row.total) if row else 0.0


# ── Eval Repository ───────────────────────────────────────────────

class EvalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def save_result(self, metrics: EvalMetrics) -> None:
        await self._s.execute(
            text("""
                INSERT INTO eval_results
                    (use_case_id, faithfulness, answer_relevancy, context_recall,
                     passed, sample_size, evaluated_at)
                VALUES
                    (:uc, :faith, :rel, :recall, :passed, :sample, :ts)
            """),
            {
                "uc": metrics.use_case_id,
                "faith": metrics.faithfulness,
                "rel": metrics.answer_relevancy,
                "recall": metrics.context_recall,
                "passed": metrics.passed,
                "sample": metrics.sample_size,
                "ts": metrics.evaluated_at,
            },
        )
        await self._s.commit()

    async def latest(self, use_case_id: str) -> EvalMetrics | None:
        result = await self._s.execute(
            text("""
                SELECT * FROM eval_results
                WHERE use_case_id = :uc
                ORDER BY evaluated_at DESC
                LIMIT 1
            """),
            {"uc": use_case_id},
        )
        row = result.fetchone()
        if not row:
            return None
        d = dict(row._mapping)
        return EvalMetrics(
            use_case_id=d["use_case_id"],
            faithfulness=d["faithfulness"],
            answer_relevancy=d["answer_relevancy"],
            context_recall=d["context_recall"],
            passed=d["passed"],
            sample_size=d["sample_size"],
            evaluated_at=d["evaluated_at"],
        )
