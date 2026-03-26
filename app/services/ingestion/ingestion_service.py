"""
app/services/ingestion/ingestion_service.py
────────────────────────────────────────────
Document ingestion pipeline:
PDF / DOCX / TXT / HTML → clean text → chunks → embeddings → Qdrant
"""
from __future__ import annotations

import contextlib
import io
import time
from typing import Any

from app.core.logging import get_logger
from app.core.plugin_registry import registry
from app.models.domain import (
    Document,
    DocumentChunk,
    DocumentStatus,
    IngestionResult,
)
from app.services.rag.embedding import get_embedding_service
from app.services.rag.vector_store import get_vector_store

logger = get_logger(__name__)


# ── Content-type validation with magic bytes ──────────────────────

# Mapping: detected MIME → canonical MIME we store
_ALLOWED_MIME_TYPES: dict[str, str] = {
    "application/pdf": "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain": "text/plain",
    "text/html": "text/html",
    "text/markdown": "text/markdown",
    "text/x-markdown": "text/markdown",   # libmagic sometimes returns this
}


def _validate_content_type(content: bytes, claimed_type: str) -> str:
    """
    FIX: Validate actual file content against magic bytes.
    Returns the canonical MIME type, raises ValueError if not allowed.
    Falls back to claimed type if python-magic is not installed (degraded mode).
    """
    try:
        import magic  # type: ignore[import]
        detected = magic.from_buffer(content[:4096], mime=True)
        canonical = _ALLOWED_MIME_TYPES.get(detected)
        if canonical is None:
            raise ValueError(
                f"File content does not match an allowed type. "
                f"Detected: {detected}, claimed: {claimed_type}"
            )
        return canonical
    except ImportError:
        # python-magic not installed — fall back to HTTP header (with warning)
        logger.warning(
            "content_type_validation.magic_unavailable",
            note="Install python-magic for proper file validation",
        )
        if claimed_type not in _ALLOWED_MIME_TYPES:
            raise ValueError(
                f"Unsupported content type: {claimed_type}"
            ) from None
        return _ALLOWED_MIME_TYPES[claimed_type]


# ── Text extractors ───────────────────────────────────────────────

async def _extract_pdf(content: bytes) -> str:
    import asyncio

    from pypdf import PdfReader

    def _extract() -> str:
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _extract)


async def _extract_docx(content: bytes) -> str:
    import asyncio

    from docx import Document as DocxDocument

    def _extract() -> str:
        doc = DocxDocument(io.BytesIO(content))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _extract)


async def _extract_text(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


EXTRACTORS: dict[str, Any] = {
    "application/pdf": _extract_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": _extract_docx,
    "text/plain": _extract_text,
    "text/html": _extract_text,
    "text/markdown": _extract_text,
}


# ── Chunker ───────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    document_id: str,
    metadata: dict[str, Any],
) -> list[DocumentChunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    raw_chunks = splitter.split_text(text)

    chunks: list[DocumentChunk] = []
    for i, raw in enumerate(raw_chunks):
        if not raw.strip():
            continue
        chunks.append(
            DocumentChunk(
                document_id=document_id,
                content=raw.strip(),
                chunk_index=i,
                metadata={**metadata, "chunk_total": len(raw_chunks)},
            )
        )
    return chunks


# ── PII Redactor ──────────────────────────────────────────────────

def redact_pii(text: str, pii_fields: list[str]) -> str:
    import re
    if not pii_fields:
        return text
    if "email" in pii_fields:
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)
    if "phone" in pii_fields:
        text = re.sub(r"\b(?:\+84|0)[0-9]{9,10}\b", "[PHONE]", text)
    if "id_number" in pii_fields:
        text = re.sub(r"\b\d{9,12}\b", "[ID_NUMBER]", text)
    return text


# ── Main Ingestion Service ────────────────────────────────────────

class IngestionService:
    def __init__(self) -> None:
        self._embedding = get_embedding_service()
        self._vector_store = get_vector_store()

    async def ingest_document(
        self,
        document: Document,
        content: bytes,
        use_case_id: str,
        chunk_size_override: int | None = None,
        chunk_overlap_override: int | None = None,
    ) -> IngestionResult:
        start_ms = int(time.monotonic() * 1000)

        plugin = registry.get(use_case_id)
        if plugin is None:
            return IngestionResult(
                document_id=document.id,
                chunks_created=0,
                status=DocumentStatus.FAILED,
                processing_time_ms=0,
                error_message=f"Unknown use_case_id: {use_case_id}",
            )

        try:
            canonical_type = _validate_content_type(content, document.content_type)
            document.content_type = canonical_type

            extractor = EXTRACTORS.get(canonical_type)
            if extractor is None:
                raise ValueError(f"Unsupported content type: {canonical_type}")

            raw_text = await extractor(content)
            if not raw_text.strip():
                raise ValueError("Document produced empty text after extraction")

            logger.info(
                "ingestion.extracted",
                doc_id=document.id,
                chars=len(raw_text),
                content_type=canonical_type,
            )

            pii_fields_raw = plugin.rbac.metadata_filters.get("pii_fields", "")
            pii_list = [f.strip() for f in pii_fields_raw.split(",")] if pii_fields_raw else []
            clean_text = redact_pii(raw_text, pii_list)

            cfg = plugin.retrieval
            chunk_size = chunk_size_override or cfg.chunk_size
            chunk_overlap = chunk_overlap_override or cfg.chunk_overlap

            import hashlib as _hashlib
            metadata = {
                "filename": document.filename,
                "document_id": document.id,
                "use_case_id": use_case_id,
                "user_id": document.user_id,
                "content_hash": _hashlib.sha256(content).hexdigest()[:16],
                **document.metadata,
            }
            chunks = chunk_text(clean_text, chunk_size, chunk_overlap, document.id, metadata)

            if not chunks:
                raise ValueError("Chunking produced no chunks")

            logger.info(
                "ingestion.chunked",
                doc_id=document.id,
                chunk_count=len(chunks),
                chunk_size=chunk_size,
            )

            texts = [c.content for c in chunks]
            embeddings = await self._embedding.embed_texts(texts)
            # strict=True ensures mismatch between chunks and embeddings raises immediately
            # rather than silently dropping un-embedded chunks
            for chunk, emb in zip(chunks, embeddings, strict=True):
                chunk.embedding = emb

            await self._vector_store.ensure_collection(plugin.collection_name)
            # Suppress error if previous version doesn't exist yet — not fatal
            with contextlib.suppress(Exception):
                await self._vector_store.delete_by_document(plugin.collection_name, document.id)

            count = await self._vector_store.upsert_chunks(plugin.collection_name, chunks)
            elapsed = int(time.monotonic() * 1000) - start_ms

            logger.info("ingestion.complete", doc_id=document.id, chunks=count, elapsed_ms=elapsed)
            return IngestionResult(
                document_id=document.id,
                chunks_created=count,
                status=DocumentStatus.INDEXED,
                processing_time_ms=elapsed,
            )

        except Exception as exc:
            elapsed = int(time.monotonic() * 1000) - start_ms
            logger.error("ingestion.failed", doc_id=document.id, error=str(exc))
            return IngestionResult(
                document_id=document.id,
                chunks_created=0,
                status=DocumentStatus.FAILED,
                processing_time_ms=elapsed,
                error_message=str(exc),
            )

    async def delete_document(self, document_id: str, use_case_id: str) -> None:
        plugin = registry.get(use_case_id)
        if plugin is None:
            raise ValueError(f"Unknown use_case_id: {use_case_id}")
        await self._vector_store.delete_by_document(
            collection_name=plugin.collection_name,
            document_id=document_id,
        )
        logger.info("ingestion.deleted", doc_id=document_id, use_case=use_case_id)


_ingestion_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service
