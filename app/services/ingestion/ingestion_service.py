"""
app/services/ingestion/ingestion_service.py
────────────────────────────────────────────
Document ingestion pipeline:
PDF / DOCX / TXT / HTML → clean text → chunks → embeddings → Qdrant

Design:
- Async throughout (no blocking IO)
- Chunk strategy per document type
- PII detection placeholder (extend with presidio)
- Progress tracking via metadata
"""
from __future__ import annotations

import io
import time
from pathlib import Path
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


# ── Text extractors ───────────────────────────────────────────────

async def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF using pypdf."""
    import asyncio
    from pypdf import PdfReader

    def _extract() -> str:
        reader = PdfReader(io.BytesIO(content))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)

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
    """
    Recursive character text splitter.
    Tries to split on paragraph → sentence → word boundaries.
    """
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


# ── PII Redactor (stub — extend with presidio) ────────────────────

def redact_pii(text: str, pii_fields: list[str]) -> str:
    """
    Basic PII redaction. In production, replace with:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    """
    import re
    if not pii_fields:
        return text

    # Email
    if "email" in pii_fields:
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)
    # Phone (Vietnamese + international)
    if "phone" in pii_fields:
        text = re.sub(r"\b(?:\+84|0)[0-9]{9,10}\b", "[PHONE]", text)
    # CCCD/CMND
    if "id_number" in pii_fields:
        text = re.sub(r"\b\d{9,12}\b", "[ID_NUMBER]", text)

    return text


# ── Main Ingestion Service ────────────────────────────────────────

class IngestionService:
    """
    Ingest a document into the vector store.
    Called by API endpoint and background workers.
    """

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
        """
        Full ingestion flow:
        1. Extract text from document
        2. Redact PII
        3. Chunk
        4. Embed
        5. Upsert to Qdrant
        """
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
            # 1. Extract
            extractor = EXTRACTORS.get(document.content_type)
            if extractor is None:
                raise ValueError(f"Unsupported content type: {document.content_type}")

            raw_text = await extractor(content)
            if not raw_text.strip():
                raise ValueError("Document produced empty text after extraction")


            logger.info(
                "ingestion.extracted",
                doc_id=document.id,
                chars=len(raw_text),
                content_type=document.content_type,
            )

            # 2. PII redaction
            pii_fields = plugin.rbac.metadata_filters.get("pii_fields", "")
            pii_list = [f.strip() for f in pii_fields.split(",")] if pii_fields else []
            clean_text = redact_pii(raw_text, pii_list)

            # 3. Chunk
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

            # 4. Embed
            texts = [c.content for c in chunks]
            embeddings = await self._embedding.embed_texts(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

            # 5. Upsert (delete existing chunks first to avoid duplicates on re-ingest)
            await self._vector_store.ensure_collection(plugin.collection_name)
            try:
                await self._vector_store.delete_by_document(
                    plugin.collection_name, document.id
                )
            except Exception:
                pass  # Collection may be empty on first ingest
            count = await self._vector_store.upsert_chunks(plugin.collection_name, chunks)

            elapsed = int(time.monotonic() * 1000) - start_ms
            logger.info(
                "ingestion.complete",
                doc_id=document.id,
                chunks=count,
                elapsed_ms=elapsed,
            )

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

    async def delete_document(
        self,
        document_id: str,
        use_case_id: str,
    ) -> None:
        """Remove all chunks for a document from vector store."""
        plugin = registry.get(use_case_id)
        if plugin is None:
            raise ValueError(f"Unknown use_case_id: {use_case_id}")

        await self._vector_store.delete_by_document(
            collection_name=plugin.collection_name,
            document_id=document_id,
        )
        logger.info("ingestion.deleted", doc_id=document_id, use_case=use_case_id)


def get_ingestion_service() -> IngestionService:
    return IngestionService()
