"""
tests/unit/test_ingestion.py
─────────────────────────────
Unit tests for ingestion service:
- Text extraction per format
- Chunking correctness
- PII redaction
- Full ingest flow (mocked Qdrant + embedding)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.models.domain import Document, DocumentStatus
from app.services.ingestion.ingestion_service import (
    IngestionService,
    chunk_text,
    redact_pii,
)

# ── Chunking tests ────────────────────────────────────────────────


class TestChunking:
    def test_basic_chunking_produces_chunks(self) -> None:
        text = "This is a sentence. " * 100
        chunks = chunk_text(
            text,
            chunk_size=200,
            chunk_overlap=20,
            document_id="doc1",
            metadata={"filename": "test.txt"},
        )
        assert len(chunks) > 1
        assert all(c.document_id == "doc1" for c in chunks)
        assert all(len(c.content) > 0 for c in chunks)

    def test_chunk_indices_are_sequential(self) -> None:
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\n" * 20
        chunks = chunk_text(text, 200, 20, "d1", {})
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_metadata_propagated(self) -> None:
        meta = {"filename": "report.pdf", "department": "Finance"}
        chunks = chunk_text("Some text. " * 50, 200, 20, "doc2", meta)
        for c in chunks:
            assert c.metadata["filename"] == "report.pdf"
            assert c.metadata["department"] == "Finance"

    def test_empty_text_returns_no_chunks(self) -> None:
        chunks = chunk_text("   \n\n   ", 512, 64, "d1", {})
        assert len(chunks) == 0

    def test_short_text_single_chunk(self) -> None:
        chunks = chunk_text("Short text.", 512, 64, "d1", {})
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_chunk_overlap_creates_continuity(self) -> None:
        # With overlap, adjacent chunks should share some content
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=30, document_id="d1", metadata={})
        if len(chunks) >= 2:
            # The end of chunk 0 and start of chunk 1 should have overlap
            words_0 = set(chunks[0].content.split())
            words_1 = set(chunks[1].content.split())
            overlap = words_0 & words_1
            assert len(overlap) > 0, "Expected some overlap between adjacent chunks"

    def test_chunk_total_in_metadata(self) -> None:
        text = "Sentence. " * 100
        chunks = chunk_text(text, 200, 20, "d1", {})
        for c in chunks:
            assert c.metadata["chunk_total"] == len(chunks)


# ── PII Redaction tests ───────────────────────────────────────────


class TestPIIRedaction:
    def test_email_redacted(self) -> None:
        text = "Email john.doe@example.com for info"
        result = redact_pii(text, ["email"])
        assert "[EMAIL]" in result
        assert "john.doe@example.com" not in result

    def test_vietnamese_phone_redacted(self) -> None:
        text = "Call 0901234567 or +84901234567"
        result = redact_pii(text, ["phone"])
        assert "[PHONE]" in result
        assert "0901234567" not in result

    def test_id_number_redacted(self) -> None:
        text = "CCCD: 079301001234"
        result = redact_pii(text, ["id_number"])
        assert "[ID_NUMBER]" in result

    def test_empty_pii_fields_no_change(self) -> None:
        text = "Email john@example.com"
        result = redact_pii(text, [])
        assert result == text  # no redaction

    def test_multiple_pii_types(self) -> None:
        text = "Contact john@co.com or 0912345678"
        result = redact_pii(text, ["email", "phone"])
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "john@co.com" not in result

    def test_no_false_positives_short_numbers(self) -> None:
        text = "The answer is 42 and code is 123"
        result = redact_pii(text, ["id_number"])
        # Short numbers should not be redacted (min 9 digits)
        assert "42" in result
        assert "123" in result


# ── Ingestion service tests ───────────────────────────────────────


class TestIngestionService:
    @pytest.fixture
    def document(self) -> Document:
        return Document(
            id="doc_test_001",
            use_case_id="knowledge_base",
            user_id="user_123",
            filename="policy.txt",
            content_type="text/plain",
            size_bytes=1000,
        )

    @pytest.mark.asyncio
    async def test_ingest_plain_text_success(self, document: Document) -> None:
        content = b"Company policy: employees receive 15 days vacation. " * 30

        with (
            patch("app.services.ingestion.ingestion_service.get_vector_store") as mock_vs,
            patch("app.services.ingestion.ingestion_service.get_embedding_service") as mock_emb,
        ):
            # Return embeddings matching however many chunks are produced
            mock_emb.return_value.embed_texts = AsyncMock(
                side_effect=lambda texts: [[0.1] * 1024] * len(texts)
            )
            mock_vs.return_value.ensure_collection = AsyncMock()
            mock_vs.return_value.upsert_chunks = AsyncMock(
                side_effect=lambda col, chunks: len(chunks)
            )

            svc = IngestionService()
            svc._embedding = mock_emb.return_value
            svc._vector_store = mock_vs.return_value

            result = await svc.ingest_document(
                document=document,
                content=content,
                use_case_id="knowledge_base",
            )

        assert result.status == DocumentStatus.INDEXED
        assert result.chunks_created > 0
        assert result.processing_time_ms >= 0
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_ingest_unknown_use_case_returns_failed(self, document: Document) -> None:
        document.use_case_id = "nonexistent_uc"
        svc = IngestionService()
        result = await svc.ingest_document(
            document=document,
            content=b"some content",
            use_case_id="nonexistent_uc",
        )
        assert result.status == DocumentStatus.FAILED
        assert "Unknown use_case_id" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_ingest_unsupported_content_type(self, document: Document) -> None:
        document.content_type = "application/octet-stream"
        with (
            patch("app.services.ingestion.ingestion_service.get_vector_store"),
            patch("app.services.ingestion.ingestion_service.get_embedding_service"),
        ):
            svc = IngestionService()
            result = await svc.ingest_document(
                document=document,
                content=b"\x00\x01\x02\x03",
                use_case_id="knowledge_base",
            )
        # Content type validation rejects binary content regardless of which
        # code path runs: the magic-bytes check (when python-magic is installed)
        # produces "File content does not match an allowed type", while the
        # fallback produces "Unsupported content type". Both result in FAILED —
        # assert on the observable behavior, not the internal message string.
        assert result.status == DocumentStatus.FAILED
        assert result.error_message is not None
        assert result.chunks_created == 0

    @pytest.mark.asyncio
    async def test_ingest_empty_document_returns_failed(self, document: Document) -> None:
        with (
            patch("app.services.ingestion.ingestion_service.get_vector_store"),
            patch("app.services.ingestion.ingestion_service.get_embedding_service"),
        ):
            svc = IngestionService()
            result = await svc.ingest_document(
                document=document,
                content=b"   \n\n   ",  # effectively empty after strip
                use_case_id="knowledge_base",
            )
        assert result.status == DocumentStatus.FAILED

    @pytest.mark.asyncio
    async def test_delete_document_calls_vector_store(self, document: Document) -> None:
        with patch("app.services.ingestion.ingestion_service.get_vector_store") as mock_vs:
            mock_vs.return_value.delete_by_document = AsyncMock()
            svc = IngestionService()
            svc._vector_store = mock_vs.return_value

            await svc.delete_document(
                document_id=document.id,
                use_case_id="knowledge_base",
            )

            mock_vs.return_value.delete_by_document.assert_called_once_with(
                collection_name="uc_knowledge_base",
                document_id=document.id,
            )
