"""
tests/unit/test_pipeline.py
────────────────────────────
Unit tests for RAG pipeline and plugin registry.
All external deps mocked — tests run with no infra.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.plugin_registry import (
    PluginRegistry,
    PluginStatus,
    RBACRule,
    RetrievalConfig,
    UseCasePlugin,
)
from app.models.domain import (
    DocumentChunk,
    QueryRequest,
    QueryStatus,
    Role,
    User,
)
from app.services.rag.pipeline import RAGPipeline

# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_plugin() -> UseCasePlugin:
    return UseCasePlugin(
        id="test_kb",
        name="Test KB",
        description="Test",
        collection_name="uc_test",
        status=PluginStatus.PRODUCTION,
        intent_patterns=[r"test|sample|example"],
        system_prompt_path="knowledge_base_system.j2",
        retrieval=RetrievalConfig(top_k=3, score_threshold=0.5),
        rbac=RBACRule(allowed_roles=["*"]),
        escalation_pattern=None,
    )


@pytest.fixture
def sample_user() -> User:
    return User(
        id=uuid4(),
        email="test@company.com",
        name="Test User",
        roles=[Role.USER],
        department="Engineering",
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            id=str(uuid4()),
            document_id="doc_001",
            content="The vacation policy allows 15 days annual leave.",
            metadata={"filename": "hr_policy.pdf"},
            score=0.92,
        ),
        DocumentChunk(
            id=str(uuid4()),
            document_id="doc_001",
            content="Employees must request leave at least 5 days in advance.",
            metadata={"filename": "hr_policy.pdf"},
            score=0.85,
        ),
    ]


# ── Plugin Registry tests ─────────────────────────────────────────

class TestPluginRegistry:
    def test_register_and_get(self, sample_plugin: UseCasePlugin) -> None:
        reg = PluginRegistry()
        reg.register(sample_plugin)
        assert reg.get("test_kb") is sample_plugin

    def test_get_unknown_returns_none(self) -> None:
        reg = PluginRegistry()
        assert reg.get("nonexistent") is None

    def test_get_active_filters_non_production(self, sample_plugin: UseCasePlugin) -> None:
        reg = PluginRegistry()
        reg.register(sample_plugin)

        dev_plugin = UseCasePlugin(
            id="dev_plugin",
            name="Dev",
            description="Dev",
            collection_name="uc_dev",
            status=PluginStatus.DEVELOPMENT,
            intent_patterns=[],
            system_prompt_path="knowledge_base_system.j2",
        )
        reg.register(dev_plugin)

        active = reg.get_active()
        assert len(active) == 1
        assert active[0].id == "test_kb"

    def test_intent_pattern_matching(self, sample_plugin: UseCasePlugin) -> None:
        assert sample_plugin.matches_intent("I need a test example") is True
        assert sample_plugin.matches_intent("unrelated query") is False

    def test_route_by_intent(self, sample_plugin: UseCasePlugin) -> None:
        reg = PluginRegistry()
        reg.register(sample_plugin)

        result = reg.route_by_intent("can you give me a sample?", ["user"])
        assert result is not None
        assert result.id == "test_kb"

    def test_route_by_intent_rbac_blocked(self) -> None:
        reg = PluginRegistry()
        restricted = UseCasePlugin(
            id="admin_only",
            name="Admin",
            description="Admin only",
            collection_name="uc_admin",
            status=PluginStatus.PRODUCTION,
            intent_patterns=[r"admin|secret"],
            system_prompt_path="knowledge_base_system.j2",
            rbac=RBACRule(allowed_roles=["admin"]),
        )
        reg.register(restricted)
        result = reg.route_by_intent("admin query", ["user"])
        assert result is None  # user role blocked

    def test_idempotent_registration(self, sample_plugin: UseCasePlugin) -> None:
        reg = PluginRegistry()
        reg.register(sample_plugin)
        reg.register(sample_plugin)  # re-register
        assert len(reg) == 1


# ── RAG Pipeline tests ────────────────────────────────────────────

class TestRAGPipeline:
    @pytest.mark.asyncio
    async def test_query_success(
        self,
        sample_plugin: UseCasePlugin,
        sample_user: User,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        """Full pipeline happy path with all deps mocked."""
        from app.core.plugin_registry import registry as global_registry
        global_registry.register(sample_plugin)

        with (
            patch("app.services.rag.pipeline.get_embedding_service") as mock_emb,
            patch("app.services.rag.pipeline.get_vector_store") as mock_vs,
            patch("app.services.rag.pipeline.get_llm_service") as mock_llm,
            patch("app.services.rag.pipeline._reranker") as mock_reranker,
        ):
            # Setup mocks
            mock_emb.return_value.embed_query = AsyncMock(return_value=[0.1] * 1024)
            mock_vs.return_value.search = AsyncMock(return_value=sample_chunks)
            mock_llm.return_value.render_prompt = MagicMock(return_value="system prompt")
            mock_llm.return_value.generate = AsyncMock(
                return_value=(
                    "You can take 15 days vacation.",
                    {"input_tokens": 100, "output_tokens": 50},
                )
            )
            mock_reranker.rerank = AsyncMock(return_value=sample_chunks)

            pipeline = RAGPipeline()
            pipeline._embedding = mock_emb.return_value
            pipeline._vector_store = mock_vs.return_value
            pipeline._llm = mock_llm.return_value

            request = QueryRequest(
                query="How many vacation days do I have?",
                use_case_id="test_kb",
            )
            response = await pipeline.query(request, sample_user)

        assert response.status == QueryStatus.COMPLETED
        assert "15 days" in response.answer
        assert response.confidence > 0
        assert len(response.citations) > 0

    @pytest.mark.asyncio
    async def test_query_no_chunks_returns_graceful_response(
        self,
        sample_plugin: UseCasePlugin,
        sample_user: User,
    ) -> None:
        """Test that when no chunks are found, system returns graceful response."""
        from app.core.plugin_registry import registry as global_registry
        global_registry.register(sample_plugin)

        with (
            patch("app.services.rag.pipeline.get_embedding_service") as mock_emb,
            patch("app.services.rag.pipeline.get_vector_store") as mock_vs,
            patch("app.services.rag.pipeline.get_llm_service"),
        ):
            mock_emb.return_value.embed_query = AsyncMock(return_value=[0.1] * 1024)
            mock_vs.return_value.search = AsyncMock(return_value=[])  # no results

            pipeline = RAGPipeline()
            pipeline._embedding = mock_emb.return_value
            pipeline._vector_store = mock_vs.return_value

            request = QueryRequest(query="xyz obscure question", use_case_id="test_kb")
            response = await pipeline.query(request, sample_user)

        # Verify graceful response
        assert response.confidence == 0.0
        assert "don't have information" in response.answer.lower()
        assert response.status == QueryStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_escalation_on_pattern_match(
        self,
        sample_user: User,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        """Test that escalation pattern triggers escalation."""
        from app.core.plugin_registry import registry as global_registry
        escalation_plugin = UseCasePlugin(
            id="escalation_test",
            name="Escalation Test",
            description="Test",
            collection_name="uc_esc_test",
            status=PluginStatus.PRODUCTION,
            intent_patterns=[r".*"],
            system_prompt_path="knowledge_base_system.j2",
            escalation_pattern=r"termination|lawsuit",
        )
        global_registry.register(escalation_plugin)

        with (
            patch("app.services.rag.pipeline.get_embedding_service") as mock_emb,
            patch("app.services.rag.pipeline.get_vector_store") as mock_vs,
            patch("app.services.rag.pipeline.get_llm_service") as mock_llm,
            patch("app.services.rag.pipeline._reranker") as mock_reranker,
        ):
            mock_emb.return_value.embed_query = AsyncMock(return_value=[0.1] * 1024)
            mock_vs.return_value.search = AsyncMock(return_value=sample_chunks)
            mock_llm.return_value.render_prompt = MagicMock(return_value="system")
            mock_llm.return_value.generate = AsyncMock(return_value=("answer", {}))
            mock_reranker.rerank = AsyncMock(return_value=sample_chunks)

            pipeline = RAGPipeline()
            pipeline._embedding = mock_emb.return_value
            pipeline._vector_store = mock_vs.return_value
            pipeline._llm = mock_llm.return_value

            request = QueryRequest(
                query="I want to file a lawsuit against the company",
                use_case_id="escalation_test",
            )
            response = await pipeline.query(request, sample_user)

        assert response.escalated is True
        assert response.status == QueryStatus.ESCALATED

    def test_confidence_computation(self) -> None:
        """Test confidence score computation."""
        pipeline = RAGPipeline()
        chunks = [
            DocumentChunk(id="1", document_id="d1", content="a", score=0.90),
            DocumentChunk(id="2", document_id="d2", content="b", score=0.80),
            DocumentChunk(id="3", document_id="d3", content="c", score=0.70),
        ]
        confidence = pipeline._compute_confidence(chunks)
        assert 0.79 < confidence < 0.81  # avg of top 3

    def test_confidence_empty_chunks(self) -> None:
        """Test confidence with empty chunks."""
        pipeline = RAGPipeline()
        assert pipeline._compute_confidence([]) == 0.0

    def test_build_citations_deduplicates_by_doc(self) -> None:
        """Test that citations are deduplicated by document ID."""
        pipeline = RAGPipeline()
        chunks = [
            DocumentChunk(id="c1", document_id="doc1", content="a",
                          metadata={"filename": "file.pdf"}, score=0.9),
            DocumentChunk(id="c2", document_id="doc1", content="b",
                          metadata={"filename": "file.pdf"}, score=0.8),  # same doc
            DocumentChunk(id="c3", document_id="doc2", content="c",
                          metadata={"filename": "other.pdf"}, score=0.7),
        ]
        citations = pipeline._build_citations(chunks)
        assert len(citations) == 2   # deduped by document_id
        assert {c.document_id for c in citations} == {"doc1", "doc2"}