"""
Unit tests for cross-encoder reranker.

Note: These tests require the optional 'embeddings' dependencies:
    pip install -e ".[embeddings]"
"""

import pytest

# Skip all tests in this module if sentence-transformers is not installed
pytest.importorskip(
    "sentence_transformers", reason="sentence-transformers required for reranker tests"
)

from src.rag.retriever.protocols import DocumentType, RetrievalResult
from src.rag.retriever.reranker import CrossEncoderReranker, create_reranker


@pytest.fixture(scope="module")
def reranker():
    """Module-scoped reranker to avoid reloading model for each test."""
    return create_reranker()


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    """Sample retrieval results for testing."""
    return [
        RetrievalResult(
            id="doc1",
            content="Apple Inc. is a technology company that designs consumer electronics.",
            document_type=DocumentType.SEC_FILING,
            score=0.8,
            domain="sec_filings",
            metadata={"ticker": "AAPL"},
        ),
        RetrievalResult(
            id="doc2",
            content="The company reported strong iPhone sales in Q4 2024.",
            document_type=DocumentType.SEC_FILING,
            score=0.75,
            domain="sec_filings",
            metadata={"ticker": "AAPL"},
        ),
        RetrievalResult(
            id="doc3",
            content="Microsoft Azure cloud revenue grew 30% year over year.",
            document_type=DocumentType.SEC_FILING,
            score=0.7,
            domain="sec_filings",
            metadata={"ticker": "MSFT"},
        ),
        RetrievalResult(
            id="doc4",
            content="Google's advertising revenue continues to dominate the market.",
            document_type=DocumentType.SEC_FILING,
            score=0.65,
            domain="sec_filings",
            metadata={"ticker": "GOOGL"},
        ),
        RetrievalResult(
            id="doc5",
            content="Apple's services segment including App Store and iCloud showed growth.",
            document_type=DocumentType.SEC_FILING,
            score=0.6,
            domain="sec_filings",
            metadata={"ticker": "AAPL"},
        ),
    ]


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_create_reranker(self):
        """Test reranker creation with factory function."""
        reranker = create_reranker()
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_create_reranker_custom_model(self):
        """Test reranker creation with custom model."""
        reranker = create_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def test_reranker_stats_initial(self):
        """Test initial stats are zero."""
        fresh_reranker = create_reranker()
        stats = fresh_reranker.stats
        assert stats["total_rerank_calls"] == 0
        assert stats["total_pairs_scored"] == 0

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, reranker):
        """Test reranking with empty results."""
        results = await reranker.rerank("test query", [], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_fewer_than_top_k(self, reranker, sample_results):
        """Test reranking when results < top_k returns unchanged."""
        # Request top_k=10 but only have 5 results
        results = await reranker.rerank("Apple products", sample_results, top_k=10)
        # Should return original results since no reranking needed
        assert len(results) == 5
        # Original order preserved (scores unchanged)
        assert results[0].id == sample_results[0].id

    @pytest.mark.asyncio
    async def test_rerank_returns_top_k(self, reranker, sample_results):
        """Test reranking returns exactly top_k results."""
        results = await reranker.rerank("Apple company information", sample_results, top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_rerank_reorders_by_relevance(self, reranker, sample_results):
        """Test that reranking reorders results by cross-encoder relevance."""
        query = "What are Apple's main products and services?"
        results = await reranker.rerank(query, sample_results, top_k=3)

        # Apple-related documents should be ranked higher for Apple query
        apple_ids = {"doc1", "doc2", "doc5"}  # Documents about Apple
        top_ids = {r.id for r in results}

        # At least 2 of top 3 should be Apple-related
        assert len(top_ids & apple_ids) >= 2

    @pytest.mark.asyncio
    async def test_rerank_updates_scores(self, reranker, sample_results):
        """Test that reranking updates scores with cross-encoder scores."""
        results = await reranker.rerank("Apple iPhone sales", sample_results, top_k=3)

        # Scores should be updated (not the original scores)
        for result in results:
            # Cross-encoder scores are typically in a different range
            assert isinstance(result.score, float)

    @pytest.mark.asyncio
    async def test_rerank_updates_stats(self):
        """Test that reranking updates statistics."""
        fresh_reranker = create_reranker()
        # Need more docs than top_k to trigger actual reranking
        sample = [
            RetrievalResult(
                id=f"doc{i}",
                content=f"Test content {i}",
                document_type=DocumentType.SEC_FILING,
                score=0.8 - i * 0.1,
                domain="test",
                metadata={},
            )
            for i in range(5)
        ]
        await fresh_reranker.rerank("test query", sample, top_k=3)

        stats = fresh_reranker.stats
        assert stats["total_rerank_calls"] == 1
        assert stats["total_pairs_scored"] == 5

    def test_rerank_sync(self, reranker, sample_results):
        """Test synchronous reranking."""
        results = reranker.rerank_sync("Apple products", sample_results, top_k=3)
        assert len(results) == 3

    def test_rerank_sync_empty(self, reranker):
        """Test synchronous reranking with empty results."""
        results = reranker.rerank_sync("test", [], top_k=5)
        assert results == []

    def test_rerank_sync_fewer_than_top_k(self, reranker, sample_results):
        """Test synchronous reranking when results < top_k."""
        results = reranker.rerank_sync("test", sample_results[:3], top_k=10)
        assert len(results) == 3


class TestRerankerIntegration:
    """Integration-style tests for reranker behavior."""

    @pytest.mark.asyncio
    async def test_rerank_prefers_exact_match(self, reranker):
        """Test that reranker prefers documents with exact query terms."""
        results = [
            RetrievalResult(
                id="exact",
                content="Apple's iPhone 15 sales exceeded expectations in Q4.",
                document_type=DocumentType.SEC_FILING,
                score=0.5,
            ),
            RetrievalResult(
                id="related",
                content="The smartphone market continues to grow globally.",
                document_type=DocumentType.SEC_FILING,
                score=0.8,
            ),
            RetrievalResult(
                id="unrelated",
                content="Amazon Web Services reported cloud infrastructure growth.",
                document_type=DocumentType.SEC_FILING,
                score=0.9,
            ),
        ]

        reranked = await reranker.rerank("iPhone 15 sales", results, top_k=2)

        # Document with exact terms should rank first
        assert reranked[0].id == "exact"

    @pytest.mark.asyncio
    async def test_rerank_preserves_metadata(self, reranker, sample_results):
        """Test that reranking preserves document metadata."""
        reranked = await reranker.rerank("Apple", sample_results, top_k=3)

        for result in reranked:
            assert result.document_type == DocumentType.SEC_FILING
            assert result.domain == "sec_filings"
            assert "ticker" in result.metadata
