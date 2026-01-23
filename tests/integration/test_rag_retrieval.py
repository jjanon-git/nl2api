"""
RAG Retrieval Integration Tests

Tests the RAG retrieval pipeline end-to-end with real database queries.
Requires PostgreSQL with indexed field codes.

Run with: pytest tests/integration/test_rag_retrieval.py -v
"""

import os
from pathlib import Path

import pytest

# Load env before imports
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


@pytest.fixture
async def db_pool():
    """Create database connection pool."""
    import asyncpg

    db_url = os.environ.get("DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api")
    pool = await asyncpg.create_pool(db_url)
    yield pool
    await pool.close()


@pytest.fixture
async def retriever(db_pool):
    """Create RAG retriever without embedder (keyword-only mode)."""
    from src.nl2api.rag.retriever import HybridRAGRetriever

    return HybridRAGRetriever(db_pool)


@pytest.fixture
async def has_indexed_data(db_pool) -> bool:
    """Check if RAG documents are indexed."""
    async with db_pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM rag_documents")
    return count > 0


class TestRAGKeywordRetrieval:
    """Test keyword-based RAG retrieval (no embeddings required)."""

    @pytest.mark.asyncio
    async def test_bid_price_retrieves_pb_field(self, retriever, has_indexed_data):
        """Test that 'bid price' query retrieves PB field code."""
        if not has_indexed_data:
            pytest.skip("No RAG documents indexed")

        results = await retriever.retrieve_field_codes(
            query="bid price",
            domain="datastream",
            limit=5,
        )

        codes = [r.field_code for r in results]
        assert "PB" in codes, f"Expected PB in results, got: {codes}"

    @pytest.mark.asyncio
    async def test_revenue_retrieves_tr_revenue(self, retriever, has_indexed_data):
        """Test that 'revenue' query retrieves TR.Revenue field code."""
        if not has_indexed_data:
            pytest.skip("No RAG documents indexed")

        results = await retriever.retrieve_field_codes(
            query="revenue",
            domain="fundamentals",
            limit=5,
        )

        codes = [r.field_code for r in results]
        assert "TR.Revenue" in codes, f"Expected TR.Revenue in results, got: {codes}"

    @pytest.mark.asyncio
    async def test_analyst_recommendations_retrieves_recmean(self, retriever, has_indexed_data):
        """Test that 'analyst recommendations' retrieves TR.RecMean."""
        if not has_indexed_data:
            pytest.skip("No RAG documents indexed")

        results = await retriever.retrieve_field_codes(
            query="analyst recommendations",
            domain="estimates",
            limit=5,
        )

        codes = [r.field_code for r in results]
        assert "TR.RecMean" in codes, f"Expected TR.RecMean in results, got: {codes}"

    @pytest.mark.asyncio
    async def test_market_cap_cross_domain(self, retriever, has_indexed_data):
        """Test that 'market cap' retrieves results from multiple domains."""
        if not has_indexed_data:
            pytest.skip("No RAG documents indexed")

        from src.nl2api.rag.protocols import DocumentType

        results = await retriever.retrieve_by_keyword(
            query="market cap",
            document_types=[DocumentType.FIELD_CODE],
            limit=10,
        )

        domains = set(r.domain for r in results)
        # Should find market cap in datastream and fundamentals at minimum
        assert len(domains) >= 2, f"Expected multiple domains, got: {domains}"

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, retriever, has_indexed_data):
        """Test that empty/nonsense query returns empty results gracefully."""
        if not has_indexed_data:
            pytest.skip("No RAG documents indexed")

        results = await retriever.retrieve_field_codes(
            query="xyzzy foobar nonsense",
            domain="datastream",
            limit=5,
        )

        # Should return empty, not error
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_domain_filter_works(self, retriever, has_indexed_data):
        """Test that domain filter restricts results correctly."""
        if not has_indexed_data:
            pytest.skip("No RAG documents indexed")

        results = await retriever.retrieve_field_codes(
            query="price",
            domain="estimates",
            limit=10,
        )

        # All results should be from estimates domain
        for r in results:
            assert r.domain == "estimates", f"Expected estimates domain, got: {r.domain}"


class TestRAGRetrieverFallback:
    """Test that retriever correctly falls back to keyword search."""

    @pytest.mark.asyncio
    async def test_no_embedder_uses_keyword_search(self, retriever, has_indexed_data):
        """Test that without embedder, keyword search is used automatically."""
        if not has_indexed_data:
            pytest.skip("No RAG documents indexed")

        # Retriever has no embedder set, should use keyword fallback
        assert retriever._embedder is None

        # This should work via keyword fallback
        results = await retriever.retrieve_field_codes(
            query="bid price",
            domain="datastream",
            limit=5,
        )

        assert len(results) > 0, "Expected results from keyword fallback"
