"""
Unit tests for HybridRAGRetriever.

Tests the hybrid retrieval combining vector similarity and keyword search.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.retriever.protocols import DocumentType
from src.rag.retriever.retriever import HybridRAGRetriever

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    mock_conn = MagicMock()
    mock_conn.fetch = AsyncMock(return_value=[])

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    return mock_pool, mock_conn


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 384)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    embedder.dimension = 384
    return embedder


@pytest.fixture
def mock_redis_cache():
    """Create a mock Redis cache."""
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


@pytest.fixture
def mock_reranker():
    """Create a mock reranker."""

    async def mock_rerank(query, results, top_k=10):
        # Just return top_k results, sorted by score descending
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        return sorted_results[:top_k]

    reranker = MagicMock()
    reranker.rerank = AsyncMock(side_effect=mock_rerank)
    return reranker


@pytest.fixture
def retriever(mock_pool):
    """Create a HybridRAGRetriever with mocked pool."""
    pool, _ = mock_pool
    return HybridRAGRetriever(pool=pool, embedding_dimension=384)


@pytest.fixture
def sample_db_rows():
    """Sample database rows for testing."""
    return [
        {
            "id": "doc-1",
            "content": "Price earnings ratio definition",
            "document_type": "field_code",
            "domain": "fundamentals",
            "field_code": "TR.PERatio",
            "example_query": None,
            "example_api_call": None,
            "metadata": json.dumps({"source": "test"}),
            "combined_score": 0.85,
        },
        {
            "id": "doc-2",
            "content": "EPS mean consensus estimate",
            "document_type": "field_code",
            "domain": "estimates",
            "field_code": "TR.EPSMean",
            "example_query": None,
            "example_api_call": None,
            "metadata": {"category": "earnings"},  # JSONB (already dict)
            "combined_score": 0.75,
        },
    ]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestHybridRAGRetrieverInit:
    """Tests for HybridRAGRetriever initialization."""

    def test_default_initialization(self, mock_pool):
        """Retriever initializes with default parameters."""
        pool, _ = mock_pool
        retriever = HybridRAGRetriever(pool=pool)

        assert retriever._embedding_dimension == 1536
        assert retriever._vector_weight == 0.7
        assert retriever._keyword_weight == 0.3
        assert retriever._embedder is None
        assert retriever._redis_cache is None
        assert retriever._reranker is None

    def test_custom_initialization(self, mock_pool, mock_redis_cache, mock_reranker):
        """Retriever accepts custom parameters."""
        pool, _ = mock_pool
        retriever = HybridRAGRetriever(
            pool=pool,
            embedding_dimension=384,
            vector_weight=0.6,
            keyword_weight=0.4,
            redis_cache=mock_redis_cache,
            cache_ttl_seconds=7200,
            reranker=mock_reranker,
            first_stage_limit=100,
        )

        assert retriever._embedding_dimension == 384
        assert retriever._vector_weight == 0.6
        assert retriever._keyword_weight == 0.4
        assert retriever._redis_cache == mock_redis_cache
        assert retriever._cache_ttl == 7200
        assert retriever._reranker == mock_reranker
        assert retriever._first_stage_limit == 100


# =============================================================================
# Embedder Management Tests
# =============================================================================


class TestEmbedderManagement:
    """Tests for embedder getter/setter."""

    def test_set_embedder(self, retriever, mock_embedder):
        """set_embedder sets the embedder."""
        retriever.set_embedder(mock_embedder)
        assert retriever._embedder == mock_embedder

    def test_set_reranker(self, retriever, mock_reranker):
        """set_reranker sets the reranker."""
        retriever.set_reranker(mock_reranker)
        assert retriever._reranker == mock_reranker


# =============================================================================
# Cache Key Generation Tests
# =============================================================================


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_cache_key_basic(self, retriever):
        """Cache key generated for basic query."""
        key = retriever._make_cache_key("test query", None, None, 10)

        assert key.startswith("rag:")
        assert "all" in key  # domain and types default to "all"
        assert "10" in key  # limit included

    def test_cache_key_with_domain(self, retriever):
        """Cache key includes domain."""
        key = retriever._make_cache_key("test query", "estimates", None, 10)

        assert "estimates" in key

    def test_cache_key_with_document_types(self, retriever):
        """Cache key includes document types."""
        doc_types = [DocumentType.FIELD_CODE, DocumentType.QUERY_EXAMPLE]
        key = retriever._make_cache_key("test query", None, doc_types, 10)

        assert "field_code" in key
        assert "query_example" in key

    def test_cache_key_deterministic(self, retriever):
        """Same inputs produce same cache key."""
        key1 = retriever._make_cache_key("test query", "domain", None, 10)
        key2 = retriever._make_cache_key("test query", "domain", None, 10)

        assert key1 == key2


# =============================================================================
# Retrieve Tests
# =============================================================================


class TestRetrieve:
    """Tests for the retrieve() method."""

    @pytest.mark.asyncio
    async def test_retrieve_requires_embedder(self, retriever):
        """retrieve() raises error without embedder."""
        with pytest.raises(RuntimeError) as exc_info:
            await retriever.retrieve("test query")
        assert "Embedder not set" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, mock_pool, mock_embedder, sample_db_rows):
        """retrieve() returns results from database."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=sample_db_rows)

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("price earnings ratio")

        assert len(results) == 2
        assert results[0].id == "doc-1"
        assert results[0].score == 0.85
        assert results[0].document_type == DocumentType.FIELD_CODE

    @pytest.mark.asyncio
    async def test_retrieve_parses_metadata_string(self, mock_pool, mock_embedder, sample_db_rows):
        """retrieve() parses JSON string metadata."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=sample_db_rows)

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("test")

        # First row has JSON string metadata
        assert results[0].metadata == {"source": "test"}
        # Second row has dict metadata (JSONB)
        assert results[1].metadata == {"category": "earnings"}

    @pytest.mark.asyncio
    async def test_retrieve_handles_null_metadata(self, mock_pool, mock_embedder):
        """retrieve() handles null metadata."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "doc-1",
                    "content": "test content",
                    "document_type": "field_code",
                    "domain": None,
                    "field_code": None,
                    "example_query": None,
                    "example_api_call": None,
                    "metadata": None,
                    "combined_score": 0.5,
                }
            ]
        )

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("test")

        assert results[0].metadata == {}

    @pytest.mark.asyncio
    async def test_retrieve_with_domain_filter(self, mock_pool, mock_embedder):
        """retrieve() passes domain filter to query."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        await retriever.retrieve("test", domain="estimates")

        # Verify parameterized domain filter is in the SQL
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "domain = $8" in sql
        # Verify domain is passed as parameter (second to last, before ticker)
        # Parameters: embedding, limit, query, vec_weight, kw_weight, threshold, types, domain, ticker
        assert call_args[0][8] == "estimates"  # $8 is domain (0-indexed: position 8)
        assert call_args[0][9] is None  # $9 is ticker (should be None when not specified)

    @pytest.mark.asyncio
    async def test_retrieve_with_document_types(self, mock_pool, mock_embedder):
        """retrieve() passes document type filter to query."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        await retriever.retrieve(
            "test", document_types=[DocumentType.FIELD_CODE, DocumentType.QUERY_EXAMPLE]
        )

        # Verify parameterized type filter is in the SQL
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "document_type = ANY($7)" in sql
        # Verify types are passed as list parameter
        type_values = call_args[0][7]
        assert type_values == ["field_code", "query_example"]

    @pytest.mark.asyncio
    async def test_retrieve_with_ticker_filter(self, mock_pool, mock_embedder):
        """retrieve() passes ticker filter to query for entity-specific retrieval."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        await retriever.retrieve("What is Apple revenue?", ticker="AAPL")

        # Verify parameterized ticker filter is in the SQL
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "metadata->>'ticker' = $9" in sql
        # Verify ticker is passed as the last parameter
        assert call_args[0][9] == "AAPL"

    @pytest.mark.asyncio
    async def test_retrieve_without_ticker_filter(self, mock_pool, mock_embedder):
        """retrieve() passes None for ticker when not specified."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        await retriever.retrieve("What is EBITDA?")

        # Verify ticker parameter is None (no filtering)
        call_args = conn.fetch.call_args
        assert call_args[0][9] is None


# =============================================================================
# Cache Tests
# =============================================================================


class TestCaching:
    """Tests for Redis caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_pool, mock_embedder, mock_redis_cache):
        """retrieve() returns cached results on cache hit."""
        pool, conn = mock_pool

        # Set up cache to return cached results
        cached_data = [
            {
                "id": "cached-1",
                "content": "cached content",
                "document_type": "field_code",
                "score": 0.9,
                "domain": "test",
                "field_code": "TEST",
                "example_query": None,
                "example_api_call": None,
                "metadata": {},
            }
        ]
        mock_redis_cache.get = AsyncMock(return_value=cached_data)

        retriever = HybridRAGRetriever(
            pool=pool, embedding_dimension=384, redis_cache=mock_redis_cache
        )
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("test query")

        # Should return cached result
        assert len(results) == 1
        assert results[0].id == "cached-1"
        # Database should not be queried
        conn.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_queries_db(
        self, mock_pool, mock_embedder, mock_redis_cache, sample_db_rows
    ):
        """retrieve() queries database on cache miss."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=sample_db_rows)

        retriever = HybridRAGRetriever(
            pool=pool, embedding_dimension=384, redis_cache=mock_redis_cache
        )
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("test query")

        # Should query database
        conn.fetch.assert_called_once()
        # Should have results from DB
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_cache_stores_results(
        self, mock_pool, mock_embedder, mock_redis_cache, sample_db_rows
    ):
        """retrieve() stores results in cache."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=sample_db_rows)

        retriever = HybridRAGRetriever(
            pool=pool, embedding_dimension=384, redis_cache=mock_redis_cache
        )
        retriever.set_embedder(mock_embedder)

        await retriever.retrieve("test query")

        # Should store in cache
        mock_redis_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_pool, mock_embedder, mock_redis_cache, sample_db_rows):
        """retrieve() skips cache when use_cache=False."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=sample_db_rows)

        # Cache would return results if checked
        mock_redis_cache.get = AsyncMock(return_value=[{"id": "cached", "content": "cached"}])

        retriever = HybridRAGRetriever(
            pool=pool, embedding_dimension=384, redis_cache=mock_redis_cache
        )
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("test query", use_cache=False)

        # Should query database, not cache
        conn.fetch.assert_called_once()
        assert results[0].id == "doc-1"  # From DB, not cache


# =============================================================================
# Two-Stage Retrieval Tests
# =============================================================================


class TestTwoStageRetrieval:
    """Tests for two-stage retrieval with reranking."""

    @pytest.mark.asyncio
    async def test_reranking_with_more_results(self, mock_pool, mock_embedder, mock_reranker):
        """Reranker is called when results exceed limit."""
        pool, conn = mock_pool

        # Return more results than limit
        db_rows = [
            {
                "id": f"doc-{i}",
                "content": f"content {i}",
                "document_type": "field_code",
                "domain": None,
                "field_code": None,
                "example_query": None,
                "example_api_call": None,
                "metadata": None,
                "combined_score": 0.5 + (i * 0.01),
            }
            for i in range(15)
        ]
        conn.fetch = AsyncMock(return_value=db_rows)

        retriever = HybridRAGRetriever(
            pool=pool,
            embedding_dimension=384,
            reranker=mock_reranker,
            first_stage_limit=50,
        )
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("test", limit=10)

        # Reranker should be called
        mock_reranker.rerank.assert_called_once()
        # Results should be limited
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_no_reranking_without_reranker(self, mock_pool, mock_embedder):
        """No reranking when reranker not configured."""
        pool, conn = mock_pool

        db_rows = [
            {
                "id": f"doc-{i}",
                "content": f"content {i}",
                "document_type": "field_code",
                "domain": None,
                "field_code": None,
                "example_query": None,
                "example_api_call": None,
                "metadata": None,
                "combined_score": 0.5 + (i * 0.01),
            }
            for i in range(15)
        ]
        conn.fetch = AsyncMock(return_value=db_rows)

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("test", limit=10)

        # Results should be trimmed to limit
        assert len(results) == 10


# =============================================================================
# Specialized Retrieve Methods Tests
# =============================================================================


class TestSpecializedRetrieveMethods:
    """Tests for retrieve_field_codes and retrieve_examples."""

    @pytest.mark.asyncio
    async def test_retrieve_field_codes(self, mock_pool, mock_embedder, sample_db_rows):
        """retrieve_field_codes filters by FIELD_CODE type."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=sample_db_rows[:1])

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        await retriever.retrieve_field_codes("price ratio", domain="fundamentals")

        # Should filter by field_code type via parameterized query
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "document_type = ANY($7)" in sql
        # Verify field_code is passed as parameter
        assert call_args[0][7] == ["field_code"]

    @pytest.mark.asyncio
    async def test_retrieve_field_codes_keyword_fallback(self, mock_pool):
        """retrieve_field_codes uses keyword search without embedder."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        # No embedder set

        await retriever.retrieve_field_codes("price ratio", domain="fundamentals")

        # Should use keyword search
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "plainto_tsquery" in sql

    @pytest.mark.asyncio
    async def test_retrieve_examples(self, mock_pool, mock_embedder):
        """retrieve_examples filters by QUERY_EXAMPLE type."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        await retriever.retrieve_examples("what is the price of Apple")

        # Should filter by query_example type via parameterized query
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "document_type = ANY($7)" in sql
        # Verify query_example is passed as parameter
        assert call_args[0][7] == ["query_example"]

    @pytest.mark.asyncio
    async def test_retrieve_by_keyword(self, mock_pool, sample_db_rows):
        """retrieve_by_keyword uses keyword-only search."""
        pool, conn = mock_pool

        # Keyword search returns different format
        keyword_rows = [
            {
                **row,
                "score": row["combined_score"],  # Different field name
            }
            for row in sample_db_rows
        ]
        del keyword_rows[0]["combined_score"]
        del keyword_rows[1]["combined_score"]

        conn.fetch = AsyncMock(return_value=keyword_rows)

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        # No embedder needed for keyword search

        await retriever.retrieve_by_keyword("price ratio")

        # Should not use vector search
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "embedding" not in sql
        assert "plainto_tsquery" in sql


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_pool, mock_embedder):
        """retrieve() handles empty results gracefully."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("nonexistent query")

        assert results == []

    @pytest.mark.asyncio
    async def test_invalid_json_metadata(self, mock_pool, mock_embedder):
        """retrieve() handles invalid JSON metadata."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "doc-1",
                    "content": "test",
                    "document_type": "field_code",
                    "domain": None,
                    "field_code": None,
                    "example_query": None,
                    "example_api_call": None,
                    "metadata": "not valid json {",  # Invalid JSON string
                    "combined_score": 0.5,
                }
            ]
        )

        retriever = HybridRAGRetriever(pool=pool, embedding_dimension=384)
        retriever.set_embedder(mock_embedder)

        results = await retriever.retrieve("test")

        # Should handle gracefully with empty dict
        assert results[0].metadata == {}
