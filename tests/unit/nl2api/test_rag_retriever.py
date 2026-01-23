"""Tests for RAG retriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.nl2api.rag.protocols import DocumentType, RetrievalResult


class TestDocumentType:
    """Test suite for DocumentType enum."""

    def test_all_types_defined(self) -> None:
        """Test all expected document types are defined."""
        assert DocumentType.FIELD_CODE.value == "field_code"
        assert DocumentType.QUERY_EXAMPLE.value == "query_example"
        assert DocumentType.ECONOMIC_INDICATOR.value == "economic_indicator"


class TestRetrievalResult:
    """Test suite for RetrievalResult dataclass."""

    def test_field_code_result(self) -> None:
        """Test creating a field code retrieval result."""
        result = RetrievalResult(
            id="fc-001",
            content="Mean EPS estimate",
            document_type=DocumentType.FIELD_CODE,
            score=0.95,
            domain="estimates",
            field_code="TR.EPSMean",
        )
        assert result.id == "fc-001"
        assert result.document_type == DocumentType.FIELD_CODE
        assert result.field_code == "TR.EPSMean"
        assert result.score == 0.95

    def test_query_example_result(self) -> None:
        """Test creating a query example retrieval result."""
        result = RetrievalResult(
            id="qe-001",
            content="Q: What is Apple's EPS?",
            document_type=DocumentType.QUERY_EXAMPLE,
            score=0.88,
            domain="estimates",
            example_query="What is Apple's EPS?",
            example_api_call="get_data(RICs=['AAPL.O'])",
        )
        assert result.id == "qe-001"
        assert result.document_type == DocumentType.QUERY_EXAMPLE
        assert result.example_query == "What is Apple's EPS?"
        assert result.example_api_call == "get_data(RICs=['AAPL.O'])"

    def test_optional_fields_default_to_none(self) -> None:
        """Test that optional fields default to None."""
        result = RetrievalResult(
            id="test-001",
            content="Test",
            document_type=DocumentType.FIELD_CODE,
            score=0.5,
        )
        assert result.domain is None
        assert result.field_code is None
        assert result.example_query is None
        # metadata defaults to empty dict in the actual implementation
        assert result.metadata == {}

    def test_with_metadata(self) -> None:
        """Test result with metadata."""
        result = RetrievalResult(
            id="test-002",
            content="Test",
            document_type=DocumentType.FIELD_CODE,
            score=0.5,
            metadata={"source": "manual", "priority": 1},
        )
        assert result.metadata == {"source": "manual", "priority": 1}


class TestHybridRAGRetriever:
    """Test suite for HybridRAGRetriever."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        return mock_pool, mock_conn

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        embedder.dimension = 1536
        return embedder

    @pytest.mark.asyncio
    async def test_retrieve_field_codes(self, mock_pool, mock_embedder) -> None:
        """Test retrieving field codes."""
        from src.nl2api.rag.retriever import HybridRAGRetriever

        pool, conn = mock_pool
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "1",
                    "content": "Mean EPS estimate",
                    "document_type": "field_code",
                    "domain": "estimates",
                    "field_code": "TR.EPSMean",
                    "example_query": None,
                    "example_api_call": None,
                    "metadata": {},
                    "combined_score": 0.95,
                }
            ]
        )

        retriever = HybridRAGRetriever(pool=pool)
        retriever.set_embedder(mock_embedder)
        results = await retriever.retrieve_field_codes(
            query="EPS estimate",
            domain="estimates",
            limit=5,
        )

        assert len(results) == 1
        assert results[0].field_code == "TR.EPSMean"
        assert results[0].document_type == DocumentType.FIELD_CODE

    @pytest.mark.asyncio
    async def test_retrieve_examples(self, mock_pool, mock_embedder) -> None:
        """Test retrieving query examples."""
        from src.nl2api.rag.retriever import HybridRAGRetriever

        pool, conn = mock_pool
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "2",
                    "content": "Q: Get EPS\nA: get_data(...)",
                    "document_type": "query_example",
                    "domain": "estimates",
                    "field_code": None,
                    "example_query": "Get EPS",
                    "example_api_call": "get_data(RICs=['AAPL.O'])",
                    "metadata": {},
                    "combined_score": 0.88,
                }
            ]
        )

        retriever = HybridRAGRetriever(pool=pool)
        retriever.set_embedder(mock_embedder)
        results = await retriever.retrieve_examples(
            query="Get EPS",
            domain="estimates",
            limit=3,
        )

        assert len(results) == 1
        assert results[0].example_query == "Get EPS"
        assert results[0].document_type == DocumentType.QUERY_EXAMPLE

    @pytest.mark.asyncio
    async def test_retrieve_with_hybrid_scoring(self, mock_pool, mock_embedder) -> None:
        """Test hybrid vector + keyword scoring."""
        from src.nl2api.rag.retriever import HybridRAGRetriever

        pool, conn = mock_pool
        # Return results with both vector and text scores
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "1",
                    "content": "Mean EPS estimate",
                    "document_type": "field_code",
                    "field_code": "TR.EPSMean",
                    "domain": "estimates",
                    "example_query": None,
                    "example_api_call": None,
                    "metadata": {},
                    "combined_score": 0.90,
                },
                {
                    "id": "2",
                    "content": "Revenue mean",
                    "document_type": "field_code",
                    "field_code": "TR.RevenueMean",
                    "domain": "estimates",
                    "example_query": None,
                    "example_api_call": None,
                    "metadata": {},
                    "combined_score": 0.75,
                },
            ]
        )

        retriever = HybridRAGRetriever(pool=pool)
        retriever.set_embedder(mock_embedder)
        results = await retriever.retrieve_field_codes(query="EPS", domain="estimates", limit=10)

        assert len(results) == 2
        # Results should be sorted by score descending
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, mock_pool, mock_embedder) -> None:
        """Test handling of empty results."""
        from src.nl2api.rag.retriever import HybridRAGRetriever

        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        retriever = HybridRAGRetriever(pool=pool)
        retriever.set_embedder(mock_embedder)
        results = await retriever.retrieve_field_codes(
            query="nonexistent query",
            domain="estimates",
            limit=5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_without_embedder_raises(self, mock_pool) -> None:
        """Test that hybrid retrieve() without embedder raises RuntimeError."""
        from src.nl2api.rag.retriever import HybridRAGRetriever

        pool, conn = mock_pool

        # Create retriever without setting embedder
        retriever = HybridRAGRetriever(pool=pool)

        # Main retrieve() should raise RuntimeError (requires embedder for vector search)
        with pytest.raises(RuntimeError, match="Embedder not set"):
            await retriever.retrieve(query="EPS", domain="estimates", limit=5)

    @pytest.mark.asyncio
    async def test_retrieve_field_codes_falls_back_to_keyword(self, mock_pool) -> None:
        """Test that retrieve_field_codes falls back to keyword search without embedder."""
        from src.nl2api.rag.retriever import HybridRAGRetriever

        pool, conn = mock_pool

        # Create retriever without setting embedder
        retriever = HybridRAGRetriever(pool=pool)

        # Set up mock to return empty results for keyword search
        conn.fetch.return_value = []

        # Should not raise - falls back to keyword search
        results = await retriever.retrieve_field_codes(query="EPS", domain="estimates", limit=5)
        assert results == []

        # Verify fetch was called (keyword search)
        conn.fetch.assert_called_once()


class TestOpenAIEmbedder:
    """Test suite for OpenAIEmbedder."""

    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        return mock_client

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_openai) -> None:
        """Test embedding a single text."""
        import asyncio

        from src.nl2api.rag.retriever import OpenAIEmbedder

        # Add usage tracking to mock response
        mock_openai.embeddings.create.return_value.usage = MagicMock()
        mock_openai.embeddings.create.return_value.usage.total_tokens = 10

        embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
        embedder._client = mock_openai
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = {"text-embedding-3-small": 1536}
        embedder._semaphore = asyncio.Semaphore(5)
        embedder._total_requests = 0
        embedder._total_tokens = 0
        embedder._rate_limit_hits = 0

        embedding = await embedder.embed("What is Apple's EPS?")

        assert len(embedding) == 1536
        mock_openai.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_openai) -> None:
        """Test embedding a batch of texts."""
        import asyncio

        from src.nl2api.rag.retriever import OpenAIEmbedder

        mock_embedding1 = MagicMock()
        mock_embedding1.embedding = [0.1] * 1536
        mock_embedding2 = MagicMock()
        mock_embedding2.embedding = [0.2] * 1536

        mock_openai.embeddings.create.return_value.data = [mock_embedding1, mock_embedding2]
        mock_openai.embeddings.create.return_value.usage = MagicMock()
        mock_openai.embeddings.create.return_value.usage.total_tokens = 20

        embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
        embedder._client = mock_openai
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = {"text-embedding-3-small": 1536}
        embedder._semaphore = asyncio.Semaphore(5)
        embedder._total_requests = 0
        embedder._total_tokens = 0
        embedder._rate_limit_hits = 0

        embeddings = await embedder.embed_batch(["Text 1", "Text 2"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536

    def test_dimension_property(self) -> None:
        """Test dimension property returns correct value for model."""
        from src.nl2api.rag.retriever import OpenAIEmbedder

        embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
        embedder._model = "text-embedding-3-small"
        embedder._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        assert embedder.dimension == 1536

        embedder._model = "text-embedding-3-large"
        assert embedder.dimension == 3072


class TestRAGRetrieverProtocol:
    """Test suite for RAGRetriever protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that RAGRetriever is runtime checkable."""
        from src.nl2api.rag.protocols import RAGRetriever

        # Create a minimal implementation with correct signatures
        class MinimalRetriever:
            async def retrieve(
                self, query, domain=None, document_types=None, limit=10, threshold=0.5
            ):
                return []

            async def retrieve_field_codes(self, query, domain, limit=5):
                return []

            async def retrieve_examples(self, query, domain=None, limit=3):
                return []

        retriever = MinimalRetriever()
        assert isinstance(retriever, RAGRetriever)


class TestLocalEmbedder:
    """Test suite for LocalEmbedder."""

    def test_local_embedder_imports(self) -> None:
        """Test that LocalEmbedder can be imported."""
        from src.nl2api.rag.embedders import Embedder, LocalEmbedder

        # LocalEmbedder should be a subclass of Embedder
        assert issubclass(LocalEmbedder, Embedder)

    def test_create_embedder_local(self) -> None:
        """Test create_embedder factory with local provider."""
        from src.nl2api.rag.embedders import LocalEmbedder, create_embedder

        embedder = create_embedder("local")
        assert isinstance(embedder, LocalEmbedder)
        assert embedder.dimension == 384  # all-MiniLM-L6-v2 default

    def test_create_embedder_invalid_provider(self) -> None:
        """Test create_embedder raises for unknown provider."""
        from src.nl2api.rag.embedders import create_embedder

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedder("invalid_provider")

    def test_create_embedder_openai_requires_api_key(self) -> None:
        """Test create_embedder raises when OpenAI API key missing."""
        from src.nl2api.rag.embedders import create_embedder

        with pytest.raises(ValueError, match="api_key"):
            create_embedder("openai")

    @pytest.mark.asyncio
    async def test_local_embedder_embed(self) -> None:
        """Test LocalEmbedder can generate embeddings."""
        from src.nl2api.rag.embedders import LocalEmbedder

        embedder = LocalEmbedder()
        embedding = await embedder.embed("Apple stock price")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_local_embedder_embed_batch(self) -> None:
        """Test LocalEmbedder can generate batch embeddings."""
        from src.nl2api.rag.embedders import LocalEmbedder

        embedder = LocalEmbedder()
        embeddings = await embedder.embed_batch(["Apple", "Microsoft", "Google"])

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)
