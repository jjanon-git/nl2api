"""
Unit tests for AzureAISearchRetriever.

Tests the Azure AI Search retriever with mocked Azure SDK.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.retriever.protocols import DocumentType, RetrievalResult  # noqa: I001

# =============================================================================
# Mock Azure SDK
# =============================================================================


class MockSearchResult:
    """Mock Azure search result."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data.get(key)

    def get(self, key, default=None):
        return self._data.get(key, default)


def create_mock_azure_modules():
    """Create mock Azure modules for testing."""
    # Mock VectorizedQuery class
    mock_vectorized_query = MagicMock()

    # Mock SearchClient class
    mock_search_client_class = MagicMock()

    # Mock AzureKeyCredential class
    mock_azure_key_credential_class = MagicMock()

    # Create mock modules
    mock_azure_core = MagicMock()
    mock_azure_core.credentials.AzureKeyCredential = mock_azure_key_credential_class

    mock_azure_search = MagicMock()
    mock_azure_search.documents.SearchClient = mock_search_client_class
    mock_azure_search.documents.models.VectorizedQuery = mock_vectorized_query

    return {
        "azure": MagicMock(),
        "azure.core": mock_azure_core,
        "azure.core.credentials": mock_azure_core.credentials,
        "azure.search": mock_azure_search,
        "azure.search.documents": mock_azure_search.documents,
        "azure.search.documents.models": mock_azure_search.documents.models,
    }


# =============================================================================
# AzureAISearchRetriever Tests
# =============================================================================


class TestAzureAISearchRetriever:
    """Tests for AzureAISearchRetriever."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.dimension = 1536
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        return embedder

    @pytest.fixture
    def mock_search_client(self):
        """Create a mock SearchClient."""
        client = MagicMock()
        client.search = MagicMock(return_value=[])
        client.close = MagicMock()
        return client

    @pytest.fixture
    def mock_vectorized_query(self):
        """Patch the VectorizedQuery getter for async tests."""
        with patch("src.rag.retriever.azure_search._get_vectorized_query_class") as mock_get_vq:
            mock_get_vq.return_value = MagicMock()
            yield mock_get_vq

    @pytest.fixture
    def retriever(self, mock_embedder, mock_search_client):
        """Create AzureAISearchRetriever with mocked dependencies."""
        mock_modules = create_mock_azure_modules()

        # Configure SearchClient mock to return our mock_search_client
        mock_modules["azure.search.documents"].SearchClient.return_value = mock_search_client

        with patch.dict(sys.modules, mock_modules):
            # Import after mocking
            from src.rag.retriever.azure_search import AzureAISearchRetriever

            retriever = AzureAISearchRetriever(
                endpoint="https://test-search.search.windows.net",
                api_key="test-api-key",
                index_name="test-index",
                embedder=mock_embedder,
                vector_field="embedding",
                content_field="content",
            )
            # Replace the client with our mock
            retriever._client = mock_search_client
            return retriever

    def test_initialization(self, retriever):
        """AzureAISearchRetriever initializes with correct configuration."""
        assert retriever._endpoint == "https://test-search.search.windows.net"
        assert retriever._index_name == "test-index"
        assert retriever._vector_field == "embedding"
        assert retriever._content_field == "content"
        assert retriever._hybrid_search is True

    def test_build_filter_no_filters(self, retriever):
        """_build_filter returns None when no filters provided."""
        result = retriever._build_filter()
        assert result is None

    def test_build_filter_domain_only(self, retriever):
        """_build_filter handles domain filter."""
        result = retriever._build_filter(domain="estimates")
        assert result == "domain eq 'estimates'"

    def test_build_filter_single_document_type(self, retriever):
        """_build_filter handles single document type."""
        result = retriever._build_filter(document_types=[DocumentType.SEC_FILING])
        assert result == "document_type eq 'sec_filing'"

    def test_build_filter_multiple_document_types(self, retriever):
        """_build_filter handles multiple document types."""
        result = retriever._build_filter(
            document_types=[DocumentType.SEC_FILING, DocumentType.FIELD_CODE]
        )
        assert "search.in(document_type" in result

    def test_build_filter_combined(self, retriever):
        """_build_filter combines domain and document type filters."""
        result = retriever._build_filter(
            domain="estimates", document_types=[DocumentType.FIELD_CODE]
        )
        assert "domain eq 'estimates'" in result
        assert "document_type eq 'field_code'" in result
        assert " and " in result

    def test_to_retrieval_result_basic(self, retriever):
        """_to_retrieval_result converts Azure result correctly."""
        azure_result = MockSearchResult(
            {
                "@search.score": 0.85,
                "id": "doc-123",
                "content": "Test content",
                "document_type": "sec_filing",
                "domain": "fundamentals",
                "metadata": {"ticker": "AAPL", "fiscal_year": 2024},
            }
        )

        result = retriever._to_retrieval_result(azure_result)

        assert isinstance(result, RetrievalResult)
        assert result.id == "doc-123"
        assert result.content == "Test content"
        assert result.document_type == DocumentType.SEC_FILING
        assert result.score == 0.85
        assert result.domain == "fundamentals"
        assert result.metadata["ticker"] == "AAPL"

    def test_to_retrieval_result_high_score_normalization(self, retriever):
        """_to_retrieval_result normalizes high scores."""
        azure_result = MockSearchResult(
            {
                "@search.score": 15.5,  # Azure can return scores > 1
                "id": "doc-123",
                "content": "Test content",
                "document_type": "sec_filing",
            }
        )

        result = retriever._to_retrieval_result(azure_result)

        # Score should be normalized to <= 1.0
        assert result.score <= 1.0

    def test_to_retrieval_result_string_metadata(self, retriever):
        """_to_retrieval_result handles JSON string metadata."""
        azure_result = MockSearchResult(
            {
                "@search.score": 0.5,
                "id": "doc-123",
                "content": "Test content",
                "document_type": "sec_filing",
                "metadata": '{"ticker": "MSFT"}',  # JSON string
            }
        )

        result = retriever._to_retrieval_result(azure_result)

        assert result.metadata["ticker"] == "MSFT"

    def test_to_retrieval_result_invalid_document_type(self, retriever):
        """_to_retrieval_result defaults to SEC_FILING for invalid types."""
        azure_result = MockSearchResult(
            {
                "@search.score": 0.5,
                "id": "doc-123",
                "content": "Test content",
                "document_type": "unknown_type",
            }
        )

        result = retriever._to_retrieval_result(azure_result)

        assert result.document_type == DocumentType.SEC_FILING

    @pytest.mark.asyncio
    async def test_retrieve_calls_embedder(
        self, retriever, mock_embedder, mock_search_client, mock_vectorized_query
    ):
        """retrieve() generates query embedding."""
        mock_search_client.search.return_value = []

        await retriever.retrieve("What is Apple's revenue?")

        mock_embedder.embed.assert_called_once_with("What is Apple's revenue?")

    @pytest.mark.asyncio
    async def test_retrieve_calls_search_client(
        self, retriever, mock_embedder, mock_search_client, mock_vectorized_query
    ):
        """retrieve() calls Azure search client."""
        mock_search_client.search.return_value = []

        await retriever.retrieve("test query", limit=5)

        mock_search_client.search.assert_called_once()
        call_kwargs = mock_search_client.search.call_args.kwargs

        # Check vector queries were included
        assert "vector_queries" in call_kwargs
        assert len(call_kwargs["vector_queries"]) == 1

        # Check search text for hybrid search
        assert call_kwargs.get("search_text") == "test query"

    @pytest.mark.asyncio
    async def test_retrieve_applies_threshold_filter(
        self, retriever, mock_embedder, mock_search_client, mock_vectorized_query
    ):
        """retrieve() filters results below threshold."""
        mock_search_client.search.return_value = [
            MockSearchResult(
                {
                    "@search.score": 0.9,
                    "id": "doc-1",
                    "content": "High score",
                    "document_type": "sec_filing",
                }
            ),
            MockSearchResult(
                {
                    "@search.score": 0.3,
                    "id": "doc-2",
                    "content": "Low score",
                    "document_type": "sec_filing",
                }
            ),
        ]

        results = await retriever.retrieve("test", threshold=0.5)

        # Only high score result should be included
        assert len(results) == 1
        assert results[0].id == "doc-1"

    @pytest.mark.asyncio
    async def test_retrieve_limits_results(
        self, retriever, mock_embedder, mock_search_client, mock_vectorized_query
    ):
        """retrieve() respects limit parameter."""
        mock_search_client.search.return_value = [
            MockSearchResult(
                {
                    "@search.score": 0.9 - i * 0.1,
                    "id": f"doc-{i}",
                    "content": f"Content {i}",
                    "document_type": "sec_filing",
                }
            )
            for i in range(10)
        ]

        results = await retriever.retrieve("test", limit=3, threshold=0.0)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_sorts_by_score(
        self, retriever, mock_embedder, mock_search_client, mock_vectorized_query
    ):
        """retrieve() returns results sorted by score descending."""
        mock_search_client.search.return_value = [
            MockSearchResult(
                {
                    "@search.score": 0.5,
                    "id": "doc-low",
                    "content": "Low",
                    "document_type": "sec_filing",
                }
            ),
            MockSearchResult(
                {
                    "@search.score": 0.9,
                    "id": "doc-high",
                    "content": "High",
                    "document_type": "sec_filing",
                }
            ),
            MockSearchResult(
                {
                    "@search.score": 0.7,
                    "id": "doc-mid",
                    "content": "Mid",
                    "document_type": "sec_filing",
                }
            ),
        ]

        results = await retriever.retrieve("test", threshold=0.0)

        assert results[0].id == "doc-high"
        assert results[1].id == "doc-mid"
        assert results[2].id == "doc-low"

    @pytest.mark.asyncio
    async def test_retrieve_with_domain_filter(
        self, retriever, mock_embedder, mock_search_client, mock_vectorized_query
    ):
        """retrieve() applies domain filter."""
        mock_search_client.search.return_value = []

        await retriever.retrieve("test", domain="estimates")

        call_kwargs = mock_search_client.search.call_args.kwargs
        assert call_kwargs.get("filter") == "domain eq 'estimates'"

    @pytest.mark.asyncio
    async def test_retrieve_with_document_type_filter(
        self, retriever, mock_embedder, mock_search_client, mock_vectorized_query
    ):
        """retrieve() applies document type filter."""
        mock_search_client.search.return_value = []

        await retriever.retrieve("test", document_types=[DocumentType.FIELD_CODE])

        call_kwargs = mock_search_client.search.call_args.kwargs
        assert "document_type eq 'field_code'" in call_kwargs.get("filter", "")

    @pytest.mark.asyncio
    async def test_retrieve_field_codes(
        self, retriever, mock_embedder, mock_search_client, mock_vectorized_query
    ):
        """retrieve_field_codes() uses correct defaults."""
        mock_search_client.search.return_value = [
            MockSearchResult(
                {
                    "@search.score": 0.8,
                    "id": "fc-1",
                    "content": "TR.EPSMean",
                    "document_type": "field_code",
                    "domain": "estimates",
                    "field_code": "TR.EPSMean",
                }
            ),
        ]

        results = await retriever.retrieve_field_codes("EPS forecast", domain="estimates")

        assert len(results) == 1
        assert results[0].document_type == DocumentType.FIELD_CODE

        # Verify correct filter was applied
        call_kwargs = mock_search_client.search.call_args.kwargs
        filter_expr = call_kwargs.get("filter", "")
        assert "estimates" in filter_expr
        assert "field_code" in filter_expr

    @pytest.mark.asyncio
    async def test_retrieve_examples(
        self, retriever, mock_embedder, mock_search_client, mock_vectorized_query
    ):
        """retrieve_examples() uses correct defaults."""
        mock_search_client.search.return_value = [
            MockSearchResult(
                {
                    "@search.score": 0.7,
                    "id": "ex-1",
                    "content": "Example query",
                    "document_type": "query_example",
                    "example_query": "What is revenue?",
                    "example_api_call": "TR.Revenue",
                }
            ),
        ]

        results = await retriever.retrieve_examples("revenue data")

        assert len(results) == 1
        assert results[0].document_type == DocumentType.QUERY_EXAMPLE

    @pytest.mark.asyncio
    async def test_close(self, retriever, mock_search_client):
        """close() closes the search client."""
        await retriever.close()

        mock_search_client.close.assert_called_once()


# =============================================================================
# Factory Tests
# =============================================================================


class TestCreateRetriever:
    """Tests for create_retriever factory function."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.dimension = 384
        return embedder

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = MagicMock()
        return pool

    def test_create_retriever_postgres_backend(self, mock_pool):
        """create_retriever returns HybridRAGRetriever for postgres backend."""
        from unittest.mock import patch

        from src.rag.ui.config import RAGUIConfig

        config = RAGUIConfig(rag_backend="postgres", embedding_provider="local")

        # Mock the LocalEmbedder and HybridRAGRetriever
        mock_modules = create_mock_azure_modules()
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {**mock_modules, "sentence_transformers": mock_st_module}):
            from src.rag.retriever.factory import create_retriever
            from src.rag.retriever.retriever import HybridRAGRetriever

            retriever = create_retriever(config, pool=mock_pool)

            assert isinstance(retriever, HybridRAGRetriever)

    def test_create_retriever_postgres_requires_pool(self):
        """create_retriever raises error if pool not provided for postgres backend."""
        from src.rag.ui.config import RAGUIConfig

        config = RAGUIConfig(rag_backend="postgres")

        # Mock the LocalEmbedder
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            from src.rag.retriever.factory import create_retriever

            with pytest.raises(ValueError, match="PostgreSQL connection pool required"):
                create_retriever(config, pool=None)

    def test_create_retriever_azure_backend(self):
        """create_retriever returns AzureAISearchRetriever for azure backend."""
        from src.rag.ui.config import RAGUIConfig

        config = RAGUIConfig(
            rag_backend="azure",
            azure_search_endpoint="https://test.search.windows.net",
            azure_search_api_key="test-key",
            azure_search_index="test-index",
            embedding_provider="local",
        )

        # Mock dependencies
        mock_modules = create_mock_azure_modules()
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {**mock_modules, "sentence_transformers": mock_st_module}):
            from src.rag.retriever.azure_search import AzureAISearchRetriever
            from src.rag.retriever.factory import create_retriever

            retriever = create_retriever(config)

            assert isinstance(retriever, AzureAISearchRetriever)

    def test_create_retriever_azure_requires_endpoint(self):
        """create_retriever raises error if endpoint not provided for azure backend."""
        from src.rag.ui.config import RAGUIConfig

        config = RAGUIConfig(
            rag_backend="azure",
            azure_search_endpoint=None,  # Missing!
            azure_search_api_key="test-key",
        )

        # Mock the LocalEmbedder
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            from src.rag.retriever.factory import create_retriever

            with pytest.raises(ValueError, match="Azure AI Search endpoint required"):
                create_retriever(config)

    def test_create_retriever_azure_requires_api_key(self):
        """create_retriever raises error if API key not provided for azure backend."""
        from src.rag.ui.config import RAGUIConfig

        config = RAGUIConfig(
            rag_backend="azure",
            azure_search_endpoint="https://test.search.windows.net",
            azure_search_api_key=None,  # Missing!
        )

        # Mock the LocalEmbedder
        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            from src.rag.retriever.factory import create_retriever

            with pytest.raises(ValueError, match="Azure AI Search API key required"):
                create_retriever(config)
