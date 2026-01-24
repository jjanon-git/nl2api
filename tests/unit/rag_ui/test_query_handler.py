"""Unit tests for RAG query handler."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.retriever.protocols import DocumentType, RetrievalResult
from src.rag.ui.config import RAGUIConfig
from src.rag.ui.query_handler import QueryResult, RAGQueryHandler


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


@pytest.fixture
def config():
    """Create test configuration."""
    return RAGUIConfig(
        database_url="postgresql://test:test@localhost:5432/test",
        anthropic_api_key="test-key",
        embedding_provider="local",
    )


@pytest.fixture
def sample_chunks():
    """Create sample retrieval results."""
    return [
        RetrievalResult(
            id="chunk-1",
            content="Apple Inc. reported total revenue of $394.3 billion for fiscal year 2022.",
            document_type=DocumentType.SEC_FILING,
            score=0.85,
            metadata={
                "ticker": "AAPL",
                "filing_type": "10-K",
                "fiscal_year": "2022",
                "section": "Item 8",
            },
        ),
        RetrievalResult(
            id="chunk-2",
            content="The Company's total net sales increased 8% year-over-year.",
            document_type=DocumentType.SEC_FILING,
            score=0.72,
            metadata={
                "ticker": "AAPL",
                "filing_type": "10-K",
                "fiscal_year": "2022",
                "section": "MD&A",
            },
        ),
    ]


class TestRAGQueryHandler:
    """Tests for RAGQueryHandler."""

    @pytest.mark.asyncio
    async def test_build_context_formats_chunks(self, mock_pool, config, sample_chunks):
        """Context should format chunks with metadata."""
        handler = RAGQueryHandler(pool=mock_pool, config=config)

        context = handler._build_context(sample_chunks)

        assert "AAPL 10-K FY2022" in context
        assert "Item 8" in context
        assert "MD&A" in context
        assert "$394.3 billion" in context
        assert "Score: 0.85" in context

    @pytest.mark.asyncio
    async def test_build_context_empty_chunks(self, mock_pool, config):
        """Empty chunks should return fallback message."""
        handler = RAGQueryHandler(pool=mock_pool, config=config)

        context = handler._build_context([])

        assert "No relevant context found" in context

    @pytest.mark.asyncio
    async def test_query_returns_result_with_sources(self, mock_pool, config, sample_chunks):
        """Query should return answer with source chunks."""
        handler = RAGQueryHandler(pool=mock_pool, config=config)

        # Mock the retriever
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=sample_chunks)
        handler._retriever = mock_retriever

        # Mock the LLM client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="The revenue was $394.3 billion in FY2022.")]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        handler._client = mock_client

        # Use a generic query that won't trigger company-specific path
        result = await handler.query("What is the company revenue?")

        assert isinstance(result, QueryResult)
        assert result.answer == "The revenue was $394.3 billion in FY2022."
        assert len(result.sources) == 2
        assert result.query == "What is the company revenue?"
        assert all(hasattr(s, "content") for s in result.sources)

    @pytest.mark.asyncio
    async def test_query_calls_retriever_with_correct_params(
        self, mock_pool, config, sample_chunks
    ):
        """Query should call retriever with SEC_FILING document type."""
        handler = RAGQueryHandler(pool=mock_pool, config=config)

        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=sample_chunks)
        handler._retriever = mock_retriever

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Answer")]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        handler._client = mock_client

        await handler.query("Test question", top_k=7)

        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["query"] == "Test question"
        assert call_kwargs["document_types"] == [DocumentType.SEC_FILING]
        assert call_kwargs["limit"] == 7

    @pytest.mark.asyncio
    async def test_query_raises_without_initialization(self, mock_pool, config):
        """Query should raise if handler not initialized."""
        handler = RAGQueryHandler(pool=mock_pool, config=config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await handler.query("Test question")


class TestRAGUIConfig:
    """Tests for RAGUIConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = RAGUIConfig(anthropic_api_key="test")

        assert config.embedding_provider == "local"
        assert config.default_top_k == 5
        assert config.vector_weight == 0.7
        assert config.keyword_weight == 0.3
        assert "postgresql" in config.database_url

    def test_embedding_dimension_local(self):
        """Local embedder should use 384 dimensions."""
        config = RAGUIConfig(
            anthropic_api_key="test",
            embedding_provider="local",
        )
        assert config.embedding_dimension == 384

    def test_embedding_dimension_openai(self):
        """OpenAI embedder should use 1536 dimensions."""
        config = RAGUIConfig(
            anthropic_api_key="test",
            embedding_provider="openai",
        )
        assert config.embedding_dimension == 1536


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self, sample_chunks):
        """QueryResult should store all fields correctly."""
        result = QueryResult(
            answer="Test answer",
            sources=sample_chunks,
            query="Test question",
            metadata={"model": "test-model"},
        )

        assert result.answer == "Test answer"
        assert len(result.sources) == 2
        assert result.query == "Test question"
        assert result.metadata["model"] == "test-model"

    def test_query_result_empty_sources(self):
        """QueryResult should handle empty sources."""
        result = QueryResult(
            answer="No information found.",
            sources=[],
            query="Unknown question",
        )

        assert result.answer == "No information found."
        assert result.sources == []
        assert result.metadata == {}


class TestTemporalDetection:
    """Tests for temporal intent detection."""

    @pytest.fixture
    def handler(self, mock_pool, config):
        """Create handler for testing."""
        return RAGQueryHandler(pool=mock_pool, config=config)

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("What is Apple's latest revenue?", True),
            ("What was the most recent earnings?", True),
            ("Tell me about Amazon's last 10-K filing", True),
            ("What are the current risk factors?", True),
            ("Show me 2024 financial results", True),
            ("What was revenue in FY2024?", True),
            ("What were the risk factors in 2020?", False),
            ("Tell me about the company", False),
            ("What is the PE ratio?", False),
        ],
    )
    def test_detect_temporal_intent(self, handler, query, expected):
        """Temporal detection should identify queries asking for recent data."""
        result = handler._detect_temporal_intent(query)
        assert result == expected, f"Query: '{query}' expected {expected}, got {result}"


class TestTickerExtraction:
    """Tests for company/ticker extraction using entity resolver."""

    @pytest.fixture
    def handler(self, mock_pool, config):
        """Create handler for testing."""
        handler = RAGQueryHandler(pool=mock_pool, config=config)
        # Add some known tickers
        handler._known_tickers = {"AAPL", "MSFT", "AMZN", "GOOGL"}
        return handler

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query,resolver_result,single_result_ticker,expected_ticker",
        [
            # Entity resolver returns RICs, resolve_single provides ticker in metadata
            ("What is Apple's revenue?", {"Apple": "AAPL.O"}, "AAPL", "AAPL"),
            ("Tell me about Microsoft's risk factors", {"Microsoft": "MSFT.O"}, "MSFT", "MSFT"),
            ("What did Amazon say in their 10-K?", {"Amazon": "AMZN.O"}, "AMZN", "AMZN"),
            ("Google's operating expenses", {"Google": "GOOGL.O"}, "GOOGL", "GOOGL"),
            ("Alphabet's latest filing", {"Alphabet": "GOOGL.O"}, "GOOGL", "GOOGL"),
            ("What is AAPL trading at?", {"AAPL": "AAPL.O"}, "AAPL", "AAPL"),
            ("MSFT revenue growth", {"MSFT": "MSFT.O"}, "MSFT", "MSFT"),
            # No entities resolved
            ("What is the market like?", {}, None, None),
            ("Revenue trends in tech sector", {}, None, None),
            # Entity resolved but not in our filings
            ("Tell me about Acme Corp", {"Acme Corp": "ACME.O"}, "ACME", None),
        ],
    )
    async def test_extract_ticker(
        self, handler, query, resolver_result, single_result_ticker, expected_ticker
    ):
        """Ticker extraction should use entity resolver and match against known tickers."""
        from src.nl2api.resolution.protocols import ResolvedEntity

        # Mock the entity resolver
        mock_resolver = AsyncMock()
        mock_resolver.resolve = AsyncMock(return_value=resolver_result)

        # Mock resolve_single to return ResolvedEntity with ticker in metadata
        if single_result_ticker:
            mock_single_result = ResolvedEntity(
                original=list(resolver_result.keys())[0] if resolver_result else "",
                identifier=list(resolver_result.values())[0] if resolver_result else "",
                entity_type="company",
                confidence=0.95,
                metadata={"ticker": single_result_ticker},
            )
            mock_resolver.resolve_single = AsyncMock(return_value=mock_single_result)
        else:
            mock_resolver.resolve_single = AsyncMock(return_value=None)

        handler._entity_resolver = mock_resolver

        result = await handler._extract_ticker(query)
        assert result == expected_ticker, (
            f"Query: '{query}' expected {expected_ticker}, got {result}"
        )

        # Verify resolver was called with the query
        mock_resolver.resolve.assert_called_once_with(query)
