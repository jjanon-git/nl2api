"""
Unit tests for RAG embedders.

Tests LocalEmbedder and OpenAIEmbedder implementations.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.rag.retriever.embedders import (
    Embedder,
    LocalEmbedder,
    OpenAIEmbedder,
    create_embedder,
)

# =============================================================================
# LocalEmbedder Tests
# =============================================================================


class TestLocalEmbedder:
    """Tests for LocalEmbedder using mocked SentenceTransformer."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"

        # Mock encode to return numpy arrays
        def mock_encode(text_or_texts, **kwargs):
            if isinstance(text_or_texts, str):
                return np.random.randn(384).astype(np.float32)
            else:
                return np.random.randn(len(text_or_texts), 384).astype(np.float32)

        mock_model.encode = mock_encode
        return mock_model

    @pytest.fixture
    def embedder(self, mock_sentence_transformer):
        """Create LocalEmbedder with mocked model."""
        # Patch the sentence_transformers module that gets imported inside LocalEmbedder.__init__
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_sentence_transformer)
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
            return embedder

    def test_initialization(self, embedder):
        """LocalEmbedder initializes with correct dimension."""
        assert embedder.dimension == 384

    def test_dimension_property(self, embedder):
        """Dimension property returns correct value."""
        assert embedder.dimension == 384

    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedder):
        """embed() returns embedding of correct dimension."""
        result = await embedder.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, embedder):
        """embed_batch() handles empty list."""
        result = await embedder.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_single(self, embedder):
        """embed_batch() handles single text."""
        result = await embedder.embed_batch(["test text"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 384

    @pytest.mark.asyncio
    async def test_embed_batch_multiple(self, embedder):
        """embed_batch() handles multiple texts."""
        texts = ["text one", "text two", "text three"]
        result = await embedder.embed_batch(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 384

    def test_stats_initial(self, embedder):
        """Stats start at zero."""
        stats = embedder.stats
        assert stats["total_requests"] == 0
        assert stats["total_texts"] == 0

    @pytest.mark.asyncio
    async def test_stats_tracking_single(self, embedder):
        """Stats track single embed calls."""
        await embedder.embed("test")

        stats = embedder.stats
        assert stats["total_requests"] == 1
        assert stats["total_texts"] == 1

    @pytest.mark.asyncio
    async def test_stats_tracking_batch(self, embedder):
        """Stats track batch embed calls."""
        await embedder.embed_batch(["text1", "text2", "text3"])

        stats = embedder.stats
        assert stats["total_requests"] == 1
        assert stats["total_texts"] == 3

    @pytest.mark.asyncio
    async def test_stats_accumulate(self, embedder):
        """Stats accumulate across calls."""
        await embedder.embed("single")
        await embedder.embed_batch(["batch1", "batch2"])

        stats = embedder.stats
        assert stats["total_requests"] == 2
        assert stats["total_texts"] == 3


class TestLocalEmbedderImportError:
    """Test LocalEmbedder when sentence-transformers is not installed."""

    def test_import_error(self):
        """LocalEmbedder raises ImportError when sentence-transformers missing."""
        # Remove sentence_transformers from modules to simulate it not being installed
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError) as exc_info:
                LocalEmbedder()
            assert "sentence-transformers" in str(exc_info.value)


# =============================================================================
# OpenAIEmbedder Tests
# =============================================================================


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder with mocked OpenAI client."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock AsyncOpenAI client."""
        mock_client = AsyncMock()

        # Mock embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_response.usage = MagicMock(total_tokens=10)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        return mock_client

    @pytest.fixture
    def embedder(self, mock_openai_client):
        """Create OpenAIEmbedder with mocked client."""
        # Patch the openai module that gets imported inside OpenAIEmbedder.__init__
        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_client)
        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            embedder = OpenAIEmbedder(api_key="test-key")
            embedder._client = mock_openai_client
            return embedder

    def test_initialization(self, embedder):
        """OpenAIEmbedder initializes with correct dimension."""
        assert embedder.dimension == 1536

    def test_dimension_by_model(self):
        """Dimension varies by model."""
        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            embedder_small = OpenAIEmbedder(api_key="test", model="text-embedding-3-small")
            assert embedder_small.dimension == 1536

            embedder_large = OpenAIEmbedder(api_key="test", model="text-embedding-3-large")
            assert embedder_large.dimension == 3072

            embedder_ada = OpenAIEmbedder(api_key="test", model="text-embedding-ada-002")
            assert embedder_ada.dimension == 1536

    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedder):
        """embed() returns embedding of correct dimension."""
        result = await embedder.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder, mock_openai_client):
        """embed_batch() returns embeddings for multiple texts."""
        # Update mock for batch response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        mock_response.usage = MagicMock(total_tokens=20)
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await embedder.embed_batch(["text1", "text2"])

        assert isinstance(result, list)
        assert len(result) == 2

    def test_stats_initial(self, embedder):
        """Stats start at zero."""
        stats = embedder.stats
        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
        assert stats["rate_limit_hits"] == 0

    @pytest.mark.asyncio
    async def test_stats_tracking(self, embedder):
        """Stats track API calls."""
        await embedder.embed("test")

        stats = embedder.stats
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 10

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, embedder, mock_openai_client):
        """Embedder retries on rate limit errors."""
        # First call fails with rate limit, second succeeds
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_response.usage = MagicMock(total_tokens=10)

        mock_openai_client.embeddings.create = AsyncMock(
            side_effect=[
                Exception("rate_limit_exceeded"),
                mock_response,
            ]
        )

        # Should succeed after retry
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await embedder.embed("test")

        assert len(result) == 1536
        assert embedder.stats["rate_limit_hits"] == 1


class TestOpenAIEmbedderImportError:
    """Test OpenAIEmbedder when openai is not installed."""

    def test_import_error(self):
        """OpenAIEmbedder raises ImportError when openai missing."""
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError) as exc_info:
                OpenAIEmbedder(api_key="test")
            assert "openai" in str(exc_info.value)


# =============================================================================
# create_embedder Factory Tests
# =============================================================================


class TestCreateEmbedder:
    """Tests for create_embedder factory function."""

    def test_create_local_embedder(self):
        """create_embedder creates LocalEmbedder for 'local' provider."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            embedder = create_embedder("local")
            assert isinstance(embedder, LocalEmbedder)

    def test_create_openai_embedder(self):
        """create_embedder creates OpenAIEmbedder for 'openai' provider."""
        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            embedder = create_embedder("openai", api_key="test-key")
            assert isinstance(embedder, OpenAIEmbedder)

    def test_create_openai_without_key_fails(self):
        """create_embedder raises ValueError without API key for OpenAI."""
        with pytest.raises(ValueError) as exc_info:
            create_embedder("openai")
        assert "api_key" in str(exc_info.value)

    def test_create_unknown_provider_fails(self):
        """create_embedder raises ValueError for unknown provider."""
        with pytest.raises(ValueError) as exc_info:
            create_embedder("unknown")
        assert "Unknown" in str(exc_info.value)

    def test_create_local_with_custom_model(self):
        """create_embedder passes custom model name to LocalEmbedder."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.device = "cpu"
        mock_st_module = MagicMock()
        mock_st_class = MagicMock(return_value=mock_model)
        mock_st_module.SentenceTransformer = mock_st_class
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            create_embedder("local", model_name="custom-model")
            mock_st_class.assert_called_once_with("custom-model", device=None)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestEmbedderProtocol:
    """Test that embedders implement the Embedder protocol."""

    def test_local_embedder_is_embedder(self):
        """LocalEmbedder satisfies Embedder protocol."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)
        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            embedder = LocalEmbedder()
            assert isinstance(embedder, Embedder)

    def test_openai_embedder_is_embedder(self):
        """OpenAIEmbedder satisfies Embedder protocol."""
        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            embedder = OpenAIEmbedder(api_key="test")
            assert isinstance(embedder, Embedder)
