"""
Embedding Providers

Abstracts embedding generation with support for multiple providers:
- LocalEmbedder: Uses sentence-transformers (free, runs locally)
- OpenAIEmbedder: Uses OpenAI API (better quality, requires API key)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    def stats(self) -> dict[str, int]:
        """Get embedder statistics (optional)."""
        return {}


class LocalEmbedder(Embedder):
    """
    Local embedder using sentence-transformers.

    Uses all-MiniLM-L6-v2 by default:
    - 384 dimensions
    - Fast inference
    - Good quality for semantic search
    - ~80MB model size

    No API key required - runs entirely locally.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize the local embedder.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
            batch_size: Batch size for encoding
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package required. "
                "Install with: pip install sentence-transformers"
            )

        self._model_name = model_name
        self._batch_size = batch_size

        # Load model
        logger.info(f"Loading local embedding model: {model_name}")
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Loaded {model_name} (dimension={self._dimension}, device={self._model.device})"
        )

        # Stats
        self._total_requests = 0
        self._total_texts = 0

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def stats(self) -> dict[str, int]:
        """Get embedder statistics."""
        return {
            "total_requests": self._total_requests,
            "total_texts": self._total_texts,
        }

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed_sync, text)
        self._total_requests += 1
        self._total_texts += 1
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._embed_batch_sync, texts)
        self._total_requests += 1
        self._total_texts += len(texts)
        return embeddings

    def _embed_sync(self, text: str) -> list[float]:
        """Synchronous embed for thread pool."""
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous batch embed for thread pool."""
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()


class OpenAIEmbedder(Embedder):
    """
    Embedder using OpenAI's text-embedding models.

    Features:
    - Concurrency control via semaphore
    - Automatic retry with exponential backoff for rate limits
    - Token tracking for rate limit awareness
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        max_concurrent: int = 5,
        requests_per_minute: int = 3000,
    ):
        """
        Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
            max_concurrent: Maximum concurrent requests
            requests_per_minute: Rate limit (for logging/monitoring)
        """
        from src.evalkit.common.llm import create_openai_client

        self._model = model
        self._client = create_openai_client(async_client=True, api_key=api_key)
        self._max_concurrent = max_concurrent
        self._requests_per_minute = requests_per_minute

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Dimension varies by model
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._rate_limit_hits = 0

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimensions.get(self._model, 1536)

    @property
    def stats(self) -> dict[str, int]:
        """Get embedder statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "rate_limit_hits": self._rate_limit_hits,
        }

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        async with self._semaphore:
            return await self._embed_with_retry(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Uses semaphore to limit concurrent requests and handles rate limits.
        """
        async with self._semaphore:
            return await self._embed_batch_with_retry(texts)

    async def _embed_with_retry(
        self,
        text: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> list[float]:
        """Embed with retry logic for rate limits."""
        import random

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=text,
                )
                self._total_requests += 1
                self._total_tokens += response.usage.total_tokens
                return response.data[0].embedding
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "rate_limit" in error_str or "429" in error_str:
                    self._rate_limit_hits += 1
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"OpenAI rate limit hit, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise last_error or RuntimeError("Embed failed after retries")

    async def _embed_batch_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> list[list[float]]:
        """Embed batch with retry logic for rate limits."""
        import random

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                self._total_requests += 1
                self._total_tokens += response.usage.total_tokens
                return [item.embedding for item in response.data]
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "rate_limit" in error_str or "429" in error_str:
                    self._rate_limit_hits += 1
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"OpenAI rate limit hit, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise last_error or RuntimeError("Embed batch failed after retries")


def create_embedder(
    provider: str = "local",
    **kwargs,
) -> Embedder:
    """
    Factory function to create an embedder.

    Args:
        provider: "local" or "openai"
        **kwargs: Provider-specific arguments

    Returns:
        Embedder instance

    Examples:
        # Local embedder (default)
        embedder = create_embedder("local")

        # OpenAI embedder
        embedder = create_embedder("openai", api_key="sk-...")
    """
    if provider == "local":
        return LocalEmbedder(
            model_name=kwargs.get("model_name", "all-MiniLM-L6-v2"),
            device=kwargs.get("device"),
            batch_size=kwargs.get("batch_size", 32),
        )
    elif provider == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("OpenAI embedder requires 'api_key' argument")
        return OpenAIEmbedder(
            api_key=api_key,
            model=kwargs.get("model", "text-embedding-3-small"),
            max_concurrent=kwargs.get("max_concurrent", 5),
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
