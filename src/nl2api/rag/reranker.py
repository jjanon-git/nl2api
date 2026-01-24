"""
Cross-Encoder Reranking for RAG

Implements two-stage retrieval with cross-encoder reranking for improved accuracy.
Research shows +20-35% retrieval accuracy improvement over single-stage retrieval.

Usage:
    reranker = CrossEncoderReranker()
    reranked = await reranker.rerank(query, candidates, top_k=10)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.evalkit.common.telemetry import get_tracer

if TYPE_CHECKING:
    from src.nl2api.rag.protocols import RetrievalResult

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Protocol for rerankers."""

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results."""
        ...


class CrossEncoderReranker:
    """
    Cross-encoder reranker for two-stage retrieval.

    Uses a cross-encoder model to score query-document pairs jointly,
    providing more accurate relevance scores than bi-encoder similarity.

    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    - BAAI/bge-reranker-large (better quality, multilingual)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (higher quality)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
            batch_size: Batch size for scoring
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers package required. "
                "Install with: pip install sentence-transformers"
            )

        self._model_name = model_name
        self._batch_size = batch_size

        logger.info(f"Loading cross-encoder model: {model_name}")
        self._model = CrossEncoder(model_name, device=device)
        logger.info(f"Loaded cross-encoder (device={self._model.model.device})")

        # Stats
        self._total_rerank_calls = 0
        self._total_pairs_scored = 0

    @property
    def stats(self) -> dict[str, int]:
        """Get reranker statistics."""
        return {
            "total_rerank_calls": self._total_rerank_calls,
            "total_pairs_scored": self._total_pairs_scored,
        }

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder scores.

        Args:
            query: The search query
            results: Candidate results from first-stage retrieval
            top_k: Number of top results to return

        Returns:
            Reranked list of RetrievalResult, limited to top_k
        """
        if not results:
            return results

        if len(results) <= top_k:
            # No need to rerank if we have fewer results than top_k
            return results

        with tracer.start_as_current_span("reranker.rerank") as span:
            span.set_attribute("reranker.model", self._model_name)
            span.set_attribute("reranker.candidates", len(results))
            span.set_attribute("reranker.top_k", top_k)

            # Create query-document pairs
            pairs = [(query, r.content) for r in results]

            # Score pairs using cross-encoder (run in thread pool)
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: self._model.predict(
                    pairs,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                ),
            )

            # Update stats
            self._total_rerank_calls += 1
            self._total_pairs_scored += len(pairs)

            # Sort by cross-encoder score (descending)
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Create new results with updated scores
            # We use dataclass replace to create new instances with updated scores
            reranked = []
            for result, score in scored_results[:top_k]:
                # Create a new RetrievalResult with the cross-encoder score
                # Since RetrievalResult is a dataclass with frozen=True (immutable),
                # we need to create a new instance
                from dataclasses import replace

                reranked.append(replace(result, score=float(score)))

            span.set_attribute("reranker.returned", len(reranked))
            logger.debug(
                f"Reranked {len(results)} candidates → top {len(reranked)} "
                f"(score range: {scores.min():.3f} to {scores.max():.3f})"
            )

            return reranked

    def rerank_sync(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Synchronous version of rerank for non-async contexts.

        Args:
            query: The search query
            results: Candidate results from first-stage retrieval
            top_k: Number of top results to return

        Returns:
            Reranked list of RetrievalResult, limited to top_k
        """
        if not results or len(results) <= top_k:
            return results

        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        self._total_rerank_calls += 1
        self._total_pairs_scored += len(pairs)

        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        from dataclasses import replace

        return [replace(result, score=float(score)) for result, score in scored_results[:top_k]]


def create_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str | None = None,
) -> CrossEncoderReranker:
    """
    Factory function to create a reranker.

    Args:
        model_name: Cross-encoder model name
        device: Device to run on

    Returns:
        CrossEncoderReranker instance
    """
    return CrossEncoderReranker(model_name=model_name, device=device)
