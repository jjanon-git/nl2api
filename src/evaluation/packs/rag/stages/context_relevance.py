"""
Context Relevance Stage

Evaluates whether the retrieved context is relevant to the query.
Reference-free: uses LLM-as-judge, no ground truth needed.

Part of the RAG Triad evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.evalkit.contracts import StageResult, TestCase

if TYPE_CHECKING:
    from src.evalkit.contracts import EvalContext


@dataclass
class ContextRelevanceStage:
    """
    Stage 2: Context Relevance Evaluation

    Uses LLM-as-judge to evaluate if retrieved context is relevant to the query.
    Reference-free: no ground truth needed.

    Evaluates each chunk/document and aggregates scores.
    """

    name: str = field(default="context_relevance", init=False)
    is_gate: bool = field(default=False, init=False)

    # Configurable thresholds
    pass_threshold: float = 0.6
    max_chunks_to_evaluate: int = 5  # Limit LLM calls

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None,
    ) -> StageResult:
        """
        Evaluate context relevance.

        Args:
            test_case: Test case with input['query']
            system_output: System output with 'retrieved_chunks' or 'context'
            context: Evaluation context with LLM judge

        Returns:
            StageResult with relevance metrics
        """
        start_time = time.perf_counter()

        # Get query
        query = test_case.input.get("query", "")
        if not query:
            return self._skip_result("No query in test case", start_time)

        # Get retrieved context
        chunks = self._extract_chunks(system_output)
        if not chunks:
            return self._skip_result("No context retrieved", start_time)

        # Get LLM judge from context
        llm_judge = self._get_llm_judge(context)
        if llm_judge is None:
            # Fall back to heuristic evaluation
            return await self._heuristic_evaluate(query, chunks, start_time)

        # Evaluate each chunk (up to limit)
        chunks_to_evaluate = chunks[: self.max_chunks_to_evaluate]
        chunk_scores: list[float] = []
        chunk_details: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks_to_evaluate):
            chunk_text = chunk if isinstance(chunk, str) else chunk.get("text", str(chunk))

            try:
                result = await llm_judge.evaluate_relevance(
                    query=query,
                    text=chunk_text,
                    context_type="context",
                )
                chunk_scores.append(result.score)
                chunk_details.append(
                    {
                        "chunk_index": i,
                        "score": result.score,
                        "reasoning": result.reasoning[:200],
                    }
                )
            except Exception as e:
                # Continue on individual chunk failure
                chunk_details.append(
                    {
                        "chunk_index": i,
                        "score": 0.5,
                        "error": str(e),
                    }
                )
                chunk_scores.append(0.5)

        # Aggregate scores (weighted by position - earlier chunks matter more)
        if chunk_scores:
            # Apply position decay: first chunk full weight, subsequent chunks decay
            weights = [1.0 / (i + 1) for i in range(len(chunk_scores))]
            total_weight = sum(weights)
            score = sum(s * w for s, w in zip(chunk_scores, weights)) / total_weight
        else:
            score = 0.0

        passed = score >= self.pass_threshold
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=score,
            reason=self._build_reason(score, chunk_scores, passed),
            metrics={
                "mean_chunk_score": sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0,
                "min_chunk_score": min(chunk_scores) if chunk_scores else 0.0,
                "max_chunk_score": max(chunk_scores) if chunk_scores else 0.0,
                "num_chunks_evaluated": len(chunk_scores),
                "num_chunks_total": len(chunks),
            },
            artifacts={
                "chunk_details": chunk_details,
                "query": query[:200],
            },
            duration_ms=duration_ms,
        )

    def _extract_chunks(self, system_output: dict[str, Any]) -> list[Any]:
        """Extract chunks from system output."""
        # Try various field names
        for field_name in ["retrieved_chunks", "context", "sources", "documents"]:
            if field_name in system_output:
                value = system_output[field_name]
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    # Single context string - split into chunks
                    return [value]
        return []

    def _get_llm_judge(self, context: EvalContext | None) -> Any:
        """Get LLM judge from context."""
        if context is None:
            return None
        return context.llm_judge or context.config.get("llm_judge")

    async def _heuristic_evaluate(
        self,
        query: str,
        chunks: list[Any],
        start_time: float,
    ) -> StageResult:
        """
        Heuristic evaluation when no LLM judge available.

        Uses simple keyword overlap as proxy for relevance.
        """
        query_words = set(query.lower().split())

        chunk_scores = []
        for chunk in chunks[: self.max_chunks_to_evaluate]:
            chunk_text = chunk if isinstance(chunk, str) else chunk.get("text", str(chunk))
            chunk_words = set(chunk_text.lower().split())

            # Jaccard-like overlap
            overlap = len(query_words & chunk_words)
            score = overlap / len(query_words) if query_words else 0.0
            chunk_scores.append(min(1.0, score * 2))  # Scale up

        score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        passed = score >= self.pass_threshold
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=score,
            reason=f"Heuristic evaluation (no LLM judge): score={score:.2f}",
            metrics={
                "evaluation_method": "heuristic",
                "mean_chunk_score": score,
                "num_chunks_evaluated": len(chunk_scores),
            },
            duration_ms=duration_ms,
        )

    def _skip_result(self, reason: str, start_time: float) -> StageResult:
        """Create a skip result."""
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return StageResult(
            stage_name=self.name,
            passed=True,
            score=1.0,
            reason=f"Skipped - {reason}",
            metrics={"skipped": True, "reason": reason},
            duration_ms=duration_ms,
        )

    def _build_reason(
        self,
        score: float,
        chunk_scores: list[float],
        passed: bool,
    ) -> str:
        """Build human-readable explanation."""
        if not chunk_scores:
            return "No chunks to evaluate"

        avg = sum(chunk_scores) / len(chunk_scores)

        if passed:
            return f"Context relevance acceptable: avg={avg:.2f} across {len(chunk_scores)} chunks"
        else:
            low_chunks = sum(1 for s in chunk_scores if s < 0.5)
            return f"Context relevance below threshold: avg={avg:.2f}, {low_chunks}/{len(chunk_scores)} chunks scored < 0.5"
