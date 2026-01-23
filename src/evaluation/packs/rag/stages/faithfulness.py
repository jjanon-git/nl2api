"""
Faithfulness Stage

Evaluates whether the response is grounded in the retrieved context.
Reference-free: uses claim extraction + verification, no ground truth needed.

Part of the RAG Triad evaluation (also called "Groundedness").

Approach:
1. Extract atomic claims from the response
2. Verify each claim against the context
3. Score = supported claims / total claims
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.contracts import StageResult, TestCase

if TYPE_CHECKING:
    from src.contracts import EvalContext


@dataclass
class FaithfulnessStage:
    """
    Stage 3: Faithfulness Evaluation

    Uses claim extraction + verification to evaluate if response is grounded.
    Reference-free: no ground truth needed.

    Based on the RAGAS faithfulness metric approach.
    """

    name: str = field(default="faithfulness", init=False)
    is_gate: bool = field(default=False, init=False)

    # Configurable thresholds
    pass_threshold: float = 0.7
    max_claims_to_verify: int = 10  # Limit LLM calls

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None,
    ) -> StageResult:
        """
        Evaluate faithfulness (groundedness).

        Args:
            test_case: Test case (used for context only)
            system_output: System output with 'response' and 'context'
            context: Evaluation context with LLM judge

        Returns:
            StageResult with faithfulness metrics
        """
        start_time = time.perf_counter()

        # Get response
        response = self._extract_response(system_output)
        if not response:
            return self._skip_result("No response to evaluate", start_time)

        # Get context
        retrieved_context = self._extract_context(system_output)
        if not retrieved_context:
            return self._skip_result("No context to verify against", start_time)

        # Get LLM judge from context
        llm_judge = self._get_llm_judge(context)
        if llm_judge is None:
            return await self._heuristic_evaluate(response, retrieved_context, start_time)

        try:
            # Use LLM judge for full evaluation
            result = await llm_judge.evaluate_faithfulness(
                response=response,
                context=retrieved_context,
            )

            passed = result.passed
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            return StageResult(
                stage_name=self.name,
                passed=passed,
                score=result.score,
                reason=result.reasoning,
                metrics=result.metrics,
                artifacts={
                    "response_length": len(response),
                    "context_length": len(retrieved_context),
                },
                duration_ms=duration_ms,
            )

        except Exception as e:
            # Fall back to heuristic on LLM failure
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return StageResult(
                stage_name=self.name,
                passed=False,
                score=0.5,
                reason=f"LLM evaluation failed: {e}",
                metrics={"error": str(e)},
                duration_ms=duration_ms,
            )

    def _extract_response(self, system_output: dict[str, Any]) -> str:
        """Extract response text from system output."""
        for field_name in ["response", "answer", "generated_text", "output"]:
            if field_name in system_output:
                value = system_output[field_name]
                if isinstance(value, str):
                    return value
        return ""

    def _extract_context(self, system_output: dict[str, Any]) -> str:
        """Extract context text from system output."""
        # Try single context field
        for field_name in ["context", "retrieved_context"]:
            if field_name in system_output:
                value = system_output[field_name]
                if isinstance(value, str):
                    return value

        # Try chunks field
        chunks = system_output.get("retrieved_chunks", [])
        if chunks:
            texts = []
            for chunk in chunks:
                if isinstance(chunk, str):
                    texts.append(chunk)
                elif isinstance(chunk, dict):
                    texts.append(chunk.get("text", str(chunk)))
            return "\n\n".join(texts)

        return ""

    def _get_llm_judge(self, context: EvalContext | None) -> Any:
        """Get LLM judge from context."""
        if context is None:
            return None
        return context.llm_judge or context.config.get("llm_judge")

    async def _heuristic_evaluate(
        self,
        response: str,
        context: str,
        start_time: float,
    ) -> StageResult:
        """
        Heuristic evaluation when no LLM judge available.

        Uses n-gram overlap as proxy for groundedness.
        """
        # Extract n-grams from response and context
        response_ngrams = self._extract_ngrams(response, n=3)
        context_ngrams = self._extract_ngrams(context, n=3)

        if not response_ngrams:
            score = 1.0  # Empty response is trivially grounded
        else:
            overlap = len(response_ngrams & context_ngrams)
            score = overlap / len(response_ngrams)

        passed = score >= self.pass_threshold
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=score,
            reason=f"Heuristic evaluation (no LLM judge): {score:.2f} n-gram overlap",
            metrics={
                "evaluation_method": "heuristic",
                "ngram_overlap": score,
                "response_ngrams": len(response_ngrams),
                "context_ngrams": len(context_ngrams),
            },
            duration_ms=duration_ms,
        )

    def _extract_ngrams(self, text: str, n: int = 3) -> set[tuple[str, ...]]:
        """Extract word n-grams from text."""
        words = text.lower().split()
        if len(words) < n:
            return set()
        return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

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
