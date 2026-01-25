"""
Answer Relevance Stage

Evaluates whether the response answers the question asked.
Reference-free: uses LLM-as-judge, no ground truth needed.

Part of the RAG Triad evaluation.

Checks:
- Does the answer address the question?
- Is it complete?
- Is it on-topic?
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.evalkit.contracts import StageResult, TestCase

if TYPE_CHECKING:
    from src.evalkit.contracts import EvalContext


@dataclass
class AnswerRelevanceStage:
    """
    Stage 4: Answer Relevance Evaluation

    Uses LLM-as-judge to evaluate if the response answers the question.
    Reference-free: no ground truth needed.
    """

    name: str = field(default="answer_relevance", init=False)
    is_gate: bool = field(default=False, init=False)

    # Configurable thresholds
    pass_threshold: float = 0.7

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None,
    ) -> StageResult:
        """
        Evaluate answer relevance.

        Args:
            test_case: Test case with input['query']
            system_output: System output with 'response' or 'answer'
            context: Evaluation context with LLM judge

        Returns:
            StageResult with relevance metrics
        """
        start_time = time.perf_counter()

        # Get query
        query = test_case.input.get("query", "")
        if not query:
            return self._skip_result("No query in test case", start_time)

        # Get response
        response = self._extract_response(system_output)
        if not response:
            return self._skip_result("No response to evaluate", start_time)

        # Check for rejection patterns first
        is_rejection = self._is_rejection(response)
        expected_behavior = test_case.expected.get("behavior", "answer")

        if is_rejection and expected_behavior == "reject":
            # Correct rejection - full score
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return StageResult(
                stage_name=self.name,
                passed=True,
                score=1.0,
                reason="Correct rejection - query should not be answered",
                metrics={"is_rejection": True, "rejection_correct": True},
                duration_ms=duration_ms,
            )

        if is_rejection and expected_behavior == "answer":
            # False positive - system rejected when it should have answered
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return StageResult(
                stage_name=self.name,
                passed=False,
                score=0.0,
                reason="False positive: System rejected query that should be answered",
                metrics={"is_rejection": True, "rejection_correct": False},
                duration_ms=duration_ms,
            )

        # Get LLM judge from context
        llm_judge = self._get_llm_judge(context)
        if llm_judge is None:
            return await self._heuristic_evaluate(query, response, start_time)

        try:
            result = await llm_judge.evaluate_relevance(
                query=query,
                text=response,
                context_type="answer",
            )

            # Override LLM judge's passed with our threshold
            passed = result.score >= self.pass_threshold
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            return StageResult(
                stage_name=self.name,
                passed=passed,
                score=result.score,
                reason=result.reasoning,
                metrics={
                    "relevance_score": result.score,
                    "is_rejection": is_rejection,
                },
                artifacts={
                    "query": query[:200],
                    "response_preview": response[:200],
                },
                duration_ms=duration_ms,
            )

        except Exception as e:
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

    def _is_rejection(self, response: str) -> bool:
        """Detect if response is a rejection."""
        response_lower = response.lower()

        rejection_patterns = [
            "i cannot",
            "i can't",
            "i'm not able to",
            "i am not able to",
            "i don't have",
            "i do not have",
            "i'm unable to",
            "i am unable to",
            "cannot provide",
            "cannot answer",
            "not available",
            "no information",
            "outside my scope",
            "beyond my knowledge",
            "i apologize",
            "unfortunately",
        ]

        return any(pattern in response_lower for pattern in rejection_patterns)

    def _get_llm_judge(self, context: EvalContext | None) -> Any:
        """Get LLM judge from context."""
        if context is None:
            return None
        return context.llm_judge or context.config.get("llm_judge")

    async def _heuristic_evaluate(
        self,
        query: str,
        response: str,
        start_time: float,
    ) -> StageResult:
        """
        Heuristic evaluation when no LLM judge available.

        Uses keyword overlap and response characteristics.
        """
        # Check for very short responses (likely insufficient)
        if len(response.strip()) < 20:
            score = 0.3
            reason = "Response too short"
        else:
            # Keyword overlap
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words & response_words)
            keyword_score = min(1.0, overlap / len(query_words)) if query_words else 0.5

            # Response length bonus (up to 0.2)
            length_bonus = min(0.2, len(response) / 500)

            score = min(1.0, keyword_score * 0.8 + length_bonus)
            reason = f"Heuristic evaluation: keyword overlap={keyword_score:.2f}"

        passed = score >= self.pass_threshold
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=score,
            reason=reason,
            metrics={
                "evaluation_method": "heuristic",
                "response_length": len(response),
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
