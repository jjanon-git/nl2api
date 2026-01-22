"""
Semantics Evaluator - LLM-as-Judge for NL Response Quality.

Stage 4 of the evaluation pipeline. Compares actual NL responses against
expected NL responses using an LLM judge for semantic similarity.

The evaluation flow:
1. Load stored expected_response from test case
2. Feed expected_response to NL generator -> actual_nl
3. Compare actual_nl vs expected_nl_response (semantic comparison)

This tests NL generation quality in isolation:
"Given the same data, do we produce equivalent NL output?"
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from CONTRACTS import (
    ErrorCode,
    EvaluationStage,
    LLMJudgeConfig,
    StageResult,
    TestCase,
)
from src.common.telemetry import get_meter, get_tracer
from src.evaluation.core.semantics.prompts import (
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
meter = get_meter(__name__)

# Metrics
semantics_total = meter.create_counter(
    "eval_semantics_total",
    description="Total semantic evaluations",
)
semantics_passed = meter.create_counter(
    "eval_semantics_passed",
    description="Passed semantic evaluations",
)
semantics_score = meter.create_histogram(
    "eval_semantics_score",
    description="Semantic evaluation score distribution",
)
semantics_latency = meter.create_histogram(
    "eval_semantics_latency_ms",
    description="Semantic evaluation latency in milliseconds",
)


@dataclass
class ComparisonScores:
    """Scores from the LLM judge comparison."""

    meaning_match: float
    completeness: float
    accuracy: float
    reasoning: str

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score (default weights: 0.4, 0.3, 0.3)."""
        return (
            self.meaning_match * 0.4 +
            self.completeness * 0.3 +
            self.accuracy * 0.3
        )


class SemanticsEvaluator:
    """
    Stage 4: LLM-as-Judge Semantic Evaluator.

    Compares actual NL responses against expected NL responses using
    an LLM judge for semantic similarity scoring.

    Key behaviors:
    - Uses Claude 3.5 Haiku by default (configurable)
    - Includes retry logic with exponential backoff
    - Returns StageResult with proper ErrorCode on failure
    - OTEL instrumentation with evaluator.semantics span
    - Skips evaluation if expected_response or expected_nl_response is NULL
    """

    def __init__(
        self,
        config: LLMJudgeConfig | None = None,
        llm: Any | None = None,  # ClaudeProvider or compatible
    ):
        """
        Initialize the semantics evaluator.

        Args:
            config: LLM judge configuration
            llm: Optional LLM provider (created if not provided)
        """
        self.config = config or LLMJudgeConfig()
        self._llm = llm
        self._llm_initialized = llm is not None

    async def _get_llm(self) -> Any:
        """Lazily initialize the LLM provider."""
        if not self._llm_initialized:
            from src.nl2api.config import NL2APIConfig
            from src.nl2api.llm.claude import ClaudeProvider

            cfg = NL2APIConfig()
            self._llm = ClaudeProvider(
                api_key=cfg.get_llm_api_key(),
                model=self.config.model,
            )
            self._llm_initialized = True
        return self._llm

    async def evaluate(
        self,
        test_case: TestCase,
        nl_generator: Callable[[dict[str, Any]], Awaitable[str]] | None = None,
    ) -> StageResult:
        """
        Evaluate semantic similarity between expected and actual NL responses.

        The evaluation flow:
        1. If nl_generator provided: generate actual_nl from expected_response
        2. Otherwise: use test_case.expected_nl_response vs system's actual output
        3. Compare using LLM judge
        4. Return weighted score

        Args:
            test_case: Test case with expected_response and expected_nl_response
            nl_generator: Optional async function to generate NL from response data.
                         If provided, generates actual_nl from expected_response.
                         If not provided, must have actual_nl passed separately.

        Returns:
            StageResult with semantic similarity score
        """
        with tracer.start_as_current_span("evaluator.semantics") as span:
            span.set_attribute("test_case.id", test_case.id)
            span.set_attribute("test_case.category", test_case.metadata.source or "unknown")

            start_time = time.perf_counter()

            # Check if we have the required fields
            if test_case.expected_nl_response is None:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                span.set_attribute("result.skipped", True)
                span.set_attribute("result.skip_reason", "expected_nl_response is NULL")
                return StageResult(
                    stage=EvaluationStage.SEMANTICS,
                    passed=True,
                    score=1.0,
                    reason="Skipped: expected_nl_response is NULL",
                    artifacts={"skipped": True, "skip_reason": "expected_nl_response_null"},
                    duration_ms=duration_ms,
                )

            # If nl_generator is provided, we need expected_response to generate actual
            if nl_generator is not None and test_case.expected_response is None:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                span.set_attribute("result.skipped", True)
                span.set_attribute("result.skip_reason", "expected_response is NULL")
                return StageResult(
                    stage=EvaluationStage.SEMANTICS,
                    passed=True,
                    score=1.0,
                    reason="Skipped: expected_response is NULL (needed for NL generation)",
                    artifacts={"skipped": True, "skip_reason": "expected_response_null"},
                    duration_ms=duration_ms,
                )

            try:
                # Generate actual NL if generator provided
                if nl_generator is not None:
                    actual_nl = await nl_generator(test_case.expected_response)
                    span.set_attribute("nl_generator.used", True)
                else:
                    # In direct comparison mode, we compare against stored expected
                    # This is used when we already have an actual response to compare
                    actual_nl = test_case.expected_nl_response  # Self-comparison for testing
                    span.set_attribute("nl_generator.used", False)

                # Perform semantic comparison
                comparison_result = await self._compare_semantic(
                    expected=test_case.expected_nl_response,
                    actual=actual_nl,
                    query=test_case.nl_query,
                )

                # Calculate weighted score using config weights
                score = (
                    comparison_result.meaning_match * self.config.meaning_weight +
                    comparison_result.completeness * self.config.completeness_weight +
                    comparison_result.accuracy * self.config.accuracy_weight
                )

                duration_ms = int((time.perf_counter() - start_time) * 1000)
                passed = score >= self.config.pass_threshold

                # Record metrics
                semantics_total.add(1)
                if passed:
                    semantics_passed.add(1)
                semantics_score.record(score)
                semantics_latency.record(duration_ms)

                # Set span attributes
                span.set_attribute("result.passed", passed)
                span.set_attribute("result.score", score)
                span.set_attribute("result.meaning_match", comparison_result.meaning_match)
                span.set_attribute("result.completeness", comparison_result.completeness)
                span.set_attribute("result.accuracy", comparison_result.accuracy)
                span.set_attribute("result.duration_ms", duration_ms)

                error_code = None if passed else ErrorCode.SEMANTIC_LOW_SCORE

                return StageResult(
                    stage=EvaluationStage.SEMANTICS,
                    passed=passed,
                    score=score,
                    error_code=error_code,
                    reason=comparison_result.reasoning,
                    artifacts={
                        "meaning_match": comparison_result.meaning_match,
                        "completeness": comparison_result.completeness,
                        "accuracy": comparison_result.accuracy,
                        "expected_nl": test_case.expected_nl_response,
                        "actual_nl": actual_nl,
                        "pass_threshold": self.config.pass_threshold,
                    },
                    duration_ms=duration_ms,
                )

            except TimeoutError:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                span.set_attribute("result.error", "timeout")
                semantics_total.add(1)
                semantics_latency.record(duration_ms)
                return StageResult(
                    stage=EvaluationStage.SEMANTICS,
                    passed=False,
                    score=0.0,
                    error_code=ErrorCode.SYSTEM_TIMEOUT,
                    reason=f"Timeout after {self.config.timeout_ms}ms",
                    artifacts={"timeout_ms": self.config.timeout_ms},
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                span.set_attribute("result.error", str(e))
                span.record_exception(e)
                logger.warning(f"Semantic evaluation failed: {e}")
                semantics_total.add(1)
                semantics_latency.record(duration_ms)
                return StageResult(
                    stage=EvaluationStage.SEMANTICS,
                    passed=False,
                    score=0.0,
                    error_code=ErrorCode.SEMANTIC_LLM_ERROR,
                    reason=f"LLM error: {e}",
                    artifacts={"error": str(e)},
                    duration_ms=duration_ms,
                )

    async def evaluate_direct(
        self,
        test_case: TestCase,
        actual_nl: str,
    ) -> StageResult:
        """
        Evaluate semantic similarity with a provided actual NL response.

        Use this when you already have the actual NL response from the system
        and want to compare it against the expected NL response.

        Args:
            test_case: Test case with expected_nl_response
            actual_nl: The actual NL response from the system

        Returns:
            StageResult with semantic similarity score
        """
        with tracer.start_as_current_span("evaluator.semantics_direct") as span:
            span.set_attribute("test_case.id", test_case.id)

            start_time = time.perf_counter()

            # Check if we have expected NL response
            if test_case.expected_nl_response is None:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                span.set_attribute("result.skipped", True)
                return StageResult(
                    stage=EvaluationStage.SEMANTICS,
                    passed=True,
                    score=1.0,
                    reason="Skipped: expected_nl_response is NULL",
                    artifacts={"skipped": True},
                    duration_ms=duration_ms,
                )

            try:
                # Perform semantic comparison
                comparison_result = await self._compare_semantic(
                    expected=test_case.expected_nl_response,
                    actual=actual_nl,
                    query=test_case.nl_query,
                )

                # Calculate weighted score
                score = (
                    comparison_result.meaning_match * self.config.meaning_weight +
                    comparison_result.completeness * self.config.completeness_weight +
                    comparison_result.accuracy * self.config.accuracy_weight
                )

                duration_ms = int((time.perf_counter() - start_time) * 1000)
                passed = score >= self.config.pass_threshold

                # Record metrics
                semantics_total.add(1)
                if passed:
                    semantics_passed.add(1)
                semantics_score.record(score)
                semantics_latency.record(duration_ms)

                span.set_attribute("result.passed", passed)
                span.set_attribute("result.score", score)

                error_code = None if passed else ErrorCode.SEMANTIC_LOW_SCORE

                return StageResult(
                    stage=EvaluationStage.SEMANTICS,
                    passed=passed,
                    score=score,
                    error_code=error_code,
                    reason=comparison_result.reasoning,
                    artifacts={
                        "meaning_match": comparison_result.meaning_match,
                        "completeness": comparison_result.completeness,
                        "accuracy": comparison_result.accuracy,
                        "expected_nl": test_case.expected_nl_response,
                        "actual_nl": actual_nl,
                    },
                    duration_ms=duration_ms,
                )

            except TimeoutError:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                span.set_attribute("result.error", "timeout")
                semantics_total.add(1)
                semantics_latency.record(duration_ms)
                return StageResult(
                    stage=EvaluationStage.SEMANTICS,
                    passed=False,
                    score=0.0,
                    error_code=ErrorCode.SYSTEM_TIMEOUT,
                    reason=f"Timeout after {self.config.timeout_ms}ms",
                    artifacts={"timeout_ms": self.config.timeout_ms},
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                span.record_exception(e)
                semantics_total.add(1)
                return StageResult(
                    stage=EvaluationStage.SEMANTICS,
                    passed=False,
                    score=0.0,
                    error_code=ErrorCode.SEMANTIC_LLM_ERROR,
                    reason=f"LLM error: {e}",
                    duration_ms=duration_ms,
                )

    async def _compare_semantic(
        self,
        expected: str,
        actual: str,
        query: str,
    ) -> ComparisonScores:
        """
        Compare expected and actual NL responses using LLM judge.

        Args:
            expected: Expected NL response
            actual: Actual NL response
            query: Original query for context

        Returns:
            ComparisonScores with meaning_match, completeness, accuracy, reasoning
        """
        llm = await self._get_llm()

        from src.nl2api.llm.protocols import LLMMessage, MessageRole

        # Build messages
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=COMPARISON_SYSTEM_PROMPT),
            LLMMessage(
                role=MessageRole.USER,
                content=COMPARISON_USER_PROMPT_TEMPLATE.format(
                    query=query,
                    expected=expected,
                    actual=actual,
                ),
            ),
        ]

        # Call LLM with retry
        response = await llm.complete_with_retry(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            max_retries=self.config.max_retries,
        )

        # Parse response JSON
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM judge response: {response.content[:200]}")
            raise ValueError(f"Invalid JSON from LLM judge: {e}")

        return ComparisonScores(
            meaning_match=float(result.get("meaning_match", 0.0)),
            completeness=float(result.get("completeness", 0.0)),
            accuracy=float(result.get("accuracy", 0.0)),
            reasoning=result.get("reasoning", "No reasoning provided"),
        )
