"""
Generic Evaluator Facade

Provides a simple API for running evaluations with any EvaluationPack.
This is the primary entry point for the evaluation framework.

Usage:
    from src.evalkit.core import Evaluator
    from src.nl2api.evaluation import NL2APIPack

    # Create evaluator with pack
    pack = NL2APIPack()
    evaluator = Evaluator(pack=pack)

    # Evaluate single test case
    scorecard = await evaluator.evaluate(test_case, system_output)

    # Evaluate batch
    scorecards = await evaluator.evaluate_batch(test_cases, target_system)
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from opentelemetry import trace as otel_trace

from CONTRACTS import (
    EvalContext,
    Scorecard,
    StageResult,
    TestCase,
)
from src.evalkit.common.telemetry import get_tracer

tracer = get_tracer(__name__)


def _get_trace_context() -> tuple[str | None, str | None]:
    """
    Extract trace_id and span_id from the current span context.

    Returns:
        Tuple of (trace_id, span_id) as hex strings, or (None, None) if no active span.
    """
    span = otel_trace.get_current_span()
    if span is None:
        return None, None

    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return None, None

    # Format as hex strings (trace_id is 128-bit/32 chars, span_id is 64-bit/16 chars)
    trace_id = format(ctx.trace_id, "032x")
    span_id = format(ctx.span_id, "016x")
    return trace_id, span_id


def _add_debug_attributes_to_span(span: Any, result: Any) -> None:
    """
    Add debugging attributes to a span from StageResult.

    Extracts commonly useful artifacts and adds them as span attributes
    for easier debugging in Jaeger/tracing UIs.

    Args:
        span: The OTEL span to add attributes to
        result: StageResult with artifacts to extract
    """
    if not result.artifacts:
        return

    artifacts = result.artifacts

    # NL2API Logic stage: tool call mismatches
    if "missing_calls" in artifacts and artifacts["missing_calls"]:
        tools = ", ".join(tc.get("tool_name", "?") for tc in artifacts["missing_calls"][:5])
        span.set_attribute("debug.missing_tools", tools)

    if "extra_calls" in artifacts and artifacts["extra_calls"]:
        tools = ", ".join(tc.get("tool_name", "?") for tc in artifacts["extra_calls"][:5])
        span.set_attribute("debug.extra_tools", tools)

    if "argument_diffs" in artifacts and artifacts["argument_diffs"]:
        diffs = artifacts["argument_diffs"]
        if isinstance(diffs, list) and diffs:
            summary = "; ".join(f"{d.get('tool', '?')}: {d.get('field', '?')}" for d in diffs[:3])
            span.set_attribute("debug.arg_diffs", summary)

    # RAG Retrieval stage: document mismatches
    if "expected_docs" in artifacts and artifacts["expected_docs"]:
        docs = ", ".join(str(d)[:50] for d in artifacts["expected_docs"][:5])
        span.set_attribute("debug.expected_docs", docs)

    if "retrieved_docs" in artifacts and artifacts["retrieved_docs"]:
        docs = ", ".join(str(d)[:50] for d in artifacts["retrieved_docs"][:5])
        span.set_attribute("debug.retrieved_docs", docs)

    # RAG IR metrics
    for metric in ["recall@5", "precision@5", "mrr", "ndcg@5"]:
        if metric in artifacts:
            attr_name = f"metrics.{metric.replace('@', '_at_')}"
            span.set_attribute(attr_name, artifacts[metric])


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to JSON-serializable form.

    Handles common non-serializable types:
    - Pydantic models -> dict
    - dataclasses -> dict
    - tuples -> lists
    - sets -> lists
    - objects with __dict__ -> dict representation

    Args:
        obj: Any object to convert

    Returns:
        JSON-serializable version of the object
    """
    import json
    from dataclasses import asdict, is_dataclass

    # None, bool, int, float, str are already serializable
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Lists and tuples
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]

    # Sets
    if isinstance(obj, set):
        return [_make_json_serializable(item) for item in obj]

    # Dicts
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}

    # Pydantic models (v2)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    # Pydantic models (v1) / objects with dict method
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()

    # Dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)

    # FrozenDict or similar
    if hasattr(obj, "items"):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}

    # Last resort: try to convert to string
    try:
        # Test if it's already serializable
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Convert to string representation
        return str(obj)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class EvaluationPack(Protocol):
    """Protocol for evaluation packs."""

    @property
    def name(self) -> str:
        """Return the pack name (e.g., 'nl2api', 'rag')."""
        ...

    def get_stages(self) -> list[Any]:
        """Return ordered list of evaluation stages."""
        ...

    def get_default_weights(self) -> dict[str, float]:
        """Return default scoring weights per stage name."""
        ...

    def validate_test_case(self, test_case: TestCase) -> list[str]:
        """Validate test case has required fields. Return error messages."""
        ...

    def compute_overall_score(
        self,
        stage_results: dict[str, StageResult],
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute weighted overall score from stage results."""
        ...

    def compute_overall_passed(
        self,
        stage_results: dict[str, StageResult],
    ) -> bool:
        """Determine if overall evaluation passed."""
        ...


@runtime_checkable
class TargetSystem(Protocol):
    """Protocol for systems being evaluated."""

    async def process(self, test_case: TestCase) -> dict[str, Any]:
        """
        Process a test case and return system output.

        Returns:
            Dict containing at minimum 'raw_output' key with the system's response.
            Additional keys depend on the evaluation pack being used.
        """
        ...


# =============================================================================
# Evaluator
# =============================================================================


@dataclass
class EvaluatorConfig:
    """Configuration for the Evaluator."""

    # Whether to validate test cases before evaluation
    validate_inputs: bool = True

    # Whether to fail fast on validation errors
    fail_on_validation_error: bool = True

    # Maximum concurrent evaluations (for batch mode)
    max_concurrency: int = 10

    # Context configuration passed to stages
    context_config: dict[str, Any] = field(default_factory=dict)

    # Worker ID for tracking
    worker_id: str = "local"


class Evaluator:
    """
    Generic evaluation facade.

    Provides a simple API for running evaluations with any EvaluationPack.
    Handles validation, execution, and result collection.

    Usage:
        pack = NL2APIPack()
        evaluator = Evaluator(pack=pack)

        # Single evaluation
        scorecard = await evaluator.evaluate(test_case, system_output)

        # Batch with target system
        scorecards = await evaluator.evaluate_batch(test_cases, target)
    """

    def __init__(
        self,
        pack: EvaluationPack,
        config: EvaluatorConfig | None = None,
    ):
        """
        Initialize the evaluator.

        Args:
            pack: The evaluation pack to use
            config: Optional configuration
        """
        self.pack = pack
        self.config = config or EvaluatorConfig()

    @property
    def pack_name(self) -> str:
        """Return the pack name."""
        return self.pack.name

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None = None,
    ) -> Scorecard:
        """
        Evaluate a single test case with provided system output.

        Args:
            test_case: The test case with expected values
            system_output: Output from the target system
            context: Optional evaluation context

        Returns:
            Scorecard with evaluation results

        Raises:
            ValueError: If test case fails validation and fail_on_validation_error is True
        """
        context = context or EvalContext(
            config=self.config.context_config,
            worker_id=self.config.worker_id,
        )

        with tracer.start_as_current_span("evaluator.evaluate") as span:
            span.set_attribute("pack.name", self.pack_name)
            span.set_attribute("test_case.id", test_case.id)

            # Add the query/question for debugging context
            # Support both NL2API (nl_query) and RAG (input.query) formats
            query = test_case.nl_query or (
                test_case.input.get("query") if test_case.input else None
            )
            if query:
                # Truncate to avoid huge spans
                query_display = query[:200] if len(query) > 200 else query
                span.set_attribute("test_case.query", query_display)

            # Validate
            if self.config.validate_inputs:
                errors = self.pack.validate_test_case(test_case)
                if errors:
                    span.set_attribute("validation.errors", len(errors))
                    if self.config.fail_on_validation_error:
                        raise ValueError(f"Test case validation failed: {errors}")

            # Run evaluation
            start_time = time.perf_counter()
            stage_results: dict[str, StageResult] = {}

            for stage in self.pack.get_stages():
                with tracer.start_as_current_span(f"evaluator.stage.{stage.name}") as stage_span:
                    result = await stage.evaluate(test_case, system_output, context)
                    stage_results[stage.name] = result

                    # Core result attributes
                    stage_span.set_attribute("result.passed", result.passed)
                    stage_span.set_attribute("result.score", result.score)
                    stage_span.set_attribute("result.duration_ms", result.duration_ms)

                    # Human-readable reason (truncate for span size limits)
                    if result.reason:
                        reason = result.reason[:500] if len(result.reason) > 500 else result.reason
                        stage_span.set_attribute("result.reason", reason)

                    # Structured error code
                    if result.error_code:
                        stage_span.set_attribute("result.error_code", result.error_code.value)

                    # Debug artifacts (pack-specific details)
                    _add_debug_attributes_to_span(stage_span, result)

                # Check for gate failure
                if stage.is_gate and not result.passed:
                    span.set_attribute("gate_failed", stage.name)
                    break

            total_latency_ms = int((time.perf_counter() - start_time) * 1000)
            overall_passed = self.pack.compute_overall_passed(stage_results)
            overall_score = self.pack.compute_overall_score(stage_results)

            span.set_attribute("result.overall_passed", overall_passed)
            span.set_attribute("result.overall_score", overall_score)
            span.set_attribute("result.total_latency_ms", total_latency_ms)

            # Capture trace context for Jaeger correlation
            trace_id, span_id = _get_trace_context()

            # Make system_output JSON-serializable for storage
            # (stages may add non-serializable objects like ToolCall, Pydantic models, etc.)
            serializable_output = _make_json_serializable(system_output)

            # Build scorecard
            return Scorecard(
                test_case_id=test_case.id,
                batch_id=context.batch_id,
                pack_name=self.pack_name,
                stage_results=stage_results,
                stage_weights=self.pack.get_default_weights(),
                generated_output=serializable_output,
                worker_id=context.worker_id,
                total_latency_ms=total_latency_ms,
                trace_id=trace_id,
                span_id=span_id,
            )

    async def evaluate_with_target(
        self,
        test_case: TestCase,
        target: TargetSystem,
        context: EvalContext | None = None,
    ) -> Scorecard:
        """
        Evaluate a single test case by running it through a target system.

        Args:
            test_case: The test case with expected values
            target: The target system to evaluate
            context: Optional evaluation context

        Returns:
            Scorecard with evaluation results
        """
        with tracer.start_as_current_span("evaluator.evaluate_with_target") as span:
            span.set_attribute("pack.name", self.pack_name)
            span.set_attribute("test_case.id", test_case.id)

            # Get system output
            system_output = await target.process(test_case)

            # Evaluate
            return await self.evaluate(test_case, system_output, context)

    async def evaluate_batch(
        self,
        test_cases: list[TestCase],
        target: TargetSystem,
        context: EvalContext | None = None,
        on_result: Callable[[Scorecard], Awaitable[None]] | None = None,
    ) -> list[Scorecard]:
        """
        Evaluate multiple test cases against a target system.

        Args:
            test_cases: List of test cases to evaluate
            target: The target system to evaluate
            context: Optional evaluation context
            on_result: Optional async callback for each result (for streaming/persistence)

        Returns:
            List of scorecards for all test cases
        """
        import asyncio

        context = context or EvalContext(
            config=self.config.context_config,
            worker_id=self.config.worker_id,
        )

        with tracer.start_as_current_span("evaluator.evaluate_batch") as span:
            span.set_attribute("pack.name", self.pack_name)
            span.set_attribute("batch.size", len(test_cases))
            span.set_attribute("batch.max_concurrency", self.config.max_concurrency)

            scorecards: list[Scorecard] = []
            semaphore = asyncio.Semaphore(self.config.max_concurrency)

            async def evaluate_one(tc: TestCase) -> Scorecard:
                async with semaphore:
                    scorecard = await self.evaluate_with_target(tc, target, context)
                    if on_result:
                        await on_result(scorecard)
                    return scorecard

            tasks = [evaluate_one(tc) for tc in test_cases]
            scorecards = await asyncio.gather(*tasks)

            # Summary stats
            passed_count = sum(
                1
                for sc in scorecards
                if all(sr.passed for sr in sc.get_all_stage_results().values())
            )
            span.set_attribute("batch.passed_count", passed_count)
            span.set_attribute("batch.failed_count", len(scorecards) - passed_count)

            return list(scorecards)

    def validate_test_cases(self, test_cases: list[TestCase]) -> dict[str, list[str]]:
        """
        Validate multiple test cases without running evaluation.

        Args:
            test_cases: List of test cases to validate

        Returns:
            Dict mapping test case IDs to lists of validation errors.
            Empty dict means all test cases are valid.
        """
        errors = {}
        for tc in test_cases:
            tc_errors = self.pack.validate_test_case(tc)
            if tc_errors:
                errors[tc.id] = tc_errors
        return errors
