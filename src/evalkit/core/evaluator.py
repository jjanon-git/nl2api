"""
Generic Evaluator Facade

Provides a simple API for running evaluations with any EvaluationPack.
This is the primary entry point for the evaluation framework.

Usage:
    from src.evalkit.core import Evaluator
    from src.evaluation.packs import NL2APIPack

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

from CONTRACTS import (
    EvalContext,
    Scorecard,
    StageResult,
    TestCase,
)
from src.evalkit.common.telemetry import get_tracer

tracer = get_tracer(__name__)


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

                    stage_span.set_attribute("result.passed", result.passed)
                    stage_span.set_attribute("result.score", result.score)

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

            # Build scorecard
            return Scorecard(
                test_case_id=test_case.id,
                batch_id=context.batch_id,
                pack_name=self.pack_name,
                stage_results=stage_results,
                stage_weights=self.pack.get_default_weights(),
                generated_output=system_output,
                worker_id=context.worker_id,
                total_latency_ms=total_latency_ms,
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
