"""
Unit tests for the Evaluator facade.

Tests the generic Evaluator class with various packs and configurations.
"""

import json
from dataclasses import dataclass
from typing import Any

import pytest

from CONTRACTS import (
    EvalContext,
    Scorecard,
    TestCase,
    ToolCall,
)
from src.evaluation.core.evaluator import Evaluator, EvaluatorConfig
from src.evaluation.packs.nl2api import NL2APIPack

# =============================================================================
# Mock Target System
# =============================================================================


@dataclass
class MockTargetSystem:
    """Mock target system for testing."""

    responses: dict[str, dict[str, Any]] | None = None
    """Map of test case IDs to responses."""

    default_response: dict[str, Any] | None = None
    """Default response if ID not in responses."""

    async def process(self, test_case: TestCase) -> dict[str, Any]:
        """Return mock response for test case."""
        if self.responses and test_case.id in self.responses:
            return self.responses[test_case.id]
        if self.default_response:
            return self.default_response
        raise ValueError(f"No mock response for test case {test_case.id}")


# =============================================================================
# Basic Evaluator Tests
# =============================================================================


class TestEvaluatorBasics:
    """Basic tests for Evaluator initialization and properties."""

    def test_evaluator_init_with_pack(self):
        """Evaluator initializes with pack."""
        pack = NL2APIPack()
        evaluator = Evaluator(pack=pack)

        assert evaluator.pack_name == "nl2api"

    def test_evaluator_init_with_config(self):
        """Evaluator accepts custom configuration."""
        pack = NL2APIPack()
        config = EvaluatorConfig(
            validate_inputs=False,
            max_concurrency=5,
            worker_id="test-worker",
        )
        evaluator = Evaluator(pack=pack, config=config)

        assert evaluator.config.validate_inputs is False
        assert evaluator.config.max_concurrency == 5
        assert evaluator.config.worker_id == "test-worker"


# =============================================================================
# Single Evaluation Tests
# =============================================================================


class TestEvaluatorEvaluate:
    """Tests for single test case evaluation."""

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        return Evaluator(pack=NL2APIPack())

    @pytest.fixture
    def test_case(self) -> TestCase:
        return TestCase(
            id="test-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

    @pytest.mark.asyncio
    async def test_evaluate_passing_case(self, evaluator, test_case):
        """Evaluate a passing test case."""
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        scorecard = await evaluator.evaluate(test_case, system_output)

        assert scorecard.test_case_id == test_case.id
        assert scorecard.pack_name == "nl2api"
        assert "syntax" in scorecard.stage_results
        assert "logic" in scorecard.stage_results
        assert scorecard.stage_results["syntax"].passed is True
        assert scorecard.stage_results["logic"].passed is True

    @pytest.mark.asyncio
    async def test_evaluate_failing_syntax(self, evaluator, test_case):
        """Evaluate a test case with syntax error."""
        system_output = {"raw_output": "invalid json"}

        scorecard = await evaluator.evaluate(test_case, system_output)

        assert scorecard.stage_results["syntax"].passed is False
        # Logic should not be in results (gate failure)
        assert "logic" not in scorecard.stage_results

    @pytest.mark.asyncio
    async def test_evaluate_failing_logic(self, evaluator, test_case):
        """Evaluate a test case with logic error."""
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_volume", "arguments": {"ticker": "AAPL"}}])
        }

        scorecard = await evaluator.evaluate(test_case, system_output)

        assert scorecard.stage_results["syntax"].passed is True
        assert scorecard.stage_results["logic"].passed is False

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, evaluator, test_case):
        """Evaluate with custom context."""
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }
        context = EvalContext(
            batch_id="batch-001",
            worker_id="custom-worker",
        )

        scorecard = await evaluator.evaluate(test_case, system_output, context)

        assert scorecard.batch_id == "batch-001"
        assert scorecard.worker_id == "custom-worker"

    @pytest.mark.asyncio
    async def test_evaluate_captures_output(self, evaluator, test_case):
        """Evaluate captures system output in scorecard."""
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]),
            "extra_data": {"key": "value"},
        }

        scorecard = await evaluator.evaluate(test_case, system_output)

        assert "raw_output" in scorecard.generated_output
        assert "extra_data" in scorecard.generated_output


# =============================================================================
# Validation Tests
# =============================================================================


class TestEvaluatorValidation:
    """Tests for input validation."""

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        return Evaluator(pack=NL2APIPack())

    @pytest.fixture
    def evaluator_no_validation(self) -> Evaluator:
        return Evaluator(
            pack=NL2APIPack(),
            config=EvaluatorConfig(validate_inputs=False),
        )

    @pytest.fixture
    def evaluator_soft_validation(self) -> Evaluator:
        return Evaluator(
            pack=NL2APIPack(),
            config=EvaluatorConfig(fail_on_validation_error=False),
        )

    @pytest.mark.asyncio
    async def test_validation_error_raises(self, evaluator):
        """Validation error raises by default."""
        test_case = TestCase(id="invalid-001")  # Missing nl_query and expected_tool_calls

        with pytest.raises(ValueError) as exc_info:
            await evaluator.evaluate(test_case, {})

        assert "validation failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validation_disabled(self, evaluator_no_validation):
        """Validation can be disabled."""
        test_case = TestCase(id="invalid-001")
        system_output = {"raw_output": "{}"}

        # Should not raise, just fail on evaluation
        scorecard = await evaluator_no_validation.evaluate(test_case, system_output)
        assert scorecard.test_case_id == test_case.id

    @pytest.mark.asyncio
    async def test_validation_soft_mode(self, evaluator_soft_validation):
        """Soft validation mode doesn't raise."""
        test_case = TestCase(id="invalid-001")
        system_output = {"raw_output": "{}"}

        # Should not raise
        scorecard = await evaluator_soft_validation.evaluate(test_case, system_output)
        assert scorecard.test_case_id == test_case.id

    def test_validate_test_cases_batch(self, evaluator):
        """Batch validation returns error map."""
        test_cases = [
            TestCase(
                id="valid-001",
                nl_query="Get price",
                expected_tool_calls=(ToolCall(tool_name="get_price", arguments={}),),
            ),
            TestCase(id="invalid-001"),  # Missing fields
            TestCase(id="invalid-002"),  # Missing fields
        ]

        errors = evaluator.validate_test_cases(test_cases)

        assert "valid-001" not in errors
        assert "invalid-001" in errors
        assert "invalid-002" in errors


# =============================================================================
# Target System Tests
# =============================================================================


class TestEvaluatorWithTarget:
    """Tests for evaluation with target system."""

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        return Evaluator(pack=NL2APIPack())

    @pytest.fixture
    def test_case(self) -> TestCase:
        return TestCase(
            id="test-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

    @pytest.mark.asyncio
    async def test_evaluate_with_target(self, evaluator, test_case):
        """Evaluate with mock target system."""
        target = MockTargetSystem(
            responses={
                "test-001": {
                    "raw_output": json.dumps(
                        [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]
                    )
                }
            }
        )

        scorecard = await evaluator.evaluate_with_target(test_case, target)

        assert scorecard.test_case_id == test_case.id
        assert scorecard.stage_results["syntax"].passed is True
        assert scorecard.stage_results["logic"].passed is True


# =============================================================================
# Batch Evaluation Tests
# =============================================================================


class TestEvaluatorBatch:
    """Tests for batch evaluation."""

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        return Evaluator(
            pack=NL2APIPack(),
            config=EvaluatorConfig(max_concurrency=2),
        )

    @pytest.fixture
    def test_cases(self) -> list[TestCase]:
        return [
            TestCase(
                id=f"test-{i:03d}",
                nl_query="Get Apple price",
                expected_tool_calls=(
                    ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),
                ),
            )
            for i in range(5)
        ]

    @pytest.mark.asyncio
    async def test_batch_evaluation(self, evaluator, test_cases):
        """Batch evaluation processes all test cases."""
        target = MockTargetSystem(
            default_response={
                "raw_output": json.dumps(
                    [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]
                )
            }
        )

        scorecards = await evaluator.evaluate_batch(test_cases, target)

        assert len(scorecards) == 5
        for sc in scorecards:
            assert sc.pack_name == "nl2api"
            assert sc.stage_results["syntax"].passed is True
            assert sc.stage_results["logic"].passed is True

    @pytest.mark.asyncio
    async def test_batch_with_callback(self, evaluator, test_cases):
        """Batch evaluation calls on_result callback."""
        target = MockTargetSystem(
            default_response={
                "raw_output": json.dumps(
                    [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]
                )
            }
        )

        results: list[Scorecard] = []

        async def on_result(sc: Scorecard) -> None:
            results.append(sc)

        scorecards = await evaluator.evaluate_batch(test_cases, target, on_result=on_result)

        assert len(results) == 5
        assert len(scorecards) == 5

    @pytest.mark.asyncio
    async def test_batch_mixed_results(self, evaluator, test_cases):
        """Batch handles mixed pass/fail results."""
        target = MockTargetSystem(
            responses={
                "test-000": {
                    "raw_output": json.dumps(
                        [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]
                    )
                },
                "test-001": {"raw_output": "invalid json"},
                "test-002": {
                    "raw_output": json.dumps(
                        [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]
                    )
                },
                "test-003": {
                    "raw_output": json.dumps([{"tool_name": "wrong_tool", "arguments": {}}])
                },
                "test-004": {
                    "raw_output": json.dumps(
                        [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]
                    )
                },
            }
        )

        scorecards = await evaluator.evaluate_batch(test_cases, target)

        assert len(scorecards) == 5

        # Count passes
        passed = sum(
            1 for sc in scorecards if all(sr.passed for sr in sc.get_all_stage_results().values())
        )
        assert passed == 3  # test-000, test-002, test-004

    @pytest.mark.asyncio
    async def test_batch_respects_concurrency(self, evaluator, test_cases):
        """Batch respects max_concurrency setting."""
        import asyncio

        concurrent_count = 0
        max_concurrent = 0

        class TrackingTarget:
            async def process(self, test_case: TestCase) -> dict[str, Any]:
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.01)  # Small delay
                concurrent_count -= 1
                return {
                    "raw_output": json.dumps(
                        [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]
                    )
                }

        target = TrackingTarget()
        await evaluator.evaluate_batch(test_cases, target)

        # max_concurrency is 2, so should never exceed that
        assert max_concurrent <= 2
