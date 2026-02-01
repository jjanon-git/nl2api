"""
Unit tests for the Evaluator facade.

Tests the generic Evaluator class with various packs and configurations.
"""

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from CONTRACTS import (
    EvalContext,
    Scorecard,
    TestCase,
    ToolCall,
)
from src.evalkit.core.evaluator import (
    Evaluator,
    EvaluatorConfig,
    _get_trace_context,
    _make_json_serializable,
)
from src.nl2api.evaluation import NL2APIPack

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


# =============================================================================
# Trace Context Tests
# =============================================================================


class TestTraceContextCapture:
    """Tests for trace_id and span_id capture in scorecards."""

    def test_get_trace_context_no_active_span(self):
        """Returns None when no active span."""
        trace_id, span_id = _get_trace_context()
        # Without explicit span, may return None or have a default context
        # depending on OTEL configuration
        assert trace_id is None or len(trace_id) == 32
        assert span_id is None or len(span_id) == 16

    def test_get_trace_context_with_mock_span(self):
        """Returns trace_id and span_id from active span."""
        # Create mock span context
        mock_span_context = MagicMock()
        mock_span_context.is_valid = True
        mock_span_context.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
        mock_span_context.span_id = 0x1234567890ABCDEF

        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_span_context

        with patch(
            "src.evalkit.core.evaluator.otel_trace.get_current_span", return_value=mock_span
        ):
            trace_id, span_id = _get_trace_context()

        assert trace_id == "1234567890abcdef1234567890abcdef"
        assert span_id == "1234567890abcdef"

    def test_get_trace_context_invalid_span(self):
        """Returns None when span context is invalid."""
        mock_span_context = MagicMock()
        mock_span_context.is_valid = False

        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_span_context

        with patch(
            "src.evalkit.core.evaluator.otel_trace.get_current_span", return_value=mock_span
        ):
            trace_id, span_id = _get_trace_context()

        assert trace_id is None
        assert span_id is None

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        return Evaluator(pack=NL2APIPack())

    @pytest.fixture
    def test_case(self) -> TestCase:
        return TestCase(
            id="trace-test-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

    @pytest.mark.asyncio
    async def test_evaluate_captures_trace_context(self, evaluator, test_case):
        """Evaluator captures trace_id and span_id in scorecard."""
        mock_span_context = MagicMock()
        mock_span_context.is_valid = True
        mock_span_context.trace_id = 0xABCDEF1234567890ABCDEF1234567890
        mock_span_context.span_id = 0xFEDCBA0987654321

        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_span_context
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=None)

        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        # Mock both the tracer and get_current_span to avoid OTEL SDK interaction
        with patch("src.evalkit.core.evaluator.tracer") as mock_tracer:
            mock_tracer.start_as_current_span.return_value = mock_span
            with patch(
                "src.evalkit.core.evaluator.otel_trace.get_current_span", return_value=mock_span
            ):
                scorecard = await evaluator.evaluate(test_case, system_output)

        assert scorecard.trace_id == "abcdef1234567890abcdef1234567890"
        assert scorecard.span_id == "fedcba0987654321"

    @pytest.mark.asyncio
    async def test_evaluate_handles_no_trace_context(self, evaluator, test_case):
        """Evaluator handles missing trace context gracefully."""
        mock_span = MagicMock()
        mock_span.get_span_context.return_value = None
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=None)

        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        # Mock both the tracer and get_current_span to avoid OTEL SDK interaction
        with patch("src.evalkit.core.evaluator.tracer") as mock_tracer:
            mock_tracer.start_as_current_span.return_value = mock_span
            with patch(
                "src.evalkit.core.evaluator.otel_trace.get_current_span", return_value=mock_span
            ):
                scorecard = await evaluator.evaluate(test_case, system_output)

        assert scorecard.trace_id is None
        assert scorecard.span_id is None

    def test_trace_id_format(self):
        """Verify trace_id is formatted as 32 hex characters."""
        mock_span_context = MagicMock()
        mock_span_context.is_valid = True
        # Test with a trace_id that needs leading zeros
        mock_span_context.trace_id = 0x00000000000000001234567890ABCDEF
        mock_span_context.span_id = 0x0000000012345678

        mock_span = MagicMock()
        mock_span.get_span_context.return_value = mock_span_context

        with patch(
            "src.evalkit.core.evaluator.otel_trace.get_current_span", return_value=mock_span
        ):
            trace_id, span_id = _get_trace_context()

        # Should be zero-padded to correct length
        assert len(trace_id) == 32
        assert len(span_id) == 16
        assert trace_id == "00000000000000001234567890abcdef"
        assert span_id == "0000000012345678"


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestMakeJsonSerializable:
    """Tests for _make_json_serializable helper."""

    def test_primitives_unchanged(self):
        """Primitives pass through unchanged."""
        assert _make_json_serializable(None) is None
        assert _make_json_serializable(True) is True
        assert _make_json_serializable(42) == 42
        assert _make_json_serializable(3.14) == 3.14
        assert _make_json_serializable("hello") == "hello"

    def test_list_recursion(self):
        """Lists are recursively processed."""
        result = _make_json_serializable([1, "two", [3, 4]])
        assert result == [1, "two", [3, 4]]

    def test_tuple_to_list(self):
        """Tuples are converted to lists."""
        result = _make_json_serializable((1, 2, 3))
        assert result == [1, 2, 3]

    def test_set_to_list(self):
        """Sets are converted to lists."""
        result = _make_json_serializable({1, 2, 3})
        assert sorted(result) == [1, 2, 3]

    def test_dict_recursion(self):
        """Dicts are recursively processed."""
        result = _make_json_serializable({"a": 1, "b": {"c": 2}})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_toolcall_serialization(self):
        """ToolCall objects are serialized via model_dump."""
        tc = ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"})
        result = _make_json_serializable(tc)

        assert isinstance(result, dict)
        assert result["tool_name"] == "get_price"
        assert result["arguments"] == {"ticker": "AAPL"}

    def test_nested_toolcalls(self):
        """Nested ToolCalls in dicts are serialized."""
        data = {
            "raw_output": '{"foo": "bar"}',
            "parsed_tool_calls": (
                ToolCall(tool_name="tool1", arguments={"a": 1}),
                ToolCall(tool_name="tool2", arguments={"b": 2}),
            ),
        }
        result = _make_json_serializable(data)

        assert result["raw_output"] == '{"foo": "bar"}'
        assert len(result["parsed_tool_calls"]) == 2
        assert result["parsed_tool_calls"][0]["tool_name"] == "tool1"
        assert result["parsed_tool_calls"][1]["tool_name"] == "tool2"

        # Verify it's JSON serializable
        json.dumps(result)  # Should not raise

    def test_dataclass_serialization(self):
        """Dataclasses are serialized."""

        @dataclass
        class SimpleData:
            name: str
            value: int

        obj = SimpleData(name="test", value=42)
        result = _make_json_serializable(obj)

        assert result == {"name": "test", "value": 42}

    def test_non_serializable_fallback(self):
        """Non-serializable objects fall back to string representation."""

        class CustomObj:
            def __str__(self):
                return "custom_string"

        result = _make_json_serializable(CustomObj())
        assert result == "custom_string"


class TestEvaluatorOutputSerialization:
    """Tests that evaluator properly serializes generated_output."""

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        return Evaluator(pack=NL2APIPack())

    @pytest.fixture
    def test_case(self) -> TestCase:
        return TestCase(
            id="serial-test-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

    @pytest.mark.asyncio
    async def test_generated_output_is_serializable(self, evaluator, test_case):
        """Evaluator ensures generated_output is JSON-serializable."""
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        scorecard = await evaluator.evaluate(test_case, system_output)

        # generated_output should be serializable
        serialized = json.dumps(scorecard.generated_output)
        assert serialized is not None

        # Should contain the raw_output
        assert "raw_output" in scorecard.generated_output

    @pytest.mark.asyncio
    async def test_toolcalls_in_output_are_serialized(self, evaluator, test_case):
        """ToolCall objects in system_output are serialized."""
        # Simulate what NL2API pack's syntax stage does
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]),
            "parsed_tool_calls": (ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        }

        scorecard = await evaluator.evaluate(test_case, system_output)

        # Should be JSON-serializable
        serialized = json.dumps(scorecard.generated_output)
        assert serialized is not None

        # parsed_tool_calls should be converted to dicts
        parsed = scorecard.generated_output.get("parsed_tool_calls")
        assert parsed is not None
        assert isinstance(parsed, list)
        assert parsed[0]["tool_name"] == "get_price"
