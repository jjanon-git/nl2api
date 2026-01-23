"""Tests for evaluators."""

import json

import pytest

from CONTRACTS import (
    ErrorCode,
    EvaluationConfig,
    EvaluationStage,
    SystemResponse,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.evaluation.core.evaluators import LogicEvaluator, SyntaxEvaluator, WaterfallEvaluator


class TestSyntaxEvaluator:
    """Test suite for SyntaxEvaluator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.evaluator = SyntaxEvaluator()

    def test_valid_json_single_call(self) -> None:
        """Test parsing valid JSON with single tool call."""
        raw_output = json.dumps([{"tool_name": "search", "arguments": {"query": "test"}}])

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is True
        assert result.stage == EvaluationStage.SYNTAX
        assert result.score == 1.0
        assert parsed is not None
        assert len(parsed) == 1
        assert parsed[0].tool_name == "search"

    def test_valid_json_multiple_calls(self) -> None:
        """Test parsing valid JSON with multiple tool calls."""
        raw_output = json.dumps(
            [
                {"tool_name": "func_a", "arguments": {"x": 1}},
                {"tool_name": "func_b", "arguments": {"y": 2}},
            ]
        )

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is True
        assert parsed is not None
        assert len(parsed) == 2

    def test_wrapper_format(self) -> None:
        """Test parsing wrapper format with tool_calls key."""
        raw_output = json.dumps({"tool_calls": [{"tool_name": "search", "arguments": {}}]})

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is True
        assert parsed is not None
        assert len(parsed) == 1

    def test_single_object_format(self) -> None:
        """Test parsing single tool call object (not in array)."""
        raw_output = json.dumps({"tool_name": "get_data", "arguments": {"id": 123}})

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is True
        assert parsed is not None
        assert len(parsed) == 1

    def test_name_field_alias(self) -> None:
        """Test that 'name' field works as alias for 'tool_name'."""
        raw_output = json.dumps([{"name": "search", "arguments": {"q": "test"}}])

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is True
        assert parsed is not None
        assert parsed[0].tool_name == "search"

    def test_invalid_json(self) -> None:
        """Test handling of invalid JSON."""
        raw_output = "{ not valid json"

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is False
        assert result.error_code == ErrorCode.SYNTAX_INVALID_JSON
        assert parsed is None

    def test_missing_tool_name(self) -> None:
        """Test handling of missing tool_name field."""
        raw_output = json.dumps(
            [
                {"arguments": {"x": 1}}  # Missing tool_name
            ]
        )

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is False
        assert result.error_code == ErrorCode.SYNTAX_SCHEMA_VIOLATION
        assert parsed is None

    def test_invalid_tool_name_type(self) -> None:
        """Test handling of non-string tool_name."""
        raw_output = json.dumps(
            [
                {"tool_name": 123, "arguments": {}}  # tool_name should be string
            ]
        )

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is False
        assert result.error_code == ErrorCode.SYNTAX_SCHEMA_VIOLATION

    def test_invalid_arguments_type(self) -> None:
        """Test handling of non-dict arguments."""
        raw_output = json.dumps([{"tool_name": "func", "arguments": "not a dict"}])

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is False
        assert result.error_code == ErrorCode.SYNTAX_SCHEMA_VIOLATION

    def test_default_empty_arguments(self) -> None:
        """Test that missing arguments defaults to empty dict."""
        raw_output = json.dumps(
            [
                {"tool_name": "no_args"}  # No arguments field
            ]
        )

        result, parsed = self.evaluator.evaluate(raw_output)

        assert result.passed is True
        assert parsed is not None
        assert parsed[0].arguments == {}

    def test_duration_tracked(self) -> None:
        """Test that duration is tracked."""
        raw_output = json.dumps([{"tool_name": "test", "arguments": {}}])

        result, _ = self.evaluator.evaluate(raw_output)

        assert result.duration_ms >= 0


class TestLogicEvaluator:
    """Test suite for LogicEvaluator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.evaluator = LogicEvaluator()

    def test_exact_match(self) -> None:
        """Test exact match passes."""
        expected = (ToolCall(tool_name="search", arguments={"q": "test"}),)
        actual = (ToolCall(tool_name="search", arguments={"q": "test"}),)

        result = self.evaluator.evaluate(expected, actual)

        assert result.passed is True
        assert result.stage == EvaluationStage.LOGIC
        assert result.score == 1.0

    def test_order_independent(self) -> None:
        """Test order-independent matching."""
        expected = (
            ToolCall(tool_name="a", arguments={}),
            ToolCall(tool_name="b", arguments={}),
        )
        actual = (
            ToolCall(tool_name="b", arguments={}),
            ToolCall(tool_name="a", arguments={}),
        )

        result = self.evaluator.evaluate(expected, actual)

        assert result.passed is True
        assert result.score == 1.0

    def test_missing_call_error_code(self) -> None:
        """Test missing call produces correct error code."""
        expected = (
            ToolCall(tool_name="a", arguments={}),
            ToolCall(tool_name="b", arguments={}),
        )
        actual = (ToolCall(tool_name="a", arguments={}),)

        result = self.evaluator.evaluate(expected, actual)

        assert result.passed is False
        assert result.error_code == ErrorCode.LOGIC_MISSING_CALL
        assert "missing_calls" in result.artifacts

    def test_extra_call_error_code(self) -> None:
        """Test extra call produces correct error code."""
        expected = (ToolCall(tool_name="a", arguments={}),)
        actual = (
            ToolCall(tool_name="a", arguments={}),
            ToolCall(tool_name="extra", arguments={}),
        )

        result = self.evaluator.evaluate(expected, actual)

        assert result.passed is False
        assert result.error_code == ErrorCode.LOGIC_EXTRA_CALL
        assert "extra_calls" in result.artifacts

    def test_arg_mismatch_error_code(self) -> None:
        """Test argument mismatch produces correct error code."""
        expected = (ToolCall(tool_name="func", arguments={"x": 1}),)
        actual = (ToolCall(tool_name="func", arguments={"x": 2}),)

        result = self.evaluator.evaluate(expected, actual)

        assert result.passed is False
        assert result.error_code == ErrorCode.LOGIC_ARG_MISMATCH
        assert "argument_diffs" in result.artifacts

    def test_type_coercion(self) -> None:
        """Test type coercion in logic evaluation."""
        expected = (ToolCall(tool_name="get", arguments={"id": 5}),)
        actual = (ToolCall(tool_name="get", arguments={"id": "5"}),)

        result = self.evaluator.evaluate(expected, actual)

        assert result.passed is True

    def test_duration_tracked(self) -> None:
        """Test that duration is tracked."""
        expected = (ToolCall(tool_name="a", arguments={}),)
        actual = (ToolCall(tool_name="a", arguments={}),)

        result = self.evaluator.evaluate(expected, actual)

        assert result.duration_ms >= 0


class TestWaterfallEvaluator:
    """Test suite for WaterfallEvaluator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = EvaluationConfig()
        self.evaluator = WaterfallEvaluator(self.config)

    def _make_test_case(
        self,
        tool_calls: list[ToolCall],
        nl_query: str = "Test query",
    ) -> TestCase:
        """Helper to create test case."""
        return TestCase(
            id="test-001",
            nl_query=nl_query,
            expected_tool_calls=tuple(tool_calls),
            expected_nl_response="Expected response",
            metadata=TestCaseMetadata(
                api_version="v1.0.0",
                complexity_level=1,
            ),
        )

    def _make_response(self, tool_calls: list[dict]) -> SystemResponse:
        """Helper to create system response."""
        return SystemResponse(
            raw_output=json.dumps(tool_calls),
            latency_ms=100,
        )

    @pytest.mark.asyncio
    async def test_full_pass(self) -> None:
        """Test full pass through pipeline."""
        test_case = self._make_test_case([ToolCall(tool_name="search", arguments={"q": "test"})])
        response = self._make_response([{"tool_name": "search", "arguments": {"q": "test"}}])

        scorecard = await self.evaluator.evaluate(test_case, response, "worker-1")

        assert scorecard.overall_passed is True
        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result is not None
        assert scorecard.logic_result.passed is True
        assert scorecard.worker_id == "worker-1"

    @pytest.mark.asyncio
    async def test_syntax_failure_halts_pipeline(self) -> None:
        """Test that syntax failure halts pipeline."""
        test_case = self._make_test_case([ToolCall(tool_name="func", arguments={})])
        response = SystemResponse(
            raw_output="invalid json {",
            latency_ms=50,
        )

        scorecard = await self.evaluator.evaluate(test_case, response, "worker-1")

        assert scorecard.overall_passed is False
        assert scorecard.syntax_result.passed is False
        assert scorecard.logic_result is None  # Pipeline halted

    @pytest.mark.asyncio
    async def test_logic_failure_continues(self) -> None:
        """Test that logic failure doesn't halt pipeline."""
        test_case = self._make_test_case([ToolCall(tool_name="expected_func", arguments={})])
        response = self._make_response([{"tool_name": "different_func", "arguments": {}}])

        scorecard = await self.evaluator.evaluate(test_case, response, "worker-1")

        assert scorecard.overall_passed is False
        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result is not None
        assert scorecard.logic_result.passed is False

    @pytest.mark.asyncio
    async def test_latency_tracked(self) -> None:
        """Test that total latency is tracked."""
        test_case = self._make_test_case([ToolCall(tool_name="func", arguments={})])
        response = self._make_response([{"tool_name": "func", "arguments": {}}])

        scorecard = await self.evaluator.evaluate(test_case, response, "worker-1")

        assert scorecard.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generated_calls_captured(self) -> None:
        """Test that generated tool calls are captured in scorecard."""
        test_case = self._make_test_case([ToolCall(tool_name="func", arguments={"x": 1})])
        response = self._make_response([{"tool_name": "func", "arguments": {"x": 1}}])

        scorecard = await self.evaluator.evaluate(test_case, response, "worker-1")

        assert scorecard.generated_tool_calls is not None
        assert len(scorecard.generated_tool_calls) == 1
        assert scorecard.generated_tool_calls[0].tool_name == "func"

    @pytest.mark.asyncio
    async def test_order_independent_matching(self) -> None:
        """Test order-independent matching in full pipeline."""
        test_case = self._make_test_case(
            [
                ToolCall(tool_name="first", arguments={}),
                ToolCall(tool_name="second", arguments={}),
            ]
        )
        # Response has calls in different order
        response = self._make_response(
            [
                {"tool_name": "second", "arguments": {}},
                {"tool_name": "first", "arguments": {}},
            ]
        )

        scorecard = await self.evaluator.evaluate(test_case, response, "worker-1")

        assert scorecard.overall_passed is True
        assert scorecard.logic_result.passed is True

    @pytest.mark.asyncio
    async def test_type_coercion_in_pipeline(self) -> None:
        """Test type coercion works through full pipeline."""
        test_case = self._make_test_case([ToolCall(tool_name="get", arguments={"id": 123})])
        # Response has string "123" instead of int 123
        response = self._make_response([{"tool_name": "get", "arguments": {"id": "123"}}])

        scorecard = await self.evaluator.evaluate(test_case, response, "worker-1")

        assert scorecard.overall_passed is True
