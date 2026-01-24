"""
Unit tests for NL2API evaluation pack - Direct import validation.

Tests the NL2API pack by importing directly from the new application location
(src.nl2api.evaluation.pack) rather than the compatibility shim.

This validates that the codebase separation refactor works correctly.
"""

import json

import pytest

# Import contracts from evalkit
from src.evalkit.contracts import (
    EvalContext,
    StageResult,
    TestCase,
    ToolCall,
)

# Direct imports from new location (NOT compatibility shim)
from src.nl2api.evaluation.pack import (
    ExecutionStage,
    LogicStage,
    NL2APIPack,
    SemanticsStage,
    SyntaxStage,
)

# =============================================================================
# SyntaxStage Tests (Direct Import)
# =============================================================================


class TestSyntaxStageDirect:
    """Tests for SyntaxStage using direct imports."""

    @pytest.fixture
    def stage(self) -> SyntaxStage:
        return SyntaxStage()

    @pytest.fixture
    def test_case(self) -> TestCase:
        return TestCase(
            id="syntax-direct-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

    @pytest.fixture
    def context(self) -> EvalContext:
        return EvalContext()

    def test_stage_properties(self, stage):
        """Verify stage properties from direct import."""
        assert stage.name == "syntax"
        assert stage.is_gate is True

    @pytest.mark.asyncio
    async def test_valid_json_passes(self, stage, test_case, context):
        """Valid JSON with correct schema passes syntax check."""
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True
        assert result.score == 1.0
        assert result.stage_name == "syntax"
        assert "parsed_tool_calls" in system_output

    @pytest.mark.asyncio
    async def test_invalid_json_fails(self, stage, test_case, context):
        """Invalid JSON fails syntax check."""
        system_output = {"raw_output": "not valid json {"}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is False
        assert result.score == 0.0
        assert result.error_code is not None
        assert "Invalid JSON" in result.reason

    @pytest.mark.asyncio
    async def test_missing_tool_name_fails(self, stage, test_case, context):
        """Missing tool_name field fails syntax check."""
        system_output = {"raw_output": json.dumps([{"arguments": {"ticker": "AAPL"}}])}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is False
        assert "missing 'tool_name'" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_wrapper_format_accepted(self, stage, test_case, context):
        """Wrapper format with tool_calls key is accepted."""
        system_output = {
            "raw_output": json.dumps(
                {"tool_calls": [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]}
            )
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True
        assert len(system_output["parsed_tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_single_object_format_accepted(self, stage, test_case, context):
        """Single tool call object (not array) is accepted."""
        system_output = {
            "raw_output": json.dumps({"tool_name": "get_price", "arguments": {"ticker": "AAPL"}})
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True
        assert len(system_output["parsed_tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_name_field_accepted(self, stage, test_case, context):
        """'name' field is accepted as alternative to 'tool_name'."""
        system_output = {
            "raw_output": json.dumps([{"name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True


# =============================================================================
# LogicStage Tests (Direct Import)
# =============================================================================


class TestLogicStageDirect:
    """Tests for LogicStage using direct imports."""

    @pytest.fixture
    def stage(self) -> LogicStage:
        return LogicStage()

    @pytest.fixture
    def test_case(self) -> TestCase:
        return TestCase(
            id="logic-direct-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

    @pytest.fixture
    def context(self) -> EvalContext:
        return EvalContext()

    def test_stage_properties(self, stage):
        """Verify stage properties from direct import."""
        assert stage.name == "logic"
        assert stage.is_gate is False

    @pytest.mark.asyncio
    async def test_exact_match_passes(self, stage, test_case, context):
        """Exact match of expected and actual tool calls passes."""
        system_output = {
            "parsed_tool_calls": (ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),)
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True
        assert result.score == 1.0
        assert result.stage_name == "logic"

    @pytest.mark.asyncio
    async def test_missing_call_fails(self, stage, test_case, context):
        """Missing expected tool call fails."""
        system_output = {"parsed_tool_calls": ()}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is False
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_extra_call_fails(self, stage, test_case, context):
        """Extra unexpected tool call fails."""
        system_output = {
            "parsed_tool_calls": (
                ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),
                ToolCall(tool_name="get_volume", arguments={"ticker": "AAPL"}),
            )
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_wrong_arguments_fails(self, stage, test_case, context):
        """Wrong arguments fail."""
        system_output = {
            "parsed_tool_calls": (ToolCall(tool_name="get_price", arguments={"ticker": "MSFT"}),)
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_uses_generic_expected_if_no_specific(self, stage, context):
        """Uses expected.tool_calls if expected_tool_calls is empty."""
        test_case = TestCase(
            id="logic-direct-002",
            input={"nl_query": "Get Apple price"},
            expected={"tool_calls": [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]},
        )

        system_output = {
            "parsed_tool_calls": (ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),)
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True


# =============================================================================
# ExecutionStage Tests (Direct Import)
# =============================================================================


class TestExecutionStageDirect:
    """Tests for ExecutionStage using direct imports."""

    @pytest.fixture
    def stage(self) -> ExecutionStage:
        return ExecutionStage()

    @pytest.fixture
    def test_case(self) -> TestCase:
        return TestCase(id="exec-direct-001", nl_query="test")

    @pytest.fixture
    def context(self) -> EvalContext:
        return EvalContext()

    def test_stage_properties(self, stage):
        """Verify stage properties from direct import."""
        assert stage.name == "execution"
        assert stage.is_gate is False

    @pytest.mark.asyncio
    async def test_deferred_returns_pass(self, stage, test_case, context):
        """Deferred execution stage returns pass."""
        result = await stage.evaluate(test_case, {}, context)

        assert result.passed is True
        assert result.score == 1.0
        assert "deferred" in result.reason.lower()


# =============================================================================
# SemanticsStage Tests (Direct Import)
# =============================================================================


class TestSemanticsStageDirect:
    """Tests for SemanticsStage using direct imports."""

    @pytest.fixture
    def stage(self) -> SemanticsStage:
        return SemanticsStage()

    @pytest.fixture
    def context(self) -> EvalContext:
        return EvalContext()

    def test_stage_properties(self, stage):
        """Verify stage properties from direct import."""
        assert stage.name == "semantics"
        assert stage.is_gate is False

    @pytest.mark.asyncio
    async def test_skips_when_no_expected_nl(self, stage, context):
        """Skips when no expected NL response."""
        test_case = TestCase(id="sem-direct-001", nl_query="test")
        system_output = {"nl_response": "Some response"}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True
        assert "skipped" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_skips_when_no_actual_nl(self, stage, context):
        """Skips when no actual NL response."""
        test_case = TestCase(
            id="sem-direct-002",
            nl_query="test",
            expected_nl_response="Expected response",
        )
        system_output = {}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True
        assert "skipped" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_skips_when_no_judge(self, stage, context):
        """Skips when no LLM judge configured."""
        test_case = TestCase(
            id="sem-direct-003",
            nl_query="test",
            expected_nl_response="Expected response",
        )
        system_output = {"nl_response": "Actual response"}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is True
        assert "skipped" in result.reason.lower()


# =============================================================================
# NL2APIPack Tests (Direct Import)
# =============================================================================


class TestNL2APIPackDirect:
    """Tests for NL2APIPack using direct imports."""

    @pytest.fixture
    def pack(self) -> NL2APIPack:
        return NL2APIPack()

    @pytest.fixture
    def pack_with_all_stages(self) -> NL2APIPack:
        return NL2APIPack(execution_enabled=True, semantics_enabled=True)

    @pytest.fixture
    def test_case(self) -> TestCase:
        return TestCase(
            id="pack-direct-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

    def test_pack_name(self, pack):
        """Pack name is 'nl2api'."""
        assert pack.name == "nl2api"

    def test_default_stages(self, pack):
        """Default pack has syntax and logic stages."""
        stages = pack.get_stages()
        stage_names = [s.name for s in stages]

        assert stage_names == ["syntax", "logic"]

    def test_all_stages_when_enabled(self, pack_with_all_stages):
        """All stages present when enabled."""
        stages = pack_with_all_stages.get_stages()
        stage_names = [s.name for s in stages]

        assert stage_names == ["syntax", "logic", "execution", "semantics"]

    def test_default_weights(self, pack):
        """Default weights are defined."""
        weights = pack.get_default_weights()

        assert "syntax" in weights
        assert "logic" in weights
        assert "execution" in weights
        assert "semantics" in weights
        assert sum(weights.values()) == 1.0

    def test_validate_test_case_valid(self, pack, test_case):
        """Valid test case has no errors."""
        errors = pack.validate_test_case(test_case)
        assert errors == []

    def test_validate_test_case_missing_query(self, pack):
        """Missing nl_query produces error."""
        test_case = TestCase(
            id="invalid-direct-001",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

        errors = pack.validate_test_case(test_case)
        assert len(errors) > 0
        assert any("nl_query" in e for e in errors)

    def test_validate_test_case_missing_tool_calls(self, pack):
        """Missing expected_tool_calls produces error."""
        test_case = TestCase(id="invalid-direct-002", nl_query="Get price")

        errors = pack.validate_test_case(test_case)
        assert len(errors) > 0
        assert any("tool_calls" in e for e in errors)

    def test_validate_test_case_accepts_generic_format(self, pack):
        """Accepts generic input/expected format."""
        test_case = TestCase(
            id="generic-direct-001",
            input={"nl_query": "Get Apple price"},
            expected={"tool_calls": [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]},
        )

        errors = pack.validate_test_case(test_case)
        assert errors == []

    def test_compute_overall_score_weighted(self, pack):
        """Overall score is weighted average."""
        stage_results = {
            "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
            "logic": StageResult(stage_name="logic", passed=True, score=0.8),
        }

        score = pack.compute_overall_score(stage_results)

        # With default weights: syntax=0.1, logic=0.3
        # Expected: (1.0*0.1 + 0.8*0.3) / (0.1 + 0.3) = 0.34 / 0.4 = 0.85
        assert 0.84 <= score <= 0.86

    def test_compute_overall_passed_all_pass(self, pack):
        """All stages passing means overall passed."""
        stage_results = {
            "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
            "logic": StageResult(stage_name="logic", passed=True, score=0.8),
        }

        passed = pack.compute_overall_passed(stage_results)
        assert passed is True

    def test_compute_overall_passed_any_fail(self, pack):
        """Any stage failing means overall failed."""
        stage_results = {
            "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
            "logic": StageResult(stage_name="logic", passed=False, score=0.5),
        }

        passed = pack.compute_overall_passed(stage_results)
        assert passed is False

    @pytest.mark.asyncio
    async def test_evaluate_full_pipeline(self, pack, test_case):
        """Full evaluation pipeline produces valid scorecard."""
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        scorecard = await pack.evaluate(test_case, system_output)

        assert scorecard.test_case_id == test_case.id
        assert scorecard.pack_name == "nl2api"
        assert "syntax" in scorecard.stage_results
        assert "logic" in scorecard.stage_results
        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_gate_failure_stops_pipeline(self, pack, test_case):
        """Gate failure (syntax) stops pipeline."""
        system_output = {"raw_output": "invalid json"}

        scorecard = await pack.evaluate(test_case, system_output)

        assert scorecard.syntax_result.passed is False
        # Logic stage should not have been run
        assert scorecard.logic_result is None

    @pytest.mark.asyncio
    async def test_evaluate_includes_context(self, pack, test_case):
        """Evaluation context is passed to stages."""
        context = EvalContext(
            batch_id="test-batch-direct-001",
            worker_id="test-worker",
        )
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        scorecard = await pack.evaluate(test_case, system_output, context)

        assert scorecard.batch_id == "test-batch-direct-001"
        assert scorecard.worker_id == "test-worker"

    @pytest.mark.asyncio
    async def test_evaluate_with_multiple_tool_calls(self, pack):
        """Evaluation handles multiple tool calls."""
        test_case = TestCase(
            id="multi-tool-001",
            nl_query="Get Apple and Microsoft prices",
            expected_tool_calls=(
                ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),
                ToolCall(tool_name="get_price", arguments={"ticker": "MSFT"}),
            ),
        )
        system_output = {
            "raw_output": json.dumps(
                [
                    {"tool_name": "get_price", "arguments": {"ticker": "AAPL"}},
                    {"tool_name": "get_price", "arguments": {"ticker": "MSFT"}},
                ]
            )
        }

        scorecard = await pack.evaluate(test_case, system_output)

        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result.passed is True


# =============================================================================
# Import Verification Tests
# =============================================================================


class TestDirectImportVerification:
    """Verify that direct imports work correctly."""

    def test_stages_importable(self):
        """All stages are importable from new location."""
        from src.nl2api.evaluation.pack import (
            ExecutionStage,
            LogicStage,
            SemanticsStage,
            SyntaxStage,
        )

        assert SyntaxStage is not None
        assert LogicStage is not None
        assert ExecutionStage is not None
        assert SemanticsStage is not None

    def test_pack_importable(self):
        """NL2APIPack is importable from new location."""
        from src.nl2api.evaluation.pack import NL2APIPack

        assert NL2APIPack is not None

    def test_pack_from_evaluation_init(self):
        """NL2APIPack is importable from evaluation __init__."""
        from src.nl2api.evaluation import NL2APIPack

        assert NL2APIPack is not None
        assert NL2APIPack().name == "nl2api"

    def test_stage_instances_are_correct_types(self):
        """Verify stage instances have correct types."""
        pack = NL2APIPack(execution_enabled=True, semantics_enabled=True)
        stages = pack.get_stages()

        assert isinstance(stages[0], SyntaxStage)
        assert isinstance(stages[1], LogicStage)
        assert isinstance(stages[2], ExecutionStage)
        assert isinstance(stages[3], SemanticsStage)

    def test_evalkit_contracts_work_with_pack(self):
        """Verify evalkit contracts work with NL2API pack."""
        from src.evalkit.contracts import TestCase, ToolCall

        test_case = TestCase(
            id="verify-001",
            nl_query="test",
            expected_tool_calls=(ToolCall(tool_name="test", arguments={}),),
        )

        pack = NL2APIPack()
        errors = pack.validate_test_case(test_case)

        assert errors == []
