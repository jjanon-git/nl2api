"""
NL2API Evaluation End-to-End Tests

Tests the complete NL2API evaluation pipeline:
1. Test case creation
2. Batch evaluation execution
3. Scorecard verification
4. Results querying

These tests validate that the codebase separation refactor works
for end-to-end evaluation workflows.

Requires:
    - NL2API_ANTHROPIC_API_KEY for tests that run real LLM
    - Docker compose up for database tests (optional)
"""

import json

import pytest

# Use direct imports from new locations
from src.evalkit.contracts import (
    EvalContext,
    Scorecard,
    StageResult,
    TestCase,
    ToolCall,
)
from src.nl2api.evaluation import NL2APIPack

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nl2api_pack() -> NL2APIPack:
    """Create NL2API pack with default settings."""
    return NL2APIPack()


@pytest.fixture
def nl2api_pack_all_stages() -> NL2APIPack:
    """Create NL2API pack with all stages enabled."""
    return NL2APIPack(execution_enabled=True, semantics_enabled=True)


@pytest.fixture
def simple_test_case() -> TestCase:
    """Simple test case for price query."""
    return TestCase(
        id="e2e-simple-001",
        nl_query="Get Apple stock price",
        expected_tool_calls=(
            ToolCall(
                tool_name="get_data",
                arguments={"RICs": ["AAPL.O"], "fields": ["P"]},
            ),
        ),
        category="lookups",
        subcategory="single_field",
    )


@pytest.fixture
def multi_tool_test_case() -> TestCase:
    """Test case requiring multiple tool calls."""
    return TestCase(
        id="e2e-multi-001",
        nl_query="Compare Apple and Microsoft stock prices",
        expected_tool_calls=(
            ToolCall(
                tool_name="get_data",
                arguments={"RICs": ["AAPL.O"], "fields": ["P"]},
            ),
            ToolCall(
                tool_name="get_data",
                arguments={"RICs": ["MSFT.O"], "fields": ["P"]},
            ),
        ),
        category="comparisons",
        subcategory="two_stock",
    )


@pytest.fixture
def context() -> EvalContext:
    """Evaluation context for tests."""
    return EvalContext(
        batch_id="e2e-test-batch",
        worker_id="e2e-test-worker",
    )


# =============================================================================
# Pack Initialization E2E Tests
# =============================================================================


class TestPackInitializationE2E:
    """End-to-end tests for pack initialization."""

    def test_pack_creates_successfully(self, nl2api_pack):
        """Pack initializes without errors."""
        assert nl2api_pack is not None
        assert nl2api_pack.name == "nl2api"

    def test_pack_with_all_stages(self, nl2api_pack_all_stages):
        """Pack with all stages initializes correctly."""
        stages = nl2api_pack_all_stages.get_stages()
        stage_names = [s.name for s in stages]

        assert len(stages) == 4
        assert "syntax" in stage_names
        assert "logic" in stage_names
        assert "execution" in stage_names
        assert "semantics" in stage_names

    def test_pack_weights_configured(self, nl2api_pack):
        """Pack weights are configured correctly."""
        weights = nl2api_pack.get_default_weights()

        assert weights["syntax"] == 0.1
        assert weights["logic"] == 0.3
        assert weights["execution"] == 0.5
        assert weights["semantics"] == 0.1


# =============================================================================
# Evaluation Pipeline E2E Tests
# =============================================================================


class TestEvaluationPipelineE2E:
    """End-to-end tests for the evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_successful_evaluation(self, nl2api_pack, simple_test_case, context):
        """Successful evaluation produces valid scorecard."""
        system_output = {
            "raw_output": json.dumps(
                [{"tool_name": "get_data", "arguments": {"RICs": ["AAPL.O"], "fields": ["P"]}}]
            )
        }

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert isinstance(scorecard, Scorecard)
        assert scorecard.test_case_id == simple_test_case.id
        assert scorecard.pack_name == "nl2api"
        assert scorecard.batch_id == "e2e-test-batch"
        assert scorecard.worker_id == "e2e-test-worker"

    @pytest.mark.asyncio
    async def test_syntax_stage_passes(self, nl2api_pack, simple_test_case, context):
        """Syntax stage passes for valid JSON."""
        system_output = {
            "raw_output": json.dumps(
                [{"tool_name": "get_data", "arguments": {"RICs": ["AAPL.O"], "fields": ["P"]}}]
            )
        }

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert scorecard.syntax_result is not None
        assert scorecard.syntax_result.passed is True
        assert scorecard.syntax_result.stage_name == "syntax"

    @pytest.mark.asyncio
    async def test_logic_stage_passes_exact_match(self, nl2api_pack, simple_test_case, context):
        """Logic stage passes for exact match."""
        system_output = {
            "raw_output": json.dumps(
                [{"tool_name": "get_data", "arguments": {"RICs": ["AAPL.O"], "fields": ["P"]}}]
            )
        }

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert scorecard.logic_result is not None
        assert scorecard.logic_result.passed is True
        assert scorecard.logic_result.score == 1.0

    @pytest.mark.asyncio
    async def test_syntax_failure_stops_pipeline(self, nl2api_pack, simple_test_case, context):
        """Syntax failure (gate) stops the pipeline."""
        system_output = {"raw_output": "not valid json {{{"}

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert scorecard.syntax_result.passed is False
        assert scorecard.logic_result is None  # Not executed

    @pytest.mark.asyncio
    async def test_logic_failure_continues_pipeline(
        self, nl2api_pack_all_stages, simple_test_case, context
    ):
        """Logic failure (non-gate) allows pipeline to continue."""
        system_output = {
            "raw_output": json.dumps([{"tool_name": "wrong_tool", "arguments": {"wrong": "args"}}])
        }

        scorecard = await nl2api_pack_all_stages.evaluate(simple_test_case, system_output, context)

        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result.passed is False
        # Execution and semantics should still run
        assert scorecard.execution_result is not None
        assert scorecard.semantics_result is not None

    @pytest.mark.asyncio
    async def test_multi_tool_evaluation(self, nl2api_pack, multi_tool_test_case, context):
        """Evaluation handles multiple tool calls."""
        system_output = {
            "raw_output": json.dumps(
                [
                    {"tool_name": "get_data", "arguments": {"RICs": ["AAPL.O"], "fields": ["P"]}},
                    {"tool_name": "get_data", "arguments": {"RICs": ["MSFT.O"], "fields": ["P"]}},
                ]
            )
        }

        scorecard = await nl2api_pack.evaluate(multi_tool_test_case, system_output, context)

        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result.passed is True


# =============================================================================
# Scorecard Structure E2E Tests
# =============================================================================


class TestScorecardStructureE2E:
    """End-to-end tests for scorecard structure."""

    @pytest.mark.asyncio
    async def test_scorecard_has_stage_results(self, nl2api_pack, simple_test_case, context):
        """Scorecard contains stage results dictionary."""
        system_output = {
            "raw_output": json.dumps(
                [{"tool_name": "get_data", "arguments": {"RICs": ["AAPL.O"], "fields": ["P"]}}]
            )
        }

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert "syntax" in scorecard.stage_results
        assert "logic" in scorecard.stage_results
        assert isinstance(scorecard.stage_results["syntax"], StageResult)

    @pytest.mark.asyncio
    async def test_scorecard_has_weights(self, nl2api_pack, simple_test_case, context):
        """Scorecard contains stage weights."""
        system_output = {
            "raw_output": json.dumps(
                [{"tool_name": "get_data", "arguments": {"RICs": ["AAPL.O"], "fields": ["P"]}}]
            )
        }

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert scorecard.stage_weights is not None
        assert "syntax" in scorecard.stage_weights
        assert "logic" in scorecard.stage_weights

    @pytest.mark.asyncio
    async def test_scorecard_captures_latency(self, nl2api_pack, simple_test_case, context):
        """Scorecard tracks total latency."""
        system_output = {
            "raw_output": json.dumps(
                [{"tool_name": "get_data", "arguments": {"RICs": ["AAPL.O"], "fields": ["P"]}}]
            )
        }

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert scorecard.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_scorecard_captures_generated_output(
        self, nl2api_pack, simple_test_case, context
    ):
        """Scorecard captures the generated output."""
        system_output = {
            "raw_output": json.dumps(
                [{"tool_name": "get_data", "arguments": {"RICs": ["AAPL.O"], "fields": ["P"]}}]
            )
        }

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert scorecard.generated_output is not None
        assert "raw_output" in scorecard.generated_output


# =============================================================================
# Scoring Computation E2E Tests
# =============================================================================


class TestScoringE2E:
    """End-to-end tests for score computation."""

    def test_perfect_score_computation(self, nl2api_pack):
        """Compute perfect overall score."""
        stage_results = {
            "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
            "logic": StageResult(stage_name="logic", passed=True, score=1.0),
        }

        score = nl2api_pack.compute_overall_score(stage_results)

        assert score == 1.0

    def test_partial_score_computation(self, nl2api_pack):
        """Compute partial overall score."""
        stage_results = {
            "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
            "logic": StageResult(stage_name="logic", passed=True, score=0.5),
        }

        score = nl2api_pack.compute_overall_score(stage_results)

        # Weighted: (1.0*0.1 + 0.5*0.3) / (0.1 + 0.3) = 0.25 / 0.4 = 0.625
        assert 0.62 <= score <= 0.63

    def test_overall_passed_all_true(self, nl2api_pack):
        """Overall passed when all stages pass."""
        stage_results = {
            "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
            "logic": StageResult(stage_name="logic", passed=True, score=1.0),
        }

        passed = nl2api_pack.compute_overall_passed(stage_results)

        assert passed is True

    def test_overall_failed_any_false(self, nl2api_pack):
        """Overall failed when any stage fails."""
        stage_results = {
            "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
            "logic": StageResult(stage_name="logic", passed=False, score=0.5),
        }

        passed = nl2api_pack.compute_overall_passed(stage_results)

        assert passed is False


# =============================================================================
# Validation E2E Tests
# =============================================================================


class TestValidationE2E:
    """End-to-end tests for test case validation."""

    def test_valid_test_case(self, nl2api_pack, simple_test_case):
        """Valid test case passes validation."""
        errors = nl2api_pack.validate_test_case(simple_test_case)

        assert errors == []

    def test_missing_query_detected(self, nl2api_pack):
        """Missing query is detected."""
        test_case = TestCase(
            id="invalid-001",
            expected_tool_calls=(ToolCall(tool_name="test", arguments={}),),
        )

        errors = nl2api_pack.validate_test_case(test_case)

        assert len(errors) > 0
        assert any("nl_query" in e for e in errors)

    def test_missing_tool_calls_detected(self, nl2api_pack):
        """Missing tool calls is detected."""
        test_case = TestCase(
            id="invalid-002",
            nl_query="Test query",
        )

        errors = nl2api_pack.validate_test_case(test_case)

        assert len(errors) > 0
        assert any("tool_calls" in e for e in errors)


# =============================================================================
# Integration with Adapters (if available)
# =============================================================================


class TestAdapterIntegrationE2E:
    """Tests for integration with NL2API target adapters."""

    def test_adapter_importable(self):
        """Target adapter is importable."""
        from src.nl2api.evaluation.adapter import NL2APITargetAdapter

        assert NL2APITargetAdapter is not None

    def test_batch_adapter_importable(self):
        """Batch adapter is importable."""
        from src.nl2api.evaluation.adapter import NL2APIBatchAdapter

        assert NL2APIBatchAdapter is not None


# =============================================================================
# Registry Integration E2E Tests
# =============================================================================


class TestRegistryIntegrationE2E:
    """Tests for pack registry integration."""

    def test_pack_in_evalkit_registry(self):
        """NL2APIPack is registered in evalkit."""
        from src.evalkit.packs import get_pack

        pack = get_pack("nl2api")

        assert pack is not None
        assert pack.name == "nl2api"

    def test_pack_in_registry(self):
        """NL2APIPack is in pack registry."""
        from src.evalkit.packs import get_available_packs

        packs = get_available_packs()
        assert "nl2api" in packs
        pack = packs["nl2api"]()
        assert pack.name == "nl2api"


# =============================================================================
# Error Handling E2E Tests
# =============================================================================


class TestErrorHandlingE2E:
    """End-to-end tests for error handling."""

    @pytest.mark.asyncio
    async def test_empty_output_handled(self, nl2api_pack, simple_test_case, context):
        """Empty output is handled gracefully."""
        system_output = {"raw_output": "[]"}

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result.passed is False

    @pytest.mark.asyncio
    async def test_malformed_tool_call_handled(self, nl2api_pack, simple_test_case, context):
        """Malformed tool call is handled in syntax stage."""
        system_output = {"raw_output": json.dumps([{"not_a_tool": "call"}])}

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        assert scorecard.syntax_result.passed is False
        assert scorecard.syntax_result.error_code is not None

    @pytest.mark.asyncio
    async def test_missing_raw_output_handled(self, nl2api_pack, simple_test_case, context):
        """Missing raw_output key is handled."""
        system_output = {}

        scorecard = await nl2api_pack.evaluate(simple_test_case, system_output, context)

        # Empty string will fail JSON parse
        assert scorecard.syntax_result.passed is False
