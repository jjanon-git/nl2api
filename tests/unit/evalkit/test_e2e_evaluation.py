"""
End-to-end tests for the evaluation framework.

Tests the complete flow from test case creation through evaluation to export.
"""

import json
import tempfile
from pathlib import Path

import pytest

from CONTRACTS import (
    EvalContext,
    TestCase,
    ToolCall,
)
from src.evalkit.core import (
    CSVExporter,
    Evaluator,
    EvaluatorConfig,
    JSONExporter,
    SummaryExporter,
)
from src.nl2api.evaluation import NL2APIPack

# =============================================================================
# E2E Test: NL2API Evaluation
# =============================================================================


class TestNL2APIE2E:
    """End-to-end test for NL2API evaluation."""

    @pytest.fixture
    def test_cases(self) -> list[TestCase]:
        """Create a variety of test cases."""
        return [
            # Pass case
            TestCase(
                id="e2e-pass-001",
                nl_query="Get Apple stock price",
                expected_tool_calls=(
                    ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),
                ),
            ),
            # Pass case with multiple calls
            TestCase(
                id="e2e-pass-002",
                nl_query="Compare Apple and Microsoft prices",
                expected_tool_calls=(
                    ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),
                    ToolCall(tool_name="get_price", arguments={"ticker": "MSFT"}),
                ),
            ),
            # Fail case - wrong tool
            TestCase(
                id="e2e-fail-tool",
                nl_query="Get Tesla volume",
                expected_tool_calls=(
                    ToolCall(tool_name="get_volume", arguments={"ticker": "TSLA"}),
                ),
            ),
            # Fail case - wrong arguments
            TestCase(
                id="e2e-fail-args",
                nl_query="Get Google price",
                expected_tool_calls=(
                    ToolCall(tool_name="get_price", arguments={"ticker": "GOOGL"}),
                ),
            ),
        ]

    @pytest.fixture
    def mock_target(self):
        """Mock target system that returns predefined responses."""

        class MockTarget:
            def __init__(self):
                self.responses = {
                    "e2e-pass-001": {
                        "raw_output": json.dumps(
                            [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]
                        )
                    },
                    "e2e-pass-002": {
                        "raw_output": json.dumps(
                            [
                                {"tool_name": "get_price", "arguments": {"ticker": "AAPL"}},
                                {"tool_name": "get_price", "arguments": {"ticker": "MSFT"}},
                            ]
                        )
                    },
                    "e2e-fail-tool": {
                        "raw_output": json.dumps(
                            [{"tool_name": "get_price", "arguments": {"ticker": "TSLA"}}]
                        )
                    },
                    "e2e-fail-args": {
                        "raw_output": json.dumps(
                            [{"tool_name": "get_price", "arguments": {"ticker": "GOOG"}}]
                        )
                    },
                }

            async def process(self, test_case):
                return self.responses.get(test_case.id, {"raw_output": "{}"})

        return MockTarget()

    @pytest.mark.asyncio
    async def test_complete_evaluation_flow(self, test_cases, mock_target):
        """Test complete flow: create evaluator, run batch, export results."""
        # 1. Create evaluator with NL2API pack
        pack = NL2APIPack()
        evaluator = Evaluator(pack=pack)

        # Verify pack name
        assert evaluator.pack_name == "nl2api"

        # 2. Validate test cases
        errors = evaluator.validate_test_cases(test_cases)
        assert len(errors) == 0, f"Unexpected validation errors: {errors}"

        # 3. Run batch evaluation
        context = EvalContext(batch_id="e2e-test-batch")
        scorecards = await evaluator.evaluate_batch(test_cases, mock_target, context)

        assert len(scorecards) == 4

        # 4. Verify results
        passed = [
            sc for sc in scorecards if all(sr.passed for sr in sc.get_all_stage_results().values())
        ]
        failed = [
            sc
            for sc in scorecards
            if not all(sr.passed for sr in sc.get_all_stage_results().values())
        ]

        assert len(passed) == 2, f"Expected 2 passes, got {len(passed)}"
        assert len(failed) == 2, f"Expected 2 failures, got {len(failed)}"

        # 5. Generate summary
        summary_exporter = SummaryExporter()
        summary = summary_exporter.summarize(scorecards)

        assert summary.total_tests == 4
        assert summary.passed_tests == 2
        assert summary.failed_tests == 2
        assert summary.pass_rate == 0.5

        # 6. Export to JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            json_exporter = JSONExporter()
            await json_exporter.export(scorecards, json_path)

            # Verify JSON
            with open(json_path) as f:
                data = json.load(f)
            assert data["total_scorecards"] == 4

            # 7. Export to CSV
            csv_path = Path(tmpdir) / "results.csv"
            csv_exporter = CSVExporter()
            await csv_exporter.export(scorecards, csv_path)

            # Verify CSV exists
            assert csv_path.exists()

        # 8. Format summary
        formatted = summary_exporter.format_summary(summary)
        assert "EVALUATION SUMMARY" in formatted
        assert "Pass" in formatted

    @pytest.mark.asyncio
    async def test_single_evaluation_with_details(self, test_cases, mock_target):
        """Test single evaluation with detailed result inspection."""
        pack = NL2APIPack()
        evaluator = Evaluator(pack=pack)

        # Evaluate single passing case
        test_case = test_cases[0]  # e2e-pass-001
        system_output = await mock_target.process(test_case)
        scorecard = await evaluator.evaluate(test_case, system_output)

        # Inspect results
        assert scorecard.test_case_id == test_case.id
        assert scorecard.pack_name == "nl2api"

        # Check all stages
        all_results = scorecard.get_all_stage_results()
        assert "syntax" in all_results
        assert "logic" in all_results

        syntax = all_results["syntax"]
        assert syntax.passed is True
        assert syntax.score == 1.0
        assert syntax.stage_name == "syntax"

        logic = all_results["logic"]
        assert logic.passed is True
        assert logic.score == 1.0

    @pytest.mark.asyncio
    async def test_gate_failure_handling(self):
        """Test that gate failures stop the pipeline."""
        pack = NL2APIPack()
        evaluator = Evaluator(
            pack=pack,
            config=EvaluatorConfig(validate_inputs=False),
        )

        test_case = TestCase(id="gate-test-001")
        system_output = {"raw_output": "not valid json"}

        scorecard = await evaluator.evaluate(test_case, system_output)

        # Syntax should fail (gate)
        all_results = scorecard.get_all_stage_results()
        assert "syntax" in all_results
        assert all_results["syntax"].passed is False

        # Logic should NOT be present (gate stopped pipeline)
        assert "logic" not in all_results


# =============================================================================
# E2E Test: Generic Pack Usage
# =============================================================================


class TestGenericPackE2E:
    """End-to-end test with generic pack usage patterns."""

    @pytest.mark.asyncio
    async def test_generic_test_case_format(self):
        """Test using generic input/expected format."""
        # Create test case using generic format
        test_case = TestCase.from_generic(
            id="generic-001",
            input={"nl_query": "Get Apple price"},
            expected={"tool_calls": [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]},
        )

        # Verify auto-population of NL2API fields
        assert test_case.nl_query == "Get Apple price"
        assert len(test_case.expected_tool_calls) == 1

        # Evaluate with NL2API pack
        pack = NL2APIPack()
        evaluator = Evaluator(pack=pack)

        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        scorecard = await evaluator.evaluate(test_case, system_output)

        # Should pass
        assert scorecard.pack_name == "nl2api"
        assert all(sr.passed for sr in scorecard.get_all_stage_results().values())

    @pytest.mark.asyncio
    async def test_to_generic_conversion(self):
        """Test converting NL2API test case to generic format."""
        # Create NL2API-style test case
        test_case = TestCase(
            id="nl2api-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
            expected_nl_response="Apple's price is $150",
        )

        # Convert to generic
        generic = test_case.to_generic()

        # Verify generic fields populated
        assert generic.input["nl_query"] == "Get Apple price"
        assert len(generic.expected["tool_calls"]) == 1
        assert generic.expected["nl_response"] == "Apple's price is $150"


# =============================================================================
# E2E Test: Callback and Streaming
# =============================================================================


class TestCallbackE2E:
    """End-to-end test for callback/streaming patterns."""

    @pytest.mark.asyncio
    async def test_on_result_callback(self):
        """Test that on_result callback is called for each result."""
        pack = NL2APIPack()
        evaluator = Evaluator(pack=pack)

        test_cases = [
            TestCase(
                id=f"callback-{i:03d}",
                nl_query="Test query",
                expected_tool_calls=(ToolCall(tool_name="test", arguments={}),),
            )
            for i in range(3)
        ]

        class MockTarget:
            async def process(self, test_case):
                return {"raw_output": json.dumps([{"tool_name": "test", "arguments": {}}])}

        results_received: list[str] = []

        async def on_result(scorecard):
            results_received.append(scorecard.test_case_id)

        await evaluator.evaluate_batch(test_cases, MockTarget(), on_result=on_result)

        # All results should be received
        assert len(results_received) == 3
        assert set(results_received) == {"callback-000", "callback-001", "callback-002"}
