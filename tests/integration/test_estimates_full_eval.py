"""
Full evaluation of EstimatesAgent against the complete estimates test case set.

This test loads all 582 estimates-tagged test cases and evaluates the
EstimatesAgent's ability to generate correct API calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.nl2api.agents.estimates import EstimatesAgent
from src.nl2api.agents.protocols import AgentContext
from src.nl2api.llm.protocols import (
    LLMResponse,
    LLMToolCall,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MockLLMProvider:
    """Mock LLM provider that returns appropriate tool calls."""

    model_name: str = "mock-model"

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        return LLMResponse(content="estimates")

    async def complete_with_retry(
        self, messages, tools=None, temperature=0.0, max_tokens=4096, max_retries=3
    ):
        # Extract user query from messages
        next((m.content for m in messages if m.role.value == "user"), "")

        # Return tool call based on query content
        if tools:
            return LLMResponse(
                tool_calls=(
                    LLMToolCall(
                        id="call-1",
                        name="get_data",
                        arguments={"RICs": ["AAPL.O"], "fields": ["TR.EPSMean"]},
                    ),
                ),
            )
        return LLMResponse(content="estimates")


@dataclass
class MockEntityResolver:
    """Mock entity resolver that extracts company names from queries.

    Uses pattern matching to extract company names and generate placeholder RICs.
    In production, this would call an external entity resolution API.
    """

    async def resolve(self, query: str) -> dict[str, str]:
        """Extract company names and generate placeholder RICs.

        Uses heuristics to find company names in queries:
        1. Look for patterns like "X's" (possessive)
        2. Look for "of X" patterns
        3. Look for common suffixes like Inc, Corp, Ltd, etc.
        """
        import re

        result = {}

        # Pattern 1: "Company's" possessive form
        possessive = re.findall(r"([A-Z][A-Za-z\s&\-\.]+?)(?:'s|'s)", query)
        for match in possessive:
            company = match.strip()
            if len(company) > 2:
                result[company] = self._generate_ric(company)

        # Pattern 2: "of Company" form
        of_pattern = re.findall(
            r"of\s+([A-Z][A-Za-z\s&\-\.]+?)(?:\s+(?:for|in|to|and|or|$)|\?|$)", query
        )
        for match in of_pattern:
            company = match.strip()
            if len(company) > 2 and company not in result:
                result[company] = self._generate_ric(company)

        # Pattern 3: Company suffixes
        suffix_pattern = re.findall(
            r"([A-Z][A-Za-z\s&\-\.]*?\s*(?:Inc|Corp|Ltd|Co|PLC|AG|SA|NV|SE|Group|Bank|Holdings?)\.?)",
            query,
        )
        for match in suffix_pattern:
            company = match.strip()
            if len(company) > 2 and company not in result:
                result[company] = self._generate_ric(company)

        # Pattern 4: Ticker symbols (all caps, 1-5 chars)
        tickers = re.findall(r"\b([A-Z]{1,5})\b", query)
        for ticker in tickers:
            # Skip common words
            if ticker not in (
                "I",
                "A",
                "THE",
                "AND",
                "OR",
                "FOR",
                "OF",
                "IN",
                "TO",
                "GET",
                "EPS",
                "PE",
            ):
                if ticker not in result:
                    result[ticker] = f"{ticker}.O"

        return result

    def _generate_ric(self, company: str) -> str:
        """Generate a placeholder RIC from company name."""
        # Extract first word/acronym for RIC
        words = company.replace(".", "").replace(",", "").split()
        if words:
            # Use first word, uppercase, max 4 chars
            ric_base = words[0].upper()[:4]
            return f"{ric_base}.O"
        return "UNKN.O"


def load_estimates_test_cases() -> list[dict]:
    """Load all estimates-tagged test cases from fixtures."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "lseg" / "generated"
    all_cases = []

    files = [
        fixtures_dir / "lookups" / "lookups.json",
        fixtures_dir / "complex" / "complex.json",
    ]

    for file_path in files:
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
                cases = data.get("test_cases", [])
                estimates = [tc for tc in cases if "estimates" in tc.get("tags", [])]
                all_cases.extend(estimates)

    # Also load hand-written fixtures
    lseg_dir = Path(__file__).parent.parent / "fixtures" / "lseg"
    for json_file in lseg_dir.glob("*.json"):
        if json_file.is_file():
            with open(json_file) as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict) and "estimates" in data.get("metadata", {}).get(
                        "tags", []
                    ):
                        all_cases.append(data)
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass  # Skip malformed or unexpected JSON files

    return all_cases


def normalize_field(field: str) -> str:
    """Normalize field code for comparison."""
    # Remove Period parameter for comparison
    field = field.upper()
    if "(" in field:
        field = field.split("(")[0]
    # Normalize common variations
    field = field.replace("TR.", "").replace(".", "")
    return field


def compare_fields(expected_fields: list[str], actual_fields: list[str]) -> dict:
    """Compare expected vs actual fields."""
    expected_normalized = {normalize_field(f) for f in expected_fields}
    actual_normalized = {normalize_field(f) for f in actual_fields}

    matching = expected_normalized & actual_normalized
    missing = expected_normalized - actual_normalized
    extra = actual_normalized - expected_normalized

    return {
        "matching": list(matching),
        "missing": list(missing),
        "extra": list(extra),
        "precision": len(matching) / len(actual_normalized) if actual_normalized else 0,
        "recall": len(matching) / len(expected_normalized) if expected_normalized else 0,
    }


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""

    test_id: str
    query: str
    expected_fields: list[str]
    actual_fields: list[str]
    field_match: bool
    field_comparison: dict
    had_entity: bool
    rule_based: bool
    error: str | None = None


@dataclass
class EvalSummary:
    """Summary of full evaluation run."""

    total_cases: int
    processed: int
    errors: int
    field_exact_match: int
    field_partial_match: int  # At least one field matches
    no_match: int
    avg_precision: float
    avg_recall: float
    rule_based_count: int
    llm_count: int


class TestEstimatesFullEvaluation:
    """Full evaluation against all estimates test cases."""

    @pytest.fixture
    def test_cases(self) -> list[dict]:
        """Load all estimates test cases."""
        return load_estimates_test_cases()

    @pytest.fixture
    def agent(self) -> EstimatesAgent:
        """Create agent with mock LLM."""
        return EstimatesAgent(llm=MockLLMProvider())

    @pytest.fixture
    def resolver(self) -> MockEntityResolver:
        """Create mock entity resolver."""
        return MockEntityResolver()

    def test_load_test_cases(self, test_cases) -> None:
        """Verify test cases are loaded."""
        assert len(test_cases) > 0, "No test cases loaded"
        logger.info(f"Loaded {len(test_cases)} estimates test cases")

    @pytest.mark.asyncio
    async def test_sample_evaluation(self, test_cases, agent, resolver) -> None:
        """Evaluate a sample of test cases."""
        sample_size = min(50, len(test_cases))
        sample = test_cases[:sample_size]

        results = []
        for tc in sample:
            result = await self._evaluate_single(tc, agent, resolver)
            results.append(result)

        # Calculate metrics
        field_matches = sum(1 for r in results if r.field_match)
        partial_matches = sum(1 for r in results if r.field_comparison.get("matching"))
        errors = sum(1 for r in results if r.error)

        logger.info(f"Sample evaluation ({sample_size} cases):")
        logger.info(
            f"  Exact field matches: {field_matches} ({100 * field_matches / sample_size:.1f}%)"
        )
        logger.info(
            f"  Partial matches: {partial_matches} ({100 * partial_matches / sample_size:.1f}%)"
        )
        logger.info(f"  Errors: {errors}")

        # At least some partial matches expected
        assert partial_matches > 0, "No field matches at all"

    @pytest.mark.asyncio
    async def test_full_evaluation(self, test_cases, agent, resolver) -> None:
        """Full evaluation of all test cases."""
        results: list[EvalResult] = []

        for i, tc in enumerate(test_cases):
            result = await self._evaluate_single(tc, agent, resolver)
            results.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(test_cases)} cases")

        summary = self._compute_summary(results)
        self._print_summary(summary, results)

        # Store results for analysis
        self._save_results(results, summary)

        # Assertions - report metrics, low threshold for CI
        assert summary.processed > 0
        # Rule-based recall should be meaningful
        rule_based_results = [r for r in results if r.rule_based]
        if rule_based_results:
            rule_recall = sum(
                r.field_comparison.get("recall", 0) for r in rule_based_results
            ) / len(rule_based_results)
            logger.info(f"Rule-based recall: {rule_recall:.2%}")
            assert rule_recall > 0.3, f"Rule-based recall too low: {rule_recall:.2%}"

    async def _evaluate_single(
        self,
        tc: dict,
        agent: EstimatesAgent,
        resolver: MockEntityResolver,
    ) -> EvalResult:
        """Evaluate a single test case."""
        query = tc.get("nl_query", "")
        test_id = tc.get("id", "unknown")

        # Extract expected fields
        expected_calls = tc.get("expected_tool_calls", [])
        expected_fields = []
        for call in expected_calls:
            args = call.get("arguments", {})
            expected_fields.extend(args.get("fields", []))

        try:
            # Resolve entities
            resolved = await resolver.resolve(query)

            # Build context
            context = AgentContext(
                query=query,
                resolved_entities=resolved,
            )

            # Try rule-based extraction
            rule_result = agent._try_rule_based_extraction(context)
            rule_based = rule_result is not None and rule_result.confidence >= 0.8

            if rule_based:
                result = rule_result
            else:
                result = await agent.process(context)

            # Extract actual fields
            actual_fields = []
            for tc_call in result.tool_calls:
                args = tc_call.arguments
                if isinstance(args, dict):
                    actual_fields.extend(args.get("fields", []))

            # Compare
            comparison = compare_fields(expected_fields, actual_fields)
            field_match = set(normalize_field(f) for f in expected_fields) == set(
                normalize_field(f) for f in actual_fields
            )

            return EvalResult(
                test_id=test_id,
                query=query,
                expected_fields=expected_fields,
                actual_fields=actual_fields,
                field_match=field_match,
                field_comparison=comparison,
                had_entity=bool(resolved),
                rule_based=rule_based,
            )

        except Exception as e:
            return EvalResult(
                test_id=test_id,
                query=query,
                expected_fields=expected_fields,
                actual_fields=[],
                field_match=False,
                field_comparison={},
                had_entity=False,
                rule_based=False,
                error=str(e),
            )

    def _compute_summary(self, results: list[EvalResult]) -> EvalSummary:
        """Compute evaluation summary."""
        total = len(results)
        errors = sum(1 for r in results if r.error)
        processed = total - errors

        exact_matches = sum(1 for r in results if r.field_match)
        partial_matches = sum(
            1 for r in results if r.field_comparison.get("matching") and not r.field_match
        )
        no_match = sum(1 for r in results if not r.field_comparison.get("matching") and not r.error)

        precisions = [r.field_comparison.get("precision", 0) for r in results if not r.error]
        recalls = [r.field_comparison.get("recall", 0) for r in results if not r.error]

        return EvalSummary(
            total_cases=total,
            processed=processed,
            errors=errors,
            field_exact_match=exact_matches,
            field_partial_match=partial_matches,
            no_match=no_match,
            avg_precision=sum(precisions) / len(precisions) if precisions else 0,
            avg_recall=sum(recalls) / len(recalls) if recalls else 0,
            rule_based_count=sum(1 for r in results if r.rule_based),
            llm_count=sum(1 for r in results if not r.rule_based and not r.error),
        )

    def _print_summary(self, summary: EvalSummary, results: list[EvalResult]) -> None:
        """Print evaluation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("ESTIMATES AGENT FULL EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total test cases: {summary.total_cases}")
        logger.info(f"Processed: {summary.processed}")
        logger.info(f"Errors: {summary.errors}")
        logger.info("")
        logger.info("Field Matching:")
        logger.info(
            f"  Exact match: {summary.field_exact_match} ({100 * summary.field_exact_match / summary.total_cases:.1f}%)"
        )
        logger.info(
            f"  Partial match: {summary.field_partial_match} ({100 * summary.field_partial_match / summary.total_cases:.1f}%)"
        )
        logger.info(
            f"  No match: {summary.no_match} ({100 * summary.no_match / summary.total_cases:.1f}%)"
        )
        logger.info("")
        logger.info(f"Average Precision: {summary.avg_precision:.2%}")
        logger.info(f"Average Recall: {summary.avg_recall:.2%}")
        logger.info("")
        logger.info(
            f"Rule-based: {summary.rule_based_count} ({100 * summary.rule_based_count / summary.total_cases:.1f}%)"
        )
        logger.info(
            f"LLM-based: {summary.llm_count} ({100 * summary.llm_count / summary.total_cases:.1f}%)"
        )
        logger.info("=" * 60)

        # Show some failures for analysis
        failures = [r for r in results if not r.field_match and not r.error][:5]
        if failures:
            logger.info("\nSample failures:")
            for r in failures:
                logger.info(f"  Query: {r.query[:60]}...")
                logger.info(f"  Expected: {r.expected_fields[:3]}")
                logger.info(f"  Actual: {r.actual_fields[:3]}")
                logger.info("")

    def _save_results(self, results: list[EvalResult], summary: EvalSummary) -> None:
        """Save results to file for analysis."""
        output_path = Path(__file__).parent / "estimates_eval_results.json"

        output = {
            "summary": {
                "total_cases": summary.total_cases,
                "processed": summary.processed,
                "errors": summary.errors,
                "exact_match": summary.field_exact_match,
                "partial_match": summary.field_partial_match,
                "no_match": summary.no_match,
                "avg_precision": summary.avg_precision,
                "avg_recall": summary.avg_recall,
                "rule_based_count": summary.rule_based_count,
                "llm_count": summary.llm_count,
            },
            "results": [
                {
                    "test_id": r.test_id,
                    "query": r.query,
                    "expected_fields": r.expected_fields,
                    "actual_fields": r.actual_fields,
                    "field_match": r.field_match,
                    "precision": r.field_comparison.get("precision", 0),
                    "recall": r.field_comparison.get("recall", 0),
                    "rule_based": r.rule_based,
                    "error": r.error,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")


# Quick test function for CLI usage
async def run_quick_eval():
    """Run a quick evaluation from CLI."""
    test_cases = load_estimates_test_cases()
    agent = EstimatesAgent(llm=MockLLMProvider())
    resolver = MockEntityResolver()

    test_class = TestEstimatesFullEvaluation()
    results = []

    for tc in test_cases[:100]:  # Quick sample
        result = await test_class._evaluate_single(tc, agent, resolver)
        results.append(result)

    summary = test_class._compute_summary(results)
    test_class._print_summary(summary, results)
    return summary


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_quick_eval())
