#!/usr/bin/env python3
"""
Medium-scale test for temporal evaluation framework.

Generates synthetic responses with absolute dates and validates
that temporal comparison correctly matches them to relative dates.
"""

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from CONTRACTS import (
    EvaluationConfig,
    SystemResponse,
    TemporalValidationMode,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.evaluation.core.evaluators import WaterfallEvaluator
from src.evaluation.core.temporal import DateResolver


def load_temporal_fixtures(limit: int = 250) -> list[dict]:
    """Load temporal test cases from fixtures."""
    fixture_path = Path("tests/fixtures/lseg/generated/temporal/temporal.json")
    with open(fixture_path) as f:
        data = json.load(f)
    return data["test_cases"][:limit]


def convert_to_test_case(tc_dict: dict) -> TestCase:
    """Convert dict to TestCase, handling missing fields."""
    return TestCase(
        id=tc_dict["id"],
        nl_query=tc_dict["nl_query"],
        expected_tool_calls=tuple(
            ToolCall(tool_name=tc["tool_name"], arguments=tc["arguments"])
            for tc in tc_dict["expected_tool_calls"]
        ),
        metadata=TestCaseMetadata(
            api_version=tc_dict.get("metadata", {}).get("api_version", "1.0.0"),
            complexity_level=tc_dict.get("metadata", {}).get("complexity_level", 1),
            source="temporal_fixture",
        ),
    )


def generate_absolute_response(
    test_case: TestCase,
    resolver: DateResolver,
) -> SystemResponse:
    """
    Generate a synthetic response with absolute dates.

    Takes the expected tool calls and converts any relative date
    expressions to absolute dates.
    """
    date_fields = {"start", "end", "SDate", "EDate", "Period"}

    tool_calls = []
    for tc in test_case.expected_tool_calls:
        args = dict(tc.arguments)
        for field in date_fields:
            if field in args:
                value = args[field]
                if isinstance(value, str):
                    # Normalize to absolute date
                    normalized = resolver.normalize(value)
                    args[field] = normalized
        tool_calls.append({"tool_name": tc.tool_name, "arguments": args})

    return SystemResponse(
        raw_output=json.dumps(tool_calls),
        latency_ms=10,
    )


async def run_temporal_test(limit: int = 250):
    """Run medium-scale temporal evaluation test."""
    print(f"Loading {limit} temporal test cases...")
    fixtures = load_temporal_fixtures(limit)
    print(f"Loaded {len(fixtures)} test cases")

    # Configure evaluator with temporal mode
    eval_date = date(2026, 1, 22)
    config = EvaluationConfig(
        temporal_mode=TemporalValidationMode.STRUCTURAL,
        evaluation_date=eval_date,
    )
    evaluator = WaterfallEvaluator(config=config)
    resolver = DateResolver(reference_date=eval_date)

    print(f"\nRunning evaluation...")
    print(f"  Reference date: {eval_date}")
    print(f"  Temporal mode: STRUCTURAL")
    print()

    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "syntax_failed": 0,
        "logic_failed": 0,
        "errors": [],
    }

    for i, tc_dict in enumerate(fixtures):
        try:
            test_case = convert_to_test_case(tc_dict)
            response = generate_absolute_response(test_case, resolver)
            scorecard = await evaluator.evaluate(test_case, response, "test-worker")

            results["total"] += 1
            if scorecard.overall_passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                if not scorecard.syntax_result.passed:
                    results["syntax_failed"] += 1
                elif scorecard.logic_result and not scorecard.logic_result.passed:
                    results["logic_failed"] += 1
                    if len(results["errors"]) < 5:
                        results["errors"].append({
                            "id": test_case.id,
                            "query": test_case.nl_query[:50],
                            "reason": scorecard.logic_result.reason,
                        })

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(fixtures)}...")

        except Exception as e:
            results["total"] += 1
            results["failed"] += 1
            if len(results["errors"]) < 5:
                results["errors"].append({
                    "id": tc_dict.get("id", "unknown"),
                    "error": str(e),
                })

    # Print results
    print("\n" + "=" * 60)
    print("TEMPORAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total:        {results['total']}")
    print(f"Passed:       {results['passed']}")
    print(f"Failed:       {results['failed']}")
    print(f"  - Syntax:   {results['syntax_failed']}")
    print(f"  - Logic:    {results['logic_failed']}")
    print(f"Pass Rate:    {results['passed']/results['total']*100:.1f}%")

    if results["errors"]:
        print("\nSample Failures:")
        for err in results["errors"]:
            print(f"  - {err.get('id', 'unknown')}: {err.get('reason', err.get('error', 'unknown'))}")

    return results["passed"] == results["total"]


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    success = asyncio.run(run_temporal_test(limit))
    sys.exit(0 if success else 1)
