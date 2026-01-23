#!/usr/bin/env python3
"""
Run EstimatesAgent evaluation with real LLM.

This script evaluates the EstimatesAgent against the estimates test cases
using actual LLM API calls (Claude or OpenAI).

Usage:
    # Set API key
    export NL2API_ANTHROPIC_API_KEY="sk-ant-..."
    # Or for OpenAI:
    export NL2API_LLM_PROVIDER="openai"
    export NL2API_OPENAI_API_KEY="sk-..."

    # Run evaluation
    python scripts/run_estimates_eval.py --limit 100

    # Run with specific model
    python scripts/run_estimates_eval.py --model claude-sonnet-4-20250514 --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nl2api.agents.estimates import EstimatesAgent
from src.nl2api.agents.protocols import AgentContext
from src.nl2api.config import NL2APIConfig
from src.nl2api.llm.factory import create_llm_provider
from src.nl2api.resolution.resolver import ExternalEntityResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""

    test_id: str
    query: str
    expected_fields: list[str]
    actual_fields: list[str]
    field_match: bool
    field_comparison: dict
    resolved_entities: dict[str, str]
    confidence: float
    reasoning: str
    latency_ms: float
    error: str | None = None


@dataclass
class EvalSummary:
    """Summary of evaluation run."""

    total_cases: int
    processed: int
    errors: int
    exact_match: int
    partial_match: int
    no_match: int
    avg_precision: float
    avg_recall: float
    avg_latency_ms: float
    total_time_s: float
    model: str
    provider: str


def load_estimates_test_cases(limit: int | None = None) -> list[dict]:
    """Load estimates-tagged test cases from fixtures."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "lseg" / "generated"
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
    lseg_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "lseg"
    for json_file in lseg_dir.glob("*.json"):
        if json_file.is_file():
            with open(json_file) as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict) and "estimates" in data.get("metadata", {}).get(
                        "tags", []
                    ):
                        all_cases.append(data)
                except Exception:
                    pass

    if limit:
        all_cases = all_cases[:limit]

    return all_cases


def normalize_field(field_str: str) -> str:
    """Normalize field code for comparison."""
    field_str = field_str.upper()
    if "(" in field_str:
        field_str = field_str.split("(")[0]
    field_str = field_str.replace("TR.", "").replace(".", "")
    return field_str


def compare_fields(expected: list[str], actual: list[str]) -> dict:
    """Compare expected vs actual fields."""
    expected_norm = {normalize_field(f) for f in expected}
    actual_norm = {normalize_field(f) for f in actual}

    matching = expected_norm & actual_norm
    missing = expected_norm - actual_norm
    extra = actual_norm - expected_norm

    precision = len(matching) / len(actual_norm) if actual_norm else 0
    recall = len(matching) / len(expected_norm) if expected_norm else 0

    return {
        "matching": list(matching),
        "missing": list(missing),
        "extra": list(extra),
        "precision": precision,
        "recall": recall,
    }


async def evaluate_single(
    tc: dict,
    agent: EstimatesAgent,
    resolver: ExternalEntityResolver,
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

    start_time = time.perf_counter()

    try:
        # Resolve entities
        resolved = await resolver.resolve(query)

        # Build context
        context = AgentContext(
            query=query,
            resolved_entities=resolved,
        )

        # Process with agent
        result = await agent.process(context)

        latency_ms = (time.perf_counter() - start_time) * 1000

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
            resolved_entities=resolved,
            confidence=result.confidence,
            reasoning=result.reasoning or "",
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Error evaluating {test_id}: {e}")
        return EvalResult(
            test_id=test_id,
            query=query,
            expected_fields=expected_fields,
            actual_fields=[],
            field_match=False,
            field_comparison={},
            resolved_entities={},
            confidence=0.0,
            reasoning="",
            latency_ms=latency_ms,
            error=str(e),
        )


def compute_summary(
    results: list[EvalResult],
    model: str,
    provider: str,
    total_time: float,
) -> EvalSummary:
    """Compute evaluation summary."""
    total = len(results)
    errors = sum(1 for r in results if r.error)
    processed = total - errors

    exact_matches = sum(1 for r in results if r.field_match)
    partial_matches = sum(
        1
        for r in results
        if r.field_comparison.get("matching") and not r.field_match and not r.error
    )
    no_match = sum(1 for r in results if not r.field_comparison.get("matching") and not r.error)

    precisions = [r.field_comparison.get("precision", 0) for r in results if not r.error]
    recalls = [r.field_comparison.get("recall", 0) for r in results if not r.error]
    latencies = [r.latency_ms for r in results if not r.error]

    return EvalSummary(
        total_cases=total,
        processed=processed,
        errors=errors,
        exact_match=exact_matches,
        partial_match=partial_matches,
        no_match=no_match,
        avg_precision=sum(precisions) / len(precisions) if precisions else 0,
        avg_recall=sum(recalls) / len(recalls) if recalls else 0,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        total_time_s=total_time,
        model=model,
        provider=provider,
    )


def print_summary(summary: EvalSummary, results: list[EvalResult]) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 70)
    print("ESTIMATES AGENT EVALUATION RESULTS (Real LLM)")
    print("=" * 70)
    print(f"Provider: {summary.provider}")
    print(f"Model: {summary.model}")
    print(f"Total test cases: {summary.total_cases}")
    print(f"Processed: {summary.processed}")
    print(f"Errors: {summary.errors}")
    print()
    print("Field Matching:")
    print(
        f"  Exact match:   {summary.exact_match:4d} ({100 * summary.exact_match / summary.total_cases:5.1f}%)"
    )
    print(
        f"  Partial match: {summary.partial_match:4d} ({100 * summary.partial_match / summary.total_cases:5.1f}%)"
    )
    print(
        f"  No match:      {summary.no_match:4d} ({100 * summary.no_match / summary.total_cases:5.1f}%)"
    )
    print()
    print(f"Average Precision: {summary.avg_precision:.2%}")
    print(f"Average Recall:    {summary.avg_recall:.2%}")
    print()
    print(f"Average Latency: {summary.avg_latency_ms:.0f}ms per query")
    print(f"Total Time: {summary.total_time_s:.1f}s")
    print("=" * 70)

    # Show sample failures
    failures = [r for r in results if not r.field_match and not r.error][:5]
    if failures:
        print("\nSample failures:")
        for r in failures:
            print(f"\n  Query: {r.query[:70]}...")
            print(f"  Expected: {r.expected_fields[:3]}")
            print(f"  Actual:   {r.actual_fields[:3]}")
            print(f"  Missing:  {r.field_comparison.get('missing', [])[:3]}")

    # Show sample successes
    successes = [r for r in results if r.field_match][:3]
    if successes:
        print("\nSample successes:")
        for r in successes:
            print(f"\n  Query: {r.query[:70]}...")
            print(f"  Fields: {r.actual_fields[:3]}")
            print(f"  Confidence: {r.confidence:.2f}")


def save_results(
    results: list[EvalResult],
    summary: EvalSummary,
    output_path: Path,
) -> None:
    """Save results to JSON file."""
    output = {
        "summary": {
            "provider": summary.provider,
            "model": summary.model,
            "total_cases": summary.total_cases,
            "processed": summary.processed,
            "errors": summary.errors,
            "exact_match": summary.exact_match,
            "exact_match_pct": summary.exact_match / summary.total_cases * 100,
            "partial_match": summary.partial_match,
            "no_match": summary.no_match,
            "avg_precision": summary.avg_precision,
            "avg_recall": summary.avg_recall,
            "avg_latency_ms": summary.avg_latency_ms,
            "total_time_s": summary.total_time_s,
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
                "confidence": r.confidence,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Run EstimatesAgent evaluation with real LLM")
    parser.add_argument("--limit", type=int, default=50, help="Number of test cases to evaluate")
    parser.add_argument(
        "--provider", choices=["claude", "openai"], default=None, help="LLM provider"
    )
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    # Load configuration
    config = NL2APIConfig()

    # Override with CLI args
    if args.provider:
        config.llm_provider = args.provider
    if args.model:
        config.llm_model = args.model

    # Get API key
    try:
        api_key = config.get_llm_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nSet the appropriate environment variable:")
        print("  export NL2API_ANTHROPIC_API_KEY='sk-ant-...'")
        print("  # or")
        print("  export NL2API_OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print(f"Using {config.llm_provider} with model {config.llm_model}")

    # Create LLM provider
    llm = create_llm_provider(
        provider=config.llm_provider,
        api_key=api_key,
        model=config.llm_model,
    )

    # Create agent and resolver
    agent = EstimatesAgent(llm=llm)
    resolver = ExternalEntityResolver()

    # Load test cases
    test_cases = load_estimates_test_cases(limit=args.limit)
    print(f"Loaded {len(test_cases)} test cases")

    if not test_cases:
        print("No test cases found!")
        sys.exit(1)

    # Run evaluation
    results: list[EvalResult] = []
    start_time = time.perf_counter()

    for i, tc in enumerate(test_cases):
        result = await evaluate_single(tc, agent, resolver)
        results.append(result)

        # Progress update
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            eta = (len(test_cases) - i - 1) / rate if rate > 0 else 0
            successes = sum(1 for r in results if r.field_match)
            print(
                f"  [{i + 1}/{len(test_cases)}] {successes} exact matches, {elapsed:.1f}s elapsed, ETA {eta:.0f}s"
            )

    total_time = time.perf_counter() - start_time

    # Compute and print summary
    summary = compute_summary(results, config.llm_model, config.llm_provider, total_time)
    print_summary(summary, results)

    # Save results
    output_path = (
        Path(args.output)
        if args.output
        else Path(f"estimates_eval_{config.llm_provider}_{len(test_cases)}.json")
    )
    save_results(results, summary, output_path)


if __name__ == "__main__":
    asyncio.run(main())
