#!/usr/bin/env python3
"""
Run NL2API evaluation with Haiku across all test fixtures.

Usage:
    # First time: set your API key
    export NL2API_ANTHROPIC_API_KEY="sk-ant-api03-..."

    # Or create a .env file with:
    # NL2API_ANTHROPIC_API_KEY=sk-ant-api03-...

    # Run evaluation (start with a small sample)
    python scripts/run_haiku_eval.py --limit 100

    # Run full evaluation (~12,887 test cases)
    python scripts/run_haiku_eval.py

    # Run specific category only
    python scripts/run_haiku_eval.py --category lookups --limit 500
"""

import asyncio
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file manually (no external dependency)
def load_env_file():
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value

load_env_file()


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    test_id: str
    query: str
    category: str
    passed: bool
    expected_function: str
    actual_function: str | None
    error: str | None = None
    latency_ms: float = 0


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    by_category: dict = field(default_factory=dict)
    failed_tests: list = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    total_tokens: int = 0

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0


async def run_evaluation(
    api_key: str,
    limit: int | None = None,
    category: str | None = None,
    concurrency: int = 5,
    verbose: bool = False,
) -> EvalSummary:
    """Run the evaluation."""

    # Import here to avoid issues if deps missing
    from src.nl2api.llm.factory import create_llm_provider
    from src.nl2api.agents.datastream import DatastreamAgent
    from src.nl2api.agents.estimates import EstimatesAgent
    from src.nl2api.agents.screening import ScreeningAgent
    from src.nl2api.agents.fundamentals import FundamentalsAgent
    from src.nl2api.agents.officers import OfficersAgent
    from src.nl2api.orchestrator import NL2APIOrchestrator
    from tests.unit.nl2api.fixture_loader import FixtureLoader, GeneratedTestCase

    print("=" * 60)
    print("NL2API Haiku Evaluation")
    print("=" * 60)

    # 1. Create Haiku LLM provider
    print("\n[1/4] Setting up Haiku LLM...")
    llm = create_llm_provider(
        provider="claude",
        api_key=api_key,
        model="claude-3-haiku-20240307",
    )
    print(f"  Model: claude-3-haiku-20240307")

    # 2. Create domain agents
    print("\n[2/4] Initializing domain agents...")
    agents = {
        "datastream": DatastreamAgent(llm=llm),
        "estimates": EstimatesAgent(llm=llm),
        "screening": ScreeningAgent(llm=llm),
        "fundamentals": FundamentalsAgent(llm=llm),
        "officers": OfficersAgent(llm=llm),
    }
    print(f"  Agents: {', '.join(agents.keys())}")

    # 3. Create orchestrator
    print("\n[3/4] Creating orchestrator...")
    orchestrator = NL2APIOrchestrator(llm=llm, agents=agents)

    # 4. Load test fixtures
    print("\n[4/4] Loading test fixtures...")
    loader = FixtureLoader()

    if category:
        test_cases = loader.load_category(category)
        print(f"  Category: {category}")
    else:
        test_cases = list(loader.iterate_all())
        print(f"  Categories: {', '.join(loader.CATEGORIES)}")

    if limit:
        test_cases = test_cases[:limit]

    print(f"  Test cases: {len(test_cases)}")
    print(f"  Concurrency: {concurrency}")

    # Estimate cost
    estimated_tokens = len(test_cases) * 1000  # ~1k tokens per test
    estimated_cost = estimated_tokens / 1_000_000 * 0.25  # Haiku input pricing
    print(f"  Estimated cost: ~${estimated_cost:.2f}")

    print("\n" + "-" * 60)
    print("Starting evaluation...")
    print("-" * 60 + "\n")

    # Run evaluation
    summary = EvalSummary(total=len(test_cases))
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_one(tc: GeneratedTestCase) -> EvalResult:
        """Evaluate a single test case."""
        async with semaphore:
            start = time.perf_counter()
            try:
                # Call the orchestrator
                response = await orchestrator.process(tc.nl_query)
                latency = (time.perf_counter() - start) * 1000

                # Check if we got tool calls
                if not response.tool_calls:
                    return EvalResult(
                        test_id=tc.id,
                        query=tc.nl_query,
                        category=tc.category,
                        passed=False,
                        expected_function=tc.expected_tool_calls[0].get("function", "unknown") if tc.expected_tool_calls else "unknown",
                        actual_function=None,
                        error="No tool calls returned",
                        latency_ms=latency,
                    )

                # Compare tool calls
                actual_func = response.tool_calls[0].tool_name
                expected_func = tc.expected_tool_calls[0].get("function", "") if tc.expected_tool_calls else ""

                # Normalize using ToolRegistry (single source of truth)
                from CONTRACTS import ToolRegistry
                actual_normalized = ToolRegistry.normalize(actual_func)
                expected_normalized = ToolRegistry.normalize(expected_func)

                passed = actual_normalized == expected_normalized

                # Also check key arguments if function matches
                if passed and tc.expected_tool_calls:
                    expected_args = tc.expected_tool_calls[0].get("arguments", {})
                    actual_args = dict(response.tool_calls[0].arguments)

                    # Check fields match
                    expected_fields = set(expected_args.get("fields", []))
                    actual_fields = set(actual_args.get("fields", []))
                    if expected_fields and actual_fields != expected_fields:
                        passed = False

                return EvalResult(
                    test_id=tc.id,
                    query=tc.nl_query,
                    category=tc.category,
                    passed=passed,
                    expected_function=expected_func,
                    actual_function=actual_func,
                    latency_ms=latency,
                )

            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                return EvalResult(
                    test_id=tc.id,
                    query=tc.nl_query,
                    category=tc.category,
                    passed=False,
                    expected_function=tc.expected_tool_calls[0].get("function", "unknown") if tc.expected_tool_calls else "unknown",
                    actual_function=None,
                    error=str(e),
                    latency_ms=latency,
                )

    # Process in batches to show progress
    batch_size = 50
    results: list[EvalResult] = []

    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i:i + batch_size]
        batch_results = await asyncio.gather(*[evaluate_one(tc) for tc in batch])
        results.extend(batch_results)

        # Update summary
        for r in batch_results:
            if r.error:
                summary.errors += 1
                summary.failed += 1
                if len(summary.failed_tests) < 20:  # Keep first 20 failures
                    summary.failed_tests.append(r)
            elif r.passed:
                summary.passed += 1
            else:
                summary.failed += 1
                if len(summary.failed_tests) < 20:  # Keep first 20 failures
                    summary.failed_tests.append(r)

            # Track by category
            if r.category not in summary.by_category:
                summary.by_category[r.category] = {"passed": 0, "failed": 0, "total": 0}
            summary.by_category[r.category]["total"] += 1
            if r.passed:
                summary.by_category[r.category]["passed"] += 1
            else:
                summary.by_category[r.category]["failed"] += 1

        # Progress update
        completed = min(i + batch_size, len(test_cases))
        pct = completed / len(test_cases) * 100
        print(f"  Progress: {completed}/{len(test_cases)} ({pct:.1f}%) - "
              f"Pass: {summary.passed}, Fail: {summary.failed}, Errors: {summary.errors}")

    summary.end_time = datetime.now()

    return summary


def print_summary(summary: EvalSummary):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(f"\nOverall Results:")
    print(f"  Total:      {summary.total}")
    print(f"  Passed:     {summary.passed} ({summary.pass_rate:.1f}%)")
    print(f"  Failed:     {summary.failed}")
    print(f"  Errors:     {summary.errors}")
    print(f"  Duration:   {summary.duration_seconds:.1f}s")

    print(f"\nResults by Category:")
    for cat, stats in sorted(summary.by_category.items()):
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat:15} {stats['passed']:4}/{stats['total']:4} ({rate:5.1f}%)")

    if summary.failed_tests:
        print(f"\nSample Failed Tests (first {len(summary.failed_tests)}):")
        for r in summary.failed_tests[:10]:
            print(f"  - [{r.category}] {r.query[:50]}...")
            print(f"    Expected: {r.expected_function}, Got: {r.actual_function or 'None'}")
            if r.error:
                print(f"    Error: {r.error[:60]}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run NL2API evaluation with Haiku")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--category", type=str, help="Run specific category only")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check for anthropic package first
    try:
        import anthropic  # noqa: F401
    except ImportError:
        print("ERROR: anthropic package not installed!")
        print("\nTo fix this, run:")
        print("  .venv/bin/pip install anthropic")
        sys.exit(1)

    # Check for API key
    api_key = os.environ.get("NL2API_ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: NL2API_ANTHROPIC_API_KEY not set!")
        print("\nTo fix this, either:")
        print("  1. Export the environment variable:")
        print('     export NL2API_ANTHROPIC_API_KEY="sk-ant-api03-..."')
        print("\n  2. Or create a .env file in the project root:")
        print('     echo \'NL2API_ANTHROPIC_API_KEY=sk-ant-api03-...\' > .env')
        print("\nGet your API key from: https://console.anthropic.com/settings/keys")
        sys.exit(1)

    # Validate API key format
    if not api_key.startswith("sk-ant-"):
        print(f"WARNING: API key doesn't look like an Anthropic key (should start with 'sk-ant-')")
        print(f"Got: {api_key[:20]}...")

    # Run evaluation
    try:
        summary = asyncio.run(run_evaluation(
            api_key=api_key,
            limit=args.limit,
            category=args.category,
            concurrency=args.concurrency,
            verbose=args.verbose,
        ))
        print_summary(summary)

        # Save results to JSON
        results_file = PROJECT_ROOT / "eval_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "errors": summary.errors,
                "pass_rate": summary.pass_rate,
                "duration_seconds": summary.duration_seconds,
                "by_category": summary.by_category,
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        # Exit with appropriate code
        sys.exit(0 if summary.pass_rate > 50 else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
