#!/usr/bin/env python3
"""
Evaluate orchestrator routing accuracy.

This is a CLI wrapper around the shared accuracy testing infrastructure.
For pytest-based testing, use: pytest tests/accuracy/routing/ -m tier1

Usage:
    # Run routing evaluation with 100 test cases
    .venv/bin/python scripts/eval_routing.py --limit 100

    # Run balanced evaluation across domains
    .venv/bin/python scripts/eval_routing.py --limit 50 --balanced

    # Dry run - show expected routing without LLM calls
    .venv/bin/python scripts/eval_routing.py --dry-run --limit 20
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared infrastructure
from tests.accuracy.core.config import AccuracyConfig
from tests.accuracy.core.evaluator import RoutingAccuracyEvaluator
from tests.accuracy.routing.test_routing_accuracy import (
    load_routing_test_cases,
    print_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate orchestrator routing accuracy",
        epilog="For pytest-based testing, use: pytest tests/accuracy/routing/ -m tier1",
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of test cases")
    parser.add_argument(
        "--model",
        type=str,
        default="haiku",
        choices=["haiku", "sonnet"],
        help="Model to use for routing",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show expected routing without LLM calls"
    )
    parser.add_argument("--balanced", action="store_true", help="Balance samples across domains")
    parser.add_argument(
        "--threshold", type=float, default=0.80, help="Accuracy threshold (default: 0.80)"
    )
    args = parser.parse_args()

    # Model mapping
    model_map = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-sonnet-4-20250514",
    }
    model_id = model_map.get(args.model, args.model)

    print("=" * 60)
    print("ORCHESTRATOR ROUTING EVALUATION")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Limit: {args.limit}")
    print(f"Balanced: {args.balanced}")
    print()

    # Load test cases
    test_cases = load_routing_test_cases(limit=args.limit, balanced=args.balanced)

    if args.dry_run:
        print("\nDry run - showing expected routing:")
        by_domain: dict = defaultdict(list)
        for tc in test_cases:
            by_domain[tc.expected_domain].append(tc)

        for domain, cases in sorted(by_domain.items()):
            print(f"\n{domain.upper()} ({len(cases)} cases):")
            for tc in cases[:3]:
                print(f"  - {tc.query[:60]}...")
        return

    # Create evaluator
    config = AccuracyConfig(model=model_id)
    evaluator = RoutingAccuracyEvaluator(config=config)

    # Progress callback
    def progress(current, total, result):
        if current % 10 == 0:
            print(
                f"  [{current}/{total}] Latest: {result.expected} -> {result.predicted} (conf={result.confidence:.2f})"
            )

    # Run evaluation
    print("\nRunning routing evaluation...")
    report = asyncio.run(evaluator.evaluate_batch(test_cases, progress_callback=progress))

    # Print report
    print_report(report, args.threshold)

    # Save results
    results_file = PROJECT_ROOT / f"routing_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Exit with appropriate code
    if report.accuracy < args.threshold:
        print(f"\nFAILED: Accuracy {report.accuracy:.1%} below threshold {args.threshold:.0%}")
        sys.exit(1)
    else:
        print(f"\nPASSED: Accuracy {report.accuracy:.1%} meets threshold {args.threshold:.0%}")


if __name__ == "__main__":
    main()
