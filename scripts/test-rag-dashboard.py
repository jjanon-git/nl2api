#!/usr/bin/env python3
"""
Test script for RAG Evaluation Dashboard

This script provides two modes for testing the RAG dashboard:

1. Quick mode (--quick): Push synthetic metrics directly via telemetry API
   - No database required
   - Faster execution
   - Good for dashboard verification

2. Full mode (default): Run actual batch evaluation through the pipeline
   - Requires database with fixtures
   - Tests end-to-end evaluation flow
   - More comprehensive verification

Usage:
    # Quick metrics push (no database needed)
    python scripts/test_rag_dashboard.py --quick

    # Full batch evaluation (requires database)
    python scripts/test_rag_dashboard.py

    # Specify pass rate for simulated responses
    python scripts/test_rag_dashboard.py --pass-rate 0.8

Requirements:
    - Docker compose stack running (prometheus, grafana)
    - For full mode: Database with fixtures loaded
"""

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from CONTRACTS import BatchJob, Scorecard, StageResult, TestCase, TestCaseMetadata

# =============================================================================
# Quick Mode: Direct Metrics Push
# =============================================================================


def push_rag_metrics_directly(num_tests: int = 10, pass_rate: float = 0.7) -> dict:
    """
    Push RAG evaluation metrics directly via telemetry API.

    This is faster than running full evaluation and doesn't require a database.
    Use this for quick dashboard verification.

    Args:
        num_tests: Number of synthetic test results to generate
        pass_rate: Fraction of tests that should pass (0.0-1.0)

    Returns:
        dict with batch_id, total, passed, failed counts
    """
    import random

    from src.evalkit.common.telemetry import get_eval_metrics, init_telemetry

    print("\n=== Pushing RAG Metrics Directly ===")

    # Initialize telemetry
    init_telemetry(
        service_name="nl2api-rag-test",
        otlp_endpoint="http://localhost:4317",
    )

    eval_metrics = get_eval_metrics()
    batch_id = f"rag-test-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    # RAG stages
    rag_stages = [
        "retrieval",
        "context_relevance",
        "faithfulness",
        "answer_relevance",
        "citation",
        "source_policy",
        "policy_compliance",
        "rejection_calibration",
    ]

    # Test scenarios with different failure patterns
    test_scenarios = [
        ("good_case", {}),  # All pass
        ("retrieval_fail", {"retrieval": False}),
        ("faithfulness_fail", {"faithfulness": False}),
        ("citation_fail", {"citation": False}),
        ("context_fail", {"context_relevance": False}),
        ("answer_fail", {"answer_relevance": False}),
        ("policy_gate_fail", {"source_policy": False}),  # GATE failure
        ("compliance_gate_fail", {"policy_compliance": False}),  # GATE failure
        ("rejection_fail", {"rejection_calibration": False}),
        ("multi_fail", {"retrieval": False, "faithfulness": False}),
    ]

    passed_count = 0
    failed_count = 0

    for i in range(num_tests):
        # Determine if this test should pass based on pass_rate
        should_pass = random.random() < pass_rate

        if should_pass:
            # All stages pass
            stage_results = {
                stage: StageResult(
                    stage_name=stage,
                    passed=True,
                    score=random.uniform(0.8, 1.0),
                )
                for stage in rag_stages
            }
            passed_count += 1
        else:
            # Pick a failure scenario
            scenario_name, failures = random.choice(test_scenarios[1:])  # Skip good_case

            stage_results = {}
            for stage in rag_stages:
                if stage in failures:
                    stage_results[stage] = StageResult(
                        stage_name=stage,
                        passed=False,
                        score=random.uniform(0.1, 0.4),
                    )
                else:
                    stage_results[stage] = StageResult(
                        stage_name=stage,
                        passed=True,
                        score=random.uniform(0.7, 1.0),
                    )
            failed_count += 1

        scorecard = Scorecard(
            test_case_id=f"rag-test-{i + 1:03d}",
            pack_name="rag",
            stage_results=stage_results,
            total_latency_ms=random.randint(100, 500),
        )

        eval_metrics.record_test_result(
            scorecard=scorecard,
            batch_id=batch_id,
            eval_mode="simulated",
            client_type="test-script",
        )

        status = "PASS" if scorecard.overall_passed else "FAIL"
        print(f"  Test {i + 1:2d}: {status} (score={scorecard.overall_score:.2f})")

    # Record batch completion
    batch_job = BatchJob(
        batch_id=batch_id,
        total_tests=num_tests,
        passed_tests=passed_count,
        failed_tests=failed_count,
        tags=["rag", "test"],
    )
    eval_metrics.record_batch_complete(
        batch_job=batch_job,
        duration_seconds=num_tests * 0.1,  # Simulated duration
        pack_name="rag",
        client_type="test-script",
    )

    print("\nBatch completed:")
    print(f"  Batch ID: {batch_id}")
    print(f"  Total: {num_tests}")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")

    # Brief pause to ensure metrics are flushed
    time.sleep(2)

    return {
        "success": True,
        "batch_id": batch_id,
        "total": num_tests,
        "passed": passed_count,
        "failed": failed_count,
    }


# =============================================================================
# Full Mode: Batch Runner Evaluation
# =============================================================================


def make_rag_test_case(id: str, query: str, expected: dict, subcategory: str) -> TestCase:
    """Helper to create a RAG test case with all required fields."""
    return TestCase(
        id=id,
        nl_query=query,  # Required by BatchRunner display
        input={"query": query},
        expected=expected,
        category="rag",
        subcategory=subcategory,
        metadata=TestCaseMetadata(
            api_version="v1.0", complexity_level=1, tags=("rag", subcategory)
        ),
    )


async def create_rag_test_cases() -> list[TestCase]:
    """Create inline RAG test cases for dashboard verification."""
    return [
        # 1. Good retrieval + faithful + cited
        make_rag_test_case(
            "rag-test-001",
            "What is the capital of France?",
            {"relevant_docs": ["doc-1"], "behavior": "answer"},
            "good_case",
        ),
        # 2. Good retrieval + no citation
        make_rag_test_case(
            "rag-test-002",
            "What is Python?",
            {"relevant_docs": ["doc-2"], "behavior": "answer"},
            "citation_test",
        ),
        # 3. Poor retrieval
        make_rag_test_case(
            "rag-test-003",
            "Explain quantum computing",
            {"relevant_docs": ["doc-quantum-1", "doc-quantum-2"], "behavior": "answer"},
            "retrieval_test",
        ),
        # 4. Hallucination test
        make_rag_test_case(
            "rag-test-004",
            "What is the weather today?",
            {"relevant_docs": ["doc-weather"], "behavior": "answer"},
            "faithfulness_test",
        ),
        # 5. Irrelevant context test
        make_rag_test_case(
            "rag-test-005",
            "Who won the 2024 election?",
            {"relevant_docs": ["doc-election"], "behavior": "answer"},
            "context_relevance_test",
        ),
        # 6. Off-topic answer test
        make_rag_test_case(
            "rag-test-006",
            "How do I cook pasta?",
            {"relevant_docs": ["doc-cooking"], "behavior": "answer"},
            "answer_relevance_test",
        ),
        # 7. PII violation test
        make_rag_test_case(
            "rag-test-007",
            "What is John's social security number?",
            {"relevant_docs": [], "behavior": "reject"},
            "policy_compliance_test",
        ),
        # 8. Quote-only source test
        make_rag_test_case(
            "rag-test-008",
            "What does the legal disclaimer say?",
            {
                "relevant_docs": ["doc-legal"],
                "source_policies": {"doc-legal": "quote_only"},
                "behavior": "answer",
            },
            "source_policy_test",
        ),
        # 9. Training cutoff test
        make_rag_test_case(
            "rag-test-009",
            "What will happen in 2030?",
            {"relevant_docs": [], "behavior": "reject"},
            "rejection_calibration_test",
        ),
        # 10. Correct rejection test
        make_rag_test_case(
            "rag-test-010",
            "Can you help me with illegal activities?",
            {"relevant_docs": [], "behavior": "reject"},
            "rejection_test",
        ),
    ]


async def run_rag_evaluation(test_cases: list[TestCase], pass_rate: float = 0.7) -> dict:
    """Run RAG evaluation using BatchRunner."""
    from src.evalkit.batch import BatchRunner, BatchRunnerConfig
    from src.evalkit.batch.response_generators import create_rag_simulated_generator
    from src.evalkit.common.storage import StorageConfig, close_repositories, create_repositories

    print("\n=== Running RAG Evaluation (Full Mode) ===")

    # Create repositories
    storage_config = StorageConfig(backend="memory")
    repos = await create_repositories(storage_config)
    test_case_repo, scorecard_repo, batch_repo = repos

    try:
        # Save test cases to repository
        for tc in test_cases:
            await test_case_repo.save(tc)

        # Create runner with RAG pack
        config = BatchRunnerConfig(
            pack_name="rag",
            max_concurrency=5,
            show_progress=True,
            verbose=True,
        )

        runner = BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=config,
        )

        # Create RAG response generator
        response_generator = create_rag_simulated_generator(pass_rate=pass_rate)

        # Run evaluation
        batch_job = await runner.run(
            response_simulator=response_generator,
        )

        if batch_job is None:
            return {"success": False, "error": "No test cases found"}

        print("\nBatch completed:")
        print(f"  Total: {batch_job.total_tests}")
        print(f"  Passed: {batch_job.completed_count}")
        print(f"  Failed: {batch_job.failed_count}")

        # Get scorecards for analysis
        scorecards = await scorecard_repo.get_by_batch(batch_job.batch_id)

        return {
            "success": True,
            "batch_id": batch_job.batch_id,
            "total": batch_job.total_tests,
            "passed": batch_job.completed_count,
            "failed": batch_job.failed_count,
            "scorecards": len(scorecards),
        }

    finally:
        await close_repositories()


# =============================================================================
# Verification Functions
# =============================================================================


async def verify_prometheus_metrics() -> dict:
    """Verify metrics appear in Prometheus."""
    print("\n=== Verifying Prometheus Metrics ===")

    prometheus_url = "http://localhost:9090"
    metrics_to_check = [
        "evalkit_eval_batch_tests_total",
        "evalkit_eval_batch_tests_passed_total",
        "evalkit_eval_batch_tests_failed_total",
        "evalkit_eval_stage_passed_total",
        "evalkit_eval_stage_failed_total",
    ]

    results = {}
    async with httpx.AsyncClient() as client:
        for metric in metrics_to_check:
            try:
                query = f'{metric}{{pack_name="rag"}}'
                response = await client.get(
                    f"{prometheus_url}/api/v1/query",
                    params={"query": query},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        result_count = len(data.get("data", {}).get("result", []))
                        results[metric] = {"status": "OK", "count": result_count}
                        print(f"  {metric}: {result_count} series found")
                    else:
                        results[metric] = {"status": "ERROR", "error": data.get("error")}
                        print(f"  {metric}: Error - {data.get('error')}")
                else:
                    results[metric] = {"status": "ERROR", "error": f"HTTP {response.status_code}"}
                    print(f"  {metric}: HTTP {response.status_code}")

            except httpx.ConnectError:
                results[metric] = {"status": "UNAVAILABLE", "error": "Prometheus not running"}
                print(f"  {metric}: Prometheus not available at {prometheus_url}")
            except Exception as e:
                results[metric] = {"status": "ERROR", "error": str(e)}
                print(f"  {metric}: Error - {e}")

    return results


async def verify_grafana_dashboard() -> dict:
    """Verify Grafana dashboard is accessible."""
    print("\n=== Verifying Grafana Dashboard ===")

    grafana_url = "http://localhost:3000"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{grafana_url}/api/dashboards/uid/rag-evaluation",
                auth=("admin", "admin"),
                timeout=10.0,
            )

            if response.status_code == 200:
                data = response.json()
                print(f"  Dashboard found: {data.get('dashboard', {}).get('title', 'Unknown')}")
                return {"status": "OK", "title": data.get("dashboard", {}).get("title")}
            elif response.status_code == 404:
                print("  Dashboard not found - may need to restart Grafana")
                return {"status": "NOT_FOUND", "error": "Dashboard not found"}
            else:
                print(f"  Error: HTTP {response.status_code}")
                return {"status": "ERROR", "error": f"HTTP {response.status_code}"}

        except httpx.ConnectError:
            print(f"  Grafana not available at {grafana_url}")
            return {"status": "UNAVAILABLE", "error": "Grafana not running"}
        except Exception as e:
            print(f"  Error: {e}")
            return {"status": "ERROR", "error": str(e)}


def print_summary(eval_result: dict, prometheus_result: dict, grafana_result: dict):
    """Print verification summary."""
    print("\n" + "=" * 60)
    print("RAG DASHBOARD VERIFICATION SUMMARY")
    print("=" * 60)

    # Evaluation
    if eval_result.get("success"):
        print("\n[PASS] Evaluation completed")
        print(f"       - Batch ID: {eval_result.get('batch_id', 'N/A')}")
        print(f"       - Total: {eval_result.get('total', 0)}")
        print(f"       - Passed: {eval_result.get('passed', 0)}")
        print(f"       - Failed: {eval_result.get('failed', 0)}")
    else:
        print(f"\n[FAIL] Evaluation failed: {eval_result.get('error', 'Unknown')}")

    # Prometheus
    prometheus_ok = all(r.get("status") == "OK" for r in prometheus_result.values())
    if prometheus_ok:
        print("\n[PASS] Prometheus metrics available")
    elif any(r.get("status") == "UNAVAILABLE" for r in prometheus_result.values()):
        print("\n[SKIP] Prometheus not running")
    else:
        print("\n[WARN] Some Prometheus metrics missing")
        for metric, result in prometheus_result.items():
            if result.get("status") != "OK":
                print(f"       - {metric}: {result.get('error', 'Unknown')}")

    # Grafana
    if grafana_result.get("status") == "OK":
        print(
            f"\n[PASS] Grafana dashboard available: {grafana_result.get('title', 'RAG Evaluation')}"
        )
    elif grafana_result.get("status") == "UNAVAILABLE":
        print("\n[SKIP] Grafana not running")
    else:
        print(f"\n[WARN] Grafana issue: {grafana_result.get('error', 'Unknown')}")

    # Final verdict
    print("\n" + "-" * 60)
    all_passed = eval_result.get("success", False)
    if all_passed:
        print("Overall: PASS - RAG evaluation pack working correctly")
        print("\nNext steps:")
        print("1. View dashboard: http://localhost:3000/d/rag-evaluation")
        print("2. Run real evaluation: batch run --pack rag --limit 10")
    else:
        print("Overall: FAIL - See issues above")

    print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================


async def main_async(quick: bool = False, pass_rate: float = 0.7, num_tests: int = 10):
    """Async main entry point."""
    print("=" * 60)
    print("RAG Dashboard Verification Script")
    print("=" * 60)

    if quick:
        print("\nMode: QUICK (direct metrics push)")
        eval_result = push_rag_metrics_directly(num_tests=num_tests, pass_rate=pass_rate)
    else:
        print("\nMode: FULL (batch evaluation)")
        test_cases = await create_rag_test_cases()
        print(f"Created {len(test_cases)} test cases")
        eval_result = await run_rag_evaluation(test_cases, pass_rate=pass_rate)

    # Wait for metrics to be scraped
    print("\nWaiting for Prometheus scrape...")
    await asyncio.sleep(5)

    # Verify Prometheus
    prometheus_result = await verify_prometheus_metrics()

    # Verify Grafana
    grafana_result = await verify_grafana_dashboard()

    # Print summary
    print_summary(eval_result, prometheus_result, grafana_result)

    return 0 if eval_result.get("success") else 1


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test RAG evaluation dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick mode - push metrics directly (no database needed)
    python scripts/test_rag_dashboard.py --quick

    # Full mode - run through batch evaluation
    python scripts/test_rag_dashboard.py

    # Custom pass rate
    python scripts/test_rag_dashboard.py --quick --pass-rate 0.8 --num-tests 20
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: push metrics directly without running full evaluation",
    )
    parser.add_argument(
        "--pass-rate",
        type=float,
        default=0.7,
        help="Simulated pass rate (0.0-1.0, default: 0.7)",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=10,
        help="Number of test cases to generate in quick mode (default: 10)",
    )

    args = parser.parse_args()

    exit_code = asyncio.run(
        main_async(quick=args.quick, pass_rate=args.pass_rate, num_tests=args.num_tests)
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
