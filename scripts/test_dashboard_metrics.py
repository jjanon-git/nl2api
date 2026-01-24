#!/usr/bin/env python3
"""
Test script to push sample metrics for all dashboards.

This script pushes synthetic metrics to verify the observability pipeline:
- OTEL Collector receives metrics
- Prometheus scrapes and stores them
- Grafana dashboards display them

Usage:
    python scripts/test_dashboard_metrics.py

Requirements:
    - Docker compose stack running (docker compose up -d)
    - OTEL Collector on localhost:4317

After running, wait 15-20 seconds for Prometheus scrape, then check:
    - NL2API Overview: http://localhost:3000/d/nl2api-overview
    - Eval Infrastructure: http://localhost:3000/d/eval-infrastructure
    - NL2API Evaluation: http://localhost:3000/d/nl2api-evaluation
    - RAG Evaluation: http://localhost:3000/d/rag-evaluation
"""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from CONTRACTS import BatchJob, Scorecard, StageResult
from src.evalkit.common.telemetry import get_eval_metrics, init_telemetry
from src.evalkit.common.telemetry.metrics import get_regression_alert_metrics


def push_nl2api_metrics(eval_metrics, batch_id: str) -> None:
    """Push NL2API evaluation metrics."""
    print("\n=== NL2API Pack Metrics ===")

    scorecards = [
        Scorecard(
            test_case_id="nl2api-test-001",
            pack_name="nl2api",
            stage_results={
                "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
                "logic": StageResult(stage_name="logic", passed=True, score=0.9),
            },
            syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            logic_result=StageResult(stage_name="logic", passed=True, score=0.9),
            total_latency_ms=150,
            input_tokens=500,
            output_tokens=200,
            estimated_cost_usd=0.0035,
        ),
        Scorecard(
            test_case_id="nl2api-test-002",
            pack_name="nl2api",
            stage_results={
                "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
                "logic": StageResult(stage_name="logic", passed=False, score=0.3),
            },
            syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            logic_result=StageResult(stage_name="logic", passed=False, score=0.3),
            total_latency_ms=200,
            input_tokens=600,
            output_tokens=250,
            estimated_cost_usd=0.0042,
        ),
        Scorecard(
            test_case_id="nl2api-test-003",
            pack_name="nl2api",
            stage_results={
                "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
                "logic": StageResult(stage_name="logic", passed=True, score=0.85),
            },
            syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            logic_result=StageResult(stage_name="logic", passed=True, score=0.85),
            total_latency_ms=180,
            input_tokens=550,
            output_tokens=220,
            estimated_cost_usd=0.0038,
        ),
    ]

    for sc in scorecards:
        eval_metrics.record_test_result(
            scorecard=sc,
            batch_id=batch_id,
            eval_mode="test",
            client_type="cli",
            client_version="1.0.0",
        )
        print(
            f"  {sc.test_case_id}: passed={sc.overall_passed}, tokens={sc.input_tokens}/{sc.output_tokens}"
        )

    # Batch completion
    batch_job = BatchJob(
        batch_id=batch_id,
        total_tests=3,
        passed_tests=2,
        failed_tests=1,
        tags=["test", "cli"],
    )
    eval_metrics.record_batch_complete(
        batch_job=batch_job,
        duration_seconds=8.5,
        pack_name="nl2api",
        client_type="cli",
    )
    print("  Batch completed: 2/3 passed")


def push_rag_metrics(eval_metrics, batch_id: str) -> None:
    """Push RAG evaluation metrics."""
    print("\n=== RAG Pack Metrics ===")

    # RAG has different stages
    rag_stages = [
        "retrieval",
        "context_relevance",
        "faithfulness",
        "answer_relevance",
        "citation",
        "source_policy",
    ]

    scorecards = [
        # Good case - all stages pass
        Scorecard(
            test_case_id="rag-test-001",
            pack_name="rag",
            stage_results={
                stage: StageResult(stage_name=stage, passed=True, score=0.9) for stage in rag_stages
            },
            total_latency_ms=250,
        ),
        # Retrieval failure
        Scorecard(
            test_case_id="rag-test-002",
            pack_name="rag",
            stage_results={
                "retrieval": StageResult(stage_name="retrieval", passed=False, score=0.2),
                **{
                    stage: StageResult(stage_name=stage, passed=True, score=0.8)
                    for stage in rag_stages[1:]
                },
            },
            total_latency_ms=180,
        ),
        # Faithfulness failure (hallucination)
        Scorecard(
            test_case_id="rag-test-003",
            pack_name="rag",
            stage_results={
                "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.9),
                "context_relevance": StageResult(
                    stage_name="context_relevance", passed=True, score=0.85
                ),
                "faithfulness": StageResult(stage_name="faithfulness", passed=False, score=0.3),
                "answer_relevance": StageResult(
                    stage_name="answer_relevance", passed=True, score=0.8
                ),
                "citation": StageResult(stage_name="citation", passed=True, score=1.0),
                "source_policy": StageResult(stage_name="source_policy", passed=True, score=1.0),
            },
            total_latency_ms=200,
        ),
    ]

    for sc in scorecards:
        eval_metrics.record_test_result(
            scorecard=sc,
            batch_id=f"rag-{batch_id}",
            eval_mode="test",
            client_type="cli",
        )
        print(f"  {sc.test_case_id}: passed={sc.overall_passed}")

    # Batch completion
    batch_job = BatchJob(
        batch_id=f"rag-{batch_id}",
        total_tests=3,
        passed_tests=1,
        failed_tests=2,
        tags=["test", "rag"],
    )
    eval_metrics.record_batch_complete(
        batch_job=batch_job,
        duration_seconds=5.2,
        pack_name="rag",
        client_type="cli",
    )
    print("  Batch completed: 1/3 passed")


def push_infrastructure_metrics(eval_metrics) -> None:
    """Push infrastructure metrics (workers, queues)."""
    print("\n=== Infrastructure Metrics ===")

    # Worker metrics
    eval_metrics.record_worker_status(
        worker_id="worker-1",
        active=True,
        tasks_processed=15,
        tasks_failed=2,
        task_duration_ms=175.0,
    )
    eval_metrics.record_worker_status(
        worker_id="worker-2",
        active=True,
        tasks_processed=12,
        tasks_failed=1,
        task_duration_ms=145.0,
    )
    print("  Workers: 2 active")

    # Queue metrics
    eval_metrics.record_queue_operation("enqueue", count=150)
    eval_metrics.record_queue_operation("ack", count=140)
    eval_metrics.record_queue_operation("nack", count=5, action="requeue")
    eval_metrics.record_queue_operation("dlq", count=5)
    print("  Queue: 150 enqueued, 140 acked, 5 requeued, 5 DLQ")


def push_regression_alerts(regression_metrics) -> None:
    """Push regression alert metrics."""
    print("\n=== Regression Alerts ===")

    regression_metrics.record_alert_created(
        severity="warning",
        metric_name="pass_rate",
        delta_pct=-5.2,
        client_type="cli",
    )
    regression_metrics.record_alert_created(
        severity="critical",
        metric_name="latency_p95",
        delta_pct=25.0,
        client_type="cli",
    )
    regression_metrics.record_alert_created(
        severity="warning",
        metric_name="cost_per_test",
        delta_pct=10.5,
        client_type="api",
    )
    # Acknowledge one alert
    regression_metrics.record_alert_acknowledged(
        severity="warning",
        metric_name="pass_rate",
    )
    print("  Alerts: 3 created, 1 acknowledged")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Dashboard Metrics Test Script")
    print("=" * 60)

    # Initialize telemetry
    init_telemetry(
        service_name="nl2api-metrics-test",
        otlp_endpoint="http://localhost:4317",
    )

    # Get metrics instances
    eval_metrics = get_eval_metrics()
    regression_metrics = get_regression_alert_metrics()

    # Generate batch ID
    batch_id = f"test-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    print(f"\nBatch ID: {batch_id}")

    # Push all metrics
    push_nl2api_metrics(eval_metrics, batch_id)
    push_rag_metrics(eval_metrics, batch_id)
    push_infrastructure_metrics(eval_metrics)
    push_regression_alerts(regression_metrics)

    print("\n" + "=" * 60)
    print("All metrics pushed successfully!")
    print("=" * 60)
    print("\nWait 15-20 seconds for Prometheus scrape, then check dashboards:")
    print("  - NL2API Overview:      http://localhost:3000/d/nl2api-overview")
    print("  - Eval Infrastructure:  http://localhost:3000/d/eval-infrastructure")
    print("  - NL2API Evaluation:    http://localhost:3000/d/nl2api-evaluation")
    print("  - RAG Evaluation:       http://localhost:3000/d/rag-evaluation")
    print("  - NL2API Accuracy:      http://localhost:3000/d/nl2api-accuracy")
    print("=" * 60)

    # Brief pause to ensure metrics are flushed
    time.sleep(2)


if __name__ == "__main__":
    main()
