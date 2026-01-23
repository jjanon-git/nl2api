#!/usr/bin/env python
"""
Test script to verify metrics export to Prometheus.

Generates sample metrics data to confirm the pipeline is working:
- Token usage metrics
- Cost metrics
- Regression alert metrics

Run with: python scripts/test_metrics_export.py
"""

import os
import sys
import time

# Ensure telemetry is enabled
os.environ["NL2API_TELEMETRY_ENABLED"] = "true"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.telemetry.metrics import (
    get_eval_metrics,
    get_regression_alert_metrics,
)
from src.common.telemetry.setup import init_telemetry


def main():
    print("Setting up telemetry...")
    init_telemetry(service_name="nl2api-test")

    eval_metrics = get_eval_metrics()
    alert_metrics = get_regression_alert_metrics()

    print("\n=== Generating Test Metrics ===\n")

    # Simulate token usage metrics
    print("1. Recording token usage metrics...")
    for i in range(5):
        # Create a mock scorecard-like object
        class MockScorecard:
            def __init__(self):
                self.overall_passed = i % 2 == 0
                self.overall_score = 0.7 + (i * 0.05)
                self.total_latency_ms = 100 + (i * 50)
                self.syntax_result = None
                self.logic_result = None
                self.execution_result = None
                self.semantics_result = None
                self.input_tokens = 500 + (i * 100)
                self.output_tokens = 200 + (i * 50)
                self.estimated_cost_usd = 0.001 + (i * 0.0005)

        scorecard = MockScorecard()
        eval_metrics.record_test_result(
            scorecard=scorecard,
            batch_id=f"test-batch-{i}",
            tags=["test"],
            client_type="test_client",
            client_version="v1.0",
            eval_mode="orchestrator",
        )
        print(
            f"   Recorded test {i + 1}: {scorecard.input_tokens} input, {scorecard.output_tokens} output tokens, ${scorecard.estimated_cost_usd:.4f}"
        )

    # Simulate regression alerts
    print("\n2. Recording regression alert metrics...")
    alert_metrics.record_alert_created(
        severity="warning",
        metric_name="pass_rate",
        delta_pct=-3.5,
        client_type="test_client",
    )
    print("   Recorded warning alert for pass_rate (-3.5%)")

    alert_metrics.record_alert_created(
        severity="critical",
        metric_name="avg_latency_ms",
        delta_pct=75.0,
        client_type="test_client",
    )
    print("   Recorded critical alert for avg_latency_ms (+75%)")

    alert_metrics.record_alert_acknowledged(
        severity="warning",
        metric_name="pass_rate",
    )
    print("   Acknowledged warning alert for pass_rate")

    print("\n=== Waiting for metrics to be exported ===")
    print("Metrics are exported every 60 seconds by default.")
    print("Waiting 65 seconds for export cycle...")

    for i in range(65, 0, -5):
        print(f"   {i} seconds remaining...")
        time.sleep(5)

    print("\n=== Verification ===")
    print("Check Prometheus at http://localhost:9090 for:")
    print('  - nl2api_eval_tokens_total{token_type="input"}')
    print('  - nl2api_eval_tokens_total{token_type="output"}')
    print("  - nl2api_eval_cost_usd_total")
    print("  - nl2api_regression_alerts_total")
    print("  - nl2api_regression_alerts_acknowledged_total")
    print("\nOr check Grafana dashboards at http://localhost:3000")


if __name__ == "__main__":
    main()
