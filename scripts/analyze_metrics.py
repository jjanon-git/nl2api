#!/usr/bin/env python
"""
Analyze NL2API request metrics from JSONL files.

Usage:
    python scripts/analyze_metrics.py metrics.jsonl
    python scripts/analyze_metrics.py metrics.jsonl --summary
    python scripts/analyze_metrics.py metrics.jsonl --by-domain
    python scripts/analyze_metrics.py metrics.jsonl --slow-requests 500
    python scripts/analyze_metrics.py metrics.jsonl --errors
    python scripts/analyze_metrics.py metrics.jsonl --export-csv output.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


@dataclass
class MetricsSummary:
    """Summary statistics for a set of metrics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    clarification_requests: int

    # Latency stats (ms)
    latency_mean: float
    latency_median: float
    latency_p95: float
    latency_p99: float

    # Token stats
    total_tokens: int
    tokens_per_request_mean: float

    # Routing stats
    cache_hit_rate: float
    llm_usage_rate: float
    rule_match_rate: float

    # Domain distribution
    domains: dict[str, int]

    # Error types
    error_types: dict[str, int]


def load_metrics(file_path: Path) -> list[dict[str, Any]]:
    """Load metrics from JSONL file."""
    metrics = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}", file=sys.stderr)
    return metrics


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def compute_summary(metrics: list[dict[str, Any]]) -> MetricsSummary:
    """Compute summary statistics from metrics."""
    if not metrics:
        return MetricsSummary(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            clarification_requests=0,
            latency_mean=0,
            latency_median=0,
            latency_p95=0,
            latency_p99=0,
            total_tokens=0,
            tokens_per_request_mean=0,
            cache_hit_rate=0,
            llm_usage_rate=0,
            rule_match_rate=0,
            domains={},
            error_types={},
        )

    total = len(metrics)
    successful = sum(1 for m in metrics if m.get("success", True))
    failed = total - successful
    clarifications = sum(1 for m in metrics if m.get("needs_clarification", False))

    latencies = [m.get("total_latency_ms", 0) for m in metrics if m.get("total_latency_ms")]
    tokens = [m.get("total_tokens", 0) for m in metrics]
    total_tokens = sum(tokens)

    cache_hits = sum(1 for m in metrics if m.get("routing_cached", False))
    llm_used = sum(1 for m in metrics if m.get("agent_used_llm", False))
    rule_matched = sum(1 for m in metrics if m.get("agent_rule_matched"))

    domains: dict[str, int] = defaultdict(int)
    for m in metrics:
        domain = m.get("routing_domain") or m.get("agent_domain") or "unknown"
        domains[domain] += 1

    error_types: dict[str, int] = defaultdict(int)
    for m in metrics:
        if not m.get("success", True):
            error_type = m.get("error_type", "unknown")
            error_types[error_type] += 1

    return MetricsSummary(
        total_requests=total,
        successful_requests=successful,
        failed_requests=failed,
        clarification_requests=clarifications,
        latency_mean=mean(latencies) if latencies else 0,
        latency_median=median(latencies) if latencies else 0,
        latency_p95=percentile(latencies, 95),
        latency_p99=percentile(latencies, 99),
        total_tokens=total_tokens,
        tokens_per_request_mean=mean(tokens) if tokens else 0,
        cache_hit_rate=cache_hits / total if total else 0,
        llm_usage_rate=llm_used / total if total else 0,
        rule_match_rate=rule_matched / total if total else 0,
        domains=dict(domains),
        error_types=dict(error_types),
    )


def print_summary(summary: MetricsSummary) -> None:
    """Print summary in readable format."""
    print("=" * 60)
    print("NL2API METRICS SUMMARY")
    print("=" * 60)

    print("\n📊 REQUEST OVERVIEW")
    print(f"  Total requests:        {summary.total_requests:,}")
    print(f"  Successful:            {summary.successful_requests:,} ({summary.successful_requests/summary.total_requests*100:.1f}%)" if summary.total_requests else "  Successful:            0")
    print(f"  Failed:                {summary.failed_requests:,} ({summary.failed_requests/summary.total_requests*100:.1f}%)" if summary.total_requests else "  Failed:                0")
    print(f"  Clarifications:        {summary.clarification_requests:,}")

    print("\n⏱️  LATENCY (ms)")
    print(f"  Mean:                  {summary.latency_mean:.1f}")
    print(f"  Median:                {summary.latency_median:.1f}")
    print(f"  P95:                   {summary.latency_p95:.1f}")
    print(f"  P99:                   {summary.latency_p99:.1f}")

    print("\n🔤 TOKENS")
    print(f"  Total:                 {summary.total_tokens:,}")
    print(f"  Per request (mean):    {summary.tokens_per_request_mean:.1f}")

    print("\n🎯 EFFICIENCY")
    print(f"  Cache hit rate:        {summary.cache_hit_rate*100:.1f}%")
    print(f"  LLM usage rate:        {summary.llm_usage_rate*100:.1f}%")
    print(f"  Rule match rate:       {summary.rule_match_rate*100:.1f}%")

    if summary.domains:
        print("\n📁 DOMAIN DISTRIBUTION")
        for domain, count in sorted(summary.domains.items(), key=lambda x: -x[1]):
            pct = count / summary.total_requests * 100
            print(f"  {domain:20} {count:6,} ({pct:5.1f}%)")

    if summary.error_types:
        print("\n❌ ERROR TYPES")
        for error_type, count in sorted(summary.error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type:20} {count:6,}")

    print()


def print_by_domain(metrics: list[dict[str, Any]]) -> None:
    """Print metrics grouped by domain."""
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in metrics:
        domain = m.get("routing_domain") or m.get("agent_domain") or "unknown"
        by_domain[domain].append(m)

    print("=" * 60)
    print("METRICS BY DOMAIN")
    print("=" * 60)

    for domain in sorted(by_domain.keys()):
        domain_metrics = by_domain[domain]
        summary = compute_summary(domain_metrics)

        print(f"\n📂 {domain.upper()}")
        print(f"   Requests: {summary.total_requests:,}")
        print(f"   Success rate: {summary.successful_requests/summary.total_requests*100:.1f}%")
        print(f"   Latency (mean): {summary.latency_mean:.1f}ms")
        print(f"   Tokens (mean): {summary.tokens_per_request_mean:.1f}")
        print(f"   LLM usage: {summary.llm_usage_rate*100:.1f}%")


def print_slow_requests(metrics: list[dict[str, Any]], threshold_ms: int) -> None:
    """Print requests slower than threshold."""
    slow = [m for m in metrics if m.get("total_latency_ms", 0) > threshold_ms]
    slow.sort(key=lambda m: -m.get("total_latency_ms", 0))

    print("=" * 60)
    print(f"SLOW REQUESTS (>{threshold_ms}ms)")
    print("=" * 60)
    print(f"\nFound {len(slow)} slow requests out of {len(metrics)} total")

    for m in slow[:20]:  # Top 20
        print(f"\n  Request: {m.get('request_id', 'unknown')[:8]}...")
        print(f"    Query: {m.get('query', '')[:50]}...")
        print(f"    Latency: {m.get('total_latency_ms', 0)}ms")
        print(f"    Domain: {m.get('routing_domain', 'unknown')}")
        print(f"    Tokens: {m.get('total_tokens', 0)}")
        print(f"    LLM: {m.get('agent_used_llm', False)}")


def print_errors(metrics: list[dict[str, Any]]) -> None:
    """Print failed requests."""
    errors = [m for m in metrics if not m.get("success", True)]

    print("=" * 60)
    print("FAILED REQUESTS")
    print("=" * 60)
    print(f"\nFound {len(errors)} failed requests out of {len(metrics)} total")

    for m in errors[:20]:  # First 20
        print(f"\n  Request: {m.get('request_id', 'unknown')[:8]}...")
        print(f"    Query: {m.get('query', '')[:50]}...")
        print(f"    Error: {m.get('error_type', 'unknown')}: {m.get('error_message', '')[:50]}")
        print(f"    Domain: {m.get('routing_domain', 'unknown')}")


def export_csv(metrics: list[dict[str, Any]], output_path: Path) -> None:
    """Export metrics to CSV for further analysis."""
    if not metrics:
        print("No metrics to export")
        return

    # Collect all unique keys
    all_keys = set()
    for m in metrics:
        all_keys.update(m.keys())

    # Define preferred column order
    priority_keys = [
        "request_id", "timestamp", "query", "routing_domain", "routing_confidence",
        "routing_cached", "agent_used_llm", "agent_rule_matched", "tool_calls_count",
        "total_latency_ms", "total_tokens", "success", "error_type", "error_message",
    ]

    # Order columns: priority first, then alphabetical
    columns = [k for k in priority_keys if k in all_keys]
    columns += sorted(all_keys - set(columns))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for m in metrics:
            # Convert complex types to strings
            row = {}
            for k, v in m.items():
                if isinstance(v, (list, dict)):
                    row[k] = json.dumps(v)
                else:
                    row[k] = v
            writer.writerow(row)

    print(f"Exported {len(metrics)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze NL2API request metrics from JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file", type=Path, help="Path to metrics JSONL file")
    parser.add_argument("--summary", action="store_true", help="Show overall summary (default)")
    parser.add_argument("--by-domain", action="store_true", help="Show metrics by domain")
    parser.add_argument("--slow-requests", type=int, metavar="MS", help="Show requests slower than MS milliseconds")
    parser.add_argument("--errors", action="store_true", help="Show failed requests")
    parser.add_argument("--export-csv", type=Path, metavar="PATH", help="Export to CSV file")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading metrics from {args.file}...")
    metrics = load_metrics(args.file)
    print(f"Loaded {len(metrics)} records\n")

    if not metrics:
        print("No metrics found in file")
        sys.exit(0)

    # Determine what to show
    show_summary = args.summary or not any([args.by_domain, args.slow_requests, args.errors, args.export_csv])

    if show_summary:
        summary = compute_summary(metrics)
        print_summary(summary)

    if args.by_domain:
        print_by_domain(metrics)

    if args.slow_requests:
        print_slow_requests(metrics, args.slow_requests)

    if args.errors:
        print_errors(metrics)

    if args.export_csv:
        export_csv(metrics, args.export_csv)


if __name__ == "__main__":
    main()
