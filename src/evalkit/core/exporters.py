"""
Result Exporters

Export evaluation results to various formats for analysis and reporting.

Usage:
    from src.evaluation.core.exporters import JSONExporter, CSVExporter, SummaryExporter

    # Export to JSON
    exporter = JSONExporter()
    await exporter.export(scorecards, "results.json")

    # Export to CSV for analysis
    exporter = CSVExporter()
    await exporter.export(scorecards, "results.csv")

    # Get summary
    summary = SummaryExporter().summarize(scorecards)
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from CONTRACTS import Scorecard, StageResult

# =============================================================================
# Base Exporter
# =============================================================================


class Exporter(ABC):
    """Base class for result exporters."""

    @abstractmethod
    async def export(
        self,
        scorecards: list[Scorecard],
        output_path: str | Path,
    ) -> None:
        """Export scorecards to the specified path."""
        ...


# =============================================================================
# JSON Exporter
# =============================================================================


class JSONExporter(Exporter):
    """Export scorecards to JSON format."""

    def __init__(self, indent: int = 2, include_full_results: bool = True):
        """
        Initialize JSON exporter.

        Args:
            indent: JSON indentation level
            include_full_results: Whether to include full stage results or just summary
        """
        self.indent = indent
        self.include_full_results = include_full_results

    async def export(
        self,
        scorecards: list[Scorecard],
        output_path: str | Path,
    ) -> None:
        """Export scorecards to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "exported_at": datetime.now(UTC).isoformat(),
            "total_scorecards": len(scorecards),
            "scorecards": [self._scorecard_to_dict(sc) for sc in scorecards],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=self.indent, default=str)

    def _scorecard_to_dict(self, scorecard: Scorecard) -> dict[str, Any]:
        """Convert scorecard to JSON-serializable dict."""
        result = {
            "scorecard_id": scorecard.scorecard_id,
            "test_case_id": scorecard.test_case_id,
            "batch_id": scorecard.batch_id,
            "pack_name": scorecard.pack_name,
            "timestamp": scorecard.timestamp.isoformat() if scorecard.timestamp else None,
            "worker_id": scorecard.worker_id,
            "total_latency_ms": scorecard.total_latency_ms,
        }

        # Add stage summary
        all_results = scorecard.get_all_stage_results()
        result["stages"] = {
            name: {
                "passed": sr.passed,
                "score": sr.score,
                "reason": sr.reason,
                "error_code": sr.error_code.value if sr.error_code else None,
                "duration_ms": sr.duration_ms,
            }
            for name, sr in all_results.items()
        }

        # Compute overall
        result["overall_passed"] = all(sr.passed for sr in all_results.values())
        result["overall_score"] = (
            sum(sr.score for sr in all_results.values()) / len(all_results) if all_results else 0.0
        )

        if self.include_full_results:
            result["stage_weights"] = dict(scorecard.stage_weights)
            result["generated_output"] = scorecard.generated_output

        return result


# =============================================================================
# CSV Exporter
# =============================================================================


class CSVExporter(Exporter):
    """Export scorecards to CSV format for analysis."""

    def __init__(self, include_stage_scores: bool = True):
        """
        Initialize CSV exporter.

        Args:
            include_stage_scores: Whether to include individual stage scores as columns
        """
        self.include_stage_scores = include_stage_scores

    async def export(
        self,
        scorecards: list[Scorecard],
        output_path: str | Path,
    ) -> None:
        """Export scorecards to CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not scorecards:
            # Write empty file with headers
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["test_case_id", "pack_name", "overall_passed", "overall_score"])
            return

        # Collect all stage names across all scorecards
        all_stages: set[str] = set()
        for sc in scorecards:
            all_stages.update(sc.get_all_stage_results().keys())
        stage_names = sorted(all_stages)

        # Build headers
        headers = [
            "scorecard_id",
            "test_case_id",
            "batch_id",
            "pack_name",
            "overall_passed",
            "overall_score",
            "total_latency_ms",
            "timestamp",
        ]

        if self.include_stage_scores:
            for stage in stage_names:
                headers.extend([f"{stage}_passed", f"{stage}_score"])

        # Write CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()

            for sc in scorecards:
                all_results = sc.get_all_stage_results()
                overall_passed = all(sr.passed for sr in all_results.values())
                overall_score = (
                    sum(sr.score for sr in all_results.values()) / len(all_results)
                    if all_results
                    else 0.0
                )

                row = {
                    "scorecard_id": sc.scorecard_id,
                    "test_case_id": sc.test_case_id,
                    "batch_id": sc.batch_id,
                    "pack_name": sc.pack_name,
                    "overall_passed": overall_passed,
                    "overall_score": round(overall_score, 4),
                    "total_latency_ms": sc.total_latency_ms,
                    "timestamp": sc.timestamp.isoformat() if sc.timestamp else "",
                }

                if self.include_stage_scores:
                    for stage in stage_names:
                        if stage in all_results:
                            row[f"{stage}_passed"] = all_results[stage].passed
                            row[f"{stage}_score"] = round(all_results[stage].score, 4)
                        else:
                            row[f"{stage}_passed"] = ""
                            row[f"{stage}_score"] = ""

                writer.writerow(row)


# =============================================================================
# Summary Exporter
# =============================================================================


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""

    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    pass_rate: float = 0.0
    average_score: float = 0.0

    # Per-stage stats
    stage_stats: dict[str, StageStats] = field(default_factory=dict)

    # Per-pack stats (if multiple packs)
    pack_stats: dict[str, PackStats] = field(default_factory=dict)

    # Timing
    total_latency_ms: int = 0
    average_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": round(self.pass_rate, 4),
            "average_score": round(self.average_score, 4),
            "total_latency_ms": self.total_latency_ms,
            "average_latency_ms": round(self.average_latency_ms, 2),
            "stage_stats": {k: asdict(v) for k, v in self.stage_stats.items()},
            "pack_stats": {k: asdict(v) for k, v in self.pack_stats.items()},
        }


@dataclass
class StageStats:
    """Statistics for a single evaluation stage."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    average_score: float = 0.0


@dataclass
class PackStats:
    """Statistics for a single evaluation pack."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0


class SummaryExporter:
    """Generate summary statistics from evaluation results."""

    def summarize(self, scorecards: list[Scorecard]) -> EvaluationSummary:
        """
        Generate summary from scorecards.

        Args:
            scorecards: List of scorecards to summarize

        Returns:
            EvaluationSummary with aggregated statistics
        """
        summary = EvaluationSummary()

        if not scorecards:
            return summary

        summary.total_tests = len(scorecards)

        # Aggregate stage and pack stats
        stage_totals: dict[str, list[StageResult]] = {}
        pack_totals: dict[str, list[bool]] = {}
        all_scores: list[float] = []
        all_latencies: list[int] = []

        for sc in scorecards:
            all_results = sc.get_all_stage_results()
            overall_passed = all(sr.passed for sr in all_results.values())

            if overall_passed:
                summary.passed_tests += 1
            else:
                summary.failed_tests += 1

            # Overall score
            if all_results:
                avg = sum(sr.score for sr in all_results.values()) / len(all_results)
                all_scores.append(avg)

            # Latency
            if sc.total_latency_ms:
                all_latencies.append(sc.total_latency_ms)

            # Stage stats
            for stage_name, result in all_results.items():
                if stage_name not in stage_totals:
                    stage_totals[stage_name] = []
                stage_totals[stage_name].append(result)

            # Pack stats
            pack = sc.pack_name
            if pack not in pack_totals:
                pack_totals[pack] = []
            pack_totals[pack].append(overall_passed)

        # Compute pass rate
        summary.pass_rate = summary.passed_tests / summary.total_tests

        # Compute average score
        if all_scores:
            summary.average_score = sum(all_scores) / len(all_scores)

        # Compute latency stats
        if all_latencies:
            summary.total_latency_ms = sum(all_latencies)
            summary.average_latency_ms = summary.total_latency_ms / len(all_latencies)

        # Compute stage stats
        for stage_name, results in stage_totals.items():
            stats = StageStats(
                total=len(results),
                passed=sum(1 for r in results if r.passed),
                failed=sum(1 for r in results if not r.passed),
            )
            stats.pass_rate = stats.passed / stats.total if stats.total > 0 else 0.0
            stats.average_score = sum(r.score for r in results) / len(results) if results else 0.0
            summary.stage_stats[stage_name] = stats

        # Compute pack stats
        for pack_name, passed_list in pack_totals.items():
            stats = PackStats(
                total=len(passed_list),
                passed=sum(1 for p in passed_list if p),
                failed=sum(1 for p in passed_list if not p),
            )
            stats.pass_rate = stats.passed / stats.total if stats.total > 0 else 0.0
            summary.pack_stats[pack_name] = stats

        return summary

    def format_summary(self, summary: EvaluationSummary) -> str:
        """Format summary as human-readable text."""
        lines = [
            "=" * 60,
            "EVALUATION SUMMARY",
            "=" * 60,
            f"Total Tests:    {summary.total_tests}",
            f"Passed:         {summary.passed_tests} ({summary.pass_rate:.1%})",
            f"Failed:         {summary.failed_tests}",
            f"Average Score:  {summary.average_score:.2%}",
            f"Avg Latency:    {summary.average_latency_ms:.1f}ms",
            "",
        ]

        if summary.stage_stats:
            lines.append("STAGE BREAKDOWN:")
            lines.append("-" * 40)
            for stage_name, stats in sorted(summary.stage_stats.items()):
                lines.append(
                    f"  {stage_name:15} {stats.passed:4}/{stats.total:4} "
                    f"({stats.pass_rate:.1%})  avg: {stats.average_score:.2%}"
                )
            lines.append("")

        if len(summary.pack_stats) > 1:
            lines.append("PACK BREAKDOWN:")
            lines.append("-" * 40)
            for pack_name, stats in sorted(summary.pack_stats.items()):
                lines.append(
                    f"  {pack_name:15} {stats.passed:4}/{stats.total:4} ({stats.pass_rate:.1%})"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
