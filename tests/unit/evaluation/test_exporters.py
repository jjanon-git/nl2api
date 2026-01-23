"""
Unit tests for result exporters.

Tests JSON, CSV, and Summary export functionality.
"""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from CONTRACTS import (
    Scorecard,
    StageResult,
)
from src.evaluation.core.exporters import (
    CSVExporter,
    JSONExporter,
    SummaryExporter,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_scorecards() -> list[Scorecard]:
    """Create sample scorecards for testing."""
    return [
        Scorecard(
            test_case_id="test-001",
            batch_id="batch-001",
            pack_name="nl2api",
            stage_results={
                "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
                "logic": StageResult(stage_name="logic", passed=True, score=0.9),
            },
            syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            logic_result=StageResult(stage_name="logic", passed=True, score=0.9),
            total_latency_ms=100,
        ),
        Scorecard(
            test_case_id="test-002",
            batch_id="batch-001",
            pack_name="nl2api",
            stage_results={
                "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
                "logic": StageResult(stage_name="logic", passed=False, score=0.5),
            },
            syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            logic_result=StageResult(stage_name="logic", passed=False, score=0.5),
            total_latency_ms=150,
        ),
        Scorecard(
            test_case_id="test-003",
            batch_id="batch-001",
            pack_name="nl2api",
            stage_results={
                "syntax": StageResult(stage_name="syntax", passed=False, score=0.0),
            },
            syntax_result=StageResult(stage_name="syntax", passed=False, score=0.0),
            total_latency_ms=50,
        ),
    ]


# =============================================================================
# JSON Exporter Tests
# =============================================================================


class TestJSONExporter:
    """Tests for JSONExporter."""

    @pytest.mark.asyncio
    async def test_export_creates_file(self, sample_scorecards):
        """Export creates JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            exporter = JSONExporter()

            await exporter.export(sample_scorecards, output_path)

            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_export_valid_json(self, sample_scorecards):
        """Export produces valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            exporter = JSONExporter()

            await exporter.export(sample_scorecards, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert "exported_at" in data
            assert "total_scorecards" in data
            assert "scorecards" in data
            assert len(data["scorecards"]) == 3

    @pytest.mark.asyncio
    async def test_export_scorecard_structure(self, sample_scorecards):
        """Exported scorecards have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            exporter = JSONExporter()

            await exporter.export(sample_scorecards, output_path)

            with open(output_path) as f:
                data = json.load(f)

            scorecard = data["scorecards"][0]
            assert "test_case_id" in scorecard
            assert "pack_name" in scorecard
            assert "stages" in scorecard
            assert "overall_passed" in scorecard
            assert "overall_score" in scorecard

    @pytest.mark.asyncio
    async def test_export_without_full_results(self, sample_scorecards):
        """Export without full results omits generated_output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            exporter = JSONExporter(include_full_results=False)

            await exporter.export(sample_scorecards, output_path)

            with open(output_path) as f:
                data = json.load(f)

            scorecard = data["scorecards"][0]
            assert "generated_output" not in scorecard

    @pytest.mark.asyncio
    async def test_export_empty_list(self):
        """Export handles empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            exporter = JSONExporter()

            await exporter.export([], output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert data["total_scorecards"] == 0
            assert data["scorecards"] == []


# =============================================================================
# CSV Exporter Tests
# =============================================================================


class TestCSVExporter:
    """Tests for CSVExporter."""

    @pytest.mark.asyncio
    async def test_export_creates_file(self, sample_scorecards):
        """Export creates CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            exporter = CSVExporter()

            await exporter.export(sample_scorecards, output_path)

            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_export_correct_rows(self, sample_scorecards):
        """Export produces correct number of rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            exporter = CSVExporter()

            await exporter.export(sample_scorecards, output_path)

            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_export_has_stage_columns(self, sample_scorecards):
        """Export includes stage score columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            exporter = CSVExporter()

            await exporter.export(sample_scorecards, output_path)

            with open(output_path) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames

            assert "syntax_passed" in fieldnames
            assert "syntax_score" in fieldnames
            assert "logic_passed" in fieldnames
            assert "logic_score" in fieldnames

    @pytest.mark.asyncio
    async def test_export_without_stage_scores(self, sample_scorecards):
        """Export can omit stage score columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            exporter = CSVExporter(include_stage_scores=False)

            await exporter.export(sample_scorecards, output_path)

            with open(output_path) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames

            assert "syntax_passed" not in fieldnames
            assert "logic_score" not in fieldnames

    @pytest.mark.asyncio
    async def test_export_overall_values(self, sample_scorecards):
        """Export calculates overall pass/score correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            exporter = CSVExporter()

            await exporter.export(sample_scorecards, output_path)

            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # First row: all pass
            assert rows[0]["overall_passed"] == "True"
            # Second row: logic fails
            assert rows[1]["overall_passed"] == "False"
            # Third row: syntax fails (only stage)
            assert rows[2]["overall_passed"] == "False"

    @pytest.mark.asyncio
    async def test_export_empty_list(self):
        """Export handles empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            exporter = CSVExporter()

            await exporter.export([], output_path)

            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 0


# =============================================================================
# Summary Exporter Tests
# =============================================================================


class TestSummaryExporter:
    """Tests for SummaryExporter."""

    @pytest.fixture
    def exporter(self) -> SummaryExporter:
        return SummaryExporter()

    def test_summarize_totals(self, exporter, sample_scorecards):
        """Summary calculates totals correctly."""
        summary = exporter.summarize(sample_scorecards)

        assert summary.total_tests == 3
        assert summary.passed_tests == 1  # Only test-001 has all stages pass
        assert summary.failed_tests == 2

    def test_summarize_pass_rate(self, exporter, sample_scorecards):
        """Summary calculates pass rate correctly."""
        summary = exporter.summarize(sample_scorecards)

        # 1 out of 3 passed
        assert summary.pass_rate == pytest.approx(1 / 3)

    def test_summarize_average_score(self, exporter, sample_scorecards):
        """Summary calculates average score correctly."""
        summary = exporter.summarize(sample_scorecards)

        # test-001: (1.0 + 0.9) / 2 = 0.95
        # test-002: (1.0 + 0.5) / 2 = 0.75
        # test-003: 0.0 / 1 = 0.0
        # Average: (0.95 + 0.75 + 0.0) / 3 ≈ 0.567
        assert summary.average_score == pytest.approx(0.567, abs=0.01)

    def test_summarize_stage_stats(self, exporter, sample_scorecards):
        """Summary calculates stage stats correctly."""
        summary = exporter.summarize(sample_scorecards)

        assert "syntax" in summary.stage_stats
        assert "logic" in summary.stage_stats

        syntax_stats = summary.stage_stats["syntax"]
        assert syntax_stats.total == 3
        assert syntax_stats.passed == 2
        assert syntax_stats.failed == 1

        logic_stats = summary.stage_stats["logic"]
        assert logic_stats.total == 2  # test-003 doesn't have logic stage
        assert logic_stats.passed == 1
        assert logic_stats.failed == 1

    def test_summarize_latency(self, exporter, sample_scorecards):
        """Summary calculates latency correctly."""
        summary = exporter.summarize(sample_scorecards)

        assert summary.total_latency_ms == 300  # 100 + 150 + 50
        assert summary.average_latency_ms == 100.0

    def test_summarize_empty_list(self, exporter):
        """Summary handles empty list."""
        summary = exporter.summarize([])

        assert summary.total_tests == 0
        assert summary.passed_tests == 0
        assert summary.pass_rate == 0.0

    def test_format_summary(self, exporter, sample_scorecards):
        """Format summary produces readable output."""
        summary = exporter.summarize(sample_scorecards)
        text = exporter.format_summary(summary)

        assert "EVALUATION SUMMARY" in text
        assert "Total Tests:" in text
        assert "Passed:" in text
        assert "STAGE BREAKDOWN" in text
        assert "syntax" in text
        assert "logic" in text

    def test_to_dict(self, exporter, sample_scorecards):
        """Summary to_dict produces valid dict."""
        summary = exporter.summarize(sample_scorecards)
        data = summary.to_dict()

        assert "total_tests" in data
        assert "pass_rate" in data
        assert "stage_stats" in data
        assert isinstance(data["stage_stats"], dict)


# =============================================================================
# Multi-Pack Tests
# =============================================================================


class TestMultiPackExport:
    """Tests for exporting scorecards from multiple packs."""

    @pytest.fixture
    def multi_pack_scorecards(self) -> list[Scorecard]:
        """Create scorecards from different packs."""
        return [
            Scorecard(
                test_case_id="nl2api-001",
                pack_name="nl2api",
                stage_results={
                    "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
                },
                syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            ),
            Scorecard(
                test_case_id="rag-001",
                pack_name="rag",
                stage_results={
                    "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.8),
                    "faithfulness": StageResult(stage_name="faithfulness", passed=True, score=0.9),
                },
            ),
        ]

    @pytest.mark.asyncio
    async def test_csv_handles_different_stages(self, multi_pack_scorecards):
        """CSV export handles scorecards with different stage sets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            exporter = CSVExporter()

            await exporter.export(multi_pack_scorecards, output_path)

            with open(output_path) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)

            # Should have columns for all stages across all scorecards
            assert "syntax_passed" in fieldnames
            assert "retrieval_passed" in fieldnames
            assert "faithfulness_passed" in fieldnames

            assert len(rows) == 2

    def test_summary_includes_pack_stats(self, multi_pack_scorecards):
        """Summary includes per-pack statistics."""
        exporter = SummaryExporter()
        summary = exporter.summarize(multi_pack_scorecards)

        assert "nl2api" in summary.pack_stats
        assert "rag" in summary.pack_stats
        assert summary.pack_stats["nl2api"].total == 1
        assert summary.pack_stats["rag"].total == 1
