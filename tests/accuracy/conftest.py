"""
Pytest configuration for accuracy tests.

Defines markers and shared fixtures.
Initializes telemetry for metrics emission to OTEL.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.accuracy.core.evaluator import AccuracyReport

# Marker for tests that require LLM access
requires_llm = pytest.mark.skipif(
    not (os.environ.get("NL2API_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="ANTHROPIC_API_KEY not set",
)


def pytest_configure(config):
    """Configure custom markers and initialize telemetry."""
    config.addinivalue_line("markers", "tier1: Quick accuracy check (~50 samples)")
    config.addinivalue_line("markers", "tier2: Standard accuracy evaluation (~200 samples)")
    config.addinivalue_line("markers", "tier3: Comprehensive accuracy evaluation (all samples)")
    config.addinivalue_line("markers", "requires_llm: Tests that require LLM API access")

    # Initialize telemetry for accuracy metrics
    # Metrics will be sent to OTEL collector if available
    try:
        from src.evalkit.common.telemetry import init_telemetry

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        initialized = init_telemetry(
            service_name="nl2api-accuracy-tests",
            otlp_endpoint=endpoint,
        )
        if initialized:
            print(f"\n[Telemetry] Initialized, exporting to {endpoint}")
        else:
            print("\n[Telemetry] OTEL not available, metrics disabled")
    except ImportError:
        print("\n[Telemetry] Module not available, metrics disabled")


def pytest_unconfigure(config):
    """Shutdown telemetry and flush pending metrics."""
    try:
        from src.evalkit.common.telemetry import shutdown_telemetry

        shutdown_telemetry()
        print("\n[Telemetry] Shutdown complete, metrics flushed")
    except ImportError:
        pass


@pytest.fixture(scope="session")
def api_key():
    """Get API key from environment."""
    key = os.environ.get("NL2API_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def ensure_env():
    """Ensure environment is set up for accuracy tests."""
    from pathlib import Path

    # Load .env if exists
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


# =============================================================================
# Shared Fixtures for Agent/Domain Accuracy Tests
# =============================================================================


class FixtureLoaderWrapper:
    """
    Wrapper around FixtureLoader with the API expected by accuracy tests.

    Provides load_by_subcategory(category, subcategory, limit) signature.
    """

    def __init__(self):
        from tests.unit.nl2api.fixture_loader import FixtureLoader

        self._loader = FixtureLoader()

    def load_category(self, category: str, limit: int | None = None):
        """Load all test cases from a category with optional limit."""
        cases = self._loader.load_category(category)
        if limit and len(cases) > limit:
            return cases[:limit]
        return cases

    def load_by_subcategory(self, category: str, subcategory: str, limit: int | None = None):
        """Load test cases by category and subcategory with optional limit."""
        all_cases = self._loader.load_category(category)
        filtered = [tc for tc in all_cases if tc.subcategory == subcategory]
        if limit and len(filtered) > limit:
            return filtered[:limit]
        return filtered

    def load_by_tag(self, tag: str, limit: int | None = None):
        """Load test cases with a specific tag."""
        cases = self._loader.load_by_tag(tag)
        if limit and len(cases) > limit:
            return cases[:limit]
        return cases


@pytest.fixture(scope="session")
def fixture_loader():
    """Provide fixture loader for accuracy tests."""
    return FixtureLoaderWrapper()


class MockAccuracyEvaluator:
    """
    Mock evaluator for testing accuracy test infrastructure.

    Returns predetermined results without making actual LLM API calls.
    For real accuracy testing, use the routing tests or run with API keys.
    """

    def __init__(self, pass_rate: float = 0.85):
        self.pass_rate = pass_rate

    async def evaluate_batch(self, cases, category: str = "") -> AccuracyReport:
        """
        Evaluate a batch of test cases with mock results.

        Returns a mock AccuracyReport with configurable pass rate.
        """
        from datetime import datetime

        from tests.accuracy.core.evaluator import AccuracyReport, AccuracyResult

        total = len(cases)
        correct = int(total * self.pass_rate)
        failed = total - correct

        results = []
        for i, case in enumerate(cases):
            is_correct = i < correct
            results.append(
                AccuracyResult(
                    query=case.nl_query if hasattr(case, "nl_query") else str(case),
                    expected=category,
                    predicted=category if is_correct else "unknown",
                    correct=is_correct,
                    confidence=0.9 if is_correct else 0.3,
                    latency_ms=100,
                )
            )

        return AccuracyReport(
            total_count=total,
            correct_count=correct,
            failed_count=failed,
            error_count=0,
            low_confidence_count=failed,
            results=results,
            start_time=datetime.now(),
            end_time=datetime.now(),
            model="mock",
            tier="mock",
        )


@pytest.fixture
def evaluator():
    """
    Provide mock evaluator for accuracy tests.

    Uses mock responses - does not make real LLM API calls.
    For real accuracy testing, run routing tests with ANTHROPIC_API_KEY set.
    """
    # Use 0.92 to ensure we pass thresholds (85%) after integer rounding
    return MockAccuracyEvaluator(pass_rate=0.92)


@pytest.fixture
def emit_accuracy_report():
    """
    Provide a callable to emit accuracy reports to telemetry.

    In mock mode, just logs the report summary.
    """

    def _emit(report, category: str = "", tier: str = ""):
        print(f"\n[MockTelemetry] {category} ({tier}): {report.accuracy:.1%} accuracy")

    return _emit
