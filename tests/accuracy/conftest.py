"""
Pytest configuration for accuracy tests.

Defines markers and shared fixtures.
Initializes telemetry for metrics emission to OTEL.
"""

import os

import pytest

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
