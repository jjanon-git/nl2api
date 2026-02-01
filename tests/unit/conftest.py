"""
Pytest configuration for unit tests.

Sets up fixture sampling for fast test runs and disables telemetry.
"""

import os


def pytest_configure(config):
    """Configure fixture sampling and telemetry for unit tests."""
    # Sample 50 fixtures per category for fast unit tests
    # This reduces 16K fixtures to ~350 total
    # Set FIXTURE_SAMPLE_SIZE=0 to load all fixtures (CI/accuracy tests)
    if "FIXTURE_SAMPLE_SIZE" not in os.environ:
        os.environ["FIXTURE_SAMPLE_SIZE"] = "50"

    # Disable telemetry for unit tests to avoid OTEL SDK conflicts with mocks
    # This ensures get_tracer() returns NoOpTracer instead of real tracer
    os.environ["EVALKIT_TELEMETRY_ENABLED"] = "false"
