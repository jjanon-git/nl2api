"""
Pytest configuration for accuracy tests.

Defines markers and shared fixtures.
"""

import os

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "tier1: Quick accuracy check (~50 samples)")
    config.addinivalue_line("markers", "tier2: Standard accuracy evaluation (~200 samples)")
    config.addinivalue_line("markers", "tier3: Comprehensive accuracy evaluation (all samples)")


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
