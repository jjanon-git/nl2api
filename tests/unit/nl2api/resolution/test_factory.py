"""Tests for entity resolver factory."""

from __future__ import annotations

import pytest

from src.nl2api.config import NL2APIConfig
from src.nl2api.resolution import (
    HttpEntityResolver,
    MockEntityResolver,
    create_entity_resolver,
)
from src.nl2api.resolution.resolver import ExternalEntityResolver


class TestCreateEntityResolver:
    """Test suite for create_entity_resolver factory."""

    def test_creates_http_resolver_when_endpoint_configured(self) -> None:
        """When entity_resolution_api_endpoint is set, use HttpEntityResolver."""
        config = NL2APIConfig(
            _env_file=None,
            entity_resolution_enabled=True,
            entity_resolution_api_endpoint="http://localhost:8085",
        )

        resolver = create_entity_resolver(config)

        assert isinstance(resolver, HttpEntityResolver)

    def test_creates_local_resolver_when_no_endpoint(self) -> None:
        """When no endpoint configured, use ExternalEntityResolver."""
        config = NL2APIConfig(
            _env_file=None,
            entity_resolution_enabled=True,
            entity_resolution_api_endpoint=None,
        )

        resolver = create_entity_resolver(config)

        assert isinstance(resolver, ExternalEntityResolver)

    def test_creates_mock_resolver_when_disabled(self) -> None:
        """When entity_resolution_enabled=False, return no-op resolver."""
        config = NL2APIConfig(
            _env_file=None,
            entity_resolution_enabled=False,
        )

        resolver = create_entity_resolver(config)

        assert isinstance(resolver, MockEntityResolver)

    def test_http_resolver_uses_config_values(self) -> None:
        """HTTP resolver should use timeout and retry settings from config."""
        config = NL2APIConfig(
            _env_file=None,
            entity_resolution_enabled=True,
            entity_resolution_api_endpoint="http://entity-service:8085",
            entity_resolution_api_key="secret-key",
            entity_resolution_timeout_seconds=10.0,
            entity_resolution_circuit_failure_threshold=10,
            entity_resolution_circuit_recovery_seconds=60.0,
            entity_resolution_retry_max_attempts=5,
        )

        resolver = create_entity_resolver(config)

        assert isinstance(resolver, HttpEntityResolver)
        assert resolver._base_url == "http://entity-service:8085"
        assert resolver._api_key == "secret-key"
        assert resolver._timeout_seconds == 10.0

    def test_local_resolver_uses_config_values(self) -> None:
        """Local resolver should use timeout and retry settings from config."""
        config = NL2APIConfig(
            _env_file=None,
            entity_resolution_enabled=True,
            entity_resolution_api_endpoint=None,
            entity_resolution_cache_enabled=False,
            entity_resolution_timeout_seconds=10.0,
        )

        resolver = create_entity_resolver(config)

        assert isinstance(resolver, ExternalEntityResolver)
        assert resolver._use_cache is False
        assert resolver._timeout_seconds == 10.0
