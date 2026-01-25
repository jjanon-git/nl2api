"""Tests for HttpEntityResolver client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.nl2api.resolution.http_client import HttpEntityResolver


class TestHttpEntityResolver:
    """Test suite for HttpEntityResolver."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.base_url = "http://localhost:8085"
        self.resolver = HttpEntityResolver(
            base_url=self.base_url,
            timeout_seconds=5.0,
        )

    async def test_resolve_single_found(self) -> None:
        """Test resolve_single returns ResolvedEntity when found."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "found": True,
            "original": "Apple",
            "identifier": "AAPL.O",
            "entity_type": "company",
            "confidence": 0.95,
            "alternatives": ["AAPL.OQ"],
            "metadata": {"ticker": "AAPL"},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(self.resolver, "_get_client", return_value=mock_client):
            result = await self.resolver.resolve_single("Apple")

        assert result is not None
        assert result.original == "Apple"
        assert result.identifier == "AAPL.O"
        assert result.entity_type == "company"
        assert result.confidence == 0.95
        assert result.alternatives == ("AAPL.OQ",)
        assert result.metadata == {"ticker": "AAPL"}

    async def test_resolve_single_not_found(self) -> None:
        """Test resolve_single returns None when not found."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "found": False,
            "original": "UnknownCompany",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(self.resolver, "_get_client", return_value=mock_client):
            result = await self.resolver.resolve_single("UnknownCompany")

        assert result is None

    async def test_resolve_extracts_entities_from_query(self) -> None:
        """Test resolve extracts entities from a natural language query."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "query": "Compare Apple and Microsoft",
            "entities": {"Apple": "AAPL.O", "Microsoft": "MSFT.O"},
            "count": 2,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(self.resolver, "_get_client", return_value=mock_client):
            result = await self.resolver.resolve("Compare Apple and Microsoft")

        assert result == {"Apple": "AAPL.O", "Microsoft": "MSFT.O"}

    async def test_resolve_batch(self) -> None:
        """Test resolve_batch resolves multiple entities."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "resolved": [
                {
                    "original": "Apple",
                    "identifier": "AAPL.O",
                    "entity_type": "company",
                    "confidence": 0.95,
                },
                {
                    "original": "Microsoft",
                    "identifier": "MSFT.O",
                    "entity_type": "company",
                    "confidence": 0.95,
                },
            ],
            "count": 2,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(self.resolver, "_get_client", return_value=mock_client):
            results = await self.resolver.resolve_batch(["Apple", "Microsoft"])

        assert len(results) == 2
        assert results[0].original == "Apple"
        assert results[0].identifier == "AAPL.O"
        assert results[1].original == "Microsoft"
        assert results[1].identifier == "MSFT.O"

    async def test_circuit_breaker_opens_on_failures(self) -> None:
        """Test circuit breaker opens after repeated failures."""
        resolver = HttpEntityResolver(
            base_url=self.base_url,
            circuit_failure_threshold=2,
            retry_max_attempts=1,
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch.object(resolver, "_get_client", return_value=mock_client):
            # First two failures should trigger circuit to open
            await resolver.resolve_single("Apple")
            await resolver.resolve_single("Microsoft")

            # Circuit should now be open
            stats = resolver.circuit_breaker_stats
            assert stats["state"] == "open"
            assert stats["failure_count"] >= 2

    async def test_handles_timeout_gracefully(self) -> None:
        """Test timeout is handled gracefully."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        with patch.object(self.resolver, "_get_client", return_value=mock_client):
            result = await self.resolver.resolve_single("Apple")

        assert result is None

    async def test_health_check_success(self) -> None:
        """Test health check returns True when service is healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(self.resolver, "_get_client", return_value=mock_client):
            result = await self.resolver.health_check()

        assert result is True

    async def test_health_check_failure(self) -> None:
        """Test health check returns False when service is down."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch.object(self.resolver, "_get_client", return_value=mock_client):
            result = await self.resolver.health_check()

        assert result is False

    async def test_close_client(self) -> None:
        """Test close properly closes the HTTP client."""
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        self.resolver._client = mock_client
        await self.resolver.close()

        mock_client.aclose.assert_called_once()
        assert self.resolver._client is None

    async def test_reset_circuit_breaker(self) -> None:
        """Test circuit breaker can be reset."""
        # Trigger some failures first
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch.object(self.resolver, "_get_client", return_value=mock_client):
            await self.resolver.resolve_single("Apple")

        # Reset
        self.resolver.reset_circuit_breaker()

        stats = self.resolver.circuit_breaker_stats
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0


class TestHttpEntityResolverWithApiKey:
    """Test HttpEntityResolver with API key authentication."""

    async def test_api_key_added_to_headers(self) -> None:
        """Test API key is added to request headers."""
        resolver = HttpEntityResolver(
            base_url="http://localhost:8085",
            api_key="test-api-key",
        )

        # Get the client - it should have the auth header
        client = await resolver._get_client()
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer test-api-key"

        await resolver.close()
