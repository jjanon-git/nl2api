"""
Tests for security middleware (rate limiting, input validation).
"""

import asyncio
import time

import pytest

from src.mcp_servers.entity_resolution.middleware import (
    InMemoryRateLimiter,
    InputValidator,
    RateLimitConfig,
    create_rate_limiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()

        assert config.requests_per_window == 100
        assert config.window_seconds == 60
        assert config.max_body_size == 1_048_576
        assert config.max_entity_length == 500
        assert config.max_batch_size == 100
        assert config.max_query_length == 2000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_window=50,
            window_seconds=30,
            max_body_size=512_000,
        )

        assert config.requests_per_window == 50
        assert config.window_seconds == 30
        assert config.max_body_size == 512_000


class TestInMemoryRateLimiter:
    """Tests for in-memory rate limiter."""

    @pytest.fixture
    def config(self):
        """Create test config with low limits."""
        return RateLimitConfig(
            requests_per_window=5,
            window_seconds=1,
        )

    @pytest.fixture
    def limiter(self, config):
        """Create rate limiter."""
        return InMemoryRateLimiter(config)

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self, limiter):
        """Test requests under limit are allowed."""
        for i in range(5):
            allowed, headers = await limiter.is_allowed("client1")
            assert allowed is True, f"Request {i + 1} should be allowed"
            assert "X-RateLimit-Limit" in headers
            assert "X-RateLimit-Remaining" in headers
            assert "X-RateLimit-Reset" in headers

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self, limiter):
        """Test requests over limit are blocked."""
        # Make 5 requests (the limit)
        for _ in range(5):
            allowed, _ = await limiter.is_allowed("client1")
            assert allowed is True

        # 6th request should be blocked
        allowed, headers = await limiter.is_allowed("client1")
        assert allowed is False
        assert headers["X-RateLimit-Remaining"] == "0"

    @pytest.mark.asyncio
    async def test_different_clients_tracked_separately(self, limiter):
        """Test each client has separate rate limit."""
        # Use up client1's limit
        for _ in range(5):
            await limiter.is_allowed("client1")

        # client2 should still be allowed
        allowed, _ = await limiter.is_allowed("client2")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_window_resets_after_expiry(self, limiter):
        """Test rate limit resets after window expires."""
        # Use up limit
        for _ in range(5):
            await limiter.is_allowed("client1")

        # Should be blocked
        allowed, _ = await limiter.is_allowed("client1")
        assert allowed is False

        # Wait for window to expire
        await asyncio.sleep(1.1)

        # Should be allowed again
        allowed, _ = await limiter.is_allowed("client1")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_headers_show_correct_remaining(self, limiter):
        """Test headers show correct remaining count."""
        _, headers = await limiter.is_allowed("client1")
        assert headers["X-RateLimit-Limit"] == "5"
        assert headers["X-RateLimit-Remaining"] == "4"

        _, headers = await limiter.is_allowed("client1")
        assert headers["X-RateLimit-Remaining"] == "3"

        _, headers = await limiter.is_allowed("client1")
        assert headers["X-RateLimit-Remaining"] == "2"


class TestInputValidator:
    """Tests for input validation."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return RateLimitConfig(
            max_entity_length=100,
            max_batch_size=10,
            max_query_length=500,
        )

    @pytest.fixture
    def validator(self, config):
        """Create validator."""
        return InputValidator(config)

    def test_validate_entity_valid(self, validator):
        """Test valid entity names pass."""
        valid, error = validator.validate_entity("Apple Inc")
        assert valid is True
        assert error is None

    def test_validate_entity_empty(self, validator):
        """Test empty entity names fail."""
        valid, error = validator.validate_entity("")
        assert valid is False
        assert "empty" in error.lower()

    def test_validate_entity_whitespace_only(self, validator):
        """Test whitespace-only entity names fail."""
        valid, error = validator.validate_entity("   ")
        assert valid is False
        assert "empty" in error.lower()

    def test_validate_entity_too_long(self, validator):
        """Test overly long entity names fail."""
        long_name = "A" * 101
        valid, error = validator.validate_entity(long_name)
        assert valid is False
        assert "maximum length" in error.lower()

    def test_validate_entity_script_injection(self, validator):
        """Test script injection patterns are blocked."""
        valid, error = validator.validate_entity("<script>alert('xss')</script>")
        assert valid is False
        assert "invalid characters" in error.lower()

    def test_validate_entity_javascript_protocol(self, validator):
        """Test javascript: protocol is blocked."""
        valid, error = validator.validate_entity("javascript:alert(1)")
        assert valid is False
        assert "invalid characters" in error.lower()

    def test_validate_batch_valid(self, validator):
        """Test valid batch passes."""
        entities = ["Apple", "Microsoft", "Google"]
        valid, error = validator.validate_batch(entities)
        assert valid is True
        assert error is None

    def test_validate_batch_empty(self, validator):
        """Test empty batch fails."""
        valid, error = validator.validate_batch([])
        assert valid is False
        assert "empty" in error.lower()

    def test_validate_batch_too_large(self, validator):
        """Test batch exceeding max size fails."""
        entities = [f"Company{i}" for i in range(11)]
        valid, error = validator.validate_batch(entities)
        assert valid is False
        assert "maximum" in error.lower()

    def test_validate_batch_with_invalid_entity(self, validator):
        """Test batch with invalid entity fails."""
        entities = ["Apple", "", "Google"]
        valid, error = validator.validate_batch(entities)
        assert valid is False
        assert "index 1" in error.lower()

    def test_validate_query_valid(self, validator):
        """Test valid query passes."""
        valid, error = validator.validate_query("What is Apple's stock price?")
        assert valid is True
        assert error is None

    def test_validate_query_empty(self, validator):
        """Test empty query fails."""
        valid, error = validator.validate_query("")
        assert valid is False
        assert "empty" in error.lower()

    def test_validate_query_too_long(self, validator):
        """Test overly long query fails."""
        long_query = "A" * 501
        valid, error = validator.validate_query(long_query)
        assert valid is False
        assert "maximum length" in error.lower()


class TestCreateRateLimiter:
    """Tests for rate limiter factory."""

    def test_creates_in_memory_when_no_redis(self):
        """Test in-memory limiter created when Redis not available."""
        config = RateLimitConfig()
        limiter = create_rate_limiter(config, redis_client=None)
        assert isinstance(limiter, InMemoryRateLimiter)

    def test_creates_redis_limiter_when_redis_available(self):
        """Test Redis limiter created when Redis client provided."""
        from src.mcp_servers.entity_resolution.middleware import RedisRateLimiter

        config = RateLimitConfig()

        # Mock Redis client
        class MockRedis:
            pass

        limiter = create_rate_limiter(config, redis_client=MockRedis())
        assert isinstance(limiter, RedisRateLimiter)


class TestPydanticFieldValidation:
    """Tests for Pydantic model field validation."""

    def test_resolve_request_valid(self):
        """Test valid ResolveRequest."""
        from src.mcp_servers.entity_resolution.transports.sse import ResolveRequest

        req = ResolveRequest(entity="Apple Inc")
        assert req.entity == "Apple Inc"
        assert req.entity_type is None

    def test_resolve_request_with_type(self):
        """Test ResolveRequest with entity_type."""
        from src.mcp_servers.entity_resolution.transports.sse import ResolveRequest

        req = ResolveRequest(entity="Apple Inc", entity_type="company")
        assert req.entity_type == "company"

    def test_resolve_request_empty_entity_fails(self):
        """Test empty entity fails validation."""
        from pydantic import ValidationError

        from src.mcp_servers.entity_resolution.transports.sse import ResolveRequest

        with pytest.raises(ValidationError):
            ResolveRequest(entity="")

    def test_resolve_request_too_long_entity_fails(self):
        """Test overly long entity fails validation."""
        from pydantic import ValidationError

        from src.mcp_servers.entity_resolution.transports.sse import ResolveRequest

        with pytest.raises(ValidationError):
            ResolveRequest(entity="A" * 501)

    def test_batch_request_valid(self):
        """Test valid BatchResolveRequest."""
        from src.mcp_servers.entity_resolution.transports.sse import BatchResolveRequest

        req = BatchResolveRequest(entities=["Apple", "Microsoft"])
        assert len(req.entities) == 2

    def test_batch_request_empty_fails(self):
        """Test empty entities list fails validation."""
        from pydantic import ValidationError

        from src.mcp_servers.entity_resolution.transports.sse import BatchResolveRequest

        with pytest.raises(ValidationError):
            BatchResolveRequest(entities=[])

    def test_batch_request_too_large_fails(self):
        """Test batch exceeding max size fails validation."""
        from pydantic import ValidationError

        from src.mcp_servers.entity_resolution.transports.sse import BatchResolveRequest

        with pytest.raises(ValidationError):
            BatchResolveRequest(entities=["Company"] * 101)

    def test_extract_request_valid(self):
        """Test valid ExtractRequest."""
        from src.mcp_servers.entity_resolution.transports.sse import ExtractRequest

        req = ExtractRequest(query="What is Apple's stock price?")
        assert req.query == "What is Apple's stock price?"

    def test_extract_request_empty_fails(self):
        """Test empty query fails validation."""
        from pydantic import ValidationError

        from src.mcp_servers.entity_resolution.transports.sse import ExtractRequest

        with pytest.raises(ValidationError):
            ExtractRequest(query="")

    def test_extract_request_too_long_fails(self):
        """Test overly long query fails validation."""
        from pydantic import ValidationError

        from src.mcp_servers.entity_resolution.transports.sse import ExtractRequest

        with pytest.raises(ValidationError):
            ExtractRequest(query="A" * 2001)
