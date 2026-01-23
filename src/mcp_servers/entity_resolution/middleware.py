"""
Security Middleware for Entity Resolution MCP Server

Provides:
- Input validation with size limits
- Rate limiting (in-memory or Redis-backed)
- Request size limits
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Maximum requests per window
    requests_per_window: int = 100

    # Window size in seconds
    window_seconds: int = 60

    # Maximum request body size in bytes (1MB default)
    max_body_size: int = 1_048_576

    # Maximum entity name length
    max_entity_length: int = 500

    # Maximum entities in batch request
    max_batch_size: int = 100

    # Maximum query length for extract endpoint
    max_query_length: int = 2000


class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter using sliding window.

    For production with multiple instances, use RedisRateLimiter instead.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        # Track requests per client: {client_id: [(timestamp, count), ...]}
        self._requests: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed for this client.

        Returns:
            Tuple of (allowed, headers) where headers contains rate limit info
        """
        now = time.time()
        window_start = now - self.config.window_seconds

        async with self._lock:
            # Clean old entries
            self._requests[client_id] = [
                (ts, count) for ts, count in self._requests[client_id] if ts > window_start
            ]

            # Count requests in window
            total_requests = sum(count for _, count in self._requests[client_id])

            # Rate limit headers
            headers = {
                "X-RateLimit-Limit": str(self.config.requests_per_window),
                "X-RateLimit-Remaining": str(
                    max(0, self.config.requests_per_window - total_requests - 1)
                ),
                "X-RateLimit-Reset": str(int(now + self.config.window_seconds)),
            }

            if total_requests >= self.config.requests_per_window:
                logger.warning(
                    f"Rate limit exceeded for client {client_id}: "
                    f"{total_requests}/{self.config.requests_per_window}"
                )
                return False, headers

            # Record this request
            self._requests[client_id].append((now, 1))
            return True, headers


class RedisRateLimiter:
    """
    Redis-backed rate limiter for distributed deployments.

    Uses sliding window algorithm with Redis sorted sets.
    """

    def __init__(self, config: RateLimitConfig, redis_client: Any):
        self.config = config
        self._redis = redis_client

    async def is_allowed(self, client_id: str) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed for this client.

        Uses Redis sorted set with timestamps as scores.
        """
        now = time.time()
        window_start = now - self.config.window_seconds
        key = f"ratelimit:{client_id}"

        try:
            pipe = self._redis.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current entries
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(now): now})

            # Set expiry
            pipe.expire(key, self.config.window_seconds + 1)

            results = await pipe.execute()
            total_requests = results[1]  # zcard result

            headers = {
                "X-RateLimit-Limit": str(self.config.requests_per_window),
                "X-RateLimit-Remaining": str(
                    max(0, self.config.requests_per_window - total_requests - 1)
                ),
                "X-RateLimit-Reset": str(int(now + self.config.window_seconds)),
            }

            if total_requests >= self.config.requests_per_window:
                logger.warning(
                    f"Rate limit exceeded for client {client_id}: "
                    f"{total_requests}/{self.config.requests_per_window}"
                )
                # Remove the request we just added
                await self._redis.zrem(key, str(now))
                return False, headers

            return True, headers

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}, allowing request")
            # Fail open - allow request if Redis is down
            return True, {}


def create_rate_limiter(
    config: RateLimitConfig, redis_client: Any | None = None
) -> InMemoryRateLimiter | RedisRateLimiter:
    """
    Create appropriate rate limiter based on available resources.

    Args:
        config: Rate limit configuration
        redis_client: Optional Redis client for distributed rate limiting

    Returns:
        Rate limiter instance
    """
    if redis_client is not None:
        logger.info("Using Redis-backed rate limiter")
        return RedisRateLimiter(config, redis_client)
    else:
        logger.info("Using in-memory rate limiter (single instance only)")
        return InMemoryRateLimiter(config)


def create_security_middleware(
    config: RateLimitConfig,
    rate_limiter: InMemoryRateLimiter | RedisRateLimiter,
    exempt_paths: set[str] | None = None,
) -> Callable:
    """
    Create FastAPI middleware for security checks.

    Args:
        config: Rate limit and validation configuration
        rate_limiter: Rate limiter instance
        exempt_paths: Paths exempt from rate limiting (e.g., health checks)

    Returns:
        Middleware function
    """
    _exempt_paths = exempt_paths or {"/health", "/", "/docs", "/openapi.json"}

    async def security_middleware(request: Any, call_next: Callable) -> Any:
        """Apply security checks to incoming requests."""
        # Import here to avoid circular imports
        from fastapi.responses import JSONResponse

        path = request.url.path

        # Skip security for exempt paths
        if path in _exempt_paths:
            return await call_next(request)

        # Get client identifier for rate limiting
        client_id = (
            request.headers.get("X-Client-ID")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.client.host
            if request.client
            else "unknown"
        )

        # Check rate limit
        allowed, headers = await rate_limiter.is_allowed(client_id)
        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Max {config.requests_per_window} requests per {config.window_seconds} seconds.",
                    "retry_after": config.window_seconds,
                },
            )
            for key, value in headers.items():
                response.headers[key] = value
            return response

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > config.max_body_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large",
                    "message": f"Request body exceeds maximum size of {config.max_body_size} bytes",
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response

    return security_middleware


class InputValidator:
    """
    Validates input for API endpoints.

    Use this in endpoint handlers to validate request bodies.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config

    def validate_entity(self, entity: str) -> tuple[bool, str | None]:
        """
        Validate a single entity name.

        Returns:
            Tuple of (valid, error_message)
        """
        if not entity or not entity.strip():
            return False, "Entity name cannot be empty"

        if len(entity) > self.config.max_entity_length:
            return False, f"Entity name exceeds maximum length of {self.config.max_entity_length}"

        # Check for potential injection patterns
        suspicious_patterns = ["<script", "javascript:", "data:", "vbscript:"]
        entity_lower = entity.lower()
        for pattern in suspicious_patterns:
            if pattern in entity_lower:
                return False, "Entity name contains invalid characters"

        return True, None

    def validate_batch(self, entities: list[str]) -> tuple[bool, str | None]:
        """
        Validate a batch of entities.

        Returns:
            Tuple of (valid, error_message)
        """
        if not entities:
            return False, "Entities list cannot be empty"

        if len(entities) > self.config.max_batch_size:
            return False, f"Batch size exceeds maximum of {self.config.max_batch_size}"

        for i, entity in enumerate(entities):
            valid, error = self.validate_entity(entity)
            if not valid:
                return False, f"Entity at index {i}: {error}"

        return True, None

    def validate_query(self, query: str) -> tuple[bool, str | None]:
        """
        Validate a query string.

        Returns:
            Tuple of (valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"

        if len(query) > self.config.max_query_length:
            return False, f"Query exceeds maximum length of {self.config.max_query_length}"

        return True, None
