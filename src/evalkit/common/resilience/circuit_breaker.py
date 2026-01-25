"""
Circuit Breaker Pattern

Prevents cascading failures by failing fast when a service is unhealthy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Service unhealthy, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


# Re-export from centralized exceptions module for backward compatibility
from src.evalkit.exceptions import CircuitOpenError


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    recovery_timeout: float = 30.0  # Seconds before trying half-open
    excluded_exceptions: tuple[type[Exception], ...] = ()  # Don't count these as failures


@dataclass
class CircuitStats:
    """Statistics for monitoring circuit breaker state."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation with configurable thresholds.

    States:
    - CLOSED: Normal operation. Requests pass through.
              After failure_threshold failures, transitions to OPEN.
    - OPEN: Failing fast. All requests raise CircuitOpenError.
            After recovery_timeout, transitions to HALF_OPEN.
    - HALF_OPEN: Testing recovery. One request allowed through.
                 Success → CLOSED, Failure → OPEN.

    Example:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

        async def call_service():
            return await breaker.call(my_api_call)

        try:
            result = await call_service()
        except CircuitOpenError:
            # Use fallback
            result = cached_value
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: float = 30.0,
        excluded_exceptions: tuple[type[Exception], ...] = (),
        name: str = "default",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Successes needed to close from half-open
            recovery_timeout: Seconds to wait before trying half-open
            excluded_exceptions: Exception types that don't count as failures
            name: Name for logging/metrics
        """
        self._config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
            excluded_exceptions=excluded_exceptions,
        )
        self._name = name
        self._lock = asyncio.Lock()

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get current statistics."""
        return CircuitStats(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=time.time() if self._total_successes > 0 else None,
            total_calls=self._total_calls,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
        )

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Any exception from the wrapped function
        """
        async with self._lock:
            await self._check_state()

        self._total_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            # Check if this exception should be excluded
            if isinstance(e, self._config.excluded_exceptions):
                raise
            await self._on_failure(e)
            raise

    async def _check_state(self) -> None:
        """Check if request should be allowed through."""
        if self._state == CircuitState.CLOSED:
            return

        if self._state == CircuitState.OPEN:
            # Check if we should try half-open
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self._config.recovery_timeout:
                    logger.info(
                        f"Circuit breaker '{self._name}' transitioning to HALF_OPEN "
                        f"after {elapsed:.1f}s recovery timeout"
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    return

            retry_after = None
            if self._last_failure_time is not None:
                retry_after = self._config.recovery_timeout - (
                    time.time() - self._last_failure_time
                )

            raise CircuitOpenError(
                f"Circuit breaker '{self._name}' is open",
                retry_after=retry_after,
            )

        # HALF_OPEN - allow request through for testing

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    logger.info(
                        f"Circuit breaker '{self._name}' closing after "
                        f"{self._success_count} successful calls"
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self._name}' recorded failure "
                f"({self._failure_count}/{self._config.failure_threshold}): {error}"
            )

            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    f"Circuit breaker '{self._name}' opening after failure in HALF_OPEN state"
                )
                self._state = CircuitState.OPEN

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    logger.warning(
                        f"Circuit breaker '{self._name}' opening after "
                        f"{self._failure_count} consecutive failures"
                    )
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info(f"Circuit breaker '{self._name}' manually reset to CLOSED")
