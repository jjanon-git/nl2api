"""
Resilience Utilities

Inlined circuit breaker and retry logic for the entity resolution service.
These are simplified versions that don't depend on external packages.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker for failing fast on unhealthy dependencies.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing fast, requests rejected immediately
    - HALF_OPEN: Testing if dependency recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: str = "circuit",
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self._stats

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._stats.state == CircuitState.OPEN

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result of function call

        Raises:
            CircuitOpenError: If circuit is open
        """
        async with self._lock:
            self._stats.total_calls += 1

            # Check if we should try to recover
            if self._stats.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._stats.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit {self._name}: attempting recovery")
                else:
                    raise CircuitOpenError(f"Circuit {self._name} is open")

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception:
            await self._record_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._stats.last_failure_time is None:
            return True
        elapsed = time.time() - self._stats.last_failure_time
        return elapsed >= self._recovery_timeout

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.success_count += 1
            self._stats.total_successes += 1
            self._stats.last_success_time = time.time()

            if self._stats.state == CircuitState.HALF_OPEN:
                # Recovery successful
                self._stats.state = CircuitState.CLOSED
                self._stats.failure_count = 0
                logger.info(f"Circuit {self._name}: recovered, now closed")

    async def _record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.failure_count += 1
            self._stats.total_failures += 1
            self._stats.last_failure_time = time.time()

            if self._stats.state == CircuitState.HALF_OPEN:
                # Recovery failed
                self._stats.state = CircuitState.OPEN
                logger.warning(f"Circuit {self._name}: recovery failed, reopening")
            elif self._stats.failure_count >= self._failure_threshold:
                self._stats.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit {self._name}: opened after {self._stats.failure_count} failures"
                )

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        self._stats = CircuitBreakerStats()
        logger.info(f"Circuit {self._name}: manually reset")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 5.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (ConnectionError, TimeoutError, OSError)
    )


async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        config: Retry configuration
        *args, **kwargs: Arguments to pass to function

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries exhausted
    """
    cfg = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(cfg.max_attempts):
        try:
            return await func(*args, **kwargs)
        except cfg.retryable_exceptions as e:
            last_exception = e

            if attempt == cfg.max_attempts - 1:
                # Last attempt, don't sleep
                break

            # Calculate delay with exponential backoff
            delay = min(
                cfg.base_delay * (cfg.exponential_base**attempt),
                cfg.max_delay,
            )

            # Add jitter to prevent thundering herd
            if cfg.jitter:
                delay = delay * (0.5 + random.random())

            logger.debug(
                f"Retry {attempt + 1}/{cfg.max_attempts} after {delay:.2f}s: {e}"
            )
            await asyncio.sleep(delay)

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry exhausted without exception")
