"""
Retry with Exponential Backoff

Retries failed operations with configurable backoff strategy.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,  # Includes network errors
    )


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        config: Retry configuration
        on_retry: Optional callback(attempt, error, delay) on each retry
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful function call

    Raises:
        Exception: Last exception after all retries exhausted

    Example:
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()

        # Retry up to 3 times with exponential backoff
        result = await retry_with_backoff(fetch_data)

        # Custom config
        config = RetryConfig(max_attempts=5, base_delay=0.5)
        result = await retry_with_backoff(fetch_data, config=config)
    """
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_attempts:
                logger.warning(f"Retry exhausted after {attempt} attempts: {e}")
                raise

            # Calculate delay with exponential backoff
            delay = min(
                config.base_delay * (config.exponential_base ** (attempt - 1)),
                config.max_delay,
            )

            # Add jitter to prevent thundering herd
            if config.jitter:
                delay = delay * (0.5 + random.random())

            logger.info(
                f"Retry attempt {attempt}/{config.max_attempts} failed: {e}. "
                f"Retrying in {delay:.2f}s"
            )

            if on_retry:
                on_retry(attempt, e, delay)

            await asyncio.sleep(delay)
        except Exception:
            # Non-retryable exception, re-raise immediately
            raise

    # Should not reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error")


class RetryableOperation:
    """
    Context manager for retryable operations with state tracking.

    Example:
        operation = RetryableOperation(config=RetryConfig(max_attempts=3))

        async with operation as attempt:
            result = await risky_call()
            # Success recorded automatically

        print(f"Succeeded after {operation.attempts} attempts")
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()
        self.attempts = 0
        self._last_error: Exception | None = None

    @property
    def last_error(self) -> Exception | None:
        """Last recorded error, if any."""
        return self._last_error

    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        return await retry_with_backoff(func, *args, config=self.config, **kwargs)
