"""
Tests for resilience patterns (circuit breaker and retry).
"""

import asyncio
import pytest

from src.common.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    retry_with_backoff,
    RetryConfig,
)
from src.common.resilience.circuit_breaker import CircuitState


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with low thresholds for testing."""
        return CircuitBreaker(
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=0.1,  # 100ms for faster tests
            name="test",
        )

    @pytest.mark.asyncio
    async def test_closed_state_allows_requests(self, breaker):
        """Circuit breaker in closed state should allow requests."""
        async def success():
            return "success"

        result = await breaker.call(success)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self, breaker):
        """Circuit breaker should open after failure threshold."""
        async def failing():
            raise ConnectionError("Connection failed")

        # First two failures
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        assert breaker.state == CircuitState.CLOSED

        # Third failure opens the circuit
        with pytest.raises(ConnectionError):
            await breaker.call(failing)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_state_rejects_requests(self, breaker):
        """Open circuit should reject requests immediately."""
        async def failing():
            raise ConnectionError("Connection failed")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        # Now requests should be rejected
        with pytest.raises(CircuitOpenError):
            await breaker.call(failing)

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self, breaker):
        """Circuit should transition to half-open after recovery timeout."""
        async def failing():
            raise ConnectionError("Connection failed")

        async def success():
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)  # 150ms > 100ms recovery timeout

        # Next request should be allowed (half-open)
        result = await breaker.call(success)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_closes_after_success_threshold(self, breaker):
        """Circuit should close after success threshold in half-open."""
        async def failing():
            raise ConnectionError("Connection failed")

        async def success():
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Success in half-open state (need 2 successes)
        await breaker.call(success)
        await breaker.call(success)

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens(self, breaker):
        """Failure in half-open should reopen the circuit."""
        async def failing():
            raise ConnectionError("Connection failed")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Failure in half-open reopens
        with pytest.raises(ConnectionError):
            await breaker.call(failing)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_excluded_exceptions_not_counted(self):
        """Excluded exceptions should not count as failures."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )

        async def raise_value_error():
            raise ValueError("Not a failure")

        # These should not count as failures
        for _ in range(5):
            with pytest.raises(ValueError):
                await breaker.call(raise_value_error)

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reset_closes_circuit(self, breaker):
        """Manual reset should close the circuit."""
        async def failing():
            raise ConnectionError("Connection failed")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    def test_stats_tracking(self, breaker):
        """Stats should track calls and failures."""
        stats = breaker.stats
        assert stats.state == CircuitState.CLOSED
        assert stats.total_calls == 0
        assert stats.total_failures == 0


class TestRetryWithBackoff:
    """Tests for retry_with_backoff."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        """Should return immediately if first call succeeds."""
        call_count = 0

        async def success():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(success)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Should retry on retryable exceptions."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        result = await retry_with_backoff(fail_then_succeed, config=config)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        """Should raise after max attempts exhausted."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection failed")

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        with pytest.raises(ConnectionError):
            await retry_with_backoff(always_fail, config=config)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_for_non_retryable(self):
        """Should not retry for non-retryable exceptions."""
        call_count = 0

        async def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        config = RetryConfig(max_attempts=3)
        with pytest.raises(ValueError):
            await retry_with_backoff(raise_value_error, config=config)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Should call on_retry callback on each retry."""
        retries = []

        async def fail_twice():
            if len(retries) < 2:
                raise ConnectionError("Connection failed")
            return "success"

        def on_retry(attempt, error, delay):
            retries.append((attempt, str(error), delay))

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        result = await retry_with_backoff(
            fail_twice, config=config, on_retry=on_retry
        )
        assert result == "success"
        assert len(retries) == 2

    @pytest.mark.asyncio
    async def test_custom_retryable_exceptions(self):
        """Should retry only specified exception types."""
        call_count = 0

        async def raise_custom():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Custom error")
            return "success"

        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(RuntimeError,),
        )
        result = await retry_with_backoff(raise_custom, config=config)
        assert result == "success"
        assert call_count == 2
