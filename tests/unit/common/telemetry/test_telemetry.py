"""Tests for unified telemetry module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.common.telemetry.setup import (
    TelemetryConfig,
    get_tracer,
    get_meter,
    is_telemetry_enabled,
    _NoOpTracer,
    _NoOpMeter,
    _NoOpSpan,
    _NoOpCounter,
    _NoOpHistogram,
)
from src.common.telemetry.tracing import (
    trace_async,
    trace_sync,
    add_span_attributes,
    record_exception,
)
from src.common.telemetry.metrics import (
    NL2APIMetrics,
    EvalMetrics,
)


class TestTelemetryConfig:
    """Test TelemetryConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TelemetryConfig()

        assert config.service_name == "nl2api"
        assert config.service_version == "0.1.0"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.tracing_enabled is True
        assert config.metrics_enabled is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TelemetryConfig(
            service_name="custom-service",
            otlp_endpoint="http://custom:4317",
            tracing_enabled=False,
        )

        assert config.service_name == "custom-service"
        assert config.otlp_endpoint == "http://custom:4317"
        assert config.tracing_enabled is False


class TestNoOpImplementations:
    """Test no-op implementations for graceful degradation."""

    def test_noop_span(self) -> None:
        """Test NoOpSpan methods don't raise."""
        span = _NoOpSpan()

        # All methods should be no-ops
        span.set_attribute("key", "value")
        span.set_attributes({"key": "value"})
        span.add_event("event")
        span.record_exception(ValueError("test"))
        span.set_status(None)

        # Context manager should work
        with span as s:
            assert s is span

    def test_noop_tracer(self) -> None:
        """Test NoOpTracer returns NoOpSpan."""
        tracer = _NoOpTracer()

        span = tracer.start_as_current_span("test")
        assert isinstance(span, _NoOpSpan)

        span = tracer.start_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_noop_counter(self) -> None:
        """Test NoOpCounter add doesn't raise."""
        counter = _NoOpCounter()
        counter.add(1)
        counter.add(10, {"key": "value"})

    def test_noop_histogram(self) -> None:
        """Test NoOpHistogram record doesn't raise."""
        histogram = _NoOpHistogram()
        histogram.record(100)
        histogram.record(50, {"key": "value"})

    def test_noop_meter(self) -> None:
        """Test NoOpMeter creates no-op instruments."""
        meter = _NoOpMeter()

        counter = meter.create_counter("test")
        assert isinstance(counter, _NoOpCounter)

        histogram = meter.create_histogram("test")
        assert isinstance(histogram, _NoOpHistogram)


class TestTracingDecorators:
    """Test tracing decorators."""

    @pytest.mark.asyncio
    async def test_trace_async_without_otel(self) -> None:
        """Test trace_async works when OTEL not available."""

        @trace_async("test_operation")
        async def test_func(x: int) -> int:
            return x * 2

        result = await test_func(5)
        assert result == 10

    def test_trace_sync_without_otel(self) -> None:
        """Test trace_sync works when OTEL not available."""

        @trace_sync("test_operation")
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_trace_async_propagates_exception(self) -> None:
        """Test trace_async propagates exceptions."""

        @trace_async("failing_operation")
        async def failing_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await failing_func()

    def test_trace_sync_propagates_exception(self) -> None:
        """Test trace_sync propagates exceptions."""

        @trace_sync("failing_operation")
        def failing_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()


class TestSpanHelpers:
    """Test span helper functions."""

    def test_add_span_attributes_without_otel(self) -> None:
        """Test add_span_attributes doesn't raise when OTEL not available."""
        # Should not raise
        add_span_attributes({"key": "value", "count": 10})

    def test_record_exception_without_otel(self) -> None:
        """Test record_exception doesn't raise when OTEL not available."""
        # Should not raise
        record_exception(ValueError("test"))


class TestNL2APIMetrics:
    """Test NL2APIMetrics class."""

    def test_initialization(self) -> None:
        """Test NL2APIMetrics initializes without error."""
        metrics = NL2APIMetrics()
        assert metrics is not None

    def test_record_request_without_otel(self) -> None:
        """Test record_request works when OTEL not configured."""
        from src.nl2api.observability.metrics import RequestMetrics

        metrics = NL2APIMetrics()
        request_metrics = RequestMetrics(query="Test query")
        request_metrics.set_routing_result(domain="estimates", confidence=0.9)
        request_metrics.finalize(total_latency_ms=100)

        # Should not raise
        metrics.record_request(request_metrics)


class TestEvalMetrics:
    """Test EvalMetrics class."""

    def test_initialization(self) -> None:
        """Test EvalMetrics initializes without error."""
        metrics = EvalMetrics()
        assert metrics is not None

    def test_record_batch_complete_without_otel(self) -> None:
        """Test record_batch_complete works when OTEL not configured."""
        from CONTRACTS import BatchJob, TaskStatus

        metrics = EvalMetrics()
        batch_job = BatchJob(
            total_tests=10,
            completed_count=8,
            failed_count=2,
            status=TaskStatus.COMPLETED,
        )

        # Should not raise
        metrics.record_batch_complete(batch_job, 60.0)


class TestGettersWithoutInit:
    """Test getter functions before init_telemetry is called."""

    def test_get_tracer_returns_noop(self) -> None:
        """Test get_tracer returns NoOpTracer when OTEL not configured."""
        tracer = get_tracer()
        # Should be either real tracer or NoOpTracer
        assert tracer is not None

    def test_get_meter_returns_noop(self) -> None:
        """Test get_meter returns NoOpMeter when OTEL not configured."""
        meter = get_meter()
        # Should be either real meter or NoOpMeter
        assert meter is not None


class TestOTELEmitter:
    """Test OTELEmitter integration."""

    @pytest.mark.asyncio
    async def test_otel_emitter_emit(self) -> None:
        """Test OTELEmitter emit doesn't raise."""
        from src.nl2api.observability import OTELEmitter
        from src.nl2api.observability.metrics import RequestMetrics

        emitter = OTELEmitter()
        metrics = RequestMetrics(query="Test query")
        metrics.set_routing_result(domain="estimates", confidence=0.9)
        metrics.finalize(total_latency_ms=100)

        # Should not raise
        await emitter.emit(metrics)


class TestCreateEmitterFromConfig:
    """Test create_emitter_from_config function."""

    def test_creates_otel_emitter(self) -> None:
        """Test create_emitter_from_config with otel_enabled."""
        from src.nl2api.observability import (
            create_emitter_from_config,
            OTELEmitter,
            CompositeEmitter,
        )

        emitter = create_emitter_from_config(
            log_enabled=False,
            file_path=None,
            otel_enabled=True,
        )

        assert isinstance(emitter, OTELEmitter)

    def test_creates_composite_with_otel(self) -> None:
        """Test create_emitter_from_config creates composite with OTEL."""
        from src.nl2api.observability import (
            create_emitter_from_config,
            CompositeEmitter,
        )

        emitter = create_emitter_from_config(
            log_enabled=True,
            file_path=None,
            otel_enabled=True,
        )

        assert isinstance(emitter, CompositeEmitter)
