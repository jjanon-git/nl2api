"""
OpenTelemetry Setup and Configuration.

Handles initialization of tracers, meters, and exporters.
Supports graceful degradation when OTEL dependencies are not installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Track whether OTEL is available
_otel_available = False
_telemetry_initialized = False

# Try to import OpenTelemetry
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _otel_available = True
except ImportError:
    logger.debug("OpenTelemetry not installed, telemetry will be disabled")


@dataclass
class TelemetryConfig:
    """Configuration for telemetry setup."""

    # Service identification
    service_name: str = "nl2api"
    service_version: str = "0.1.0"
    environment: str = field(default_factory=lambda: os.getenv("NL2API_ENV", "development"))

    # OTLP exporter settings
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    )
    otlp_insecure: bool = True

    # Feature flags
    tracing_enabled: bool = True
    metrics_enabled: bool = True

    # Export settings
    metrics_export_interval_ms: int = 10000  # 10 seconds

    # Additional resource attributes
    resource_attributes: dict[str, str] = field(default_factory=dict)


# Global state
_config: TelemetryConfig | None = None
_tracer_provider: Any = None
_meter_provider: Any = None


def init_telemetry(
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
    config: TelemetryConfig | None = None,
) -> bool:
    """
    Initialize OpenTelemetry instrumentation.

    Call this once at application startup. Can be disabled by setting
    EVALKIT_TELEMETRY_ENABLED=false environment variable.

    Args:
        service_name: Service name for telemetry (overrides config)
        otlp_endpoint: OTLP collector endpoint (overrides config)
        config: Full telemetry configuration

    Returns:
        True if telemetry was initialized, False if OTEL not available or disabled
    """
    global _telemetry_initialized, _config, _tracer_provider, _meter_provider

    if _telemetry_initialized:
        logger.debug("Telemetry already initialized")
        return _otel_available

    # Check if telemetry is disabled via environment variable
    telemetry_enabled = os.getenv("EVALKIT_TELEMETRY_ENABLED", "true").lower()
    if telemetry_enabled in ("false", "0", "no", "off"):
        logger.info("Telemetry disabled via EVALKIT_TELEMETRY_ENABLED")
        _telemetry_initialized = True
        return False

    if not _otel_available:
        logger.warning("OpenTelemetry not installed, telemetry disabled")
        _telemetry_initialized = True
        return False

    # Build config
    _config = config or TelemetryConfig()
    if service_name:
        _config.service_name = service_name
    if otlp_endpoint:
        _config.otlp_endpoint = otlp_endpoint

    try:
        # Create resource with service info
        resource_attrs = {
            SERVICE_NAME: _config.service_name,
            SERVICE_VERSION: _config.service_version,
            "deployment.environment": _config.environment,
        }
        resource_attrs.update(_config.resource_attributes)
        resource = Resource.create(resource_attrs)

        # Initialize tracing
        if _config.tracing_enabled:
            _tracer_provider = TracerProvider(resource=resource)
            span_exporter = OTLPSpanExporter(
                endpoint=_config.otlp_endpoint,
                insecure=_config.otlp_insecure,
            )
            _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
            trace.set_tracer_provider(_tracer_provider)
            logger.info(f"Tracing initialized, exporting to {_config.otlp_endpoint}")

        # Initialize metrics
        if _config.metrics_enabled:
            # Import temporality preference and SDK instrument types for cumulative counters
            # (OTLP defaults to delta which doesn't work well with Prometheus)
            # NOTE: Must use SDK instrument classes, not API classes (opentelemetry.sdk.metrics.*)
            from opentelemetry.sdk.metrics import (
                Counter,
                Histogram,
                ObservableCounter,
                ObservableGauge,
                ObservableUpDownCounter,
                UpDownCounter,
            )
            from opentelemetry.sdk.metrics.export import AggregationTemporality

            # Use cumulative temporality for all metric types
            # This ensures Prometheus can properly aggregate counters
            preferred_temporality = {
                Counter: AggregationTemporality.CUMULATIVE,
                UpDownCounter: AggregationTemporality.CUMULATIVE,
                Histogram: AggregationTemporality.CUMULATIVE,
                ObservableCounter: AggregationTemporality.CUMULATIVE,
                ObservableUpDownCounter: AggregationTemporality.CUMULATIVE,
                ObservableGauge: AggregationTemporality.CUMULATIVE,
            }

            metric_exporter = OTLPMetricExporter(
                endpoint=_config.otlp_endpoint,
                insecure=_config.otlp_insecure,
                preferred_temporality=preferred_temporality,
            )
            reader = PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=_config.metrics_export_interval_ms,
            )

            # Custom bucket boundaries for score histograms (0-1 range)
            score_buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            score_view = View(
                instrument_name="eval_test_score",
                aggregation=ExplicitBucketHistogramAggregation(score_buckets),
            )

            _meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[reader],
                views=[score_view],
            )
            metrics.set_meter_provider(_meter_provider)
            logger.info(f"Metrics initialized, exporting to {_config.otlp_endpoint}")

        _telemetry_initialized = True
        return True

    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
        _telemetry_initialized = True
        return False


def shutdown_telemetry() -> None:
    """
    Shutdown telemetry and flush pending data.

    Call this at application shutdown. Forces a flush of all pending
    metrics and traces before shutting down providers.
    """
    global _tracer_provider, _meter_provider, _telemetry_initialized

    if not _otel_available or not _telemetry_initialized:
        return

    try:
        # Force flush before shutdown to ensure all data is exported
        if _meter_provider:
            _meter_provider.force_flush(timeout_millis=5000)
            logger.debug("Meter provider flushed")
            _meter_provider.shutdown()
            logger.debug("Meter provider shut down")

        if _tracer_provider:
            _tracer_provider.force_flush(timeout_millis=5000)
            logger.debug("Tracer provider flushed")
            _tracer_provider.shutdown()
            logger.debug("Tracer provider shut down")

        _telemetry_initialized = False

    except Exception as e:
        logger.warning(f"Error during telemetry shutdown: {e}")


def _is_telemetry_disabled_by_env() -> bool:
    """Check if telemetry is disabled via environment variable."""
    telemetry_enabled = os.getenv("EVALKIT_TELEMETRY_ENABLED", "true").lower()
    return telemetry_enabled in ("false", "0", "no", "off")


def get_tracer(name: str = "nl2api") -> Any:
    """
    Get a tracer for creating spans.

    Args:
        name: Tracer name (typically module or component name)

    Returns:
        OpenTelemetry Tracer or NoOpTracer if OTEL not available or disabled
    """
    if not _otel_available or _is_telemetry_disabled_by_env():
        return _NoOpTracer()

    return trace.get_tracer(name)


def get_meter(name: str = "nl2api") -> Any:
    """
    Get a meter for creating metrics.

    Args:
        name: Meter name (typically module or component name)

    Returns:
        OpenTelemetry Meter or NoOpMeter if OTEL not available or disabled
    """
    if not _otel_available or _is_telemetry_disabled_by_env():
        return _NoOpMeter()

    return metrics.get_meter(name)


def is_telemetry_enabled() -> bool:
    """Check if telemetry is available and initialized."""
    return _otel_available and _telemetry_initialized


# === No-Op Implementations for Graceful Degradation ===


class _NoOpSpan:
    """No-op span for when OTEL is not available."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer for when OTEL is not available."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpCounter:
    """No-op counter for when OTEL is not available."""

    def add(self, _amount: int | float, _attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpHistogram:
    """No-op histogram for when OTEL is not available."""

    def record(self, _value: int | float, _attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpGauge:
    """No-op gauge for when OTEL is not available."""

    def set(self, _value: int | float, _attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpMeter:
    """No-op meter for when OTEL is not available."""

    def create_counter(self, name: str, **kwargs: Any) -> _NoOpCounter:
        return _NoOpCounter()

    def create_histogram(self, name: str, **kwargs: Any) -> _NoOpHistogram:
        return _NoOpHistogram()

    def create_up_down_counter(self, name: str, **kwargs: Any) -> _NoOpCounter:
        return _NoOpCounter()

    def create_observable_gauge(self, name: str, **kwargs: Any) -> _NoOpGauge:
        return _NoOpGauge()
