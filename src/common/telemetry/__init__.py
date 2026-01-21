"""
Unified Telemetry Module for NL2API.

Provides centralized OpenTelemetry instrumentation for:
- Request tracing (spans)
- Metrics (counters, histograms, gauges)
- Structured logging

Usage:
    from src.common.telemetry import init_telemetry, get_tracer, get_meter

    # Initialize at application startup
    init_telemetry(service_name="nl2api", otlp_endpoint="http://localhost:4317")

    # Get tracer for spans
    tracer = get_tracer()
    with tracer.start_as_current_span("process_query") as span:
        span.set_attribute("query", query)
        ...

    # Get meter for metrics
    meter = get_meter()
    requests_counter = meter.create_counter("nl2api.requests")
    requests_counter.add(1, {"domain": "estimates"})
"""

from src.common.telemetry.setup import (
    init_telemetry,
    shutdown_telemetry,
    get_tracer,
    get_meter,
    is_telemetry_enabled,
    TelemetryConfig,
)
from src.common.telemetry.metrics import (
    NL2APIMetrics,
    get_nl2api_metrics,
    EvalMetrics,
    get_eval_metrics,
    AccuracyMetrics,
    get_accuracy_metrics,
)
from src.common.telemetry.tracing import (
    trace_async,
    trace_sync,
    add_span_attributes,
    record_exception,
)

__all__ = [
    # Setup
    "init_telemetry",
    "shutdown_telemetry",
    "get_tracer",
    "get_meter",
    "is_telemetry_enabled",
    "TelemetryConfig",
    # Metrics
    "NL2APIMetrics",
    "get_nl2api_metrics",
    "EvalMetrics",
    "get_eval_metrics",
    "AccuracyMetrics",
    "get_accuracy_metrics",
    # Tracing
    "trace_async",
    "trace_sync",
    "add_span_attributes",
    "record_exception",
]
