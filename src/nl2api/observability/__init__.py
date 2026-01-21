"""
Observability module for NL2API.

Provides:
- Request metrics collection and emission
- Structured logging for analytics
- OpenTelemetry integration for traces and metrics
"""

from src.nl2api.observability.metrics import RequestMetrics
from src.nl2api.observability.emitter import (
    MetricsEmitter,
    LoggingEmitter,
    FileEmitter,
    OTELEmitter,
    CompositeEmitter,
    NullEmitter,
    configure_emitter,
    get_emitter,
    emit_metrics,
    set_metrics_enabled,
    create_emitter_from_config,
)

__all__ = [
    # Metrics
    "RequestMetrics",
    # Emitters
    "MetricsEmitter",
    "LoggingEmitter",
    "FileEmitter",
    "OTELEmitter",
    "CompositeEmitter",
    "NullEmitter",
    "configure_emitter",
    "get_emitter",
    "emit_metrics",
    "set_metrics_enabled",
    "create_emitter_from_config",
]
