"""
Ingestion Telemetry

Provides OTEL tracing for the entity ingestion pipeline.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Try to import telemetry, gracefully degrade if not available
_telemetry_available = False
_tracer = None
_meter = None

try:
    from src.common.telemetry import get_tracer, get_meter, is_telemetry_enabled

    _telemetry_available = True
except ImportError:
    logger.debug("Telemetry not available, spans will be no-ops")


def get_ingestion_tracer():
    """Get tracer for ingestion operations."""
    global _tracer
    if _telemetry_available and _tracer is None:
        _tracer = get_tracer("nl2api.ingestion")
    return _tracer


def get_ingestion_meter():
    """Get meter for ingestion metrics."""
    global _meter
    if _telemetry_available and _meter is None:
        _meter = get_meter("nl2api.ingestion")
    return _meter


@contextmanager
def trace_ingestion_operation(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """
    Context manager for tracing ingestion operations.

    Gracefully degrades to no-op if telemetry is not available.

    Args:
        name: Span name (e.g., "download_gleif", "bulk_load")
        attributes: Initial span attributes

    Yields:
        Span object (or None if telemetry unavailable)

    Example:
        with trace_ingestion_operation("download_file", {"source": "gleif"}) as span:
            # ... download logic
            if span:
                span.set_attribute("file.size_bytes", file_size)
    """
    if not _telemetry_available:
        try:
            yield None
        except Exception:
            raise
        return

    try:
        from src.common.telemetry import is_telemetry_enabled

        if not is_telemetry_enabled():
            try:
                yield None
            except Exception:
                raise
            return
    except ImportError:
        try:
            yield None
        except Exception:
            raise
        return

    tracer = get_ingestion_tracer()
    if tracer is None:
        try:
            yield None
        except Exception:
            raise
        return

    with tracer.start_as_current_span(f"ingestion.{name}") as span:
        if attributes:
            span.set_attributes(attributes)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            raise


def record_ingestion_metric(
    name: str,
    value: int | float,
    attributes: dict[str, Any] | None = None,
) -> None:
    """
    Record an ingestion metric.

    Args:
        name: Metric name (e.g., "entities_loaded", "download_duration_ms")
        value: Metric value
        attributes: Metric attributes/labels
    """
    if not _telemetry_available:
        return

    meter = get_ingestion_meter()
    if meter is None:
        return

    # Create counter or histogram based on metric name
    if name.endswith("_total") or name.endswith("_count"):
        counter = meter.create_counter(f"nl2api.ingestion.{name}")
        counter.add(int(value), attributes or {})
    else:
        histogram = meter.create_histogram(f"nl2api.ingestion.{name}")
        histogram.record(value, attributes or {})


# Standard span attribute names for consistency
class SpanAttributes:
    """Standard attribute names for ingestion spans."""

    # Source identification
    SOURCE = "ingestion.source"  # gleif, sec_edgar
    SOURCE_URL = "ingestion.source_url"

    # File operations
    FILE_PATH = "file.path"
    FILE_SIZE_BYTES = "file.size_bytes"
    FILE_FORMAT = "file.format"

    # Database operations
    DB_OPERATION = "db.operation"  # insert, update, upsert, copy
    DB_TABLE = "db.table"
    DB_ROWS_AFFECTED = "db.rows_affected"
    DB_BATCH_SIZE = "db.batch_size"

    # Processing stats
    RECORDS_PROCESSED = "ingestion.records_processed"
    RECORDS_INSERTED = "ingestion.records_inserted"
    RECORDS_UPDATED = "ingestion.records_updated"
    RECORDS_SKIPPED = "ingestion.records_skipped"
    RECORDS_FAILED = "ingestion.records_failed"

    # Timing
    DURATION_MS = "duration_ms"

    # Checkpoint
    CHECKPOINT_OFFSET = "checkpoint.offset"
    CHECKPOINT_STATE = "checkpoint.state"
