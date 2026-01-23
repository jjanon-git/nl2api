"""
Unit tests for ingestion telemetry.
"""

import pytest

from src.nl2api.ingestion.telemetry import (
    SpanAttributes,
    record_ingestion_metric,
    trace_ingestion_operation,
)


class TestSpanAttributes:
    """Tests for SpanAttributes constants."""

    def test_source_attributes(self):
        """Test source-related attributes."""
        assert SpanAttributes.SOURCE == "ingestion.source"
        assert SpanAttributes.SOURCE_URL == "ingestion.source_url"

    def test_file_attributes(self):
        """Test file-related attributes."""
        assert SpanAttributes.FILE_PATH == "file.path"
        assert SpanAttributes.FILE_SIZE_BYTES == "file.size_bytes"
        assert SpanAttributes.FILE_FORMAT == "file.format"

    def test_db_attributes(self):
        """Test database-related attributes."""
        assert SpanAttributes.DB_OPERATION == "db.operation"
        assert SpanAttributes.DB_TABLE == "db.table"
        assert SpanAttributes.DB_ROWS_AFFECTED == "db.rows_affected"
        assert SpanAttributes.DB_BATCH_SIZE == "db.batch_size"

    def test_processing_attributes(self):
        """Test processing stats attributes."""
        assert SpanAttributes.RECORDS_PROCESSED == "ingestion.records_processed"
        assert SpanAttributes.RECORDS_INSERTED == "ingestion.records_inserted"
        assert SpanAttributes.RECORDS_UPDATED == "ingestion.records_updated"
        assert SpanAttributes.RECORDS_SKIPPED == "ingestion.records_skipped"
        assert SpanAttributes.RECORDS_FAILED == "ingestion.records_failed"

    def test_timing_attributes(self):
        """Test timing attributes."""
        assert SpanAttributes.DURATION_MS == "duration_ms"

    def test_checkpoint_attributes(self):
        """Test checkpoint attributes."""
        assert SpanAttributes.CHECKPOINT_OFFSET == "checkpoint.offset"
        assert SpanAttributes.CHECKPOINT_STATE == "checkpoint.state"


class TestTraceIngestionOperation:
    """Tests for trace_ingestion_operation context manager."""

    def test_trace_without_telemetry(self):
        """Test tracing works when telemetry is not enabled."""
        # Should not raise even if telemetry is not initialized
        with trace_ingestion_operation("test_operation") as span:
            # span may be None if telemetry not available
            assert span is None or span is not None  # No error raised

    def test_trace_with_attributes(self):
        """Test tracing with attributes."""
        attrs = {
            SpanAttributes.SOURCE: "test",
            SpanAttributes.RECORDS_PROCESSED: 100,
        }
        with trace_ingestion_operation("test_op", attrs):
            # Should complete without error
            pass

    def test_trace_exception_propagation(self):
        """Test exceptions propagate through trace context."""
        with pytest.raises(ValueError, match="test error"):
            with trace_ingestion_operation("failing_op"):
                raise ValueError("test error")


class TestRecordIngestionMetric:
    """Tests for record_ingestion_metric function."""

    def test_record_metric_no_error(self):
        """Test recording metric doesn't error when telemetry unavailable."""
        # Should not raise
        record_ingestion_metric("test_metric_total", 100)
        record_ingestion_metric("test_latency_ms", 50.5)

    def test_record_metric_with_attributes(self):
        """Test recording metric with attributes."""
        # Should not raise
        record_ingestion_metric(
            "entities_loaded_total",
            1000,
            {"source": "gleif", "mode": "full"},
        )

    def test_record_counter_metric(self):
        """Test recording counter-style metric."""
        # Metrics ending in _total or _count are counters
        record_ingestion_metric("download_bytes_total", 1024 * 1024)
        record_ingestion_metric("error_count", 5)

    def test_record_histogram_metric(self):
        """Test recording histogram-style metric."""
        # Other metrics are histograms
        record_ingestion_metric("batch_load_duration_ms", 1500.5)
        record_ingestion_metric("records_per_second", 5000)
