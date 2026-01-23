"""Tests for metrics emitters."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.nl2api.observability.emitter import (
    CompositeEmitter,
    FileEmitter,
    LoggingEmitter,
    MetricsEmitter,
    NullEmitter,
    configure_emitter,
    create_emitter_from_config,
    emit_metrics,
    get_emitter,
    set_metrics_enabled,
)
from src.nl2api.observability.metrics import RequestMetrics


class TestLoggingEmitter:
    """Test LoggingEmitter."""

    @pytest.fixture
    def metrics(self) -> RequestMetrics:
        """Create sample metrics for testing."""
        m = RequestMetrics(query="Test query")
        m.set_routing_result(domain="estimates", confidence=0.9)
        m.finalize(total_latency_ms=100)
        return m

    @pytest.mark.asyncio
    async def test_emit_logs_summary(
        self, metrics: RequestMetrics, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that emit logs a summary."""
        emitter = LoggingEmitter()

        with caplog.at_level(logging.INFO, logger="nl2api.metrics"):
            await emitter.emit(metrics)

        assert len(caplog.records) >= 1
        assert "Request:" in caplog.text

    @pytest.mark.asyncio
    async def test_emit_with_custom_logger(
        self, metrics: RequestMetrics, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test emit with custom logger name."""
        emitter = LoggingEmitter(logger_name="custom.metrics")

        with caplog.at_level(logging.INFO, logger="custom.metrics"):
            await emitter.emit(metrics)

        assert len(caplog.records) >= 1

    @pytest.mark.asyncio
    async def test_emit_handles_exception(self, metrics: RequestMetrics) -> None:
        """Test emit handles exceptions gracefully."""
        emitter = LoggingEmitter()

        # Mock logger to raise exception
        with patch.object(emitter, "_logger") as mock_logger:
            mock_logger.info.side_effect = Exception("Log error")

            # Should not raise
            await emitter.emit(metrics)


class TestFileEmitter:
    """Test FileEmitter."""

    @pytest.fixture
    def metrics(self) -> RequestMetrics:
        """Create sample metrics for testing."""
        m = RequestMetrics(query="Test query", request_id="test-123")
        m.set_routing_result(domain="estimates", confidence=0.9)
        m.finalize(total_latency_ms=100)
        return m

    @pytest.mark.asyncio
    async def test_emit_writes_to_file(self, metrics: RequestMetrics) -> None:
        """Test that emit writes JSON lines to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            emitter = FileEmitter(file_path, buffer_size=1)  # Immediate flush
            await emitter.emit(metrics)
            await emitter.close()

            # Read back and verify
            with open(file_path) as f:
                lines = f.readlines()

            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["request_id"] == "test-123"
            assert parsed["query"] == "Test query"
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_emit_buffers_writes(self, metrics: RequestMetrics) -> None:
        """Test that emit buffers writes before flushing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            emitter = FileEmitter(file_path, buffer_size=3)

            # Write 2 metrics (less than buffer)
            await emitter.emit(metrics)
            await emitter.emit(metrics)

            # File should be empty (buffered)
            with open(file_path) as f:
                assert f.read() == ""

            # Write third to trigger flush
            await emitter.emit(metrics)

            # Now file should have content
            with open(file_path) as f:
                lines = f.readlines()
            assert len(lines) == 3

            await emitter.close()
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_close_flushes_buffer(self, metrics: RequestMetrics) -> None:
        """Test that close flushes remaining buffer."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            emitter = FileEmitter(file_path, buffer_size=10)  # Large buffer

            await emitter.emit(metrics)
            await emitter.emit(metrics)

            # Close should flush
            await emitter.close()

            with open(file_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
        finally:
            Path(file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, metrics: RequestMetrics) -> None:
        """Test that emitter creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "metrics.jsonl"

            emitter = FileEmitter(file_path, create_dirs=True, buffer_size=1)
            await emitter.emit(metrics)
            await emitter.close()

            assert file_path.exists()

    @pytest.mark.asyncio
    async def test_handles_write_error(self, metrics: RequestMetrics) -> None:
        """Test that emit handles write errors gracefully."""
        emitter = FileEmitter("/nonexistent/path/metrics.jsonl", create_dirs=False)

        # Should not raise (errors are logged)
        await emitter.emit(metrics)


class TestCompositeEmitter:
    """Test CompositeEmitter."""

    @pytest.fixture
    def metrics(self) -> RequestMetrics:
        """Create sample metrics for testing."""
        return RequestMetrics(query="Test query")

    @pytest.mark.asyncio
    async def test_emits_to_all_backends(self, metrics: RequestMetrics) -> None:
        """Test that emit sends to all backends."""
        emitter1 = AsyncMock(spec=MetricsEmitter)
        emitter2 = AsyncMock(spec=MetricsEmitter)

        composite = CompositeEmitter([emitter1, emitter2])
        await composite.emit(metrics)

        emitter1.emit.assert_called_once_with(metrics)
        emitter2.emit.assert_called_once_with(metrics)

    @pytest.mark.asyncio
    async def test_continues_on_backend_failure(self, metrics: RequestMetrics) -> None:
        """Test that failure in one backend doesn't affect others."""
        emitter1 = AsyncMock(spec=MetricsEmitter)
        emitter1.emit.side_effect = Exception("Backend error")

        emitter2 = AsyncMock(spec=MetricsEmitter)

        composite = CompositeEmitter([emitter1, emitter2])
        await composite.emit(metrics)

        # Second emitter should still be called
        emitter2.emit.assert_called_once_with(metrics)

    @pytest.mark.asyncio
    async def test_close_closes_all_emitters(self) -> None:
        """Test that close closes all emitters."""
        emitter1 = AsyncMock(spec=MetricsEmitter)
        emitter2 = AsyncMock(spec=MetricsEmitter)

        composite = CompositeEmitter([emitter1, emitter2])
        await composite.close()

        emitter1.close.assert_called_once()
        emitter2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_errors(self) -> None:
        """Test that close handles errors in individual emitters."""
        emitter1 = AsyncMock(spec=MetricsEmitter)
        emitter1.close.side_effect = Exception("Close error")

        emitter2 = AsyncMock(spec=MetricsEmitter)

        composite = CompositeEmitter([emitter1, emitter2])
        await composite.close()  # Should not raise

        emitter2.close.assert_called_once()


class TestNullEmitter:
    """Test NullEmitter."""

    @pytest.mark.asyncio
    async def test_emit_does_nothing(self) -> None:
        """Test that emit is a no-op."""
        emitter = NullEmitter()
        metrics = RequestMetrics(query="Test")

        # Should not raise
        await emitter.emit(metrics)


class TestGlobalEmitterConfiguration:
    """Test global emitter configuration functions."""

    def setup_method(self) -> None:
        """Reset global state before each test."""
        import src.nl2api.observability.emitter as emitter_module

        emitter_module._emitter = None
        emitter_module._metrics_enabled = True

    @pytest.mark.asyncio
    async def test_get_emitter_returns_default(self) -> None:
        """Test get_emitter returns LoggingEmitter by default."""
        emitter = get_emitter()

        assert isinstance(emitter, LoggingEmitter)

    @pytest.mark.asyncio
    async def test_configure_emitter_sets_global(self) -> None:
        """Test configure_emitter sets the global emitter."""
        custom_emitter = NullEmitter()
        configure_emitter(custom_emitter)

        assert get_emitter() is custom_emitter

    @pytest.mark.asyncio
    async def test_emit_metrics_uses_configured_emitter(self) -> None:
        """Test emit_metrics uses the configured emitter."""
        mock_emitter = AsyncMock(spec=MetricsEmitter)
        configure_emitter(mock_emitter)

        metrics = RequestMetrics(query="Test")
        await emit_metrics(metrics)

        mock_emitter.emit.assert_called_once_with(metrics)

    @pytest.mark.asyncio
    async def test_emit_metrics_respects_disabled(self) -> None:
        """Test emit_metrics does nothing when disabled."""
        mock_emitter = AsyncMock(spec=MetricsEmitter)
        configure_emitter(mock_emitter)
        set_metrics_enabled(False)

        metrics = RequestMetrics(query="Test")
        await emit_metrics(metrics)

        mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_emit_metrics_handles_errors(self) -> None:
        """Test emit_metrics handles errors gracefully."""
        mock_emitter = AsyncMock(spec=MetricsEmitter)
        mock_emitter.emit.side_effect = Exception("Emit error")
        configure_emitter(mock_emitter)

        metrics = RequestMetrics(query="Test")

        # Should not raise
        await emit_metrics(metrics)


class TestCreateEmitterFromConfig:
    """Test create_emitter_from_config factory function."""

    def test_creates_logging_emitter(self) -> None:
        """Test creates LoggingEmitter when log_enabled."""
        emitter = create_emitter_from_config(log_enabled=True, file_path=None)

        assert isinstance(emitter, LoggingEmitter)

    def test_creates_file_emitter(self) -> None:
        """Test creates FileEmitter when file_path provided."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            emitter = create_emitter_from_config(log_enabled=False, file_path=file_path)

            assert isinstance(emitter, FileEmitter)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_creates_composite_emitter(self) -> None:
        """Test creates CompositeEmitter when both enabled."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            file_path = f.name

        try:
            emitter = create_emitter_from_config(log_enabled=True, file_path=file_path)

            assert isinstance(emitter, CompositeEmitter)
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_creates_null_emitter_when_nothing_enabled(self) -> None:
        """Test creates NullEmitter when nothing enabled."""
        emitter = create_emitter_from_config(log_enabled=False, file_path=None)

        assert isinstance(emitter, NullEmitter)
