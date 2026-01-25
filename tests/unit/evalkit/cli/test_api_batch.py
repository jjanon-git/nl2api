"""
Unit tests for Anthropic Batch API CLI commands.

Tests the api_batch.py CLI commands with mocked Anthropic client.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.evalkit.cli.commands.api_batch import api_app

runner = CliRunner()


class MockBatch:
    """Mock Anthropic batch object."""

    def __init__(
        self,
        batch_id: str = "batch_123",
        processing_status: str = "in_progress",
        succeeded: int = 0,
        errored: int = 0,
        processing: int = 10,
        canceled: int = 0,
        expired: int = 0,
    ):
        self.id = batch_id
        self.processing_status = processing_status
        self.created_at = datetime.now()
        self.ended_at = None if processing_status != "ended" else datetime.now()
        self.request_counts = MagicMock()
        self.request_counts.succeeded = succeeded
        self.request_counts.errored = errored
        self.request_counts.processing = processing
        self.request_counts.canceled = canceled
        self.request_counts.expired = expired


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = MagicMock()
    client.beta.messages.batches.create.return_value = MockBatch()
    client.beta.messages.batches.retrieve.return_value = MockBatch()
    client.beta.messages.batches.results.return_value = []
    return client


@pytest.fixture
def mock_env(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("NL2API_ANTHROPIC_API_KEY", "test-api-key")


class TestApiStatus:
    """Tests for the api status command."""

    def test_status_shows_batch_info(self, mock_anthropic_client, mock_env):
        """Status command should display batch information."""
        with patch(
            "src.evalkit.cli.commands.api_batch._get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            result = runner.invoke(api_app, ["status", "batch_123"])

        assert result.exit_code == 0
        assert "batch_123" in result.output
        assert "Status" in result.output

    def test_status_shows_request_counts(self, mock_anthropic_client, mock_env):
        """Status command should display request counts."""
        mock_anthropic_client.beta.messages.batches.retrieve.return_value = MockBatch(
            succeeded=5,
            errored=1,
            processing=4,
        )
        with patch(
            "src.evalkit.cli.commands.api_batch._get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            result = runner.invoke(api_app, ["status", "batch_123"])

        assert result.exit_code == 0
        assert "5" in result.output  # succeeded
        assert "1" in result.output  # errored

    def test_status_handles_completed_batch(self, mock_anthropic_client, mock_env):
        """Status command should show green for completed batches."""
        mock_anthropic_client.beta.messages.batches.retrieve.return_value = MockBatch(
            processing_status="ended",
            succeeded=10,
            processing=0,
        )
        with patch(
            "src.evalkit.cli.commands.api_batch._get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            result = runner.invoke(api_app, ["status", "batch_123"])

        assert result.exit_code == 0
        assert "ended" in result.output

    def test_status_calls_retrieve_with_batch_id(self, mock_anthropic_client, mock_env):
        """Status command should call retrieve with the correct batch ID."""
        with patch(
            "src.evalkit.cli.commands.api_batch._get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            runner.invoke(api_app, ["status", "my-custom-batch-id"])

        mock_anthropic_client.beta.messages.batches.retrieve.assert_called_once_with(
            "my-custom-batch-id"
        )


class TestApiPoll:
    """Tests for the api poll command."""

    def test_poll_exits_when_complete(self, mock_anthropic_client, mock_env):
        """Poll command should exit when batch completes."""
        mock_anthropic_client.beta.messages.batches.retrieve.return_value = MockBatch(
            processing_status="ended",
            succeeded=10,
            processing=0,
        )
        with patch(
            "src.evalkit.cli.commands.api_batch._get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            result = runner.invoke(api_app, ["poll", "batch_123", "--interval", "1"])

        assert result.exit_code == 0
        assert "Complete" in result.output or "complete" in result.output

    def test_poll_shows_progress(self, mock_anthropic_client, mock_env):
        """Poll command should show progress status."""
        mock_anthropic_client.beta.messages.batches.retrieve.return_value = MockBatch(
            processing_status="ended",
            succeeded=8,
            errored=2,
            processing=0,
        )
        with patch(
            "src.evalkit.cli.commands.api_batch._get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            result = runner.invoke(api_app, ["poll", "batch_123", "--interval", "1"])

        assert result.exit_code == 0
        # Should show succeeded count
        assert "8" in result.output
        # Should show error count
        assert "2" in result.output


class TestCliHelp:
    """Tests for CLI help text."""

    def test_api_app_has_help(self):
        """API app should have help text."""
        result = runner.invoke(api_app, ["--help"])
        assert result.exit_code == 0
        assert "Anthropic" in result.output or "batch" in result.output.lower()

    def test_status_has_help(self):
        """Status command should have help text."""
        result = runner.invoke(api_app, ["status", "--help"])
        assert result.exit_code == 0
        assert "batch" in result.output.lower() or "status" in result.output.lower()

    def test_poll_has_help(self):
        """Poll command should have help text."""
        result = runner.invoke(api_app, ["poll", "--help"])
        assert result.exit_code == 0
        assert "interval" in result.output.lower()

    def test_submit_has_help(self):
        """Submit command should have help text."""
        result = runner.invoke(api_app, ["submit", "--help"])
        assert result.exit_code == 0
        assert "limit" in result.output.lower()

    def test_results_has_help(self):
        """Results command should have help text."""
        result = runner.invoke(api_app, ["results", "--help"])
        assert result.exit_code == 0
        assert "batch" in result.output.lower()


class TestApiStatusErrorHandling:
    """Tests for error handling in status command."""

    def test_status_handles_api_error(self, mock_anthropic_client, mock_env):
        """Status command should handle API errors gracefully."""
        mock_anthropic_client.beta.messages.batches.retrieve.side_effect = Exception("API Error")
        with patch(
            "src.evalkit.cli.commands.api_batch._get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            result = runner.invoke(api_app, ["status", "batch_123"])

        # Should exit with non-zero code on error
        assert result.exit_code != 0


class TestMockBatch:
    """Tests for the MockBatch helper class itself."""

    def test_mock_batch_defaults(self):
        """MockBatch should have sensible defaults."""
        batch = MockBatch()
        assert batch.id == "batch_123"
        assert batch.processing_status == "in_progress"
        assert batch.request_counts.succeeded == 0
        assert batch.request_counts.processing == 10

    def test_mock_batch_custom_values(self):
        """MockBatch should accept custom values."""
        batch = MockBatch(
            batch_id="custom_id",
            processing_status="ended",
            succeeded=100,
            errored=5,
        )
        assert batch.id == "custom_id"
        assert batch.processing_status == "ended"
        assert batch.request_counts.succeeded == 100
        assert batch.request_counts.errored == 5
        assert batch.ended_at is not None
