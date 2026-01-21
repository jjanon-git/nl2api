"""
Unit tests for ingestion configuration.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.nl2api.ingestion.config import (
    EXCHANGE_SUFFIX_MAP,
    EntityIngestionConfig,
    generate_ric,
)


class TestEntityIngestionConfig:
    """Tests for EntityIngestionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EntityIngestionConfig()

        assert config.batch_size == 50000
        assert config.max_errors == 1000
        assert config.checkpoint_interval == 100000
        assert config.refresh_enabled is False
        assert config.refresh_interval_days == 30
        assert config.default_exchange_suffix == ".O"

    def test_ensure_data_dir(self):
        """Test data directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EntityIngestionConfig(data_dir=Path(tmpdir) / "entity_data")

            result = config.ensure_data_dir()

            assert result.exists()
            assert result.is_dir()

    def test_checkpoint_dir(self):
        """Test checkpoint directory property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EntityIngestionConfig(data_dir=Path(tmpdir) / "entity_data")

            checkpoint_dir = config.checkpoint_dir

            assert checkpoint_dir.exists()
            assert checkpoint_dir.name == "checkpoints"

    def test_download_dir(self):
        """Test download directory property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EntityIngestionConfig(data_dir=Path(tmpdir) / "entity_data")

            download_dir = config.download_dir

            assert download_dir.exists()
            assert download_dir.name == "downloads"

    def test_error_log_dir(self):
        """Test error log directory property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EntityIngestionConfig(data_dir=Path(tmpdir) / "entity_data")

            error_dir = config.error_log_dir

            assert error_dir.exists()
            assert error_dir.name == "errors"

    def test_batch_size_validation(self):
        """Test batch size validation bounds."""
        # Valid batch sizes
        config = EntityIngestionConfig(batch_size=1000)
        assert config.batch_size == 1000

        config = EntityIngestionConfig(batch_size=500000)
        assert config.batch_size == 500000

        # Invalid batch sizes should raise validation error
        with pytest.raises(ValueError):
            EntityIngestionConfig(batch_size=999)  # Below minimum

        with pytest.raises(ValueError):
            EntityIngestionConfig(batch_size=500001)  # Above maximum

    def test_env_variable_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("ENTITY_BATCH_SIZE", "75000")
        monkeypatch.setenv("ENTITY_MAX_ERRORS", "500")
        monkeypatch.setenv("ENTITY_REFRESH_ENABLED", "true")

        config = EntityIngestionConfig()

        assert config.batch_size == 75000
        assert config.max_errors == 500
        assert config.refresh_enabled is True


class TestGenerateRic:
    """Tests for RIC generation."""

    def test_nyse_ric(self):
        """Test NYSE RIC generation."""
        assert generate_ric("IBM", "NYSE") == "IBM.N"
        assert generate_ric("GE", "N") == "GE.N"

    def test_nasdaq_ric(self):
        """Test NASDAQ RIC generation."""
        assert generate_ric("AAPL", "NASDAQ") == "AAPL.O"
        assert generate_ric("MSFT", "Q") == "MSFT.O"

    def test_amex_ric(self):
        """Test AMEX RIC generation."""
        assert generate_ric("SPY", "AMEX") == "SPY.A"
        assert generate_ric("SPY", "A") == "SPY.A"

    def test_unknown_exchange_defaults_to_nasdaq(self):
        """Test unknown exchange defaults to NASDAQ."""
        assert generate_ric("XYZ", "UNKNOWN") == "XYZ.O"
        assert generate_ric("XYZ", "") == "XYZ.O"

    def test_no_exchange_defaults_to_nasdaq(self):
        """Test no exchange defaults to NASDAQ."""
        assert generate_ric("TEST", None) == "TEST.O"

    def test_ticker_normalization(self):
        """Test ticker is uppercased and stripped."""
        assert generate_ric("aapl", "NASDAQ") == "AAPL.O"
        assert generate_ric("  msft  ", "NASDAQ") == "MSFT.O"

    def test_empty_ticker_returns_empty(self):
        """Test empty ticker returns empty string."""
        assert generate_ric("", "NYSE") == ""
        assert generate_ric(None, "NYSE") == ""

    def test_custom_default_suffix(self):
        """Test custom default suffix."""
        assert generate_ric("TEST", "UNKNOWN", default_suffix=".X") == "TEST.X"

    def test_international_exchanges(self):
        """Test international exchange mappings."""
        assert generate_ric("BP", "LSE") == "BP.L"
        assert generate_ric("TOYOTA", "TSE") == "TOYOTA.T"
        assert generate_ric("SAP", "XETRA") == "SAP.DE"


class TestExchangeSuffixMap:
    """Tests for exchange suffix mapping."""

    def test_us_exchanges_present(self):
        """Test US exchanges are in mapping."""
        assert "NYSE" in EXCHANGE_SUFFIX_MAP
        assert "NASDAQ" in EXCHANGE_SUFFIX_MAP
        assert "AMEX" in EXCHANGE_SUFFIX_MAP

    def test_short_codes_present(self):
        """Test short exchange codes are in mapping."""
        assert "N" in EXCHANGE_SUFFIX_MAP  # NYSE
        assert "Q" in EXCHANGE_SUFFIX_MAP  # NASDAQ
        assert "A" in EXCHANGE_SUFFIX_MAP  # AMEX

    def test_suffixes_start_with_dot(self):
        """Test all suffixes start with dot."""
        for exchange, suffix in EXCHANGE_SUFFIX_MAP.items():
            assert suffix.startswith("."), f"Suffix for {exchange} should start with '.'"
