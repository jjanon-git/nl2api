"""
Unit tests for entity validation.
"""

import tempfile
from pathlib import Path

import pytest

from src.nl2api.ingestion.validation import (
    EntityValidator,
    IngestionAbortError,
    IngestionErrorHandler,
    ValidationResult,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_invalidates(self):
        """Test that adding error invalidates result."""
        result = ValidationResult(is_valid=True)
        result.add_error("Missing field")

        assert not result.is_valid
        assert "Missing field" in result.errors

    def test_warnings_dont_invalidate(self):
        """Test that warnings don't invalidate result."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Unusual format")

        assert result.is_valid
        assert "Unusual format" in result.warnings


class TestEntityValidator:
    """Tests for EntityValidator."""

    def test_valid_entity(self):
        """Test validating a complete valid entity."""
        validator = EntityValidator()

        record = {
            "primary_name": "Apple Inc",
            "data_source": "gleif",
            "lei": "HWUPKR0MPOU8FGXBT394",
            "cik": "0000320193",
            "country_code": "US",
            "ticker": "AAPL",
        }

        result = validator.validate(record)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_primary_name(self):
        """Test error on missing primary_name."""
        validator = EntityValidator()

        record = {
            "data_source": "gleif",
            "lei": "HWUPKR0MPOU8FGXBT394",
        }

        result = validator.validate(record)
        assert not result.is_valid
        assert any("primary_name" in e for e in result.errors)

    def test_missing_data_source(self):
        """Test error on missing data_source."""
        validator = EntityValidator()

        record = {
            "primary_name": "Apple Inc",
        }

        result = validator.validate(record)
        assert not result.is_valid
        assert any("data_source" in e for e in result.errors)

    def test_invalid_lei_format(self):
        """Test error on invalid LEI format."""
        validator = EntityValidator()

        record = {
            "primary_name": "Apple Inc",
            "data_source": "gleif",
            "lei": "INVALID",  # Should be 20 alphanumeric
        }

        result = validator.validate(record)
        assert not result.is_valid
        assert any("LEI" in e for e in result.errors)

    def test_valid_lei_format(self):
        """Test valid LEI passes."""
        validator = EntityValidator()

        record = {
            "primary_name": "Apple Inc",
            "data_source": "gleif",
            "lei": "HWUPKR0MPOU8FGXBT394",  # 20 alphanumeric
        }

        result = validator.validate(record)
        assert result.is_valid

    def test_invalid_cik_format(self):
        """Test error on invalid CIK format."""
        validator = EntityValidator()

        record = {
            "primary_name": "Apple Inc",
            "data_source": "sec_edgar",
            "cik": "ABC123",  # Should be digits only
        }

        result = validator.validate(record)
        assert not result.is_valid
        assert any("CIK" in e for e in result.errors)

    def test_valid_cik_format(self):
        """Test valid CIK passes."""
        validator = EntityValidator()

        record = {
            "primary_name": "Apple Inc",
            "data_source": "sec_edgar",
            "cik": "320193",  # Valid digits
        }

        result = validator.validate(record)
        assert result.is_valid

    def test_unknown_country_code_warning(self):
        """Test warning on unknown country code."""
        validator = EntityValidator()

        record = {
            "primary_name": "Test Company",
            "data_source": "gleif",
            "country_code": "XX",  # Unknown
        }

        result = validator.validate(record)
        assert result.is_valid  # Warning doesn't invalidate
        assert any("country" in w.lower() for w in result.warnings)

    def test_name_too_short(self):
        """Test error on name too short."""
        validator = EntityValidator(min_name_length=3)

        record = {
            "primary_name": "X",
            "data_source": "gleif",
        }

        result = validator.validate(record)
        assert not result.is_valid
        assert any("too short" in e.lower() for e in result.errors)

    def test_name_too_long(self):
        """Test error on name too long."""
        validator = EntityValidator(max_name_length=50)

        record = {
            "primary_name": "A" * 100,
            "data_source": "gleif",
        }

        result = validator.validate(record)
        assert not result.is_valid
        assert any("too long" in e.lower() for e in result.errors)

    def test_placeholder_name_warning(self):
        """Test warning on placeholder names."""
        validator = EntityValidator()

        placeholder_names = ["test", "N/A", "unknown", "???", "---"]

        for name in placeholder_names:
            record = {
                "primary_name": name,
                "data_source": "gleif",
            }
            result = validator.validate(record)
            assert any("placeholder" in w.lower() for w in result.warnings), (
                f"Should warn on '{name}'"
            )

    def test_strict_mode(self):
        """Test strict mode turns warnings into errors."""
        validator = EntityValidator(strict_mode=True)

        record = {
            "primary_name": "Test Company",
            "data_source": "gleif",
            "country_code": "XX",  # Unknown - would be warning in normal mode
        }

        result = validator.validate(record)
        assert not result.is_valid  # Warning becomes error in strict mode

    def test_figi_format_warning(self):
        """Test warning on invalid FIGI format."""
        validator = EntityValidator()

        record = {
            "primary_name": "Test Company",
            "data_source": "gleif",
            "figi": "INVALID",  # Should be BBG + 9 chars
        }

        result = validator.validate(record)
        assert result.is_valid  # FIGI is warning, not error
        assert any("FIGI" in w for w in result.warnings)

    def test_valid_figi_format(self):
        """Test valid FIGI passes."""
        validator = EntityValidator()

        record = {
            "primary_name": "Apple Inc",
            "data_source": "gleif",
            "figi": "BBG000B9XRY4",  # Valid FIGI
        }

        result = validator.validate(record)
        assert result.is_valid
        assert len(result.warnings) == 0


class TestIngestionErrorHandler:
    """Tests for IngestionErrorHandler."""

    def test_handle_error_counts(self):
        """Test error counting."""
        handler = IngestionErrorHandler(max_errors=10)

        for i in range(5):
            result = handler.handle_error(f"record_{i}", "Some error")
            assert result is True

        assert handler.error_count == 5

    def test_max_errors_abort(self):
        """Test abort when max errors exceeded."""
        handler = IngestionErrorHandler(max_errors=5)

        with pytest.raises(IngestionAbortError) as exc_info:
            for i in range(10):
                handler.handle_error(f"record_{i}", "Some error")

        assert "Too many errors" in str(exc_info.value)
        assert "5" in str(exc_info.value)

    def test_error_log_file(self):
        """Test error logging to file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            error_log_path = f.name

        try:
            handler = IngestionErrorHandler(
                max_errors=100,
                error_log_path=error_log_path,
            )

            handler.handle_error("LEI123", "Invalid format")
            handler.handle_error("LEI456", "Missing field")
            handler.close()

            # Check log file
            with open(error_log_path) as f:
                lines = f.readlines()

            assert len(lines) == 2
            assert "LEI123" in lines[0]
            assert "Invalid format" in lines[0]
        finally:
            Path(error_log_path).unlink(missing_ok=True)

    def test_context_manager(self):
        """Test using error handler as context manager."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            error_log_path = f.name

        try:
            with IngestionErrorHandler(max_errors=100, error_log_path=error_log_path) as handler:
                handler.handle_error("LEI123", "Test error")

            # File should be closed after context exit
            with open(error_log_path) as f:
                content = f.read()
            assert "LEI123" in content
        finally:
            Path(error_log_path).unlink(missing_ok=True)
