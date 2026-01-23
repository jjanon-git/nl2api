"""
Unit tests for ingestion errors.
"""


from src.nl2api.ingestion.errors import (
    CheckpointError,
    DownloadError,
    IngestionAbortError,
    IngestionError,
    LoadError,
    TransformationError,
    ValidationError,
)


class TestIngestionError:
    """Tests for base IngestionError."""

    def test_create_error(self):
        """Test creating base error."""
        error = IngestionError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.source is None

    def test_create_error_with_source(self):
        """Test creating error with source."""
        error = IngestionError("Failed", source="gleif")
        assert str(error) == "Failed"
        assert error.source == "gleif"

    def test_inheritance(self):
        """Test all errors inherit from IngestionError."""
        assert issubclass(DownloadError, IngestionError)
        assert issubclass(TransformationError, IngestionError)
        assert issubclass(ValidationError, IngestionError)
        assert issubclass(LoadError, IngestionError)
        assert issubclass(CheckpointError, IngestionError)
        assert issubclass(IngestionAbortError, IngestionError)


class TestDownloadError:
    """Tests for DownloadError."""

    def test_create_download_error(self):
        """Test creating download error."""
        error = DownloadError(
            "Failed to download",
            source="sec_edgar",
            url="https://example.com/data.json",
            status_code=404,
        )
        assert str(error) == "Failed to download"
        assert error.source == "sec_edgar"
        assert error.url == "https://example.com/data.json"
        assert error.status_code == 404

    def test_download_error_without_status(self):
        """Test download error without status code."""
        error = DownloadError("Connection failed", url="https://example.com")
        assert error.status_code is None
        assert error.url == "https://example.com"


class TestTransformationError:
    """Tests for TransformationError."""

    def test_create_transformation_error(self):
        """Test creating transformation error."""
        error = TransformationError(
            "Invalid data format",
            source="gleif",
            record_id="LEI123",
            field="country_code",
        )
        assert str(error) == "Invalid data format"
        assert error.source == "gleif"
        assert error.record_id == "LEI123"
        assert error.field == "country_code"


class TestValidationError:
    """Tests for ValidationError."""

    def test_create_validation_error(self):
        """Test creating validation error."""
        error = ValidationError(
            "Entity validation failed",
            source="gleif",
            record_id="LEI456",
            validation_errors=["Missing LEI", "Invalid country code"],
        )
        assert str(error) == "Entity validation failed"
        assert error.record_id == "LEI456"
        assert len(error.validation_errors) == 2
        assert "Missing LEI" in error.validation_errors

    def test_validation_error_empty_errors(self):
        """Test validation error with no specific errors."""
        error = ValidationError("Failed")
        assert error.validation_errors == []


class TestLoadError:
    """Tests for LoadError."""

    def test_create_load_error(self):
        """Test creating load error."""
        error = LoadError(
            "Bulk insert failed",
            source="gleif",
            batch_number=5,
            rows_affected=50000,
        )
        assert str(error) == "Bulk insert failed"
        assert error.source == "gleif"
        assert error.batch_number == 5
        assert error.rows_affected == 50000


class TestCheckpointError:
    """Tests for CheckpointError."""

    def test_create_checkpoint_error(self):
        """Test creating checkpoint error."""
        error = CheckpointError(
            "Failed to save checkpoint",
            source="gleif",
            checkpoint_path="/tmp/checkpoint.json",
        )
        assert str(error) == "Failed to save checkpoint"
        assert error.checkpoint_path == "/tmp/checkpoint.json"


class TestIngestionAbortError:
    """Tests for IngestionAbortError."""

    def test_create_abort_error(self):
        """Test creating abort error."""
        error = IngestionAbortError(
            "Too many errors",
            source="sec_edgar",
            error_count=1000,
            max_errors=1000,
        )
        assert str(error) == "Too many errors"
        assert error.error_count == 1000
        assert error.max_errors == 1000

    def test_exception_chaining(self):
        """Test exception chaining works."""
        original = ValueError("Original error")
        error = IngestionAbortError("Aborted due to error")

        try:
            try:
                raise original
            except ValueError as e:
                raise error from e
        except IngestionAbortError as caught:
            assert caught.__cause__ is original
