"""
Ingestion Errors

Custom exception hierarchy for the entity ingestion pipeline.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all ingestion errors."""

    def __init__(self, message: str, source: str | None = None):
        """
        Initialize ingestion error.

        Args:
            message: Error description
            source: Data source (gleif, sec_edgar, etc.)
        """
        self.source = source
        super().__init__(message)


class DownloadError(IngestionError):
    """Failed to download data from external source."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        url: str | None = None,
        status_code: int | None = None,
    ):
        """
        Initialize download error.

        Args:
            message: Error description
            source: Data source name
            url: URL that failed
            status_code: HTTP status code if applicable
        """
        self.url = url
        self.status_code = status_code
        super().__init__(message, source)


class TransformationError(IngestionError):
    """Failed to transform source data to entity format."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        record_id: str | None = None,
        field: str | None = None,
    ):
        """
        Initialize transformation error.

        Args:
            message: Error description
            source: Data source name
            record_id: ID of the record that failed
            field: Field that caused the error
        """
        self.record_id = record_id
        self.field = field
        super().__init__(message, source)


class ValidationError(IngestionError):
    """Entity validation failed."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        record_id: str | None = None,
        validation_errors: list[str] | None = None,
    ):
        """
        Initialize validation error.

        Args:
            message: Error description
            source: Data source name
            record_id: ID of the record that failed
            validation_errors: List of specific validation failures
        """
        self.record_id = record_id
        self.validation_errors = validation_errors or []
        super().__init__(message, source)


class LoadError(IngestionError):
    """Failed to load data into database."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        batch_number: int | None = None,
        rows_affected: int | None = None,
    ):
        """
        Initialize load error.

        Args:
            message: Error description
            source: Data source name
            batch_number: Batch that failed
            rows_affected: Number of rows in the failed batch
        """
        self.batch_number = batch_number
        self.rows_affected = rows_affected
        super().__init__(message, source)


class CheckpointError(IngestionError):
    """Failed to save or load checkpoint."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """
        Initialize checkpoint error.

        Args:
            message: Error description
            source: Data source name
            checkpoint_path: Path to checkpoint file
        """
        self.checkpoint_path = checkpoint_path
        super().__init__(message, source)


class IngestionAbortError(IngestionError):
    """Ingestion must abort due to unrecoverable error or too many failures."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        error_count: int | None = None,
        max_errors: int | None = None,
    ):
        """
        Initialize abort error.

        Args:
            message: Error description
            source: Data source name
            error_count: Number of errors encountered
            max_errors: Maximum errors threshold
        """
        self.error_count = error_count
        self.max_errors = max_errors
        super().__init__(message, source)
