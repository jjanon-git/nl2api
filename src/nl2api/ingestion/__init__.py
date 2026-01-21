"""
Entity Ingestion Pipeline

Handles bulk import of entity data from external sources:
- GLEIF (Legal Entity Identifier database)
- SEC EDGAR (US public company filings)

Designed for reliability, scalability, and repeatability.
"""

from src.nl2api.ingestion.checkpoint import CheckpointManager, IngestionCheckpoint
from src.nl2api.ingestion.config import EntityIngestionConfig
from src.nl2api.ingestion.errors import (
    CheckpointError,
    DownloadError,
    IngestionAbortError,
    IngestionError,
    LoadError,
    TransformationError,
    ValidationError,
)
from src.nl2api.ingestion.progress import ProgressTracker
from src.nl2api.ingestion.telemetry import (
    SpanAttributes,
    record_ingestion_metric,
    trace_ingestion_operation,
)
from src.nl2api.ingestion.validation import EntityValidator, ValidationResult

__all__ = [
    # Checkpoint
    "CheckpointManager",
    "IngestionCheckpoint",
    # Config
    "EntityIngestionConfig",
    # Errors
    "CheckpointError",
    "DownloadError",
    "IngestionAbortError",
    "IngestionError",
    "LoadError",
    "TransformationError",
    "ValidationError",
    # Progress
    "ProgressTracker",
    # Telemetry
    "SpanAttributes",
    "record_ingestion_metric",
    "trace_ingestion_operation",
    # Validation
    "EntityValidator",
    "ValidationResult",
]
