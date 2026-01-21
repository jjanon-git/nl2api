"""
Entity Ingestion Pipeline

Handles bulk import of entity data from external sources:
- GLEIF (Legal Entity Identifier database)
- SEC EDGAR (US public company filings)

Designed for reliability, scalability, and repeatability.
"""

from src.nl2api.ingestion.checkpoint import CheckpointManager, IngestionCheckpoint
from src.nl2api.ingestion.config import EntityIngestionConfig
from src.nl2api.ingestion.progress import ProgressTracker
from src.nl2api.ingestion.validation import EntityValidator, ValidationResult

__all__ = [
    "CheckpointManager",
    "IngestionCheckpoint",
    "EntityIngestionConfig",
    "EntityValidator",
    "ValidationResult",
    "ProgressTracker",
]
