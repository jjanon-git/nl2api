"""
Ingestion Checkpoint Manager

Provides resume capability for long-running ingestion jobs.
Checkpoints are saved atomically to ensure consistency.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class IngestionCheckpoint:
    """Tracks ingestion progress for resume capability."""

    source: str  # gleif, sec_edgar
    started_at: datetime
    last_offset: int = 0  # Rows processed
    last_entity_id: str | None = None  # Last successfully imported entity
    state: str = "initialized"  # initialized, downloading, loading, complete, failed
    error_message: str | None = None
    stats: dict = field(default_factory=dict)

    # Stats tracked during ingestion
    total_records: int = 0
    imported_count: int = 0
    skipped_count: int = 0
    error_count: int = 0

    # Timestamps
    last_updated_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert datetimes to ISO strings
        d["started_at"] = self.started_at.isoformat()
        if self.last_updated_at:
            d["last_updated_at"] = self.last_updated_at.isoformat()
        if self.completed_at:
            d["completed_at"] = self.completed_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> IngestionCheckpoint:
        """Create from dictionary."""
        # Parse ISO datetime strings
        d["started_at"] = datetime.fromisoformat(d["started_at"])
        if d.get("last_updated_at"):
            d["last_updated_at"] = datetime.fromisoformat(d["last_updated_at"])
        if d.get("completed_at"):
            d["completed_at"] = datetime.fromisoformat(d["completed_at"])
        return cls(**d)

    @property
    def is_complete(self) -> bool:
        """Check if ingestion is complete."""
        return self.state == "complete"

    @property
    def is_failed(self) -> bool:
        """Check if ingestion failed."""
        return self.state == "failed"

    @property
    def is_resumable(self) -> bool:
        """Check if this checkpoint can be resumed."""
        return self.state not in ("complete", "failed", "initialized")

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_records == 0:
            return 0.0
        return (self.last_offset / self.total_records) * 100

    def mark_downloading(self) -> None:
        """Mark state as downloading."""
        self.state = "downloading"
        self.last_updated_at = datetime.now(UTC)

    def mark_loading(self, total_records: int) -> None:
        """Mark state as loading with total record count."""
        self.state = "loading"
        self.total_records = total_records
        self.last_updated_at = datetime.now(UTC)

    def update_progress(
        self,
        offset: int,
        imported: int = 0,
        skipped: int = 0,
        errors: int = 0,
        last_entity_id: str | None = None,
    ) -> None:
        """Update progress counters."""
        self.last_offset = offset
        self.imported_count += imported
        self.skipped_count += skipped
        self.error_count += errors
        if last_entity_id:
            self.last_entity_id = last_entity_id
        self.last_updated_at = datetime.now(UTC)

    def mark_complete(self) -> None:
        """Mark ingestion as complete."""
        self.state = "complete"
        self.completed_at = datetime.now(UTC)
        self.last_updated_at = self.completed_at

    def mark_failed(self, error_message: str) -> None:
        """Mark ingestion as failed."""
        self.state = "failed"
        self.error_message = error_message
        self.last_updated_at = datetime.now(UTC)


class CheckpointManager:
    """Manages ingestion checkpoints for reliability."""

    def __init__(self, checkpoint_dir: Path, source: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            source: Data source name (gleif, sec_edgar)
        """
        self.checkpoint_dir = checkpoint_dir
        self.source = source
        self.path = checkpoint_dir / f"ingestion_{source}.json"

    def save(self, checkpoint: IngestionCheckpoint) -> None:
        """
        Atomically save checkpoint.

        Writes to temp file first, then renames for atomic update.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")

        with open(temp_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        # Atomic rename on POSIX systems
        os.replace(temp_path, self.path)

    def load(self) -> IngestionCheckpoint | None:
        """Load existing checkpoint if present."""
        if not self.path.exists():
            return None

        try:
            with open(self.path) as f:
                data = json.load(f)
            return IngestionCheckpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            # Corrupted checkpoint - remove it
            self.path.unlink(missing_ok=True)
            return None

    def should_resume(self) -> bool:
        """Check if we should resume from checkpoint."""
        checkpoint = self.load()
        return checkpoint is not None and checkpoint.is_resumable

    def create_new(self) -> IngestionCheckpoint:
        """Create a new checkpoint for this source."""
        return IngestionCheckpoint(
            source=self.source,
            started_at=datetime.now(UTC),
        )

    def delete(self) -> bool:
        """Delete checkpoint file. Returns True if deleted."""
        if self.path.exists():
            self.path.unlink()
            return True
        return False

    def get_resume_offset(self) -> int:
        """Get the offset to resume from (0 if no checkpoint)."""
        checkpoint = self.load()
        if checkpoint and checkpoint.is_resumable:
            return checkpoint.last_offset
        return 0
