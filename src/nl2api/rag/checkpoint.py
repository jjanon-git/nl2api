"""
RAG Indexing Checkpoint

Provides checkpoint/resume functionality for large indexing jobs.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


class CheckpointStatus(Enum):
    """Status of an indexing checkpoint."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class IndexingCheckpoint:
    """
    Checkpoint state for a resumable indexing job.

    Tracks progress and allows resumption after failures or interruptions.
    """
    job_id: str
    total_items: int = 0
    processed_items: int = 0
    last_offset: int = 0
    status: CheckpointStatus = CheckpointStatus.RUNNING
    error_message: str | None = None
    error_count: int = 0
    domain: str | None = None
    batch_size: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def remaining_items(self) -> int:
        """Get number of items remaining."""
        return max(0, self.total_items - self.processed_items)

    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.status == CheckpointStatus.COMPLETED

    @property
    def is_resumable(self) -> bool:
        """Check if job can be resumed."""
        return self.status in (CheckpointStatus.RUNNING, CheckpointStatus.PAUSED, CheckpointStatus.FAILED)


class CheckpointManager:
    """
    Manages indexing checkpoints for resumable operations.

    Provides:
    - Create/update checkpoints during indexing
    - Resume from last successful offset
    - Track progress and errors
    """

    def __init__(self, pool: "asyncpg.Pool"):
        """
        Initialize checkpoint manager.

        Args:
            pool: asyncpg connection pool
        """
        self._pool = pool

    async def create_checkpoint(
        self,
        total_items: int,
        domain: str | None = None,
        batch_size: int = 100,
        job_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IndexingCheckpoint:
        """
        Create a new indexing checkpoint.

        Args:
            total_items: Total number of items to index
            domain: Domain being indexed (optional)
            batch_size: Batch size for processing
            job_id: Optional job ID (auto-generated if not provided)
            metadata: Additional metadata

        Returns:
            New IndexingCheckpoint
        """
        job_id = job_id or str(uuid.uuid4())
        checkpoint = IndexingCheckpoint(
            job_id=job_id,
            total_items=total_items,
            domain=domain,
            batch_size=batch_size,
            metadata=metadata or {},
        )

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO indexing_checkpoints (
                    job_id, total_items, processed_items, last_offset,
                    status, domain, batch_size, metadata, started_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                """,
                checkpoint.job_id,
                checkpoint.total_items,
                checkpoint.processed_items,
                checkpoint.last_offset,
                checkpoint.status.value,
                checkpoint.domain,
                checkpoint.batch_size,
                checkpoint.metadata,
                checkpoint.started_at,
            )

        logger.info(
            f"Created checkpoint {job_id} for {total_items} items "
            f"(domain={domain}, batch_size={batch_size})"
        )
        return checkpoint

    async def update_progress(
        self,
        job_id: str,
        processed_items: int,
        last_offset: int,
    ) -> None:
        """
        Update checkpoint progress.

        Args:
            job_id: Job ID
            processed_items: Total items processed so far
            last_offset: Last successfully processed offset
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE indexing_checkpoints
                SET processed_items = $2, last_offset = $3
                WHERE job_id = $1
                """,
                job_id,
                processed_items,
                last_offset,
            )

    async def mark_completed(self, job_id: str) -> None:
        """Mark checkpoint as completed."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE indexing_checkpoints
                SET status = 'completed', completed_at = NOW()
                WHERE job_id = $1
                """,
                job_id,
            )
        logger.info(f"Checkpoint {job_id} marked as completed")

    async def mark_failed(self, job_id: str, error_message: str) -> None:
        """Mark checkpoint as failed with error message."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE indexing_checkpoints
                SET status = 'failed', error_message = $2, error_count = error_count + 1
                WHERE job_id = $1
                """,
                job_id,
                error_message,
            )
        logger.warning(f"Checkpoint {job_id} marked as failed: {error_message}")

    async def mark_paused(self, job_id: str) -> None:
        """Mark checkpoint as paused (for graceful interruption)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE indexing_checkpoints
                SET status = 'paused'
                WHERE job_id = $1
                """,
                job_id,
            )
        logger.info(f"Checkpoint {job_id} paused")

    async def get_checkpoint(self, job_id: str) -> IndexingCheckpoint | None:
        """
        Get checkpoint by job ID.

        Args:
            job_id: Job ID to look up

        Returns:
            IndexingCheckpoint if found, None otherwise
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM indexing_checkpoints WHERE job_id = $1",
                job_id,
            )

        if not row:
            return None

        return self._row_to_checkpoint(row)

    async def get_resumable_checkpoints(
        self,
        domain: str | None = None,
    ) -> list[IndexingCheckpoint]:
        """
        Get all resumable checkpoints.

        Args:
            domain: Optional domain filter

        Returns:
            List of resumable checkpoints
        """
        query = """
            SELECT * FROM indexing_checkpoints
            WHERE status IN ('running', 'paused', 'failed')
        """
        params = []

        if domain:
            query += " AND domain = $1"
            params.append(domain)

        query += " ORDER BY last_updated DESC"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [self._row_to_checkpoint(row) for row in rows]

    async def delete_checkpoint(self, job_id: str) -> bool:
        """Delete a checkpoint. Returns True if deleted."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM indexing_checkpoints WHERE job_id = $1",
                job_id,
            )
        return result == "DELETE 1"

    async def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        Clean up old completed/failed checkpoints.

        Args:
            days: Delete checkpoints older than this many days

        Returns:
            Number of checkpoints deleted
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM indexing_checkpoints
                WHERE status IN ('completed', 'failed')
                AND last_updated < NOW() - INTERVAL '1 day' * $1
                """,
                days,
            )
        count = int(result.split()[-1])
        if count > 0:
            logger.info(f"Cleaned up {count} old checkpoints")
        return count

    def _row_to_checkpoint(self, row) -> IndexingCheckpoint:
        """Convert database row to IndexingCheckpoint."""
        started_at = row["started_at"]
        if started_at and started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)

        last_updated = row["last_updated"]
        if last_updated and last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)

        completed_at = row["completed_at"]
        if completed_at and completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)

        return IndexingCheckpoint(
            job_id=row["job_id"],
            total_items=row["total_items"],
            processed_items=row["processed_items"],
            last_offset=row["last_offset"],
            status=CheckpointStatus(row["status"]),
            error_message=row["error_message"],
            error_count=row["error_count"],
            domain=row["domain"],
            batch_size=row["batch_size"],
            metadata=row["metadata"] or {},
            started_at=started_at,
            last_updated=last_updated,
            completed_at=completed_at,
        )
