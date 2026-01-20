"""
PostgreSQL Batch Job Repository

Implements BatchJobRepository protocol for PostgreSQL.
"""

from __future__ import annotations

import uuid
from datetime import timezone

import asyncpg

from CONTRACTS import BatchJob, TaskPriority, TaskStatus


class PostgresBatchJobRepository:
    """
    PostgreSQL implementation of BatchJobRepository.

    Stores batch job tracking information for progress monitoring.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize repository with connection pool.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def create(self, batch_job: BatchJob) -> None:
        """Create a new batch job record."""
        batch_uuid = uuid.UUID(batch_job.batch_id)

        await self.pool.execute(
            """
            INSERT INTO batch_jobs (
                id, total_tests, completed_count, failed_count,
                status, submitted_by, priority, tags,
                created_at, started_at, completed_at
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8,
                $9, $10, $11
            )
            """,
            batch_uuid,
            batch_job.total_tests,
            batch_job.completed_count,
            batch_job.failed_count,
            batch_job.status.value,
            batch_job.submitted_by,
            batch_job.priority.value,
            list(batch_job.tags),
            batch_job.created_at,
            batch_job.started_at,
            batch_job.completed_at,
        )

    async def get(self, batch_id: str) -> BatchJob | None:
        """Fetch a batch job by ID."""
        try:
            batch_uuid = uuid.UUID(batch_id)
        except ValueError:
            return None

        row = await self.pool.fetchrow(
            "SELECT * FROM batch_jobs WHERE id = $1",
            batch_uuid,
        )
        return self._row_to_batch_job(row) if row else None

    async def update(self, batch_job: BatchJob) -> None:
        """Update an existing batch job."""
        batch_uuid = uuid.UUID(batch_job.batch_id)

        await self.pool.execute(
            """
            UPDATE batch_jobs SET
                completed_count = $2,
                failed_count = $3,
                status = $4,
                started_at = $5,
                completed_at = $6
            WHERE id = $1
            """,
            batch_uuid,
            batch_job.completed_count,
            batch_job.failed_count,
            batch_job.status.value,
            batch_job.started_at,
            batch_job.completed_at,
        )

    async def list_recent(self, limit: int = 10) -> list[BatchJob]:
        """List recent batch jobs."""
        rows = await self.pool.fetch(
            """
            SELECT * FROM batch_jobs
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [self._row_to_batch_job(row) for row in rows]

    def _row_to_batch_job(self, row: asyncpg.Record) -> BatchJob:
        """Convert database row to BatchJob model."""
        # Handle timestamps
        created_at = row["created_at"]
        if created_at and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        started_at = row["started_at"]
        if started_at and started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)

        completed_at = row["completed_at"]
        if completed_at and completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)

        return BatchJob(
            batch_id=str(row["id"]),
            total_tests=row["total_tests"],
            completed_count=row["completed_count"],
            failed_count=row["failed_count"],
            status=TaskStatus(row["status"]),
            submitted_by=row["submitted_by"],
            priority=TaskPriority(row["priority"]),
            tags=tuple(row["tags"]) if row["tags"] else (),
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
        )
