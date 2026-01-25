"""
Integration tests for BatchJobRepository.

Tests lifecycle operations, status transitions, and listing against real PostgreSQL.
"""

import uuid
from datetime import UTC, datetime

import pytest

from CONTRACTS import (
    BatchJob,
    TaskPriority,
    TaskStatus,
)
from src.evalkit.common.storage import close_repositories, create_repositories
from src.evalkit.common.storage.config import StorageConfig


class TestBatchJobRepository:
    """Integration tests for PostgresBatchJobRepository."""

    @pytest.mark.asyncio
    async def test_batch_job_repository_full_lifecycle(self):
        """
        Comprehensive test of batch job repository operations.

        Tests:
        1. Create and retrieve a batch job
        2. Get non-existent returns None
        3. Get with invalid UUID returns None
        4. Update progress (completed/failed counts)
        5. Full status lifecycle (PENDING -> IN_PROGRESS -> COMPLETED)
        6. Status lifecycle with failures
        7. List recent batch jobs
        8. List recent with limit
        9. Priority levels preserved
        10. Git metadata preserved
        11. Tags preserved
        12. Progress calculation
        13. Timestamps timezone-aware
        """
        # Get repository
        config = StorageConfig(backend="postgres")
        repos = await create_repositories(config)
        test_case_repo, _, batch_job_repo = repos
        pool = test_case_repo.pool

        # Track batch jobs for cleanup
        batch_ids_to_cleanup = []

        # =================================================================
        # Test 1: Create and retrieve a batch job
        # =================================================================
        batch_id_1 = str(uuid.uuid4())
        batch_job_1 = BatchJob(
            batch_id=batch_id_1,
            total_tests=100,
            completed_count=0,
            failed_count=0,
            status=TaskStatus.PENDING,
            submitted_by="test-user",
            priority=TaskPriority.NORMAL,
            tags=("integration", "test"),
            run_label="integration-test-run",
            run_description="Testing batch job repository",
        )

        await batch_job_repo.create(batch_job_1)
        batch_ids_to_cleanup.append(batch_id_1)

        retrieved_1 = await batch_job_repo.get(batch_id_1)
        assert retrieved_1 is not None
        assert retrieved_1.batch_id == batch_id_1
        assert retrieved_1.total_tests == 100
        assert retrieved_1.completed_count == 0
        assert retrieved_1.failed_count == 0
        assert retrieved_1.status == TaskStatus.PENDING
        assert retrieved_1.submitted_by == "test-user"
        assert retrieved_1.priority == TaskPriority.NORMAL
        assert "integration" in retrieved_1.tags
        assert retrieved_1.run_label == "integration-test-run"
        assert retrieved_1.run_description == "Testing batch job repository"

        # =================================================================
        # Test 2: Get non-existent returns None
        # =================================================================
        nonexistent_id = str(uuid.uuid4())
        result = await batch_job_repo.get(nonexistent_id)
        assert result is None

        # =================================================================
        # Test 3: Get with invalid UUID returns None
        # =================================================================
        result = await batch_job_repo.get("not-a-valid-uuid")
        assert result is None

        # =================================================================
        # Test 4: Update progress (completed/failed counts)
        # =================================================================
        progress_batch_id = str(uuid.uuid4())
        progress_batch = BatchJob(
            batch_id=progress_batch_id,
            total_tests=50,
            status=TaskStatus.PENDING,
        )
        await batch_job_repo.create(progress_batch)
        batch_ids_to_cleanup.append(progress_batch_id)

        # Start the batch
        started_batch = progress_batch.model_copy(
            update={
                "status": TaskStatus.IN_PROGRESS,
                "started_at": datetime.now(UTC),
            }
        )
        await batch_job_repo.update(started_batch)

        retrieved = await batch_job_repo.get(progress_batch_id)
        assert retrieved.status == TaskStatus.IN_PROGRESS
        assert retrieved.started_at is not None

        # Progress halfway
        progress_update = started_batch.model_copy(
            update={
                "completed_count": 20,
                "failed_count": 5,
            }
        )
        await batch_job_repo.update(progress_update)

        retrieved = await batch_job_repo.get(progress_batch_id)
        assert retrieved.completed_count == 20
        assert retrieved.failed_count == 5
        assert retrieved.progress_pct == 50.0  # (20+5)/50 * 100

        # =================================================================
        # Test 5: Full status lifecycle (PENDING -> IN_PROGRESS -> COMPLETED)
        # =================================================================
        lifecycle_batch_id = str(uuid.uuid4())
        lifecycle_batch = BatchJob(
            batch_id=lifecycle_batch_id,
            total_tests=10,
            status=TaskStatus.PENDING,
        )
        await batch_job_repo.create(lifecycle_batch)
        batch_ids_to_cleanup.append(lifecycle_batch_id)

        # Start
        started = lifecycle_batch.model_copy(
            update={
                "status": TaskStatus.IN_PROGRESS,
                "started_at": datetime.now(UTC),
            }
        )
        await batch_job_repo.update(started)

        # Complete all tests
        completed = started.model_copy(
            update={
                "status": TaskStatus.COMPLETED,
                "completed_count": 10,
                "failed_count": 0,
                "completed_at": datetime.now(UTC),
            }
        )
        await batch_job_repo.update(completed)

        retrieved = await batch_job_repo.get(lifecycle_batch_id)
        assert retrieved.status == TaskStatus.COMPLETED
        assert retrieved.completed_count == 10
        assert retrieved.failed_count == 0
        assert retrieved.completed_at is not None
        assert retrieved.progress_pct == 100.0

        # =================================================================
        # Test 6: Status lifecycle with failures
        # =================================================================
        failure_batch_id = str(uuid.uuid4())
        failure_batch = BatchJob(
            batch_id=failure_batch_id,
            total_tests=20,
            status=TaskStatus.PENDING,
        )
        await batch_job_repo.create(failure_batch)
        batch_ids_to_cleanup.append(failure_batch_id)

        # Complete with failures
        completed_with_failures = failure_batch.model_copy(
            update={
                "status": TaskStatus.COMPLETED,
                "started_at": datetime.now(UTC),
                "completed_at": datetime.now(UTC),
                "completed_count": 15,
                "failed_count": 5,
            }
        )
        await batch_job_repo.update(completed_with_failures)

        retrieved = await batch_job_repo.get(failure_batch_id)
        assert retrieved.completed_count == 15
        assert retrieved.failed_count == 5
        assert retrieved.progress_pct == 100.0

        # =================================================================
        # Test 7: List recent batch jobs
        # =================================================================
        list_batches = []
        for i in range(3):
            bj = BatchJob(
                batch_id=str(uuid.uuid4()),
                total_tests=10 + i,
                status=TaskStatus.PENDING,
                tags=("list-recent-test-unique",),
                run_label=f"list-test-{i}",
            )
            await batch_job_repo.create(bj)
            list_batches.append(bj)
            batch_ids_to_cleanup.append(bj.batch_id)

        recent = await batch_job_repo.list_recent(limit=10)

        our_ids = {bj.batch_id for bj in list_batches}
        found_ids = {bj.batch_id for bj in recent if bj.batch_id in our_ids}
        assert len(found_ids) == 3

        # Should be ordered by created_at DESC
        timestamps = [bj.created_at for bj in recent if bj.batch_id in our_ids]
        assert timestamps == sorted(timestamps, reverse=True)

        # =================================================================
        # Test 8: List recent with limit
        # =================================================================
        limit_batches = []
        for i in range(5):
            bj = BatchJob(
                batch_id=str(uuid.uuid4()),
                total_tests=10,
                status=TaskStatus.PENDING,
                tags=("limit-test-unique",),
            )
            await batch_job_repo.create(bj)
            limit_batches.append(bj)
            batch_ids_to_cleanup.append(bj.batch_id)

        recent = await batch_job_repo.list_recent(limit=2)
        assert len(recent) >= 2  # At least 2 results

        # =================================================================
        # Test 9: Priority levels preserved
        # =================================================================
        priority_batches = []
        for priority in [TaskPriority.LOW, TaskPriority.NORMAL, TaskPriority.HIGH]:
            bj = BatchJob(
                batch_id=str(uuid.uuid4()),
                total_tests=10,
                status=TaskStatus.PENDING,
                priority=priority,
            )
            await batch_job_repo.create(bj)
            priority_batches.append((bj, priority))
            batch_ids_to_cleanup.append(bj.batch_id)

        for bj, expected_priority in priority_batches:
            retrieved = await batch_job_repo.get(bj.batch_id)
            assert retrieved.priority == expected_priority

        # =================================================================
        # Test 10: Git metadata preserved
        # =================================================================
        git_batch_id = str(uuid.uuid4())
        git_batch = BatchJob(
            batch_id=git_batch_id,
            total_tests=10,
            status=TaskStatus.PENDING,
            git_commit="abc123def456",
            git_branch="feature/test-branch",
            run_label="git-metadata-test",
        )
        await batch_job_repo.create(git_batch)
        batch_ids_to_cleanup.append(git_batch_id)

        retrieved = await batch_job_repo.get(git_batch_id)
        assert retrieved.git_commit == "abc123def456"
        assert retrieved.git_branch == "feature/test-branch"

        # =================================================================
        # Test 11: Tags preserved
        # =================================================================
        tags_batch_id = str(uuid.uuid4())
        tags = ("entity-resolution", "accuracy-test", "tier1")
        tags_batch = BatchJob(
            batch_id=tags_batch_id,
            total_tests=50,
            status=TaskStatus.PENDING,
            tags=tags,
        )
        await batch_job_repo.create(tags_batch)
        batch_ids_to_cleanup.append(tags_batch_id)

        retrieved = await batch_job_repo.get(tags_batch_id)
        assert set(retrieved.tags) == set(tags)

        # =================================================================
        # Test 12: Progress calculation
        # =================================================================
        calc_batch_id = str(uuid.uuid4())
        calc_batch = BatchJob(
            batch_id=calc_batch_id,
            total_tests=100,
            completed_count=0,
            failed_count=0,
            status=TaskStatus.PENDING,
        )
        await batch_job_repo.create(calc_batch)
        batch_ids_to_cleanup.append(calc_batch_id)

        # 0% progress
        retrieved = await batch_job_repo.get(calc_batch_id)
        assert retrieved.progress_pct == 0.0

        # 25% progress
        updated_25 = calc_batch.model_copy(
            update={"completed_count": 20, "failed_count": 5}
        )
        await batch_job_repo.update(updated_25)

        retrieved = await batch_job_repo.get(calc_batch_id)
        assert retrieved.progress_pct == 25.0

        # 100% progress
        updated_100 = calc_batch.model_copy(
            update={"completed_count": 90, "failed_count": 10}
        )
        await batch_job_repo.update(updated_100)

        retrieved = await batch_job_repo.get(calc_batch_id)
        assert retrieved.progress_pct == 100.0

        # =================================================================
        # Test 13: Timestamps timezone-aware
        # =================================================================
        tz_batch_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        tz_batch = BatchJob(
            batch_id=tz_batch_id,
            total_tests=10,
            status=TaskStatus.IN_PROGRESS,
            started_at=now,
        )
        await batch_job_repo.create(tz_batch)
        batch_ids_to_cleanup.append(tz_batch_id)

        retrieved = await batch_job_repo.get(tz_batch_id)
        assert retrieved.created_at.tzinfo is not None
        assert retrieved.started_at.tzinfo is not None

        # =================================================================
        # Cleanup
        # =================================================================
        for batch_id in batch_ids_to_cleanup:
            await pool.execute("DELETE FROM batch_jobs WHERE id = $1", uuid.UUID(batch_id))

        # Close repositories (also resets the connection pool singleton)
        await close_repositories()
