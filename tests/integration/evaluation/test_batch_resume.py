"""
Integration tests for batch evaluation checkpoint/resume functionality.

Tests the full resume flow with real PostgreSQL database:
1. Create batch, evaluate some tests, simulate interrupt
2. Resume batch, verify only remaining tests are evaluated
3. Verify final counts are correct
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from CONTRACTS import (
    BatchJob,
    Scorecard,
    StageResult,
    TaskStatus,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.evalkit.batch.config import BatchRunnerConfig
from src.evalkit.batch.runner import BatchRunner
from src.evalkit.common.storage import close_repositories, create_repositories
from src.evalkit.common.storage.config import StorageConfig


def create_test_case(test_id: str, tag: str = "resume-test") -> TestCase:
    """Create a test case for testing."""
    return TestCase(
        id=test_id,
        nl_query=f"Query for {test_id}",
        expected_tool_calls=(ToolCall(tool_name="test_tool", arguments={"arg": test_id}),),
        metadata=TestCaseMetadata(
            category="test",
            subcategory="resume",
            api_version="1.0",
            complexity_level=1,
            tags=(tag,),
        ),
    )


class TestBatchResume:
    """Integration tests for batch resume functionality."""

    @pytest.mark.asyncio
    async def test_resume_continues_from_checkpoint(self):
        """
        Test that resuming a batch skips already-evaluated test cases.

        Flow:
        1. Create 10 test cases
        2. Start batch, evaluate 5 test cases
        3. Manually stop (simulating interrupt)
        4. Resume batch
        5. Verify only 5 remaining tests are evaluated
        6. Verify final counts are correct (10 total, 10 completed)
        """
        # Setup
        config = StorageConfig(backend="postgres")
        repos = await create_repositories(config)
        test_case_repo, scorecard_repo, batch_repo = repos
        pool = test_case_repo.pool

        # Create unique tag for this test run to avoid conflicts
        unique_tag = f"resume-test-{uuid.uuid4().hex[:8]}"

        # Track items for cleanup
        test_case_ids = []
        batch_ids = []

        try:
            # Create 10 test cases
            test_cases = []
            for i in range(10):
                tc_id = str(uuid.uuid4())
                tc = create_test_case(tc_id, unique_tag)
                await test_case_repo.save(tc)
                test_cases.append(tc)
                test_case_ids.append(tc_id)

            # Create batch runner config with checkpoint disabled (we'll manually control)
            runner_config = BatchRunnerConfig(
                pack_name="nl2api",
                max_concurrency=1,  # Sequential for predictable behavior
                show_progress=False,
                checkpoint_interval=0,  # Disable automatic checkpoints
                run_label="resume-integration-test",
            )

            # Track evaluation counts
            eval_count = 0
            first_batch_id = None

            # Mock the pack to track evaluations
            with patch("src.evalkit.batch.runner.get_pack") as mock_get_pack:

                def create_scorecard(tc_id: str, batch_id: str) -> Scorecard:
                    return Scorecard(
                        test_case_id=tc_id,
                        batch_id=batch_id,
                        pack_name="nl2api",
                        overall_passed=True,
                        overall_score=1.0,
                        syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
                    )

                mock_pack = MagicMock()

                async def mock_evaluate(test_case, system_output, context):
                    nonlocal eval_count
                    eval_count += 1
                    return create_scorecard(test_case.id, context.batch_id)

                mock_pack.evaluate = mock_evaluate
                mock_get_pack.return_value = mock_pack

                # Create runner
                runner = BatchRunner(
                    test_case_repo=test_case_repo,
                    scorecard_repo=scorecard_repo,
                    batch_repo=batch_repo,
                    config=runner_config,
                )

                # Start batch (will evaluate all 10, but we'll manually create "partial" state)
                # First, create a batch job manually with in_progress status
                first_batch_id = str(uuid.uuid4())
                batch_job = BatchJob(
                    batch_id=first_batch_id,
                    total_tests=10,
                    completed_count=0,
                    failed_count=0,
                    status=TaskStatus.IN_PROGRESS,
                    started_at=datetime.now(UTC),
                    tags=(unique_tag,),
                    run_label="resume-integration-test",
                )
                await batch_repo.create(batch_job)
                batch_ids.append(first_batch_id)

                # Manually create scorecards for first 5 test cases (simulating partial run)
                for i in range(5):
                    scorecard = create_scorecard(test_cases[i].id, first_batch_id)
                    await scorecard_repo.save(scorecard)

                # Update batch progress to reflect partial completion
                await batch_repo.update_progress(first_batch_id, 5, 0)

                # Verify partial state
                evaluated_ids = await scorecard_repo.get_evaluated_test_case_ids(first_batch_id)
                assert len(evaluated_ids) == 5

                # Reset eval counter before resume
                eval_count = 0

                # Now resume the batch
                result = await runner.run(
                    tags=[unique_tag],
                    resume_batch_id=first_batch_id,
                )

                # Verify only 5 remaining tests were evaluated
                assert eval_count == 5, f"Expected 5 evaluations, got {eval_count}"

                # Verify final batch state
                assert result is not None
                assert result.batch_id == first_batch_id
                assert result.status == TaskStatus.COMPLETED
                assert result.total_tests == 10
                assert result.completed_count == 10  # 5 resumed + 5 new
                assert result.failed_count == 0

                # Verify all 10 test cases have scorecards
                all_evaluated = await scorecard_repo.get_evaluated_test_case_ids(first_batch_id)
                assert len(all_evaluated) == 10

        finally:
            # Cleanup
            for tc_id in test_case_ids:
                await pool.execute("DELETE FROM test_cases WHERE id = $1", uuid.UUID(tc_id))
            for batch_id in batch_ids:
                # Delete scorecards first (FK constraint)
                await pool.execute("DELETE FROM scorecards WHERE batch_id = $1", batch_id)
                await pool.execute("DELETE FROM batch_jobs WHERE id = $1", uuid.UUID(batch_id))

            await close_repositories()

    @pytest.mark.asyncio
    async def test_update_progress_persists_checkpoint(self):
        """Test that update_progress correctly persists checkpoint data."""
        # Setup
        config = StorageConfig(backend="postgres")
        repos = await create_repositories(config)
        _, _, batch_repo = repos
        pool = batch_repo.pool

        batch_id = str(uuid.uuid4())

        try:
            # Create batch
            batch_job = BatchJob(
                batch_id=batch_id,
                total_tests=100,
                completed_count=0,
                failed_count=0,
                status=TaskStatus.IN_PROGRESS,
                started_at=datetime.now(UTC),
            )
            await batch_repo.create(batch_job)

            # Update progress
            await batch_repo.update_progress(batch_id, 50, 10)

            # Verify progress was persisted
            retrieved = await batch_repo.get(batch_id)
            assert retrieved.completed_count == 50
            assert retrieved.failed_count == 10

            # Verify last_checkpoint_at was set (check raw row)
            row = await pool.fetchrow(
                "SELECT last_checkpoint_at FROM batch_jobs WHERE id = $1",
                uuid.UUID(batch_id),
            )
            assert row["last_checkpoint_at"] is not None

        finally:
            await pool.execute("DELETE FROM batch_jobs WHERE id = $1", uuid.UUID(batch_id))
            await close_repositories()

    @pytest.mark.asyncio
    async def test_get_evaluated_test_case_ids_returns_correct_set(self):
        """Test that get_evaluated_test_case_ids returns correct IDs."""
        # Setup
        config = StorageConfig(backend="postgres")
        repos = await create_repositories(config)
        test_case_repo, scorecard_repo, batch_repo = repos
        pool = test_case_repo.pool

        batch_id = str(uuid.uuid4())
        test_case_ids = []

        try:
            # Create batch
            batch_job = BatchJob(
                batch_id=batch_id,
                total_tests=5,
                status=TaskStatus.IN_PROGRESS,
                started_at=datetime.now(UTC),
            )
            await batch_repo.create(batch_job)

            # Create test cases and scorecards
            for i in range(5):
                tc_id = str(uuid.uuid4())
                tc = create_test_case(tc_id)
                await test_case_repo.save(tc)
                test_case_ids.append(tc_id)

                # Create scorecard for first 3
                if i < 3:
                    scorecard = Scorecard(
                        test_case_id=tc_id,
                        batch_id=batch_id,
                        pack_name="nl2api",
                        overall_passed=True,
                        overall_score=1.0,
                        syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
                    )
                    await scorecard_repo.save(scorecard)

            # Get evaluated IDs
            evaluated_ids = await scorecard_repo.get_evaluated_test_case_ids(batch_id)

            # Verify correct IDs returned
            assert len(evaluated_ids) == 3
            assert test_case_ids[0] in evaluated_ids
            assert test_case_ids[1] in evaluated_ids
            assert test_case_ids[2] in evaluated_ids
            assert test_case_ids[3] not in evaluated_ids
            assert test_case_ids[4] not in evaluated_ids

        finally:
            # Cleanup
            await pool.execute("DELETE FROM scorecards WHERE batch_id = $1", batch_id)
            await pool.execute("DELETE FROM batch_jobs WHERE id = $1", uuid.UUID(batch_id))
            for tc_id in test_case_ids:
                await pool.execute("DELETE FROM test_cases WHERE id = $1", uuid.UUID(tc_id))
            await close_repositories()
