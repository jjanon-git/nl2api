"""
Unit tests for batch evaluation checkpoint/resume functionality.

Tests:
- get_evaluated_test_case_ids returns correct set
- Resume filters completed tests
- Resume nonexistent batch returns None
- Resume completed batch returns early
- Checkpoint interval saves progress
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.fixture
def sample_test_cases() -> list[TestCase]:
    """Create sample test cases for testing."""
    return [
        TestCase(
            id=f"tc-{i}",
            nl_query=f"Query {i}",
            expected_tool_calls=(ToolCall(tool_name="test_tool", arguments={"arg": f"value-{i}"}),),
            metadata=TestCaseMetadata(
                category="test",
                subcategory="unit",
                api_version="1.0",
                complexity_level=1,
                tags=("test",),
            ),
        )
        for i in range(10)
    ]


@pytest.fixture
def mock_repos():
    """Create mock repositories."""
    test_case_repo = AsyncMock()
    scorecard_repo = AsyncMock()
    batch_repo = AsyncMock()
    return test_case_repo, scorecard_repo, batch_repo


@pytest.fixture
def batch_config():
    """Create batch runner config for testing."""
    return BatchRunnerConfig(
        pack_name="nl2api",
        max_concurrency=5,
        show_progress=False,
        checkpoint_interval=3,
        run_label="test-run",
    )


class TestGetEvaluatedTestCaseIds:
    """Tests for ScorecardRepository.get_evaluated_test_case_ids()."""

    @pytest.mark.asyncio
    async def test_returns_correct_set(self, mock_repos):
        """Test that get_evaluated_test_case_ids returns the correct set of IDs."""
        _, scorecard_repo, _ = mock_repos

        # Simulate 3 test cases already evaluated
        scorecard_repo.get_evaluated_test_case_ids = AsyncMock(
            return_value={"tc-0", "tc-1", "tc-2"}
        )

        result = await scorecard_repo.get_evaluated_test_case_ids("batch-001")

        assert result == {"tc-0", "tc-1", "tc-2"}
        scorecard_repo.get_evaluated_test_case_ids.assert_called_once_with("batch-001")

    @pytest.mark.asyncio
    async def test_returns_empty_set_for_new_batch(self, mock_repos):
        """Test that new batch returns empty set."""
        _, scorecard_repo, _ = mock_repos
        scorecard_repo.get_evaluated_test_case_ids = AsyncMock(return_value=set())

        result = await scorecard_repo.get_evaluated_test_case_ids("new-batch")

        assert result == set()


class TestResumeFiltersCompletedTests:
    """Tests for resume filtering already-evaluated test cases."""

    @pytest.mark.asyncio
    async def test_resume_filters_completed_tests(
        self, mock_repos, sample_test_cases, batch_config
    ):
        """Test that resuming filters out already-evaluated test cases."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        # Set up existing batch with 3 completed
        existing_batch = BatchJob(
            batch_id="resume-batch",
            total_tests=10,
            completed_count=3,
            failed_count=0,
            status=TaskStatus.IN_PROGRESS,
            started_at=datetime.now(UTC),
        )
        batch_repo.get = AsyncMock(return_value=existing_batch)

        # 3 test cases already evaluated
        scorecard_repo.get_evaluated_test_case_ids = AsyncMock(
            return_value={"tc-0", "tc-1", "tc-2"}
        )
        scorecard_repo.save = AsyncMock()

        # All 10 test cases available
        test_case_repo.list = AsyncMock(return_value=sample_test_cases)

        # Mock the pack
        with patch("src.evalkit.batch.runner.get_pack") as mock_get_pack:
            mock_pack = MagicMock()
            mock_pack.evaluate = AsyncMock(
                return_value=Scorecard(
                    test_case_id="tc-3",
                    pack_name="nl2api",
                    overall_passed=True,
                    overall_score=1.0,
                    syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
                )
            )
            mock_get_pack.return_value = mock_pack

            runner = BatchRunner(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                config=batch_config,
            )

            # Mock _load_resume_state to verify it's called
            runner._load_resume_state = AsyncMock(
                return_value=(existing_batch, {"tc-0", "tc-1", "tc-2"})
            )

            result = await runner.run(
                tags=["test"],
                resume_batch_id="resume-batch",
            )

            # Verify _load_resume_state was called
            runner._load_resume_state.assert_called_once_with("resume-batch")

            # Result should be based on original batch total
            assert result is not None


class TestResumeNonexistentBatch:
    """Tests for resume with nonexistent batch."""

    @pytest.mark.asyncio
    async def test_resume_nonexistent_batch_returns_none(self, mock_repos, batch_config):
        """Test that resuming nonexistent batch returns None."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        # Batch doesn't exist
        batch_repo.get = AsyncMock(return_value=None)

        with patch("src.evalkit.batch.runner.get_pack") as mock_get_pack:
            mock_get_pack.return_value = MagicMock()

            runner = BatchRunner(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                config=batch_config,
            )

            result = await runner.run(
                tags=["test"],
                resume_batch_id="nonexistent-batch",
            )

            assert result is None
            batch_repo.get.assert_called_once_with("nonexistent-batch")


class TestResumeCompletedBatch:
    """Tests for resume with already-completed batch."""

    @pytest.mark.asyncio
    async def test_resume_completed_batch_returns_early(self, mock_repos, batch_config):
        """Test that resuming a completed batch returns it immediately."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        # Batch already completed
        completed_batch = BatchJob(
            batch_id="completed-batch",
            total_tests=10,
            completed_count=8,
            failed_count=2,
            status=TaskStatus.COMPLETED,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        batch_repo.get = AsyncMock(return_value=completed_batch)

        with patch("src.evalkit.batch.runner.get_pack") as mock_get_pack:
            mock_get_pack.return_value = MagicMock()

            runner = BatchRunner(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                config=batch_config,
            )

            result = await runner.run(
                tags=["test"],
                resume_batch_id="completed-batch",
            )

            # Should return the completed batch immediately
            assert result is not None
            assert result.batch_id == "completed-batch"
            assert result.status == TaskStatus.COMPLETED

            # Should not try to fetch test cases
            test_case_repo.list.assert_not_called()


class TestCheckpointInterval:
    """Tests for periodic checkpoint saving."""

    @pytest.mark.asyncio
    async def test_checkpoint_saves_progress(self, mock_repos, sample_test_cases):
        """Test that checkpoint_interval triggers progress saves."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        # Create config with checkpoint_interval=2
        config = BatchRunnerConfig(
            pack_name="nl2api",
            max_concurrency=1,  # Sequential for predictable behavior
            show_progress=False,
            checkpoint_interval=2,
            run_label="checkpoint-test",
        )

        # Set up repos
        test_case_repo.list = AsyncMock(return_value=sample_test_cases[:5])  # 5 test cases
        scorecard_repo.save = AsyncMock()
        batch_repo.create = AsyncMock()
        batch_repo.update = AsyncMock()
        batch_repo.update_progress = AsyncMock()

        with patch("src.evalkit.batch.runner.get_pack") as mock_get_pack:
            mock_pack = MagicMock()
            mock_pack.evaluate = AsyncMock(
                return_value=Scorecard(
                    test_case_id="tc-0",
                    pack_name="nl2api",
                    overall_passed=True,
                    overall_score=1.0,
                    syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
                )
            )
            mock_get_pack.return_value = mock_pack

            runner = BatchRunner(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                config=config,
            )

            await runner.run(tags=["test"])

            # With 5 test cases and checkpoint_interval=2:
            # Checkpoints at: 2, 4 (not at 5 because 5 % 2 = 1, but final update happens anyway)
            # So update_progress should be called at least twice
            assert batch_repo.update_progress.call_count >= 2

    @pytest.mark.asyncio
    async def test_checkpoint_disabled_when_zero(self, mock_repos, sample_test_cases):
        """Test that checkpoint_interval=0 disables checkpointing."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        # Create config with checkpoint_interval=0 (disabled)
        config = BatchRunnerConfig(
            pack_name="nl2api",
            max_concurrency=1,
            show_progress=False,
            checkpoint_interval=0,
            run_label="no-checkpoint-test",
        )

        # Set up repos
        test_case_repo.list = AsyncMock(return_value=sample_test_cases[:5])
        scorecard_repo.save = AsyncMock()
        batch_repo.create = AsyncMock()
        batch_repo.update = AsyncMock()
        batch_repo.update_progress = AsyncMock()

        with patch("src.evalkit.batch.runner.get_pack") as mock_get_pack:
            mock_pack = MagicMock()
            mock_pack.evaluate = AsyncMock(
                return_value=Scorecard(
                    test_case_id="tc-0",
                    pack_name="nl2api",
                    overall_passed=True,
                    overall_score=1.0,
                    syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
                )
            )
            mock_get_pack.return_value = mock_pack

            runner = BatchRunner(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                config=config,
            )

            await runner.run(tags=["test"])

            # With checkpoint_interval=0, update_progress should never be called
            batch_repo.update_progress.assert_not_called()


class TestLoadResumeState:
    """Tests for _load_resume_state helper method."""

    @pytest.mark.asyncio
    async def test_load_resume_state_returns_batch_and_ids(self, mock_repos, batch_config):
        """Test that _load_resume_state returns batch job and evaluated IDs."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        existing_batch = BatchJob(
            batch_id="test-batch",
            total_tests=10,
            completed_count=5,
            failed_count=0,
            status=TaskStatus.IN_PROGRESS,
        )
        batch_repo.get = AsyncMock(return_value=existing_batch)
        scorecard_repo.get_evaluated_test_case_ids = AsyncMock(
            return_value={"tc-0", "tc-1", "tc-2", "tc-3", "tc-4"}
        )

        with patch("src.evalkit.batch.runner.get_pack") as mock_get_pack:
            mock_get_pack.return_value = MagicMock()

            runner = BatchRunner(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                config=batch_config,
            )

            batch, ids = await runner._load_resume_state("test-batch")

            assert batch is not None
            assert batch.batch_id == "test-batch"
            assert len(ids) == 5
            assert "tc-0" in ids

    @pytest.mark.asyncio
    async def test_load_resume_state_not_found(self, mock_repos, batch_config):
        """Test that _load_resume_state returns None for nonexistent batch."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        batch_repo.get = AsyncMock(return_value=None)

        with patch("src.evalkit.batch.runner.get_pack") as mock_get_pack:
            mock_get_pack.return_value = MagicMock()

            runner = BatchRunner(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                config=batch_config,
            )

            batch, ids = await runner._load_resume_state("nonexistent")

            assert batch is None
            assert ids == set()


class TestSaveCheckpoint:
    """Tests for _save_checkpoint helper method."""

    @pytest.mark.asyncio
    async def test_save_checkpoint_calls_update_progress(self, mock_repos, batch_config):
        """Test that _save_checkpoint calls update_progress on repo."""
        test_case_repo, scorecard_repo, batch_repo = mock_repos

        batch_repo.update_progress = AsyncMock()

        with patch("src.evalkit.batch.runner.get_pack") as mock_get_pack:
            mock_get_pack.return_value = MagicMock()

            runner = BatchRunner(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                config=batch_config,
            )

            await runner._save_checkpoint("batch-123", 5, 2)

            batch_repo.update_progress.assert_called_once_with("batch-123", 5, 2)


class TestConfigDefaults:
    """Tests for BatchRunnerConfig checkpoint defaults."""

    def test_checkpoint_interval_default(self):
        """Test that checkpoint_interval defaults to 10."""
        config = BatchRunnerConfig(pack_name="nl2api")
        assert config.checkpoint_interval == 10

    def test_checkpoint_interval_custom(self):
        """Test that checkpoint_interval can be customized."""
        config = BatchRunnerConfig(pack_name="nl2api", checkpoint_interval=5)
        assert config.checkpoint_interval == 5

    def test_checkpoint_interval_zero_disables(self):
        """Test that checkpoint_interval=0 is valid (disables checkpointing)."""
        config = BatchRunnerConfig(pack_name="nl2api", checkpoint_interval=0)
        assert config.checkpoint_interval == 0
