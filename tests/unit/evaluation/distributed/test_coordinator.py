"""
Unit tests for BatchCoordinator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.contracts.core import TestCase, TestCaseMetadata, ToolCall
from src.contracts.worker import BatchJob
from src.evaluation.distributed.config import CoordinatorConfig
from src.evaluation.distributed.coordinator import BatchCoordinator, BatchResult
from src.evaluation.distributed.models import BatchProgress


@pytest.fixture
def mock_queue():
    """Create a mock TaskQueue."""
    queue = AsyncMock()
    queue.ensure_stream = AsyncMock()
    queue.enqueue_batch = AsyncMock(return_value=["msg-1", "msg-2", "msg-3"])
    queue.get_pending_count = AsyncMock(return_value=0)
    queue.get_processing_count = AsyncMock(return_value=0)
    queue.get_dlq_count = AsyncMock(return_value=0)
    queue.get_dlq_messages = AsyncMock(return_value=[])
    queue.retry_from_dlq = AsyncMock()
    queue.delete_stream = AsyncMock()
    return queue


@pytest.fixture
def mock_repos():
    """Create mock repositories."""
    batch_repo = AsyncMock()
    batch_repo.get = AsyncMock(return_value=BatchJob(total_tests=10))

    test_case_repo = AsyncMock()
    scorecard_repo = AsyncMock()
    scorecard_repo.get_batch_summary = AsyncMock(
        return_value={"passed": 0, "failed": 0, "avg_score": 0.0}
    )

    return batch_repo, test_case_repo, scorecard_repo


@pytest.fixture
def test_cases():
    """Create sample test cases."""
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
            ),
        )
        for i in range(3)
    ]


@pytest.fixture
def coordinator(mock_queue, mock_repos):
    """Create a BatchCoordinator with mocked dependencies."""
    batch_repo, test_case_repo, scorecard_repo = mock_repos
    config = CoordinatorConfig(
        progress_poll_interval_seconds=1,
        batch_timeout_seconds=10,
    )
    return BatchCoordinator(
        queue=mock_queue,
        batch_repo=batch_repo,
        test_case_repo=test_case_repo,
        scorecard_repo=scorecard_repo,
        config=config,
    )


class TestStartBatch:
    """Tests for BatchCoordinator.start_batch()."""

    @pytest.mark.asyncio
    async def test_start_batch_enqueues_tasks(self, coordinator, mock_queue, test_cases):
        """Test that start_batch enqueues all test cases as WorkerTasks."""
        batch_id = "test-batch-001"

        count = await coordinator.start_batch(test_cases, batch_id)

        assert count == 3
        mock_queue.ensure_stream.assert_called_once_with(batch_id)
        mock_queue.enqueue_batch.assert_called_once()

        # Verify WorkerTasks were created correctly
        call_args = mock_queue.enqueue_batch.call_args
        tasks = call_args[0][0]
        assert len(tasks) == 3
        assert tasks[0].test_case_id == "tc-0"
        assert tasks[0].batch_id == batch_id

    @pytest.mark.asyncio
    async def test_start_batch_empty_list(self, coordinator, mock_queue):
        """Test that start_batch with empty list returns 0."""
        count = await coordinator.start_batch([], "batch-001")

        assert count == 0
        mock_queue.enqueue_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_batch_large_batch_chunked(self, coordinator, mock_queue):
        """Test that large batches are chunked."""
        # Create 250 test cases (more than chunk size of 100)
        test_cases = [
            TestCase(
                id=f"tc-{i}",
                nl_query=f"Query {i}",
                expected_tool_calls=(ToolCall(tool_name="test_tool", arguments={}),),
                metadata=TestCaseMetadata(
                    category="test",
                    subcategory="unit",
                    api_version="1.0",
                    complexity_level=1,
                ),
            )
            for i in range(250)
        ]

        # Mock to return message IDs for each chunk
        mock_queue.enqueue_batch = AsyncMock(
            side_effect=[
                [f"msg-{i}" for i in range(100)],  # First chunk
                [f"msg-{i}" for i in range(100)],  # Second chunk
                [f"msg-{i}" for i in range(50)],  # Third chunk (remaining)
            ]
        )

        count = await coordinator.start_batch(test_cases, "batch-001")

        assert count == 250
        assert mock_queue.enqueue_batch.call_count == 3


class TestGetProgress:
    """Tests for BatchCoordinator.get_progress()."""

    @pytest.mark.asyncio
    async def test_get_progress_returns_batch_progress(self, coordinator, mock_queue, mock_repos):
        """Test that get_progress returns correct BatchProgress."""
        batch_repo, _, scorecard_repo = mock_repos

        # Configure mocks
        scorecard_repo.get_batch_summary = AsyncMock(
            return_value={"passed": 5, "failed": 2, "avg_score": 0.8}
        )
        mock_queue.get_pending_count = AsyncMock(return_value=2)
        mock_queue.get_processing_count = AsyncMock(return_value=1)
        mock_queue.get_dlq_count = AsyncMock(return_value=1)

        progress = await coordinator.get_progress("batch-001", total_tasks=10)

        assert isinstance(progress, BatchProgress)
        assert progress.batch_id == "batch-001"
        assert progress.total == 10
        assert progress.completed == 7  # passed + failed
        assert progress.passed == 5
        assert progress.failed == 2
        assert progress.pending == 3  # pending + processing
        assert progress.in_dlq == 1

    @pytest.mark.asyncio
    async def test_get_progress_without_total_queries_batch(self, coordinator, mock_repos):
        """Test that get_progress queries batch job when total not provided."""
        batch_repo, _, scorecard_repo = mock_repos
        batch_repo.get = AsyncMock(return_value=BatchJob(total_tests=20))
        scorecard_repo.get_batch_summary = AsyncMock(
            return_value={"passed": 10, "failed": 5, "avg_score": 0.7}
        )

        progress = await coordinator.get_progress("batch-001")

        batch_repo.get.assert_called_once_with("batch-001")
        assert progress.total == 20


class TestWaitForCompletion:
    """Tests for BatchCoordinator.wait_for_completion()."""

    @pytest.mark.asyncio
    async def test_wait_for_completion_immediate(self, coordinator, mock_repos):
        """Test wait_for_completion returns immediately when batch is complete."""
        _, _, scorecard_repo = mock_repos

        # Batch is already complete
        scorecard_repo.get_batch_summary = AsyncMock(
            return_value={"passed": 8, "failed": 2, "avg_score": 0.8}
        )

        result = await coordinator.wait_for_completion(
            batch_id="batch-001",
            total_tasks=10,
            timeout_seconds=60,
        )

        assert isinstance(result, BatchResult)
        assert result.completed is True
        assert result.total == 10
        assert result.passed == 8
        assert result.failed == 2

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self, coordinator, mock_repos):
        """Test wait_for_completion returns with completed=False on timeout."""
        _, _, scorecard_repo = mock_repos

        # Batch never completes
        scorecard_repo.get_batch_summary = AsyncMock(
            return_value={"passed": 3, "failed": 1, "avg_score": 0.75}
        )

        result = await coordinator.wait_for_completion(
            batch_id="batch-001",
            total_tasks=10,
            timeout_seconds=0.1,  # Very short timeout
            poll_interval=0.05,
        )

        assert result.completed is False
        assert result.passed == 3
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_wait_for_completion_progress_callback(self, coordinator, mock_repos):
        """Test wait_for_completion calls progress callback."""
        _, _, scorecard_repo = mock_repos

        # Simulate progress: 5 completed, then 10 completed
        call_count = 0

        async def mock_summary(batch_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"passed": 5, "failed": 0, "avg_score": 1.0}
            else:
                return {"passed": 10, "failed": 0, "avg_score": 1.0}

        scorecard_repo.get_batch_summary = mock_summary

        progress_calls = []

        def on_progress(completed, total):
            progress_calls.append((completed, total))

        result = await coordinator.wait_for_completion(
            batch_id="batch-001",
            total_tasks=10,
            on_progress=on_progress,
            poll_interval=0.05,
        )

        assert result.completed is True
        assert len(progress_calls) >= 1
        # First callback should be for 5 completed
        assert progress_calls[0] == (5, 10)


class TestRetryFailed:
    """Tests for BatchCoordinator.retry_failed()."""

    @pytest.mark.asyncio
    async def test_retry_failed_requeues_dlq_messages(self, coordinator, mock_queue):
        """Test retry_failed moves DLQ messages back to queue."""
        dlq_messages = [
            MagicMock(message_id="dlq-1"),
            MagicMock(message_id="dlq-2"),
        ]
        mock_queue.get_dlq_messages = AsyncMock(return_value=dlq_messages)
        mock_queue.retry_from_dlq = AsyncMock(return_value="new-msg-id")

        count = await coordinator.retry_failed("batch-001")

        assert count == 2
        assert mock_queue.retry_from_dlq.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_failed_empty_dlq(self, coordinator, mock_queue):
        """Test retry_failed with empty DLQ returns 0."""
        mock_queue.get_dlq_messages = AsyncMock(return_value=[])

        count = await coordinator.retry_failed("batch-001")

        assert count == 0
        mock_queue.retry_from_dlq.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_failed_handles_errors(self, coordinator, mock_queue):
        """Test retry_failed handles individual retry errors gracefully."""
        dlq_messages = [
            MagicMock(message_id="dlq-1"),
            MagicMock(message_id="dlq-2"),
            MagicMock(message_id="dlq-3"),
        ]
        mock_queue.get_dlq_messages = AsyncMock(return_value=dlq_messages)
        # Second message fails to retry
        mock_queue.retry_from_dlq = AsyncMock(
            side_effect=["new-1", Exception("Retry failed"), "new-3"]
        )

        count = await coordinator.retry_failed("batch-001")

        # Only 2 succeeded
        assert count == 2


class TestCleanup:
    """Tests for BatchCoordinator.cleanup()."""

    @pytest.mark.asyncio
    async def test_cleanup_deletes_stream(self, coordinator, mock_queue):
        """Test cleanup deletes the queue stream."""
        await coordinator.cleanup("batch-001")

        mock_queue.delete_stream.assert_called_once_with("batch-001")
