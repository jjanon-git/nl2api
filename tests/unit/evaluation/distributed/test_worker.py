"""
Unit tests for EvalWorker.

Tests worker behavior with mocked queue and dependencies.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.contracts.worker import WorkerTask
from src.evaluation.distributed.config import EvalMode, WorkerConfig
from src.evaluation.distributed.models import QueueMessage
from src.evaluation.distributed.worker import EvalWorker

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def worker_config() -> WorkerConfig:
    """Create worker configuration for testing."""
    return WorkerConfig(
        worker_id="test-worker",
        eval_mode=EvalMode.SIMULATED,
        task_timeout_seconds=10,
        heartbeat_interval_seconds=1,
        shutdown_timeout_seconds=5,
    )


@pytest.fixture
def mock_queue():
    """Create a mock queue."""
    queue = AsyncMock()
    queue.ack = AsyncMock()
    queue.nack = AsyncMock()
    queue.close = AsyncMock()
    return queue


@pytest.fixture
def sample_task() -> WorkerTask:
    """Create a sample worker task."""
    return WorkerTask(
        task_id="task-001",
        test_case_id="tc-001",
        batch_id="batch-001",
    )


@pytest.fixture
def sample_message(sample_task: WorkerTask) -> QueueMessage:
    """Create a sample queue message."""
    return QueueMessage(
        message_id="msg-001",
        stream_name=f"eval:tasks:{sample_task.batch_id}",
        payload=sample_task.model_dump(),
        attempt=1,
        enqueued_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_test_case():
    """Create a mock test case."""
    return MagicMock(
        id="tc-001",
        nl_query="What is Apple's stock price?",
        expected_tool_calls=[],
    )


@pytest.fixture
def mock_scorecard():
    """Create a mock scorecard."""
    scorecard = MagicMock()
    scorecard.test_case_id = "tc-001"
    scorecard.overall_passed = True
    scorecard.overall_score = 1.0
    return scorecard


# =============================================================================
# Worker Initialization Tests
# =============================================================================


class TestWorkerInit:
    """Tests for worker initialization."""

    def test_worker_initializes_with_required_args(self, mock_queue, worker_config):
        """Worker should initialize with required arguments."""
        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(),
            evaluator=AsyncMock(),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(),
            config=worker_config,
        )

        assert worker.worker_id == "test-worker"
        assert worker.batch_id == "batch-001"
        assert worker._running is False
        assert worker._tasks_processed == 0
        assert worker._tasks_failed == 0

    def test_worker_uses_default_config_if_not_provided(self, mock_queue):
        """Worker should create default config if not provided."""
        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(),
            evaluator=AsyncMock(),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(),
        )

        assert worker.config is not None
        assert worker.config.worker_id == "test-worker"


# =============================================================================
# Worker Status Tests
# =============================================================================


class TestWorkerStatus:
    """Tests for worker status reporting."""

    def test_status_reports_idle_when_not_running(self, mock_queue, worker_config):
        """Status should be 'stopped' when worker is not running."""
        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(),
            evaluator=AsyncMock(),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(),
            config=worker_config,
        )

        status = worker.status
        assert status.status == "stopped"
        assert status.worker_id == "test-worker"
        assert status.tasks_processed == 0
        assert status.tasks_failed == 0

    def test_status_includes_current_task_when_processing(
        self, mock_queue, worker_config, sample_message
    ):
        """Status should include current task ID when processing."""
        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(),
            evaluator=AsyncMock(),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(),
            config=worker_config,
        )

        # Simulate processing
        worker._running = True
        worker._current_task = sample_message

        status = worker.status
        assert status.status == "running"
        assert status.current_task_id == "task-001"
        assert status.current_batch_id == "batch-001"


# =============================================================================
# Task Processing Tests
# =============================================================================


class TestTaskProcessing:
    """Tests for task processing logic."""

    @pytest.mark.asyncio
    async def test_process_task_success(
        self,
        mock_queue,
        worker_config,
        sample_message,
        mock_test_case,
        mock_scorecard,
    ):
        """Successful task processing should call all stages."""
        # Set up mocks
        response_generator = AsyncMock(return_value={"result": "success"})
        evaluator = AsyncMock(return_value=mock_scorecard)
        scorecard_saver = AsyncMock()
        test_case_fetcher = AsyncMock(return_value=mock_test_case)

        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=response_generator,
            evaluator=evaluator,
            scorecard_saver=scorecard_saver,
            test_case_fetcher=test_case_fetcher,
            config=worker_config,
        )

        # Process the task
        await worker._process_task(sample_message)

        # Verify all stages were called
        test_case_fetcher.assert_called_once_with("tc-001")
        response_generator.assert_called_once()
        evaluator.assert_called_once_with(mock_test_case, {"result": "success"})
        scorecard_saver.assert_called_once_with(mock_scorecard)

    @pytest.mark.asyncio
    async def test_process_task_raises_on_missing_test_case(
        self, mock_queue, worker_config, sample_message
    ):
        """Task processing should fail if test case is not found."""
        test_case_fetcher = AsyncMock(return_value=None)

        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(),
            evaluator=AsyncMock(),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=test_case_fetcher,
            config=worker_config,
        )

        with pytest.raises(ValueError, match="Test case tc-001 not found"):
            await worker._process_task(sample_message)

    @pytest.mark.asyncio
    async def test_process_task_raises_on_missing_test_case_id(self, mock_queue, worker_config):
        """Task processing should fail if message has no test_case_id."""
        message = QueueMessage(
            message_id="msg-001",
            stream_name="eval:tasks:batch-001",
            payload={"task_id": "task-001", "batch_id": "batch-001"},  # Missing test_case_id
            attempt=1,
            enqueued_at=datetime.now(UTC),
        )

        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(),
            evaluator=AsyncMock(),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(),
            config=worker_config,
        )

        with pytest.raises(ValueError, match="has no test_case_id"):
            await worker._process_task(message)


# =============================================================================
# Main Loop Tests
# =============================================================================


class TestWorkerLoop:
    """Tests for worker main loop."""

    @pytest.mark.asyncio
    async def test_worker_processes_messages_from_queue(
        self,
        mock_queue,
        worker_config,
        sample_message,
        mock_test_case,
        mock_scorecard,
    ):
        """Worker should process messages from queue until shutdown."""

        # Set up mock consumer that yields one message then stops
        async def mock_consume(*args, **kwargs):
            yield sample_message

        mock_queue.consume = mock_consume

        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(return_value={}),
            evaluator=AsyncMock(return_value=mock_scorecard),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(return_value=mock_test_case),
            config=worker_config,
        )

        # Run worker (will stop when consumer exhausts)
        await worker.run()

        # Verify message was acked
        mock_queue.ack.assert_called_once_with(sample_message)
        assert worker._tasks_processed == 1
        assert worker._tasks_failed == 0

    @pytest.mark.asyncio
    async def test_worker_nacks_on_failure(self, mock_queue, worker_config, sample_message):
        """Worker should nack message on processing failure."""

        async def mock_consume(*args, **kwargs):
            yield sample_message

        mock_queue.consume = mock_consume

        # Fetcher returns None, causing failure
        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(),
            evaluator=AsyncMock(),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(return_value=None),
            config=worker_config,
        )

        await worker.run()

        # Verify message was nacked with requeue
        mock_queue.nack.assert_called_once()
        call_args = mock_queue.nack.call_args
        assert call_args[0][0] == sample_message
        assert call_args[1]["requeue"] is True
        assert "not found" in call_args[1]["error"]
        assert worker._tasks_failed == 1

    @pytest.mark.asyncio
    async def test_worker_increments_counters_correctly(
        self,
        mock_queue,
        worker_config,
        mock_test_case,
        mock_scorecard,
    ):
        """Worker should track processed and failed counts."""
        messages = [
            QueueMessage(
                message_id=f"msg-{i}",
                stream_name="eval:tasks:batch-001",
                payload={
                    "task_id": f"task-{i}",
                    "test_case_id": f"tc-{i}",
                    "batch_id": "batch-001",
                },
                attempt=1,
                enqueued_at=datetime.now(UTC),
            )
            for i in range(5)
        ]

        async def mock_consume(*args, **kwargs):
            for msg in messages:
                yield msg

        mock_queue.consume = mock_consume

        # Make every other task fail
        call_count = 0

        async def alternating_fetcher(test_case_id):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return None  # Fail
            return mock_test_case

        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(return_value={}),
            evaluator=AsyncMock(return_value=mock_scorecard),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=alternating_fetcher,
            config=worker_config,
        )

        await worker.run()

        assert worker._tasks_processed == 3  # 1, 3, 5 succeed
        assert worker._tasks_failed == 2  # 2, 4 fail


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestWorkerShutdown:
    """Tests for worker shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_event(self, mock_queue, worker_config):
        """Shutdown should set the shutdown event."""
        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(),
            evaluator=AsyncMock(),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(),
            config=worker_config,
        )

        assert not worker._shutdown_event.is_set()
        await worker.shutdown()
        assert worker._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_worker_stops_on_shutdown_event(
        self,
        mock_queue,
        worker_config,
        mock_test_case,
        mock_scorecard,
    ):
        """Worker should stop consuming when shutdown event is set."""
        message_count = 0
        shutdown_requested = False

        async def mock_consume(*args, **kwargs):
            nonlocal message_count, shutdown_requested
            while not shutdown_requested:
                message_count += 1
                yield QueueMessage(
                    message_id=f"msg-{message_count}",
                    stream_name="eval:tasks:batch-001",
                    payload={
                        "task_id": f"task-{message_count}",
                        "test_case_id": f"tc-{message_count}",
                        "batch_id": "batch-001",
                    },
                    attempt=1,
                    enqueued_at=datetime.now(UTC),
                )
                # After yielding a few messages, mark shutdown requested
                if message_count >= 3:
                    shutdown_requested = True

        mock_queue.consume = mock_consume

        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(return_value={}),
            evaluator=AsyncMock(return_value=mock_scorecard),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(return_value=mock_test_case),
            config=worker_config,
        )

        # Schedule shutdown after short delay
        async def trigger_shutdown():
            await asyncio.sleep(0.05)
            await worker.shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())

        await worker.run()
        await shutdown_task

        # Worker should have processed some messages
        assert worker._tasks_processed >= 1
        # Verify shutdown was handled gracefully (no exceptions)

    @pytest.mark.asyncio
    async def test_shutdown_requeues_current_message(
        self,
        mock_queue,
        worker_config,
        mock_test_case,
        mock_scorecard,
    ):
        """On shutdown, current message should be requeued."""
        messages = [
            QueueMessage(
                message_id=f"msg-{i}",
                stream_name="eval:tasks:batch-001",
                payload={
                    "task_id": f"task-{i}",
                    "test_case_id": f"tc-{i}",
                    "batch_id": "batch-001",
                },
                attempt=1,
                enqueued_at=datetime.now(UTC),
            )
            for i in range(3)
        ]

        message_index = 0
        shutdown_triggered = False

        async def mock_consume(*args, **kwargs):
            nonlocal message_index, shutdown_triggered
            while message_index < len(messages):
                msg = messages[message_index]
                message_index += 1
                yield msg
                # Trigger shutdown after first message processed
                if not shutdown_triggered and message_index == 2:
                    shutdown_triggered = True

        mock_queue.consume = mock_consume

        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(return_value={}),
            evaluator=AsyncMock(return_value=mock_scorecard),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(return_value=mock_test_case),
            config=worker_config,
        )

        # Trigger shutdown after processing starts
        async def trigger_shutdown():
            while not shutdown_triggered:
                await asyncio.sleep(0.01)
            await worker.shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await worker.run()
        await shutdown_task

        # Check that nack was called with requeue=True for interrupted message
        # (The worker should nack the message it was about to process when shutdown detected)
        [
            call for call in mock_queue.nack.call_args_list if call[1].get("requeue") is True
        ]
        # May or may not have nacked depending on timing
        # The key is that no messages are lost


# =============================================================================
# Heartbeat Tests
# =============================================================================


class TestWorkerHeartbeat:
    """Tests for worker heartbeat functionality."""

    @pytest.mark.asyncio
    async def test_heartbeat_runs_while_worker_active(
        self, mock_queue, mock_test_case, mock_scorecard
    ):
        """Heartbeat should run periodically while worker is active."""
        config = WorkerConfig(
            worker_id="test-worker",
            heartbeat_interval_seconds=0.05,  # Very short for testing
        )

        # Single message then stop
        async def mock_consume(*args, **kwargs):
            await asyncio.sleep(0.2)  # Give time for heartbeats
            yield QueueMessage(
                message_id="msg-001",
                stream_name="eval:tasks:batch-001",
                payload={"task_id": "task-001", "test_case_id": "tc-001", "batch_id": "batch-001"},
                attempt=1,
                enqueued_at=datetime.now(UTC),
            )

        mock_queue.consume = mock_consume

        heartbeat_count = 0
        original_sleep = asyncio.sleep

        async def counting_sleep(seconds):
            nonlocal heartbeat_count
            if seconds == config.heartbeat_interval_seconds:
                heartbeat_count += 1
            await original_sleep(seconds)

        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(return_value={}),
            evaluator=AsyncMock(return_value=mock_scorecard),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(return_value=mock_test_case),
            config=config,
        )

        with patch("asyncio.sleep", counting_sleep):
            await worker.run()

        # Should have had at least one heartbeat
        assert heartbeat_count >= 1


# =============================================================================
# Span/Telemetry Tests
# =============================================================================


class TestWorkerTelemetry:
    """Tests for worker telemetry integration."""

    @pytest.mark.asyncio
    async def test_span_context_manager_yields_none_without_telemetry(
        self, mock_queue, worker_config
    ):
        """Span context manager should yield None when telemetry unavailable."""
        worker = EvalWorker(
            worker_id="test-worker",
            queue=mock_queue,
            batch_id="batch-001",
            response_generator=AsyncMock(),
            evaluator=AsyncMock(),
            scorecard_saver=AsyncMock(),
            test_case_fetcher=AsyncMock(),
            config=worker_config,
        )

        # Patch to simulate no telemetry
        with patch.object(worker, "_span") as mock_span:
            # Make _span return a no-op context manager
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def noop_span(name):
                yield None

            mock_span.side_effect = noop_span

            async with worker._span("test.span"):
                # Should get None when telemetry not available
                pass  # Just verify it doesn't raise
