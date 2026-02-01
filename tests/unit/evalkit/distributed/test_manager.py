"""
Unit tests for LocalWorkerManager.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.evalkit.distributed.manager import LocalWorkerManager


class TestLocalWorkerManagerInit:
    """Tests for LocalWorkerManager initialization."""

    def test_init_default_values(self):
        """Test manager initializes with default values."""
        manager = LocalWorkerManager()

        assert manager.worker_count == 4
        assert manager.redis_url == "redis://localhost:6379"
        assert manager.eval_mode == "resolver"
        assert manager.max_retries == 3
        assert manager.verbose is False
        assert manager._workers == []

    def test_init_custom_values(self):
        """Test manager initializes with custom values."""
        manager = LocalWorkerManager(
            worker_count=8,
            redis_url="redis://custom:6380",
            eval_mode="orchestrator",
            max_retries=5,
            verbose=True,
        )

        assert manager.worker_count == 8
        assert manager.redis_url == "redis://custom:6380"
        assert manager.eval_mode == "orchestrator"
        assert manager.max_retries == 5
        assert manager.verbose is True


class TestBuildWorkerCommand:
    """Tests for LocalWorkerManager._build_worker_command()."""

    def test_build_command_basic(self):
        """Test basic command building."""
        manager = LocalWorkerManager(
            worker_count=2,
            redis_url="redis://localhost:6379",
            eval_mode="resolver",
        )

        cmd = manager._build_worker_command("worker-0", "batch-001")

        assert cmd[0] == sys.executable
        assert "-m" in cmd
        assert "src.evalkit.distributed" in cmd
        assert "--worker-id" in cmd
        assert "worker-0" in cmd
        assert "--batch-id" in cmd
        assert "batch-001" in cmd
        assert "--redis-url" in cmd
        assert "redis://localhost:6379" in cmd
        assert "--mode" in cmd
        assert "resolver" in cmd
        assert "--verbose" not in cmd

    def test_build_command_with_verbose(self):
        """Test command building with verbose flag."""
        manager = LocalWorkerManager(verbose=True)

        cmd = manager._build_worker_command("worker-0", "batch-001")

        assert "--verbose" in cmd


class TestStartWorkers:
    """Tests for LocalWorkerManager.start()."""

    @patch("subprocess.Popen")
    def test_start_spawns_workers(self, mock_popen):
        """Test start spawns the correct number of workers."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = LocalWorkerManager(worker_count=3)
        manager.start("batch-001")

        assert mock_popen.call_count == 3
        assert len(manager._workers) == 3
        assert all(w.worker_id.startswith("worker-") for w in manager._workers)
        assert manager._batch_id == "batch-001"

    @patch("subprocess.Popen")
    def test_start_with_existing_workers_stops_first(self, mock_popen):
        """Test start stops existing workers before spawning new ones."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        # Return None (running) first, then 0 (exited) after terminate
        poll_values = [None, 0]
        mock_process.poll.side_effect = lambda: poll_values.pop(0) if poll_values else 0
        mock_popen.return_value = mock_process

        manager = LocalWorkerManager(worker_count=2)

        # First start
        manager.start("batch-001")
        assert len(manager._workers) == 2

        # Second start should stop first
        manager.start("batch-002")
        assert len(manager._workers) == 2
        assert manager._batch_id == "batch-002"

    @patch("subprocess.Popen")
    def test_start_failure_stops_already_started(self, mock_popen):
        """Test that if one worker fails to start, all are stopped."""
        # First two succeed, third fails
        mock_process = MagicMock()
        mock_process.pid = 12345
        # Return 0 (exited) immediately to avoid 30s timeout in stop()
        mock_process.poll.return_value = 0
        mock_process.terminate = MagicMock()
        mock_process.wait = MagicMock()

        def popen_side_effect(*args, **kwargs):
            if mock_popen.call_count <= 2:
                return mock_process
            raise OSError("Failed to start")

        mock_popen.side_effect = popen_side_effect

        manager = LocalWorkerManager(worker_count=3)

        with pytest.raises(RuntimeError, match="Failed to start workers"):
            manager.start("batch-001")


class TestStopWorkers:
    """Tests for LocalWorkerManager.stop()."""

    def test_stop_with_no_workers(self):
        """Test stop with no workers does nothing."""
        manager = LocalWorkerManager()

        # Should not raise
        manager.stop()

        assert manager._workers == []

    @patch("subprocess.Popen")
    def test_stop_terminates_workers(self, mock_popen):
        """Test stop sends SIGTERM to all workers."""
        # Create separate mocks for each worker
        processes = []
        for i in range(2):
            p = MagicMock()
            p.pid = 12345 + i
            # Use a function that returns None first (for start), then 0 (for stop)
            call_count = [0]  # Use mutable list to track calls

            def make_poll(count_tracker):
                def poll_func():
                    count_tracker[0] += 1
                    # First call returns None (running), subsequent calls return 0 (exited)
                    return None if count_tracker[0] <= 1 else 0

                return poll_func

            p.poll = make_poll(call_count)
            p.terminate = MagicMock()
            p.wait = MagicMock()
            processes.append(p)

        mock_popen.side_effect = processes

        manager = LocalWorkerManager(worker_count=2)
        manager.start("batch-001")

        manager.stop(timeout=1)

        # Both processes should have terminate called
        for p in processes:
            p.terminate.assert_called_once()
        assert manager._workers == []

    @patch("subprocess.Popen")
    def test_stop_force_kills_unresponsive(self, mock_popen):
        """Test stop kills workers that don't respond to SIGTERM."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Never exits on terminate
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = MagicMock(side_effect=subprocess.TimeoutExpired("cmd", 1))
        mock_popen.return_value = mock_process

        manager = LocalWorkerManager(worker_count=1)
        manager.start("batch-001")

        manager.stop(timeout=0.1)  # Very short timeout

        mock_process.terminate.assert_called()
        mock_process.kill.assert_called()


class TestIsHealthy:
    """Tests for LocalWorkerManager.is_healthy()."""

    def test_is_healthy_no_workers(self):
        """Test is_healthy returns False with no workers."""
        manager = LocalWorkerManager()

        assert manager.is_healthy() is False

    @patch("subprocess.Popen")
    def test_is_healthy_all_running(self, mock_popen):
        """Test is_healthy returns True when all workers are running."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Running
        mock_popen.return_value = mock_process

        manager = LocalWorkerManager(worker_count=2)
        manager.start("batch-001")

        assert manager.is_healthy() is True

    @patch("subprocess.Popen")
    def test_is_healthy_one_crashed(self, mock_popen):
        """Test is_healthy returns False when one worker has crashed."""
        processes = []
        for i in range(2):
            p = MagicMock()
            p.pid = 12345 + i
            # First running, second crashed
            p.poll.return_value = None if i == 0 else 1
            processes.append(p)

        mock_popen.side_effect = processes

        manager = LocalWorkerManager(worker_count=2)
        manager.start("batch-001")

        assert manager.is_healthy() is False


class TestGetRunningCount:
    """Tests for LocalWorkerManager.get_running_count()."""

    def test_get_running_count_no_workers(self):
        """Test get_running_count returns 0 with no workers."""
        manager = LocalWorkerManager()

        assert manager.get_running_count() == 0

    @patch("subprocess.Popen")
    def test_get_running_count_all_running(self, mock_popen):
        """Test get_running_count returns correct count."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = LocalWorkerManager(worker_count=3)
        manager.start("batch-001")

        assert manager.get_running_count() == 3

    @patch("subprocess.Popen")
    def test_get_running_count_some_exited(self, mock_popen):
        """Test get_running_count with some workers exited."""
        processes = []
        for i in range(3):
            p = MagicMock()
            p.pid = 12345 + i
            # First two running, third exited
            p.poll.return_value = None if i < 2 else 0
            processes.append(p)

        mock_popen.side_effect = processes

        manager = LocalWorkerManager(worker_count=3)
        manager.start("batch-001")

        assert manager.get_running_count() == 2


class TestGetWorkerPids:
    """Tests for LocalWorkerManager.get_worker_pids()."""

    @patch("subprocess.Popen")
    def test_get_worker_pids(self, mock_popen):
        """Test get_worker_pids returns PIDs of running workers."""
        processes = []
        for i in range(3):
            p = MagicMock()
            p.pid = 12345 + i
            p.poll.return_value = None  # All running
            processes.append(p)

        mock_popen.side_effect = processes

        manager = LocalWorkerManager(worker_count=3)
        manager.start("batch-001")

        pids = manager.get_worker_pids()

        assert len(pids) == 3
        assert set(pids) == {12345, 12346, 12347}


class TestWaitForExit:
    """Tests for LocalWorkerManager.wait_for_exit()."""

    @patch("subprocess.Popen")
    def test_wait_for_exit_all_success(self, mock_popen):
        """Test wait_for_exit returns exit codes when all workers exit."""
        # Create workers that exit immediately with code 0
        processes = []
        for i in range(2):
            p = MagicMock()
            p.pid = 12345 + i
            p.poll.return_value = 0  # Exited with success
            processes.append(p)

        mock_popen.side_effect = processes

        manager = LocalWorkerManager(worker_count=2)
        manager.start("batch-001")

        results = manager.wait_for_exit(timeout=1)

        assert len(results) == 2
        assert all(code == 0 for code in results.values())

    @patch("subprocess.Popen")
    def test_wait_for_exit_timeout(self, mock_popen):
        """Test wait_for_exit returns -1 for workers that don't exit."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Never exits
        mock_popen.return_value = mock_process

        manager = LocalWorkerManager(worker_count=1)
        manager.start("batch-001")

        results = manager.wait_for_exit(timeout=0.1)

        assert results["worker-0"] == -1
