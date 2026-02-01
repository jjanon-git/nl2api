"""
Local Worker Manager.

Manages multiple worker subprocesses locally for distributed evaluation.
"""

from __future__ import annotations

import atexit
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorkerProcess:
    """Information about a worker subprocess."""

    worker_id: str
    process: subprocess.Popen
    started_at: float = field(default_factory=time.time)


class LocalWorkerManager:
    """
    Manages multiple worker subprocesses locally.

    Spawns worker processes using subprocess.Popen and manages their lifecycle.
    Designed for local development and testing - for production, use K8s or
    other orchestration.

    Features:
    - Spawn N worker processes with unique IDs
    - Graceful shutdown with SIGTERM then SIGKILL
    - Health monitoring (check if workers are still running)
    - Automatic cleanup on exit (via atexit handler)
    """

    def __init__(
        self,
        worker_count: int = 4,
        redis_url: str = "redis://localhost:6379",
        eval_mode: str = "resolver",
        max_retries: int = 3,
        verbose: bool = False,
        *,
        _register_atexit: bool = True,
    ):
        """
        Initialize the worker manager.

        Args:
            worker_count: Number of worker subprocesses to spawn
            redis_url: Redis URL for queue connection
            eval_mode: Evaluation mode for workers (resolver, orchestrator, etc.)
            max_retries: Maximum retry attempts per task
            verbose: Enable verbose worker logging
            _register_atexit: Register atexit cleanup handler (disable for tests)
        """
        self.worker_count = worker_count
        self.redis_url = redis_url
        self.eval_mode = eval_mode
        self.max_retries = max_retries
        self.verbose = verbose
        self._workers: list[WorkerProcess] = []
        self._batch_id: str | None = None

        # Register cleanup handler (can be disabled for unit tests)
        if _register_atexit:
            atexit.register(self._cleanup_on_exit)

    def start(self, batch_id: str) -> None:
        """
        Spawn worker subprocesses.

        Args:
            batch_id: Batch ID for workers to process
        """
        if self._workers:
            logger.warning("Workers already running, stopping existing workers first")
            self.stop()

        self._batch_id = batch_id

        for i in range(self.worker_count):
            worker_id = f"worker-{i}"
            cmd = self._build_worker_command(worker_id, batch_id)

            try:
                # Start worker process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE if not self.verbose else None,
                    stderr=subprocess.STDOUT if not self.verbose else None,
                    # Don't create new process group - we want signals to propagate
                    start_new_session=False,
                )

                worker = WorkerProcess(
                    worker_id=worker_id,
                    process=process,
                )
                self._workers.append(worker)
                logger.info(f"Started {worker_id} (PID {process.pid})")

            except Exception as e:
                logger.error(f"Failed to start {worker_id}: {e}")
                # Stop any workers we did start
                self.stop()
                raise RuntimeError(f"Failed to start workers: {e}") from e

        logger.info(f"Started {len(self._workers)} workers for batch {batch_id}")

    def _build_worker_command(self, worker_id: str, batch_id: str) -> list[str]:
        """Build the command to run a worker subprocess."""
        cmd = [
            sys.executable,
            "-m",
            "src.evalkit.distributed",
            "--worker-id",
            worker_id,
            "--batch-id",
            batch_id,
            "--redis-url",
            self.redis_url,
            "--mode",
            self.eval_mode,
            "--max-retries",
            str(self.max_retries),
        ]

        if self.verbose:
            cmd.append("--verbose")

        return cmd

    def stop(self, timeout: int = 30) -> None:
        """
        Gracefully stop all workers.

        Sends SIGTERM first, waits for graceful shutdown, then SIGKILL
        if workers don't exit within timeout.

        Args:
            timeout: Seconds to wait for graceful shutdown before SIGKILL
        """
        if not self._workers:
            return

        logger.info(f"Stopping {len(self._workers)} workers...")

        # Send SIGTERM to all workers
        for worker in self._workers:
            if worker.process.poll() is None:
                try:
                    worker.process.terminate()
                    logger.debug(f"Sent SIGTERM to {worker.worker_id} (PID {worker.process.pid})")
                except Exception as e:
                    logger.warning(f"Failed to terminate {worker.worker_id}: {e}")

        # Wait for graceful shutdown
        deadline = time.time() + timeout
        while time.time() < deadline:
            all_stopped = all(w.process.poll() is not None for w in self._workers)
            if all_stopped:
                break
            time.sleep(0.5)

        # Force kill any remaining workers
        for worker in self._workers:
            if worker.process.poll() is None:
                try:
                    logger.warning(f"Force killing {worker.worker_id} (PID {worker.process.pid})")
                    worker.process.kill()
                except Exception as e:
                    logger.warning(f"Failed to kill {worker.worker_id}: {e}")

        # Wait for kill to complete
        for worker in self._workers:
            try:
                worker.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error(f"Worker {worker.worker_id} still running after kill")

        logger.info("All workers stopped")
        self._workers = []
        self._batch_id = None

    def is_healthy(self) -> bool:
        """
        Check if all workers are still running.

        Returns:
            True if all workers are alive, False if any have exited
        """
        if not self._workers:
            return False

        return all(w.process.poll() is None for w in self._workers)

    def get_running_count(self) -> int:
        """
        Get count of currently running workers.

        Returns:
            Number of workers still running
        """
        return sum(1 for w in self._workers if w.process.poll() is None)

    def get_worker_pids(self) -> list[int]:
        """
        Get PIDs of running workers.

        Returns:
            List of PIDs for running workers
        """
        return [w.process.pid for w in self._workers if w.process.poll() is None]

    def wait_for_exit(self, timeout: float | None = None) -> dict[str, int]:
        """
        Wait for all workers to exit.

        Args:
            timeout: Maximum time to wait (None for indefinite)

        Returns:
            Dict mapping worker_id to exit code
        """
        deadline = time.time() + timeout if timeout else None
        results = {}

        while self._workers:
            for worker in self._workers[:]:
                exit_code = worker.process.poll()
                if exit_code is not None:
                    results[worker.worker_id] = exit_code
                    self._workers.remove(worker)
                    logger.info(f"{worker.worker_id} exited with code {exit_code}")

            if not self._workers:
                break

            if deadline and time.time() >= deadline:
                # Timeout - record remaining workers as -1
                for worker in self._workers:
                    results[worker.worker_id] = -1
                break

            time.sleep(0.5)

        return results

    def _cleanup_on_exit(self) -> None:
        """Cleanup handler called on program exit."""
        if self._workers:
            logger.debug("Cleaning up workers on exit...")
            self.stop(timeout=10)


__all__ = ["LocalWorkerManager"]
