"""
Distributed Evaluation Infrastructure.

Provides scalable, fault-tolerant batch evaluation using Redis Streams
for task distribution and worker coordination.

Key components:
- TaskQueue protocol and implementations (Redis Streams, in-memory)
- EvalWorker for processing evaluation tasks
- BatchCoordinator for orchestrating distributed batches
- DistributedMetrics for observability

Usage:
    from src.evalkit.distributed import (
        create_queue,
        EvalWorker,
        BatchCoordinator,
        WorkerConfig,
    )

    # Create queue
    queue = await create_queue(QueueConfig(backend="redis"))

    # Start worker
    worker = EvalWorker(worker_id="worker-0", queue=queue, ...)
    await worker.run()
"""

from src.evalkit.distributed.config import (
    AlertConfig,
    CoordinatorConfig,
    EvalMode,
    QueueBackend,
    QueueConfig,
    WorkerConfig,
)
from src.evalkit.distributed.coordinator import BatchCoordinator, BatchResult
from src.evalkit.distributed.manager import LocalWorkerManager
from src.evalkit.distributed.models import BatchProgress, QueueMessage, WorkerStatus

# Queue imports (protocol and factory)
from src.evalkit.distributed.queue import (
    InMemoryQueue,
    TaskQueue,
    create_queue,
)
from src.evalkit.distributed.worker import EvalWorker

__all__ = [
    # Config
    "QueueConfig",
    "WorkerConfig",
    "CoordinatorConfig",
    "AlertConfig",
    "EvalMode",
    "QueueBackend",
    # Models
    "QueueMessage",
    "BatchProgress",
    "WorkerStatus",
    # Queue
    "TaskQueue",
    "create_queue",
    "InMemoryQueue",
    # Coordinator
    "BatchCoordinator",
    "BatchResult",
    # Manager
    "LocalWorkerManager",
    # Worker
    "EvalWorker",
]
