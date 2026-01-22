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
    from src.evaluation.distributed import (
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

from src.evaluation.distributed.config import (
    QueueConfig,
    WorkerConfig,
    CoordinatorConfig,
    AlertConfig,
)
from src.evaluation.distributed.models import QueueMessage

# Queue imports (protocol and factory)
from src.evaluation.distributed.queue import (
    TaskQueue,
    create_queue,
    InMemoryQueue,
)

__all__ = [
    # Config
    "QueueConfig",
    "WorkerConfig",
    "CoordinatorConfig",
    "AlertConfig",
    # Models
    "QueueMessage",
    # Queue
    "TaskQueue",
    "create_queue",
    "InMemoryQueue",
]
