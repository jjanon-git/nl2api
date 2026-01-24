"""
Task Queue Factory.

Creates the appropriate TaskQueue implementation based on configuration.
"""

from __future__ import annotations

import logging

from src.evaluation.distributed.config import QueueBackend, QueueConfig
from src.evaluation.distributed.queue.memory import InMemoryQueue
from src.evaluation.distributed.queue.protocol import QueueError, TaskQueue

logger = logging.getLogger(__name__)


async def create_queue(config: QueueConfig | None = None) -> TaskQueue:
    """
    Create a TaskQueue instance based on configuration.

    Args:
        config: Queue configuration. If None, uses defaults (Redis).

    Returns:
        TaskQueue implementation

    Raises:
        QueueError: If queue creation fails
        NotImplementedError: If backend is not yet implemented

    Example:
        # In-memory for testing
        queue = await create_queue(QueueConfig(backend=QueueBackend.MEMORY))

        # Redis for production
        queue = await create_queue(QueueConfig(
            backend=QueueBackend.REDIS,
            redis_url="redis://localhost:6379",
        ))
    """
    if config is None:
        config = QueueConfig()

    logger.info(f"Creating queue with backend: {config.backend}")

    if config.backend == QueueBackend.MEMORY:
        return InMemoryQueue(
            max_retries=config.max_retries,
            consumer_group=config.consumer_group,
        )

    elif config.backend == QueueBackend.REDIS:
        # Lazy import to avoid dependency when not using Redis
        try:
            from src.evaluation.distributed.queue.redis_stream import RedisStreamQueue
        except ImportError as e:
            raise QueueError(
                "Redis backend requires 'redis' package. Install with: pip install redis"
            ) from e

        return await RedisStreamQueue.create(config)

    elif config.backend == QueueBackend.AZURE_SB:
        raise NotImplementedError(
            "Azure Service Bus backend not yet implemented. "
            "See docs/plans/distributed-evaluation-plan.md for roadmap."
        )

    else:
        raise QueueError(f"Unknown queue backend: {config.backend}")


__all__ = ["create_queue"]
