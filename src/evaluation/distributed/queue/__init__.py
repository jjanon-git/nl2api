"""
Task Queue Implementations.

Provides the TaskQueue protocol and implementations for distributed task processing.

Implementations:
- InMemoryQueue: For unit testing (no external dependencies)
- RedisStreamQueue: For production (Redis Streams with consumer groups)
- AzureServiceBusQueue: Future (Azure Service Bus)
"""

from src.evaluation.distributed.queue.protocol import TaskQueue
from src.evaluation.distributed.queue.memory import InMemoryQueue
from src.evaluation.distributed.queue.factory import create_queue

__all__ = [
    "TaskQueue",
    "InMemoryQueue",
    "create_queue",
]
