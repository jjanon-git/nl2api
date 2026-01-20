"""
In-Memory Storage Backend

Simple in-memory implementation for unit tests.
No external dependencies required - perfect for fast, isolated tests.
"""

from src.storage.memory.repositories import (
    InMemoryTestCaseRepository,
    InMemoryScorecardRepository,
)

__all__ = [
    "InMemoryTestCaseRepository",
    "InMemoryScorecardRepository",
]
