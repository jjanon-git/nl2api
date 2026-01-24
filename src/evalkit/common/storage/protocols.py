"""
Repository Protocol Definitions

Uses typing.Protocol for duck-typed interface definitions.
No inheritance required - any class implementing these methods qualifies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from CONTRACTS import BatchJob, Scorecard, TestCase


@runtime_checkable
class TestCaseRepository(Protocol):
    """
    Repository for test case storage (Gold Store).

    Provides CRUD operations plus text and vector similarity search.
    Local implementation uses PostgreSQL + pgvector.
    Production implementation will use Azure AI Search.
    """

    async def get(self, test_case_id: str) -> TestCase | None:
        """Fetch a single test case by ID."""
        ...

    async def get_many(self, test_case_ids: list[str]) -> list[TestCase]:
        """Fetch multiple test cases by IDs. Returns found cases (may be fewer than requested)."""
        ...

    async def save(self, test_case: TestCase) -> None:
        """Save or update a test case (upsert behavior)."""
        ...

    async def delete(self, test_case_id: str) -> bool:
        """Delete a test case. Returns True if deleted, False if not found."""
        ...

    async def list(
        self,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TestCase]:
        """
        List test cases with optional filters.

        Args:
            tags: Filter by any matching tag (OR logic)
            complexity_min: Minimum complexity level (1-5)
            complexity_max: Maximum complexity level (1-5)
            limit: Maximum results to return
            offset: Number of results to skip (for pagination)
        """
        ...

    async def search_text(self, query: str, limit: int = 10) -> list[TestCase]:
        """
        Full-text search on nl_query field.

        Args:
            query: Search query string
            limit: Maximum results to return
        """
        ...

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[TestCase, float]]:
        """
        Vector similarity search using embeddings.

        Args:
            embedding: Query embedding vector (1536 dimensions for ada-002)
            limit: Maximum results to return
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of (test_case, similarity_score) tuples, ordered by similarity descending
        """
        ...

    async def count(
        self,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
    ) -> int:
        """Count test cases matching the given filters."""
        ...


@runtime_checkable
class ScorecardRepository(Protocol):
    """
    Repository for scorecard storage (Results Store).

    Provides persistence for evaluation results with batch and test case queries.
    Local implementation uses PostgreSQL.
    Production implementation will use Azure Table Storage.
    """

    async def get(self, scorecard_id: str) -> Scorecard | None:
        """Fetch a single scorecard by ID."""
        ...

    async def get_by_test_case(
        self,
        test_case_id: str,
        batch_id: str | None = None,
    ) -> list[Scorecard]:
        """
        Get all scorecards for a test case.

        Args:
            test_case_id: The test case ID
            batch_id: Optional batch filter

        Returns:
            List of scorecards ordered by creation time descending
        """
        ...

    async def get_by_batch(self, batch_id: str) -> list[Scorecard]:
        """
        Get all scorecards for a batch.

        Args:
            batch_id: The batch ID

        Returns:
            List of scorecards ordered by creation time descending
        """
        ...

    async def save(self, scorecard: Scorecard) -> None:
        """Save a scorecard (insert or update)."""
        ...

    async def get_latest(self, test_case_id: str) -> Scorecard | None:
        """
        Get the most recent scorecard for a test case.

        Args:
            test_case_id: The test case ID

        Returns:
            Most recent scorecard or None if no scorecards exist
        """
        ...

    async def delete(self, scorecard_id: str) -> bool:
        """Delete a scorecard. Returns True if deleted, False if not found."""
        ...

    async def count_by_batch(self, batch_id: str) -> int:
        """Count scorecards in a batch."""
        ...

    async def get_batch_summary(self, batch_id: str) -> dict[str, int | float]:
        """
        Get summary statistics for a batch.

        Returns:
            Dict with keys: total, passed, failed, avg_score
        """
        ...


@runtime_checkable
class BatchJobRepository(Protocol):
    """
    Repository for batch job tracking.

    Tracks batch submissions, progress, and completion status.
    Local implementation uses PostgreSQL.
    Production implementation will use Azure Table Storage.
    """

    async def create(self, batch_job: BatchJob) -> None:
        """Create a new batch job record."""
        ...

    async def get(self, batch_id: str) -> BatchJob | None:
        """Fetch a batch job by ID."""
        ...

    async def update(self, batch_job: BatchJob) -> None:
        """Update an existing batch job."""
        ...

    async def list_recent(self, limit: int = 10) -> list[BatchJob]:
        """
        List recent batch jobs.

        Args:
            limit: Maximum results to return

        Returns:
            List of batch jobs ordered by creation time descending
        """
        ...
