"""
In-Memory Repository Implementations

Simple dict-based storage for unit testing.
Implements the same protocols as production backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CONTRACTS import BatchJob, DataSourceType, ReviewStatus, Scorecard, TestCase


class InMemoryTestCaseRepository:
    """
    In-memory implementation of TestCaseRepository.

    Perfect for unit tests - no database required.
    """

    def __init__(self):
        self._store: dict[str, TestCase] = {}

    async def get(self, test_case_id: str) -> TestCase | None:
        """Fetch a single test case by ID."""
        return self._store.get(test_case_id)

    async def get_many(self, test_case_ids: list[str]) -> list[TestCase]:
        """Fetch multiple test cases by IDs."""
        return [self._store[tid] for tid in test_case_ids if tid in self._store]

    async def save(self, test_case: TestCase) -> None:
        """Save or update a test case."""
        self._store[test_case.id] = test_case

    async def delete(self, test_case_id: str) -> bool:
        """Delete a test case. Returns True if deleted."""
        if test_case_id in self._store:
            del self._store[test_case_id]
            return True
        return False

    async def list(
        self,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
        source_type: DataSourceType | None = None,
        review_status: ReviewStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        exclude_ids: set[str] | None = None,
    ) -> list[TestCase]:
        """List test cases with optional filters."""
        results = list(self._store.values())

        # Apply exclusion filter first (most efficient)
        if exclude_ids:
            results = [tc for tc in results if tc.id not in exclude_ids]

        # Apply filters
        if tags:
            tag_set = set(tags)
            results = [tc for tc in results if tag_set & set(tc.metadata.tags)]

        if complexity_min is not None:
            results = [tc for tc in results if tc.metadata.complexity_level >= complexity_min]

        if complexity_max is not None:
            results = [tc for tc in results if tc.metadata.complexity_level <= complexity_max]

        if source_type is not None:
            results = [
                tc
                for tc in results
                if tc.metadata.source_metadata
                and tc.metadata.source_metadata.source_type == source_type
            ]

        if review_status is not None:
            results = [
                tc
                for tc in results
                if tc.metadata.source_metadata
                and tc.metadata.source_metadata.review_status == review_status
            ]

        # Sort by created_at descending (newest first)
        results.sort(key=lambda tc: tc.metadata.created_at, reverse=True)

        # Apply pagination
        return results[offset : offset + limit]

    async def search_text(self, query: str, limit: int = 10) -> list[TestCase]:
        """Simple text search on nl_query field."""
        query_lower = query.lower()
        results = [tc for tc in self._store.values() if query_lower in tc.nl_query.lower()]
        return results[:limit]

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[TestCase, float]]:
        """
        Vector similarity search.

        Note: This is a naive implementation using cosine similarity.
        For unit tests only - not optimized for large datasets.
        """
        import math

        def cosine_similarity(a: tuple[float, ...], b: list[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            if len(a) != len(b):
                return 0.0
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)

        results: list[tuple[TestCase, float]] = []
        for tc in self._store.values():
            if tc.embedding is not None:
                similarity = cosine_similarity(tc.embedding, embedding)
                if similarity >= threshold:
                    results.append((tc, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def count(
        self,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
        source_type: DataSourceType | None = None,
        review_status: ReviewStatus | None = None,
    ) -> int:
        """Count test cases matching the given filters."""
        results = await self.list(
            tags=tags,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
            source_type=source_type,
            review_status=review_status,
            limit=len(self._store),
        )
        return len(results)

    def clear(self) -> None:
        """Clear all test cases (useful for test setup/teardown)."""
        self._store.clear()


class InMemoryScorecardRepository:
    """
    In-memory implementation of ScorecardRepository.

    Perfect for unit tests - no database required.
    """

    def __init__(self):
        self._store: dict[str, Scorecard] = {}

    async def get(self, scorecard_id: str) -> Scorecard | None:
        """Fetch a single scorecard by ID."""
        return self._store.get(scorecard_id)

    async def get_by_test_case(
        self,
        test_case_id: str,
        batch_id: str | None = None,
    ) -> list[Scorecard]:
        """Get all scorecards for a test case."""
        results = [sc for sc in self._store.values() if sc.test_case_id == test_case_id]
        if batch_id:
            results = [sc for sc in results if sc.batch_id == batch_id]

        # Sort by timestamp descending
        results.sort(key=lambda sc: sc.timestamp, reverse=True)
        return results

    async def get_by_batch(self, batch_id: str) -> list[Scorecard]:
        """Get all scorecards for a batch."""
        results = [sc for sc in self._store.values() if sc.batch_id == batch_id]
        # Sort by timestamp descending
        results.sort(key=lambda sc: sc.timestamp, reverse=True)
        return results

    async def save(self, scorecard: Scorecard) -> None:
        """Save a scorecard (insert or update)."""
        self._store[scorecard.scorecard_id] = scorecard

    async def get_latest(self, test_case_id: str) -> Scorecard | None:
        """Get the most recent scorecard for a test case."""
        scorecards = await self.get_by_test_case(test_case_id)
        return scorecards[0] if scorecards else None

    async def delete(self, scorecard_id: str) -> bool:
        """Delete a scorecard. Returns True if deleted."""
        if scorecard_id in self._store:
            del self._store[scorecard_id]
            return True
        return False

    async def count_by_batch(self, batch_id: str) -> int:
        """Count scorecards in a batch."""
        return len([sc for sc in self._store.values() if sc.batch_id == batch_id])

    async def get_batch_summary(self, batch_id: str) -> dict[str, int | float]:
        """Get summary statistics for a batch."""
        scorecards = await self.get_by_batch(batch_id)
        if not scorecards:
            return {"total": 0, "passed": 0, "failed": 0, "avg_score": 0.0}

        passed = sum(1 for sc in scorecards if sc.overall_passed)
        failed = len(scorecards) - passed
        avg_score = sum(sc.overall_score for sc in scorecards) / len(scorecards)

        return {
            "total": len(scorecards),
            "passed": passed,
            "failed": failed,
            "avg_score": avg_score,
        }

    async def get_evaluated_test_case_ids(self, batch_id: str) -> set[str]:
        """Get test_case_ids already evaluated in this batch."""
        return {sc.test_case_id for sc in self._store.values() if sc.batch_id == batch_id}

    def clear(self) -> None:
        """Clear all scorecards (useful for test setup/teardown)."""
        self._store.clear()


class InMemoryBatchJobRepository:
    """
    In-memory implementation of BatchJobRepository.

    Perfect for unit tests - no database required.
    """

    def __init__(self) -> None:
        self._store: dict[str, BatchJob] = {}

    async def create(self, batch_job: BatchJob) -> None:
        """Create a new batch job record."""
        self._store[batch_job.batch_id] = batch_job

    async def get(self, batch_id: str) -> BatchJob | None:
        """Fetch a batch job by ID."""
        return self._store.get(batch_id)

    async def update(self, batch_job: BatchJob) -> None:
        """Update an existing batch job."""
        self._store[batch_job.batch_id] = batch_job

    async def list_recent(self, limit: int = 10) -> list[BatchJob]:
        """List recent batch jobs."""
        results = list(self._store.values())
        # Sort by created_at descending
        results.sort(key=lambda bj: bj.created_at, reverse=True)
        return results[:limit]

    async def update_progress(
        self,
        batch_id: str,
        completed: int,
        failed: int,
    ) -> None:
        """Update checkpoint progress for a batch job."""
        if batch_id in self._store:
            batch = self._store[batch_id]
            # Update the batch with new counts
            self._store[batch_id] = batch.model_copy(
                update={
                    "completed_count": completed,
                    "failed_count": failed,
                }
            )

    def clear(self) -> None:
        """Clear all batch jobs (useful for test setup/teardown)."""
        self._store.clear()
