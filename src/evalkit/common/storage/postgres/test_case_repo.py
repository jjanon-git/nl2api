"""
PostgreSQL Test Case Repository

Implements TestCaseRepository protocol for PostgreSQL + pgvector.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg

from CONTRACTS import (
    DataSourceMetadata,
    DataSourceType,
    ReviewStatus,
    TestCase,
    TestCaseMetadata,
    TestCaseStatus,
    ToolCall,
)
from src.evalkit.common.telemetry import get_tracer
from src.evalkit.exceptions import StorageQueryError, StorageWriteError

tracer = get_tracer(__name__)


class PostgresTestCaseRepository:
    """
    PostgreSQL implementation of TestCaseRepository.

    Uses pgvector for similarity search and PostgreSQL full-text search.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize repository with connection pool.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def get(self, test_case_id: str) -> TestCase | None:
        """Fetch a single test case by ID."""
        with tracer.start_as_current_span("db.test_case.get") as span:
            span.set_attribute("db.operation", "get")
            span.set_attribute("db.test_case_id", test_case_id[:36])

            try:
                test_uuid = uuid.UUID(test_case_id)
            except ValueError:
                span.set_attribute("db.result", "invalid_id")
                return None

            try:
                row = await self.pool.fetchrow(
                    "SELECT * FROM test_cases WHERE id = $1",
                    test_uuid,
                )
                span.set_attribute("db.found", row is not None)
                return self._row_to_test_case(row) if row else None
            except asyncpg.PostgresError as e:
                span.set_attribute("db.error", str(e))
                raise StorageQueryError("get_test_case", str(e)) from e

    async def get_many(self, test_case_ids: list[str]) -> list[TestCase]:
        """Fetch multiple test cases by IDs."""
        with tracer.start_as_current_span("db.test_case.get_many") as span:
            span.set_attribute("db.operation", "get_many")
            span.set_attribute("db.requested_count", len(test_case_ids))

            if not test_case_ids:
                span.set_attribute("db.result_count", 0)
                return []

            # Convert to UUIDs, skipping invalid ones
            valid_uuids = []
            for tid in test_case_ids:
                try:
                    valid_uuids.append(uuid.UUID(tid))
                except ValueError:
                    continue

            if not valid_uuids:
                span.set_attribute("db.result_count", 0)
                return []

            try:
                rows = await self.pool.fetch(
                    "SELECT * FROM test_cases WHERE id = ANY($1::uuid[])",
                    valid_uuids,
                )
                span.set_attribute("db.result_count", len(rows))
                return [self._row_to_test_case(row) for row in rows]
            except asyncpg.PostgresError as e:
                span.set_attribute("db.error", str(e))
                raise StorageQueryError("get_many_test_cases", str(e)) from e

    async def save(self, test_case: TestCase) -> None:
        """Save or update a test case (upsert)."""
        with tracer.start_as_current_span("db.test_case.save") as span:
            span.set_attribute("db.operation", "save")
            span.set_attribute("db.test_case_id", test_case.id[:36])

            test_uuid = uuid.UUID(test_case.id)

            # Serialize complex fields with error handling
            try:
                tool_calls_json = json.dumps(
                    [
                        {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                        for tc in test_case.expected_tool_calls
                    ]
                )
                raw_data_json = (
                    json.dumps(test_case.expected_response) if test_case.expected_response else None
                )
            except (TypeError, ValueError) as e:
                raise StorageWriteError("test_case", f"Failed to serialize: {e}") from e

            tags_list = list(test_case.metadata.tags)

            # Convert embedding to list if present
            embedding = list(test_case.embedding) if test_case.embedding else None

            # Extract source metadata fields
            source_type = None
            source_metadata_json = None
            review_status = None
            quality_score = None

            if test_case.metadata.source_metadata:
                sm = test_case.metadata.source_metadata
                source_type = sm.source_type.value
                review_status = sm.review_status.value
                quality_score = sm.quality_score

                # Serialize source_metadata to JSON
                source_metadata_dict = {
                    "source_type": sm.source_type.value,
                    "review_status": sm.review_status.value,
                }
                if sm.origin_system:
                    source_metadata_dict["origin_system"] = sm.origin_system
                if sm.origin_id:
                    source_metadata_dict["origin_id"] = sm.origin_id
                if sm.generator_name:
                    source_metadata_dict["generator_name"] = sm.generator_name
                if sm.generator_version:
                    source_metadata_dict["generator_version"] = sm.generator_version
                if sm.domain_expert:
                    source_metadata_dict["domain_expert"] = sm.domain_expert
                if sm.migrated_from:
                    source_metadata_dict["migrated_from"] = sm.migrated_from
                if sm.anonymized:
                    source_metadata_dict["anonymized"] = sm.anonymized
                if sm.customer_segment:
                    source_metadata_dict["customer_segment"] = sm.customer_segment
                if sm.quality_score is not None:
                    source_metadata_dict["quality_score"] = sm.quality_score

                source_metadata_json = json.dumps(source_metadata_dict)

            try:
                await self.pool.execute(
                    """
                    INSERT INTO test_cases (
                        id, nl_query, expected_tool_calls, expected_response, expected_nl_response,
                        api_version, complexity_level, tags, author, source,
                        status, stale_reason, content_hash, embedding, created_at, updated_at,
                        source_type, source_metadata, review_status, quality_score
                    ) VALUES (
                        $1, $2, $3::jsonb, $4::jsonb, $5,
                        $6, $7, $8, $9, $10,
                        $11, $12, $13, $14::vector, $15, $16,
                        $17::data_source_type, $18::jsonb, $19::review_status_type, $20
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        nl_query = EXCLUDED.nl_query,
                        expected_tool_calls = EXCLUDED.expected_tool_calls,
                        expected_response = EXCLUDED.expected_response,
                        expected_nl_response = EXCLUDED.expected_nl_response,
                        api_version = EXCLUDED.api_version,
                        complexity_level = EXCLUDED.complexity_level,
                        tags = EXCLUDED.tags,
                        author = EXCLUDED.author,
                        source = EXCLUDED.source,
                        status = EXCLUDED.status,
                        stale_reason = EXCLUDED.stale_reason,
                        content_hash = EXCLUDED.content_hash,
                        embedding = EXCLUDED.embedding,
                        source_type = EXCLUDED.source_type,
                        source_metadata = EXCLUDED.source_metadata,
                        review_status = EXCLUDED.review_status,
                        quality_score = EXCLUDED.quality_score,
                        updated_at = NOW()
                    """,
                    test_uuid,
                    test_case.nl_query,
                    tool_calls_json,
                    raw_data_json,
                    test_case.expected_nl_response,
                    test_case.metadata.api_version,
                    test_case.metadata.complexity_level,
                    tags_list,
                    test_case.metadata.author,
                    test_case.metadata.source,
                    test_case.status.value,
                    test_case.stale_reason,
                    test_case.content_hash,
                    embedding,
                    test_case.metadata.created_at,
                    test_case.metadata.updated_at,
                    source_type,
                    source_metadata_json,
                    review_status,
                    quality_score,
                )
            except asyncpg.PostgresError as e:
                span.set_attribute("db.error", str(e))
                raise StorageWriteError("test_case", str(e)) from e

    async def delete(self, test_case_id: str) -> bool:
        """Delete a test case. Returns True if deleted."""
        try:
            test_uuid = uuid.UUID(test_case_id)
        except ValueError:
            return False

        try:
            result = await self.pool.execute(
                "DELETE FROM test_cases WHERE id = $1",
                test_uuid,
            )
            return result == "DELETE 1"
        except asyncpg.PostgresError as e:
            raise StorageWriteError("delete_test_case", str(e)) from e

    async def list(
        self,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
        source_type: DataSourceType | None = None,
        review_status: ReviewStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TestCase]:
        """List test cases with optional filters."""
        query, params = self._build_filter_query(
            base_query="SELECT * FROM test_cases",
            tags=tags,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
            source_type=source_type,
            review_status=review_status,
            order_by="ORDER BY created_at DESC",
            limit=limit,
            offset=offset,
        )
        try:
            rows = await self.pool.fetch(query, *params)
            return [self._row_to_test_case(row) for row in rows]
        except asyncpg.PostgresError as e:
            raise StorageQueryError("list_test_cases", str(e)) from e

    async def get_by_source_type(
        self,
        source_type: DataSourceType,
        review_status: ReviewStatus | None = None,
        tags: list[str] | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[TestCase]:
        """
        Get test cases filtered by source type.

        Args:
            source_type: Filter by data source type (customer, sme, synthetic, hybrid)
            review_status: Optional filter by review status
            tags: Optional filter by tags
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of matching test cases
        """
        with tracer.start_as_current_span("db.test_case.get_by_source_type") as span:
            span.set_attribute("db.operation", "get_by_source_type")
            span.set_attribute("db.source_type", source_type.value)

            conditions = ["status = 'active'", "source_type = $1"]
            params: list[Any] = [source_type.value]
            param_idx = 2

            if review_status is not None:
                conditions.append(f"review_status = ${param_idx}")
                params.append(review_status.value)
                param_idx += 1

            if tags is not None:
                conditions.append(f"tags && ${param_idx}::text[]")
                params.append(tags)
                param_idx += 1

            query = f"""
                SELECT * FROM test_cases
                WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([limit, offset])

            try:
                rows = await self.pool.fetch(query, *params)
                span.set_attribute("db.result_count", len(rows))
                return [self._row_to_test_case(row) for row in rows]
            except asyncpg.PostgresError as e:
                span.set_attribute("db.error", str(e))
                raise StorageQueryError("get_by_source_type", str(e)) from e

    async def get_source_statistics(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics grouped by source type.

        Returns:
            Dict mapping source_type to stats (count, avg_quality, review_status breakdown)
        """
        with tracer.start_as_current_span("db.test_case.get_source_statistics") as span:
            span.set_attribute("db.operation", "get_source_statistics")

            try:
                rows = await self.pool.fetch("""
                    SELECT
                        source_type,
                        review_status,
                        COUNT(*) as count,
                        AVG(quality_score) as avg_quality
                    FROM test_cases
                    WHERE status = 'active'
                    GROUP BY source_type, review_status
                    ORDER BY source_type, review_status
                """)

                stats: dict[str, dict[str, Any]] = {}
                for row in rows:
                    source = row["source_type"] or "unknown"
                    if source not in stats:
                        stats[source] = {
                            "total": 0,
                            "avg_quality": None,
                            "by_review_status": {},
                        }

                    review = row["review_status"] or "pending"
                    count = row["count"]
                    stats[source]["total"] += count
                    stats[source]["by_review_status"][review] = count

                    if row["avg_quality"] is not None:
                        # Update average (weighted)
                        current_avg = stats[source]["avg_quality"]
                        if current_avg is None:
                            stats[source]["avg_quality"] = float(row["avg_quality"])
                        else:
                            # Simple average for now
                            stats[source]["avg_quality"] = (
                                current_avg + float(row["avg_quality"])
                            ) / 2

                span.set_attribute("db.source_types", len(stats))
                return stats
            except asyncpg.PostgresError as e:
                span.set_attribute("db.error", str(e))
                raise StorageQueryError("get_source_statistics", str(e)) from e

    async def search_text(self, query: str, limit: int = 10) -> list[TestCase]:
        """Full-text search on nl_query field."""
        try:
            rows = await self.pool.fetch(
                """
                SELECT *, ts_rank(to_tsvector('english', nl_query), plainto_tsquery('english', $1)) as rank
                FROM test_cases
                WHERE to_tsvector('english', nl_query) @@ plainto_tsquery('english', $1)
                  AND status = 'active'
                ORDER BY rank DESC
                LIMIT $2
                """,
                query,
                limit,
            )
            return [self._row_to_test_case(row) for row in rows]
        except asyncpg.PostgresError as e:
            raise StorageQueryError("search_text", str(e)) from e

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[TestCase, float]]:
        """Vector similarity search using embeddings."""
        try:
            rows = await self.pool.fetch(
                """
                SELECT *, 1 - (embedding <=> $1::vector) as similarity
                FROM test_cases
                WHERE embedding IS NOT NULL
                  AND status = 'active'
                  AND 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding,
                threshold,
                limit,
            )
            return [(self._row_to_test_case(row), row["similarity"]) for row in rows]
        except asyncpg.PostgresError as e:
            raise StorageQueryError("search_similar", str(e)) from e

    async def count(
        self,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
        source_type: DataSourceType | None = None,
        review_status: ReviewStatus | None = None,
    ) -> int:
        """Count test cases matching the given filters."""
        query, params = self._build_filter_query(
            base_query="SELECT COUNT(*) FROM test_cases",
            tags=tags,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
            source_type=source_type,
            review_status=review_status,
        )
        try:
            result = await self.pool.fetchval(query, *params)
            return result or 0
        except asyncpg.PostgresError as e:
            raise StorageQueryError("count_test_cases", str(e)) from e

    def _build_filter_query(
        self,
        base_query: str,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
        source_type: DataSourceType | None = None,
        review_status: ReviewStatus | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> tuple[str, list[Any]]:
        """
        Build a parameterized query with filters.

        Uses a whitelist approach for conditions to prevent SQL injection.
        All dynamic values are passed as parameters, never interpolated.

        Returns:
            Tuple of (query_string, params_list)
        """
        # Whitelist of allowed condition templates
        # Each template uses a placeholder that will be replaced with the actual param index
        CONDITION_TEMPLATES = {
            "tags": "tags && ${idx}::text[]",
            "complexity_min": "complexity_level >= ${idx}",
            "complexity_max": "complexity_level <= ${idx}",
            "source_type": "source_type = ${idx}",
            "review_status": "review_status = ${idx}",
        }

        conditions = ["status = 'active'"]
        params: list[Any] = []
        param_idx = 1

        # Build conditions using whitelist
        if tags is not None:
            template = CONDITION_TEMPLATES["tags"]
            conditions.append(template.replace("${idx}", f"${param_idx}"))
            params.append(tags)
            param_idx += 1

        if complexity_min is not None:
            template = CONDITION_TEMPLATES["complexity_min"]
            conditions.append(template.replace("${idx}", f"${param_idx}"))
            params.append(complexity_min)
            param_idx += 1

        if complexity_max is not None:
            template = CONDITION_TEMPLATES["complexity_max"]
            conditions.append(template.replace("${idx}", f"${param_idx}"))
            params.append(complexity_max)
            param_idx += 1

        if source_type is not None:
            template = CONDITION_TEMPLATES["source_type"]
            conditions.append(template.replace("${idx}", f"${param_idx}"))
            params.append(source_type.value)
            param_idx += 1

        if review_status is not None:
            template = CONDITION_TEMPLATES["review_status"]
            conditions.append(template.replace("${idx}", f"${param_idx}"))
            params.append(review_status.value)
            param_idx += 1

        # Build query with static structure
        query_parts = [base_query, "WHERE", " AND ".join(conditions)]

        if order_by:
            query_parts.append(order_by)

        if limit is not None:
            query_parts.append(f"LIMIT ${param_idx}")
            params.append(limit)
            param_idx += 1

        if offset is not None:
            query_parts.append(f"OFFSET ${param_idx}")
            params.append(offset)
            param_idx += 1

        return " ".join(query_parts), params

    def _row_to_test_case(self, row: asyncpg.Record) -> TestCase:
        """Convert database row to TestCase model."""
        # Parse tool calls from JSONB
        tool_calls_data = row["expected_tool_calls"]
        if isinstance(tool_calls_data, str):
            tool_calls_data = json.loads(tool_calls_data)

        tool_calls = tuple(
            ToolCall(tool_name=tc["tool_name"], arguments=tc.get("arguments", {}))
            for tc in tool_calls_data
        )

        # Parse raw data
        raw_data = row["expected_response"]
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        # Parse generic input/expected fields (for RAG and other packs)
        # Note: asyncpg.Record doesn't support .get(), so we check keys directly
        input_data = {}
        if "input_json" in row.keys() and row["input_json"]:
            input_data = row["input_json"]
            if isinstance(input_data, str):
                input_data = json.loads(input_data)

        expected_data = {}
        if "expected_json" in row.keys() and row["expected_json"]:
            expected_data = row["expected_json"]
            if isinstance(expected_data, str):
                expected_data = json.loads(expected_data)

        # Parse embedding
        embedding = None
        if row["embedding"] is not None:
            embedding = tuple(row["embedding"])

        # Parse source_metadata if present
        source_metadata = None
        if "source_metadata" in row.keys() and row["source_metadata"]:
            source_meta_data = row["source_metadata"]
            if isinstance(source_meta_data, str):
                source_meta_data = json.loads(source_meta_data)

            # Only create DataSourceMetadata if we have meaningful data
            if source_meta_data and source_meta_data != {}:
                # Get source_type from column or metadata
                source_type_str = None
                if "source_type" in row.keys() and row["source_type"]:
                    source_type_str = row["source_type"]
                elif "source_type" in source_meta_data:
                    source_type_str = source_meta_data["source_type"]

                if source_type_str:
                    try:
                        source_metadata = DataSourceMetadata(
                            source_type=DataSourceType(source_type_str),
                            origin_system=source_meta_data.get("origin_system"),
                            origin_id=source_meta_data.get("origin_id"),
                            review_status=ReviewStatus(
                                source_meta_data.get("review_status", "pending")
                            ),
                            quality_score=source_meta_data.get("quality_score"),
                            generator_name=source_meta_data.get("generator_name"),
                            generator_version=source_meta_data.get("generator_version"),
                            migrated_from=source_meta_data.get("migrated_from"),
                            domain_expert=source_meta_data.get("domain_expert"),
                            anonymized=source_meta_data.get("anonymized", False),
                            customer_segment=source_meta_data.get("customer_segment"),
                        )
                    except (ValueError, KeyError):
                        # If parsing fails, leave source_metadata as None
                        pass

        # Build metadata
        metadata = TestCaseMetadata(
            api_version=row["api_version"],
            complexity_level=row["complexity_level"],
            tags=tuple(row["tags"] or []),
            created_at=row["created_at"].replace(tzinfo=UTC)
            if row["created_at"]
            else datetime.now(UTC),
            updated_at=row["updated_at"].replace(tzinfo=UTC)
            if row["updated_at"]
            else datetime.now(UTC),
            author=row["author"],
            source=row["source"],
            source_metadata=source_metadata,
        )

        return TestCase(
            id=str(row["id"]),
            # Generic fields (for RAG and other packs)
            input=input_data,
            expected=expected_data,
            # NL2API-specific fields (backwards compatible)
            nl_query=row["nl_query"],
            expected_tool_calls=tool_calls,
            expected_response=raw_data,
            expected_nl_response=row["expected_nl_response"],
            metadata=metadata,
            status=TestCaseStatus(row["status"]),
            stale_reason=row["stale_reason"],
            embedding=embedding,
        )
