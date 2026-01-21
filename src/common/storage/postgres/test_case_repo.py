"""
PostgreSQL Test Case Repository

Implements TestCaseRepository protocol for PostgreSQL + pgvector.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from CONTRACTS import TestCase, TestCaseMetadata, TestCaseStatus, ToolCall
from src.common.telemetry import get_tracer

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

            row = await self.pool.fetchrow(
                "SELECT * FROM test_cases WHERE id = $1",
                test_uuid,
            )
            span.set_attribute("db.found", row is not None)
            return self._row_to_test_case(row) if row else None

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

            rows = await self.pool.fetch(
                "SELECT * FROM test_cases WHERE id = ANY($1::uuid[])",
                valid_uuids,
            )
            span.set_attribute("db.result_count", len(rows))
            return [self._row_to_test_case(row) for row in rows]

    async def save(self, test_case: TestCase) -> None:
        """Save or update a test case (upsert)."""
        with tracer.start_as_current_span("db.test_case.save") as span:
            span.set_attribute("db.operation", "save")
            span.set_attribute("db.test_case_id", test_case.id[:36])

            test_uuid = uuid.UUID(test_case.id)

            # Serialize complex fields
            tool_calls_json = json.dumps([
                {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                for tc in test_case.expected_tool_calls
            ])
            raw_data_json = json.dumps(test_case.expected_response) if test_case.expected_response else None
            tags_list = list(test_case.metadata.tags)

            # Convert embedding to list if present
            embedding = list(test_case.embedding) if test_case.embedding else None

            await self.pool.execute(
                """
                INSERT INTO test_cases (
                    id, nl_query, expected_tool_calls, expected_response, expected_nl_response,
                    api_version, complexity_level, tags, author, source,
                    status, stale_reason, content_hash, embedding, created_at, updated_at
                ) VALUES (
                    $1, $2, $3::jsonb, $4::jsonb, $5,
                    $6, $7, $8, $9, $10,
                    $11, $12, $13, $14::vector, $15, $16
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
            )

    async def delete(self, test_case_id: str) -> bool:
        """Delete a test case. Returns True if deleted."""
        try:
            test_uuid = uuid.UUID(test_case_id)
        except ValueError:
            return False

        result = await self.pool.execute(
            "DELETE FROM test_cases WHERE id = $1",
            test_uuid,
        )
        return result == "DELETE 1"

    async def list(
        self,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TestCase]:
        """List test cases with optional filters."""
        query, params = self._build_filter_query(
            base_query="SELECT * FROM test_cases",
            tags=tags,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
            order_by="ORDER BY created_at DESC",
            limit=limit,
            offset=offset,
        )
        rows = await self.pool.fetch(query, *params)
        return [self._row_to_test_case(row) for row in rows]

    async def search_text(self, query: str, limit: int = 10) -> list[TestCase]:
        """Full-text search on nl_query field."""
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

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[TestCase, float]]:
        """Vector similarity search using embeddings."""
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

    async def count(
        self,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
    ) -> int:
        """Count test cases matching the given filters."""
        query, params = self._build_filter_query(
            base_query="SELECT COUNT(*) FROM test_cases",
            tags=tags,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
        )
        result = await self.pool.fetchval(query, *params)
        return result or 0

    def _build_filter_query(
        self,
        base_query: str,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
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

        # Parse embedding
        embedding = None
        if row["embedding"] is not None:
            embedding = tuple(row["embedding"])

        # Build metadata
        metadata = TestCaseMetadata(
            api_version=row["api_version"],
            complexity_level=row["complexity_level"],
            tags=tuple(row["tags"] or []),
            created_at=row["created_at"].replace(tzinfo=timezone.utc) if row["created_at"] else datetime.now(timezone.utc),
            updated_at=row["updated_at"].replace(tzinfo=timezone.utc) if row["updated_at"] else datetime.now(timezone.utc),
            author=row["author"],
            source=row["source"],
        )

        return TestCase(
            id=str(row["id"]),
            nl_query=row["nl_query"],
            expected_tool_calls=tool_calls,
            expected_response=raw_data,
            expected_nl_response=row["expected_nl_response"],
            metadata=metadata,
            status=TestCaseStatus(row["status"]),
            stale_reason=row["stale_reason"],
            embedding=embedding,
        )
