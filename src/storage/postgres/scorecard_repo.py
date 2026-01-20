"""
PostgreSQL Scorecard Repository

Implements ScorecardRepository protocol for PostgreSQL.
"""

from __future__ import annotations

import json
import uuid
from datetime import timezone
from typing import Any

import asyncpg

from CONTRACTS import (
    ErrorCode,
    EvaluationStage,
    Scorecard,
    StageResult,
    ToolCall,
)


class PostgresScorecardRepository:
    """
    PostgreSQL implementation of ScorecardRepository.

    Stores evaluation results with batch/test case indexing.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize repository with connection pool.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def get(self, scorecard_id: str) -> Scorecard | None:
        """Fetch a single scorecard by ID."""
        try:
            sc_uuid = uuid.UUID(scorecard_id)
        except ValueError:
            return None

        row = await self.pool.fetchrow(
            "SELECT * FROM scorecards WHERE id = $1",
            sc_uuid,
        )
        return self._row_to_scorecard(row) if row else None

    async def get_by_test_case(
        self,
        test_case_id: str,
        batch_id: str | None = None,
    ) -> list[Scorecard]:
        """Get all scorecards for a test case."""
        try:
            tc_uuid = uuid.UUID(test_case_id)
        except ValueError:
            return []

        if batch_id:
            rows = await self.pool.fetch(
                """
                SELECT * FROM scorecards
                WHERE test_case_id = $1 AND batch_id = $2
                ORDER BY created_at DESC
                """,
                tc_uuid,
                batch_id,
            )
        else:
            rows = await self.pool.fetch(
                """
                SELECT * FROM scorecards
                WHERE test_case_id = $1
                ORDER BY created_at DESC
                """,
                tc_uuid,
            )
        return [self._row_to_scorecard(row) for row in rows]

    async def get_by_batch(self, batch_id: str) -> list[Scorecard]:
        """Get all scorecards for a batch."""
        rows = await self.pool.fetch(
            """
            SELECT * FROM scorecards
            WHERE batch_id = $1
            ORDER BY created_at DESC
            """,
            batch_id,
        )
        return [self._row_to_scorecard(row) for row in rows]

    async def save(self, scorecard: Scorecard) -> None:
        """Save a scorecard (insert or update)."""
        sc_uuid = uuid.UUID(scorecard.scorecard_id)
        tc_uuid = uuid.UUID(scorecard.test_case_id)

        # Serialize stage results
        syntax_json = self._stage_result_to_json(scorecard.syntax_result)
        logic_json = self._stage_result_to_json(scorecard.logic_result) if scorecard.logic_result else None
        execution_json = self._stage_result_to_json(scorecard.execution_result) if scorecard.execution_result else None
        semantics_json = self._stage_result_to_json(scorecard.semantics_result) if scorecard.semantics_result else None

        # Serialize tool calls
        tool_calls_json = None
        if scorecard.generated_tool_calls:
            tool_calls_json = json.dumps([
                {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                for tc in scorecard.generated_tool_calls
            ])

        await self.pool.execute(
            """
            INSERT INTO scorecards (
                id, test_case_id, batch_id, run_id,
                syntax_result, logic_result, execution_result, semantics_result,
                generated_tool_calls, generated_nl_response,
                overall_passed, overall_score,
                worker_id, attempt_number, message_id, total_latency_ms,
                created_at, completed_at
            ) VALUES (
                $1, $2, $3, $4,
                $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb,
                $9::jsonb, $10,
                $11, $12,
                $13, $14, $15, $16,
                $17, $18
            )
            ON CONFLICT (id) DO UPDATE SET
                syntax_result = EXCLUDED.syntax_result,
                logic_result = EXCLUDED.logic_result,
                execution_result = EXCLUDED.execution_result,
                semantics_result = EXCLUDED.semantics_result,
                generated_tool_calls = EXCLUDED.generated_tool_calls,
                generated_nl_response = EXCLUDED.generated_nl_response,
                overall_passed = EXCLUDED.overall_passed,
                overall_score = EXCLUDED.overall_score,
                attempt_number = EXCLUDED.attempt_number,
                total_latency_ms = EXCLUDED.total_latency_ms,
                completed_at = EXCLUDED.completed_at
            """,
            sc_uuid,
            tc_uuid,
            scorecard.batch_id,
            None,  # run_id placeholder
            syntax_json,
            logic_json,
            execution_json,
            semantics_json,
            tool_calls_json,
            scorecard.generated_nl_response,
            scorecard.overall_passed,
            scorecard.overall_score,
            scorecard.worker_id,
            scorecard.attempt_number,
            scorecard.message_id,
            scorecard.total_latency_ms,
            scorecard.timestamp,
            scorecard.completed_at,
        )

    async def get_latest(self, test_case_id: str) -> Scorecard | None:
        """Get the most recent scorecard for a test case."""
        try:
            tc_uuid = uuid.UUID(test_case_id)
        except ValueError:
            return None

        row = await self.pool.fetchrow(
            """
            SELECT * FROM scorecards
            WHERE test_case_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            tc_uuid,
        )
        return self._row_to_scorecard(row) if row else None

    async def delete(self, scorecard_id: str) -> bool:
        """Delete a scorecard. Returns True if deleted."""
        try:
            sc_uuid = uuid.UUID(scorecard_id)
        except ValueError:
            return False

        result = await self.pool.execute(
            "DELETE FROM scorecards WHERE id = $1",
            sc_uuid,
        )
        return result == "DELETE 1"

    async def count_by_batch(self, batch_id: str) -> int:
        """Count scorecards in a batch."""
        result = await self.pool.fetchval(
            "SELECT COUNT(*) FROM scorecards WHERE batch_id = $1",
            batch_id,
        )
        return result or 0

    async def get_batch_summary(self, batch_id: str) -> dict[str, int | float]:
        """Get summary statistics for a batch."""
        row = await self.pool.fetchrow(
            """
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE overall_passed = true) as passed,
                COUNT(*) FILTER (WHERE overall_passed = false) as failed,
                COALESCE(AVG(overall_score), 0.0) as avg_score
            FROM scorecards
            WHERE batch_id = $1
            """,
            batch_id,
        )
        return {
            "total": row["total"],
            "passed": row["passed"],
            "failed": row["failed"],
            "avg_score": float(row["avg_score"]),
        }

    def _stage_result_to_json(self, result: StageResult) -> str:
        """Convert StageResult to JSON string."""
        data = {
            "stage": result.stage.value,
            "passed": result.passed,
            "score": result.score,
            "error_code": result.error_code.value if result.error_code else None,
            "reason": result.reason,
            "artifacts": result.artifacts,
            "duration_ms": result.duration_ms,
        }
        return json.dumps(data)

    def _json_to_stage_result(self, data: dict[str, Any] | str) -> StageResult:
        """Convert JSON dict to StageResult."""
        if isinstance(data, str):
            data = json.loads(data)

        return StageResult(
            stage=EvaluationStage(data["stage"]),
            passed=data["passed"],
            score=data["score"],
            error_code=ErrorCode(data["error_code"]) if data.get("error_code") else None,
            reason=data.get("reason"),
            artifacts=data.get("artifacts", {}),
            duration_ms=data.get("duration_ms", 0),
        )

    def _row_to_scorecard(self, row: asyncpg.Record) -> Scorecard:
        """Convert database row to Scorecard model."""
        # Parse stage results
        syntax_result = self._json_to_stage_result(row["syntax_result"])
        logic_result = self._json_to_stage_result(row["logic_result"]) if row["logic_result"] else None
        execution_result = self._json_to_stage_result(row["execution_result"]) if row["execution_result"] else None
        semantics_result = self._json_to_stage_result(row["semantics_result"]) if row["semantics_result"] else None

        # Parse tool calls
        generated_tool_calls = None
        if row["generated_tool_calls"]:
            tool_calls_data = row["generated_tool_calls"]
            if isinstance(tool_calls_data, str):
                tool_calls_data = json.loads(tool_calls_data)
            generated_tool_calls = tuple(
                ToolCall(tool_name=tc["tool_name"], arguments=tc.get("arguments", {}))
                for tc in tool_calls_data
            )

        # Handle timestamps
        timestamp = row["created_at"]
        if timestamp and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        completed_at = row["completed_at"]
        if completed_at and completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)

        return Scorecard(
            test_case_id=str(row["test_case_id"]),
            batch_id=row["batch_id"],
            scorecard_id=str(row["id"]),
            timestamp=timestamp,
            completed_at=completed_at,
            syntax_result=syntax_result,
            logic_result=logic_result,
            execution_result=execution_result,
            semantics_result=semantics_result,
            generated_tool_calls=generated_tool_calls,
            generated_nl_response=row["generated_nl_response"],
            worker_id=row["worker_id"],
            attempt_number=row["attempt_number"],
            message_id=row["message_id"],
            total_latency_ms=row["total_latency_ms"],
        )
