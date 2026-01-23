"""
PostgreSQL Scorecard Repository

Implements ScorecardRepository protocol for PostgreSQL.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import asyncpg

from CONTRACTS import (
    ErrorCode,
    EvaluationStage,
    Scorecard,
    StageResult,
    ToolCall,
)
from src.common.telemetry import get_tracer

tracer = get_tracer(__name__)


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
        with tracer.start_as_current_span("db.scorecard.get") as span:
            span.set_attribute("db.operation", "get")
            span.set_attribute("db.scorecard_id", scorecard_id[:36])

            try:
                sc_uuid = uuid.UUID(scorecard_id)
            except ValueError:
                span.set_attribute("db.result", "invalid_id")
                return None

            row = await self.pool.fetchrow(
                "SELECT * FROM scorecards WHERE id = $1",
                sc_uuid,
            )
            span.set_attribute("db.found", row is not None)
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
        with tracer.start_as_current_span("db.scorecard.save") as span:
            span.set_attribute("db.operation", "save")
            span.set_attribute("db.scorecard_id", scorecard.scorecard_id[:36])
            span.set_attribute("db.test_case_id", scorecard.test_case_id[:36])
            span.set_attribute("db.overall_passed", scorecard.overall_passed)
            span.set_attribute("db.pack_name", scorecard.pack_name)

            sc_uuid = uuid.UUID(scorecard.scorecard_id)
            tc_uuid = uuid.UUID(scorecard.test_case_id)

            # Serialize NL2API-specific stage results (backwards compatibility)
            syntax_json = self._stage_result_to_json(scorecard.syntax_result)
            logic_json = self._stage_result_to_json(scorecard.logic_result)
            execution_json = self._stage_result_to_json(scorecard.execution_result)
            semantics_json = self._stage_result_to_json(scorecard.semantics_result)

            # Serialize generic stage_results
            stage_results_json = (
                self._stage_results_to_json(scorecard.stage_results)
                if scorecard.stage_results
                else "{}"
            )
            stage_weights_json = json.dumps(scorecard.stage_weights or {})

            # Serialize tool calls
            tool_calls_json = None
            if scorecard.generated_tool_calls:
                tool_calls_json = json.dumps(
                    [
                        {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                        for tc in scorecard.generated_tool_calls
                    ]
                )

            # Serialize generated_output
            generated_output_json = json.dumps(scorecard.generated_output or {})

            await self.pool.execute(
                """
                INSERT INTO scorecards (
                    id, test_case_id, batch_id, run_id,
                    pack_name, stage_results, stage_weights,
                    syntax_result, logic_result, execution_result, semantics_result,
                    generated_tool_calls, generated_nl_response, generated_output,
                    overall_passed, overall_score,
                    worker_id, attempt_number, message_id, total_latency_ms,
                    created_at, completed_at,
                    client_type, client_version, eval_mode,
                    input_tokens, output_tokens, estimated_cost_usd
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6::jsonb, $7::jsonb,
                    $8::jsonb, $9::jsonb, $10::jsonb, $11::jsonb,
                    $12::jsonb, $13, $14::jsonb,
                    $15, $16,
                    $17, $18, $19, $20,
                    $21, $22,
                    $23, $24, $25,
                    $26, $27, $28
                )
                ON CONFLICT (id) DO UPDATE SET
                    pack_name = EXCLUDED.pack_name,
                    stage_results = EXCLUDED.stage_results,
                    stage_weights = EXCLUDED.stage_weights,
                    syntax_result = EXCLUDED.syntax_result,
                    logic_result = EXCLUDED.logic_result,
                    execution_result = EXCLUDED.execution_result,
                    semantics_result = EXCLUDED.semantics_result,
                    generated_tool_calls = EXCLUDED.generated_tool_calls,
                    generated_nl_response = EXCLUDED.generated_nl_response,
                    generated_output = EXCLUDED.generated_output,
                    overall_passed = EXCLUDED.overall_passed,
                    overall_score = EXCLUDED.overall_score,
                    attempt_number = EXCLUDED.attempt_number,
                    total_latency_ms = EXCLUDED.total_latency_ms,
                    completed_at = EXCLUDED.completed_at,
                    client_type = EXCLUDED.client_type,
                    client_version = EXCLUDED.client_version,
                    eval_mode = EXCLUDED.eval_mode,
                    input_tokens = EXCLUDED.input_tokens,
                    output_tokens = EXCLUDED.output_tokens,
                    estimated_cost_usd = EXCLUDED.estimated_cost_usd
                """,
                sc_uuid,
                tc_uuid,
                scorecard.batch_id,
                None,  # run_id placeholder
                scorecard.pack_name,
                stage_results_json,
                stage_weights_json,
                syntax_json,
                logic_json,
                execution_json,
                semantics_json,
                tool_calls_json,
                scorecard.generated_nl_response,
                generated_output_json,
                scorecard.overall_passed,
                scorecard.overall_score,
                scorecard.worker_id,
                scorecard.attempt_number,
                scorecard.message_id,
                scorecard.total_latency_ms,
                scorecard.timestamp,
                scorecard.completed_at,
                scorecard.client_type,
                scorecard.client_version,
                scorecard.eval_mode,
                scorecard.input_tokens,
                scorecard.output_tokens,
                scorecard.estimated_cost_usd,
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

    async def save_batch(self, scorecards: list[Scorecard]) -> int:
        """
        Save multiple scorecards in a single transaction.

        More efficient than calling save() multiple times for batch operations.

        Args:
            scorecards: List of scorecards to save

        Returns:
            Number of scorecards saved
        """
        if not scorecards:
            return 0

        # Prepare all records
        records = []
        for scorecard in scorecards:
            sc_uuid = uuid.UUID(scorecard.scorecard_id)
            tc_uuid = uuid.UUID(scorecard.test_case_id)

            # Serialize NL2API-specific stage results
            syntax_json = self._stage_result_to_json(scorecard.syntax_result)
            logic_json = self._stage_result_to_json(scorecard.logic_result)
            execution_json = self._stage_result_to_json(scorecard.execution_result)
            semantics_json = self._stage_result_to_json(scorecard.semantics_result)

            # Serialize generic stage_results
            stage_results_json = (
                self._stage_results_to_json(scorecard.stage_results)
                if scorecard.stage_results
                else "{}"
            )
            stage_weights_json = json.dumps(scorecard.stage_weights or {})

            # Serialize tool calls
            tool_calls_json = None
            if scorecard.generated_tool_calls:
                tool_calls_json = json.dumps(
                    [
                        {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                        for tc in scorecard.generated_tool_calls
                    ]
                )

            # Serialize generated_output
            generated_output_json = json.dumps(scorecard.generated_output or {})

            records.append(
                (
                    sc_uuid,
                    tc_uuid,
                    scorecard.batch_id,
                    None,  # run_id placeholder
                    scorecard.pack_name,
                    stage_results_json,
                    stage_weights_json,
                    syntax_json,
                    logic_json,
                    execution_json,
                    semantics_json,
                    tool_calls_json,
                    scorecard.generated_nl_response,
                    generated_output_json,
                    scorecard.overall_passed,
                    scorecard.overall_score,
                    scorecard.worker_id,
                    scorecard.attempt_number,
                    scorecard.message_id,
                    scorecard.total_latency_ms,
                    scorecard.timestamp,
                    scorecard.completed_at,
                    scorecard.client_type,
                    scorecard.client_version,
                    scorecard.eval_mode,
                    scorecard.input_tokens,
                    scorecard.output_tokens,
                    scorecard.estimated_cost_usd,
                )
            )

        # Execute batch insert in a transaction
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Use executemany for batch insert with ON CONFLICT
                await conn.executemany(
                    """
                    INSERT INTO scorecards (
                        id, test_case_id, batch_id, run_id,
                        pack_name, stage_results, stage_weights,
                        syntax_result, logic_result, execution_result, semantics_result,
                        generated_tool_calls, generated_nl_response, generated_output,
                        overall_passed, overall_score,
                        worker_id, attempt_number, message_id, total_latency_ms,
                        created_at, completed_at,
                        client_type, client_version, eval_mode,
                        input_tokens, output_tokens, estimated_cost_usd
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6::jsonb, $7::jsonb,
                        $8::jsonb, $9::jsonb, $10::jsonb, $11::jsonb,
                        $12::jsonb, $13, $14::jsonb,
                        $15, $16,
                        $17, $18, $19, $20,
                        $21, $22,
                        $23, $24, $25,
                        $26, $27, $28
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        pack_name = EXCLUDED.pack_name,
                        stage_results = EXCLUDED.stage_results,
                        stage_weights = EXCLUDED.stage_weights,
                        syntax_result = EXCLUDED.syntax_result,
                        logic_result = EXCLUDED.logic_result,
                        execution_result = EXCLUDED.execution_result,
                        semantics_result = EXCLUDED.semantics_result,
                        generated_tool_calls = EXCLUDED.generated_tool_calls,
                        generated_nl_response = EXCLUDED.generated_nl_response,
                        generated_output = EXCLUDED.generated_output,
                        overall_passed = EXCLUDED.overall_passed,
                        overall_score = EXCLUDED.overall_score,
                        attempt_number = EXCLUDED.attempt_number,
                        total_latency_ms = EXCLUDED.total_latency_ms,
                        completed_at = EXCLUDED.completed_at,
                        client_type = EXCLUDED.client_type,
                        client_version = EXCLUDED.client_version,
                        eval_mode = EXCLUDED.eval_mode,
                        input_tokens = EXCLUDED.input_tokens,
                        output_tokens = EXCLUDED.output_tokens,
                        estimated_cost_usd = EXCLUDED.estimated_cost_usd
                    """,
                    records,
                )

        return len(records)

    def _stage_result_to_json(self, result: StageResult | None) -> str | None:
        """Convert StageResult to JSON string."""
        if result is None:
            return None

        data = {
            "stage_name": result.get_stage_name(),
            "stage": result.stage.value if result.stage else None,
            "passed": result.passed,
            "score": result.score,
            "error_code": result.error_code.value if result.error_code else None,
            "reason": result.reason,
            "artifacts": result.artifacts,
            "metrics": result.metrics,
            "duration_ms": result.duration_ms,
        }
        return json.dumps(data)

    def _json_to_stage_result(self, data: dict[str, Any] | str) -> StageResult:
        """Convert JSON dict to StageResult."""
        if isinstance(data, str):
            data = json.loads(data)

        # Support both old format (stage enum) and new format (stage_name string)
        stage = None
        stage_name = data.get("stage_name", "")

        if data.get("stage"):
            try:
                stage = EvaluationStage(data["stage"])
                if not stage_name:
                    stage_name = stage.value
            except ValueError:
                pass

        return StageResult(
            stage_name=stage_name,
            stage=stage,
            passed=data["passed"],
            score=data["score"],
            error_code=ErrorCode(data["error_code"]) if data.get("error_code") else None,
            reason=data.get("reason"),
            artifacts=data.get("artifacts", {}),
            metrics=data.get("metrics", {}),
            duration_ms=data.get("duration_ms", 0),
        )

    def _stage_results_to_json(self, stage_results: dict[str, StageResult]) -> str:
        """Convert stage_results dict to JSON string."""
        data = {}
        for name, result in stage_results.items():
            data[name] = {
                "stage_name": result.get_stage_name(),
                "stage": result.stage.value if result.stage else None,
                "passed": result.passed,
                "score": result.score,
                "error_code": result.error_code.value if result.error_code else None,
                "reason": result.reason,
                "artifacts": result.artifacts,
                "metrics": result.metrics,
                "duration_ms": result.duration_ms,
            }
        return json.dumps(data)

    def _json_to_stage_results(self, data: dict[str, Any] | str | None) -> dict[str, StageResult]:
        """Convert JSON dict to stage_results dict."""
        if not data:
            return {}
        if isinstance(data, str):
            data = json.loads(data)

        results = {}
        for name, result_data in data.items():
            results[name] = self._json_to_stage_result(result_data)
        return results

    def _row_to_scorecard(self, row: asyncpg.Record) -> Scorecard:
        """Convert database row to Scorecard model."""
        # Parse NL2API-specific stage results (backwards compatibility)
        syntax_result = (
            self._json_to_stage_result(row["syntax_result"]) if row["syntax_result"] else None
        )
        logic_result = (
            self._json_to_stage_result(row["logic_result"]) if row["logic_result"] else None
        )
        execution_result = (
            self._json_to_stage_result(row["execution_result"]) if row["execution_result"] else None
        )
        semantics_result = (
            self._json_to_stage_result(row["semantics_result"]) if row["semantics_result"] else None
        )

        # Parse generic stage_results (new format)
        stage_results = self._json_to_stage_results(row.get("stage_results"))

        # Parse stage_weights
        stage_weights = row.get("stage_weights") or {}
        if isinstance(stage_weights, str):
            stage_weights = json.loads(stage_weights)

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

        # Parse generated_output (generic)
        generated_output = row.get("generated_output") or {}
        if isinstance(generated_output, str):
            generated_output = json.loads(generated_output)

        # Handle timestamps
        timestamp = row["created_at"]
        if timestamp and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        completed_at = row["completed_at"]
        if completed_at and completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=UTC)

        # Handle Decimal to float conversion for estimated_cost_usd
        estimated_cost = row.get("estimated_cost_usd")
        if isinstance(estimated_cost, Decimal):
            estimated_cost = float(estimated_cost)

        return Scorecard(
            test_case_id=str(row["test_case_id"]),
            batch_id=row["batch_id"],
            scorecard_id=str(row["id"]),
            # Generic fields
            pack_name=row.get("pack_name") or "nl2api",
            stage_results=stage_results,
            stage_weights=stage_weights,
            # Timestamps
            timestamp=timestamp,
            completed_at=completed_at,
            # NL2API-specific fields
            syntax_result=syntax_result,
            logic_result=logic_result,
            execution_result=execution_result,
            semantics_result=semantics_result,
            # Captured outputs
            generated_tool_calls=generated_tool_calls,
            generated_nl_response=row["generated_nl_response"],
            generated_output=generated_output,
            # Execution context
            worker_id=row["worker_id"],
            attempt_number=row["attempt_number"],
            message_id=row["message_id"],
            total_latency_ms=row["total_latency_ms"],
            # Client tracking
            client_type=row.get("client_type"),
            client_version=row.get("client_version"),
            eval_mode=row.get("eval_mode"),
            # Cost tracking
            input_tokens=row.get("input_tokens"),
            output_tokens=row.get("output_tokens"),
            estimated_cost_usd=estimated_cost,
        )

    # =========================================================================
    # Client-based Query Methods (Multi-Client Evaluation)
    # =========================================================================

    async def get_by_client(
        self,
        client_type: str,
        client_version: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Scorecard]:
        """
        Get scorecards for a specific client type and optional version.

        Args:
            client_type: Client type to filter by
            client_version: Optional client version to filter by
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            List of scorecards matching the criteria
        """
        with tracer.start_as_current_span("db.scorecard.get_by_client") as span:
            span.set_attribute("db.operation", "get_by_client")
            span.set_attribute("db.client_type", client_type)
            if client_version:
                span.set_attribute("db.client_version", client_version)

            if client_version:
                rows = await self.pool.fetch(
                    """
                    SELECT * FROM scorecards
                    WHERE client_type = $1 AND client_version = $2
                    ORDER BY created_at DESC
                    LIMIT $3 OFFSET $4
                    """,
                    client_type,
                    client_version,
                    limit,
                    offset,
                )
            else:
                rows = await self.pool.fetch(
                    """
                    SELECT * FROM scorecards
                    WHERE client_type = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    client_type,
                    limit,
                    offset,
                )

            span.set_attribute("db.result_count", len(rows))
            return [self._row_to_scorecard(row) for row in rows]

    async def get_comparison_summary(
        self,
        client_types: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get comparison summary across multiple client types.

        Args:
            client_types: List of client types to compare
            start_date: Start of date range (defaults to 7 days ago)
            end_date: End of date range (defaults to now)

        Returns:
            List of summary dicts with metrics per client type
        """
        with tracer.start_as_current_span("db.scorecard.get_comparison_summary") as span:
            span.set_attribute("db.operation", "get_comparison_summary")
            span.set_attribute("db.client_types", ",".join(client_types))

            if start_date is None:
                start_date = datetime.now(UTC) - timedelta(days=7)
            if end_date is None:
                end_date = datetime.now(UTC)

            rows = await self.pool.fetch(
                """
                SELECT
                    client_type,
                    client_version,
                    COUNT(*) as total_tests,
                    COUNT(*) FILTER (WHERE overall_passed = true) as passed_count,
                    COUNT(*) FILTER (WHERE overall_passed = false) as failed_count,
                    ROUND(AVG(overall_score)::numeric, 4) as avg_score,
                    ROUND((COUNT(*) FILTER (WHERE overall_passed = true)::numeric /
                           NULLIF(COUNT(*), 0))::numeric, 4) as pass_rate,
                    SUM(COALESCE(input_tokens, 0)) as total_input_tokens,
                    SUM(COALESCE(output_tokens, 0)) as total_output_tokens,
                    SUM(COALESCE(estimated_cost_usd, 0)) as total_cost_usd,
                    AVG(total_latency_ms) as avg_latency_ms
                FROM scorecards
                WHERE client_type = ANY($1)
                  AND created_at >= $2
                  AND created_at <= $3
                GROUP BY client_type, client_version
                ORDER BY client_type, client_version
                """,
                client_types,
                start_date,
                end_date,
            )

            span.set_attribute("db.result_count", len(rows))

            results = []
            for row in rows:
                results.append(
                    {
                        "client_type": row["client_type"],
                        "client_version": row["client_version"],
                        "total_tests": row["total_tests"],
                        "passed_count": row["passed_count"],
                        "failed_count": row["failed_count"],
                        "avg_score": float(row["avg_score"]) if row["avg_score"] else 0.0,
                        "pass_rate": float(row["pass_rate"]) if row["pass_rate"] else 0.0,
                        "total_input_tokens": int(row["total_input_tokens"]),
                        "total_output_tokens": int(row["total_output_tokens"]),
                        "total_cost_usd": float(row["total_cost_usd"])
                        if row["total_cost_usd"]
                        else 0.0,
                        "avg_latency_ms": float(row["avg_latency_ms"])
                        if row["avg_latency_ms"]
                        else 0.0,
                    }
                )

            return results

    async def get_client_trend(
        self,
        client_type: str,
        metric: str = "pass_rate",
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Get daily trend data for a specific client type and metric.

        Args:
            client_type: Client type to get trend for
            metric: Metric to track (pass_rate, avg_score, avg_latency_ms, total_cost_usd)
            days: Number of days to include

        Returns:
            List of daily data points with date and metric value
        """
        with tracer.start_as_current_span("db.scorecard.get_client_trend") as span:
            span.set_attribute("db.operation", "get_client_trend")
            span.set_attribute("db.client_type", client_type)
            span.set_attribute("db.metric", metric)
            span.set_attribute("db.days", days)

            start_date = datetime.now(UTC) - timedelta(days=days)

            rows = await self.pool.fetch(
                """
                SELECT
                    DATE_TRUNC('day', created_at) as eval_date,
                    COUNT(*) as total_tests,
                    COUNT(*) FILTER (WHERE overall_passed = true) as passed_count,
                    ROUND(AVG(overall_score)::numeric, 4) as avg_score,
                    ROUND((COUNT(*) FILTER (WHERE overall_passed = true)::numeric /
                           NULLIF(COUNT(*), 0))::numeric, 4) as pass_rate,
                    AVG(total_latency_ms) as avg_latency_ms,
                    SUM(COALESCE(estimated_cost_usd, 0)) as total_cost_usd
                FROM scorecards
                WHERE client_type = $1
                  AND created_at >= $2
                GROUP BY DATE_TRUNC('day', created_at)
                ORDER BY eval_date ASC
                """,
                client_type,
                start_date,
            )

            span.set_attribute("db.result_count", len(rows))

            results = []
            for row in rows:
                data_point = {
                    "date": row["eval_date"].isoformat(),
                    "total_tests": row["total_tests"],
                }

                # Add the requested metric
                if metric == "pass_rate":
                    data_point["value"] = float(row["pass_rate"]) if row["pass_rate"] else 0.0
                elif metric == "avg_score":
                    data_point["value"] = float(row["avg_score"]) if row["avg_score"] else 0.0
                elif metric == "avg_latency_ms":
                    data_point["value"] = (
                        float(row["avg_latency_ms"]) if row["avg_latency_ms"] else 0.0
                    )
                elif metric == "total_cost_usd":
                    data_point["value"] = (
                        float(row["total_cost_usd"]) if row["total_cost_usd"] else 0.0
                    )
                else:
                    data_point["value"] = float(row["pass_rate"]) if row["pass_rate"] else 0.0

                results.append(data_point)

            return results
