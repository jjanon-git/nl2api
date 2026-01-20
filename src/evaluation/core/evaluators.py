"""
Evaluation Pipeline Implementations

Stage 1: SyntaxEvaluator - Validates JSON structure and schema
Stage 2: LogicEvaluator - AST-based comparison of tool calls
Stage 3: ExecutionEvaluator - Deferred to Sprint 4
Stage 4: SemanticsEvaluator - Deferred to Sprint 4
"""

from __future__ import annotations

import json
import time
from typing import Any

from CONTRACTS import (
    ErrorCode,
    EvaluationConfig,
    EvaluationStage,
    Evaluator,
    Scorecard,
    StageResult,
    SystemResponse,
    TestCase,
    ToolCall,
)

from src.evaluation.core.ast_comparator import ASTComparator, ComparisonResult


class SyntaxEvaluator:
    """
    Stage 1: Syntax Validation

    Validates that the raw output from the target system:
    1. Is valid JSON
    2. Conforms to the expected tool call schema
    3. Can be parsed into ToolCall objects

    This is a GATE stage - failure halts the pipeline.
    """

    def evaluate(self, raw_output: str) -> tuple[StageResult, tuple[ToolCall, ...] | None]:
        """
        Validate and parse raw output.

        Args:
            raw_output: Raw string output from target system

        Returns:
            Tuple of (StageResult, parsed_tool_calls or None)
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Parse JSON
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return StageResult(
                stage=EvaluationStage.SYNTAX,
                passed=False,
                score=0.0,
                error_code=ErrorCode.SYNTAX_INVALID_JSON,
                reason=f"Invalid JSON: {e}",
                artifacts={"raw_output": raw_output[:500], "error_position": e.pos},
                duration_ms=duration_ms,
            ), None

        # Step 2: Validate structure
        try:
            tool_calls = self._parse_tool_calls(parsed)
        except ValueError as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return StageResult(
                stage=EvaluationStage.SYNTAX,
                passed=False,
                score=0.0,
                error_code=ErrorCode.SYNTAX_SCHEMA_VIOLATION,
                reason=str(e),
                artifacts={"parsed_json": parsed},
                duration_ms=duration_ms,
            ), None

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return StageResult(
            stage=EvaluationStage.SYNTAX,
            passed=True,
            score=1.0,
            reason="Valid JSON and schema",
            artifacts={"tool_call_count": len(tool_calls)},
            duration_ms=duration_ms,
        ), tool_calls

    def _parse_tool_calls(self, parsed: Any) -> tuple[ToolCall, ...]:
        """
        Parse JSON into ToolCall objects.

        Accepts multiple formats:
        - List of tool calls: [{"tool_name": "...", "arguments": {...}}, ...]
        - Single tool call: {"tool_name": "...", "arguments": {...}}
        - Wrapper format: {"tool_calls": [...]}
        """
        # Handle wrapper format
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            parsed = parsed["tool_calls"]

        # Normalize to list
        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            raise ValueError(f"Expected list or dict, got {type(parsed).__name__}")

        tool_calls = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"Tool call {i} is not a dict: {type(item).__name__}")

            # Validate required fields
            if "tool_name" not in item and "name" not in item:
                raise ValueError(f"Tool call {i} missing 'tool_name' or 'name' field")

            tool_name = item.get("tool_name") or item.get("name")
            arguments = item.get("arguments", {})

            if not isinstance(tool_name, str):
                raise ValueError(f"Tool call {i} 'tool_name' must be string")

            if not isinstance(arguments, dict):
                raise ValueError(f"Tool call {i} 'arguments' must be dict")

            tool_calls.append(ToolCall(tool_name=tool_name, arguments=arguments))

        return tuple(tool_calls)


class LogicEvaluator:
    """
    Stage 2: Logic Comparison

    Compares actual tool calls against expected using AST-based comparison.
    Handles order-independent matching and type coercion.

    This is a HIGH priority stage but uses soft continue - failures are logged
    but the pipeline continues to gather diagnostic data.
    """

    def __init__(self, numeric_tolerance: float = 0.0001):
        self.comparator = ASTComparator(numeric_tolerance=numeric_tolerance)

    def evaluate(
        self,
        expected: tuple[ToolCall, ...],
        actual: tuple[ToolCall, ...],
    ) -> StageResult:
        """
        Compare expected vs actual tool calls.

        Args:
            expected: Expected tool calls from test case
            actual: Actual tool calls from system response

        Returns:
            StageResult with comparison details
        """
        start_time = time.perf_counter()

        result: ComparisonResult = self.comparator.compare(expected, actual)

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        if result.matches:
            return StageResult(
                stage=EvaluationStage.LOGIC,
                passed=True,
                score=result.score,
                reason=result.summary,
                artifacts={
                    "matched_calls": len(result.matched_calls),
                },
                duration_ms=duration_ms,
            )

        # Determine primary error code
        error_code = ErrorCode.LOGIC_TOOL_MISMATCH
        if result.missing_calls:
            error_code = ErrorCode.LOGIC_MISSING_CALL
        elif result.extra_calls:
            error_code = ErrorCode.LOGIC_EXTRA_CALL
        elif result.argument_diffs:
            error_code = ErrorCode.LOGIC_ARG_MISMATCH

        return StageResult(
            stage=EvaluationStage.LOGIC,
            passed=False,
            score=result.score,
            error_code=error_code,
            reason=result.summary,
            artifacts={
                "matched_calls": len(result.matched_calls),
                "missing_calls": [tc.model_dump() for tc in result.missing_calls],
                "extra_calls": [tc.model_dump() for tc in result.extra_calls],
                "argument_diffs": result.argument_diffs,
            },
            duration_ms=duration_ms,
        )


class WaterfallEvaluator(Evaluator):
    """
    Complete evaluation pipeline implementing the waterfall pattern.

    Sprint 1 implements Stage 1 (Syntax) and Stage 2 (Logic).
    Stage 3 (Execution) and Stage 4 (Semantics) are stubs for now.
    """

    def __init__(self, config: EvaluationConfig | None = None):
        super().__init__(config)
        self.syntax_evaluator = SyntaxEvaluator()
        self.logic_evaluator = LogicEvaluator(
            numeric_tolerance=self.config.numeric_tolerance
        )

    async def evaluate(
        self,
        test_case: TestCase,
        system_response: SystemResponse,
        worker_id: str,
    ) -> Scorecard:
        """
        Run the complete evaluation pipeline.

        Pipeline flow:
        1. Syntax (GATE) - If fails, stop and return
        2. Logic (HIGH) - Continue even on failure
        3. Execution (CRITICAL) - Skipped in Sprint 1
        4. Semantics (LOW) - Skipped in Sprint 1
        """
        start_time = time.perf_counter()

        # Stage 1: Syntax
        syntax_result, parsed_calls = self.evaluate_syntax(system_response.raw_output)

        if not syntax_result.passed:
            # GATE failure - halt pipeline
            total_latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Scorecard(
                test_case_id=test_case.id,
                syntax_result=syntax_result,
                logic_result=None,
                execution_result=None,
                semantics_result=None,
                generated_tool_calls=None,
                generated_nl_response=system_response.nl_response,
                worker_id=worker_id,
                total_latency_ms=total_latency_ms,
            )

        # Stage 2: Logic
        logic_result = self.evaluate_logic(
            test_case.expected_tool_calls,
            parsed_calls or (),
        )

        # Stage 3 & 4: Skipped in Sprint 1
        execution_result = None
        semantics_result = None

        total_latency_ms = int((time.perf_counter() - start_time) * 1000)

        return Scorecard(
            test_case_id=test_case.id,
            syntax_result=syntax_result,
            logic_result=logic_result,
            execution_result=execution_result,
            semantics_result=semantics_result,
            generated_tool_calls=parsed_calls,
            generated_nl_response=system_response.nl_response,
            worker_id=worker_id,
            total_latency_ms=total_latency_ms,
        )

    def evaluate_syntax(self, raw_output: str) -> tuple[StageResult, tuple[ToolCall, ...] | None]:
        """Stage 1: Validate JSON structure and schema."""
        return self.syntax_evaluator.evaluate(raw_output)

    def evaluate_logic(
        self,
        expected: tuple[ToolCall, ...],
        actual: tuple[ToolCall, ...],
    ) -> StageResult:
        """Stage 2: AST-based comparison of tool calls."""
        return self.logic_evaluator.evaluate(expected, actual)

    async def evaluate_execution(
        self,
        expected_data: dict[str, Any] | None,
        actual_data: dict[str, Any] | None,
    ) -> StageResult:
        """Stage 3: Compare execution results. Deferred to Sprint 4."""
        return StageResult(
            stage=EvaluationStage.EXECUTION,
            passed=True,
            score=1.0,
            reason="Execution stage skipped (Sprint 1)",
            duration_ms=0,
        )

    async def evaluate_semantics(
        self,
        expected_text: str,
        actual_text: str,
    ) -> StageResult:
        """Stage 4: LLM-as-Judge semantic comparison. Deferred to Sprint 4."""
        return StageResult(
            stage=EvaluationStage.SEMANTICS,
            passed=True,
            score=1.0,
            reason="Semantics stage skipped (Sprint 1)",
            duration_ms=0,
        )
