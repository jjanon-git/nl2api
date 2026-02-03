"""
NL2API Evaluation Pack

Evaluates tool-calling LLM systems using a 4-stage waterfall pipeline:
1. Syntax: Validates JSON structure and schema (GATE)
2. Logic: AST-based comparison of tool calls
3. Execution: Compares API execution results (deferred)
4. Semantics: LLM-as-judge NL response comparison

This pack implements the EvaluationPack protocol for backwards compatibility
with the existing NL2API evaluation infrastructure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from CONTRACTS import (
    ErrorCode,
    EvalContext,
    EvaluationStage,
    Scorecard,
    StageResult,
    TemporalValidationMode,
    TestCase,
    ToolCall,
)
from src.evalkit.common.telemetry import get_tracer
from src.evalkit.core.ast_comparator import ASTComparator, ComparisonResult
from src.evalkit.core.temporal import DateResolver, TemporalComparator

tracer = get_tracer(__name__)


def _truncate(text: str, max_length: int = 500) -> str:
    """Truncate text for trace attributes, preserving useful prefix."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... ({len(text) - max_length} more chars)"


# =============================================================================
# Stage Implementations
# =============================================================================


@dataclass
class SyntaxStage:
    """
    Stage 1: Syntax Validation

    Validates that the raw output from the target system:
    1. Is valid JSON
    2. Conforms to the expected tool call schema
    3. Can be parsed into ToolCall objects

    This is a GATE stage - failure halts the pipeline.
    """

    @property
    def name(self) -> str:
        return "syntax"

    @property
    def is_gate(self) -> bool:
        return True

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext,
    ) -> StageResult:
        """
        Validate and parse raw output.

        Expects system_output to have 'raw_output' key with the raw string.
        Sets 'parsed_tool_calls' in system_output on success.
        """
        start_time = time.perf_counter()
        raw_output = system_output.get("raw_output", "")

        try:
            # Step 1: Parse JSON
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return StageResult(
                stage_name="syntax",
                stage=EvaluationStage.SYNTAX,
                passed=False,
                score=0.0,
                error_code=ErrorCode.SYNTAX_INVALID_JSON,
                reason=f"Invalid JSON: {e}",
                artifacts={"raw_output": raw_output[:500], "error_position": e.pos},
                duration_ms=duration_ms,
            )

        # Step 2: Validate structure
        try:
            tool_calls = self._parse_tool_calls(parsed)
            # Store parsed calls in system_output for downstream stages
            system_output["parsed_tool_calls"] = tool_calls
        except ValueError as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return StageResult(
                stage_name="syntax",
                stage=EvaluationStage.SYNTAX,
                passed=False,
                score=0.0,
                error_code=ErrorCode.SYNTAX_SCHEMA_VIOLATION,
                reason=str(e),
                artifacts={"parsed_json": parsed},
                duration_ms=duration_ms,
            )

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return StageResult(
            stage_name="syntax",
            stage=EvaluationStage.SYNTAX,
            passed=True,
            score=1.0,
            reason="Valid JSON and schema",
            artifacts={"tool_call_count": len(tool_calls)},
            duration_ms=duration_ms,
        )

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


@dataclass
class LogicStage:
    """
    Stage 2: Logic Comparison

    Compares actual tool calls against expected using AST-based comparison.
    Handles order-independent matching and type coercion.

    This is a HIGH priority stage but not a gate - failures are logged
    but the pipeline continues to gather diagnostic data.
    """

    numeric_tolerance: float = 0.0001

    @property
    def name(self) -> str:
        return "logic"

    @property
    def is_gate(self) -> bool:
        return False

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext,
    ) -> StageResult:
        """
        Compare expected vs actual tool calls.
        """
        start_time = time.perf_counter()

        # Get expected from test case (support both generic and NL2API formats)
        expected = test_case.expected_tool_calls
        if not expected and test_case.expected.get("tool_calls"):
            # Convert from generic format
            expected = tuple(
                ToolCall(
                    tool_name=tc["tool_name"],
                    arguments=tc.get("arguments", {}),
                )
                for tc in test_case.expected["tool_calls"]
            )

        # Get actual from system_output
        actual = system_output.get("parsed_tool_calls", ())

        # Use custom comparator from context if provided, otherwise default
        comparator = context.config.get("comparator") or ASTComparator(
            numeric_tolerance=self.numeric_tolerance
        )
        result: ComparisonResult = comparator.compare(expected, actual)

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        if result.matches:
            return StageResult(
                stage_name="logic",
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
            stage_name="logic",
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


@dataclass
class ExecutionStage:
    """
    Stage 3: Execution Comparison

    Compares actual API execution results against expected data.
    Currently deferred - returns pass with note.
    """

    @property
    def name(self) -> str:
        return "execution"

    @property
    def is_gate(self) -> bool:
        return False

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext,
    ) -> StageResult:
        """
        Compare execution results. Currently deferred.
        """
        return StageResult(
            stage_name="execution",
            stage=EvaluationStage.EXECUTION,
            passed=True,
            score=1.0,
            reason="Execution stage deferred - not implemented",
            duration_ms=0,
        )


@dataclass
class SemanticsStage:
    """
    Stage 4: Semantics Comparison

    Uses LLM-as-judge to compare natural language responses.
    Optional stage - skipped if no expected NL response.
    """

    @property
    def name(self) -> str:
        return "semantics"

    @property
    def is_gate(self) -> bool:
        return False

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext,
    ) -> StageResult:
        """
        Compare NL responses using LLM-as-judge.
        """
        start_time = time.perf_counter()

        # Get expected NL response
        expected_nl = test_case.expected_nl_response
        if not expected_nl:
            expected_nl = test_case.expected.get("nl_response")

        # Get actual NL response
        actual_nl = system_output.get("nl_response")

        # Skip if either is missing
        if not expected_nl or not actual_nl:
            return StageResult(
                stage_name="semantics",
                stage=EvaluationStage.SEMANTICS,
                passed=True,
                score=1.0,
                reason="Skipped - no NL responses to compare",
                artifacts={
                    "has_expected": expected_nl is not None,
                    "has_actual": actual_nl is not None,
                },
                duration_ms=0,
            )

        # Use semantics evaluator from context if available
        semantics_evaluator = context.config.get("semantics_evaluator")
        if semantics_evaluator:
            result = await semantics_evaluator.evaluate_direct(
                test_case=test_case,
                actual_nl=actual_nl,
            )
            return result

        # Default: simple comparison (for testing without LLM)
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return StageResult(
            stage_name="semantics",
            stage=EvaluationStage.SEMANTICS,
            passed=True,
            score=1.0,
            reason="Semantics stage skipped - no LLM judge configured",
            duration_ms=duration_ms,
        )


# =============================================================================
# NL2API Pack
# =============================================================================


class NL2APIPack:
    """
    Evaluation pack for NL2API tool-calling LLM systems.

    Implements the EvaluationPack protocol with 4 stages:
    1. Syntax (GATE): JSON validation and schema checking
    2. Logic: AST-based tool call comparison
    3. Execution: API result comparison (deferred)
    4. Semantics: LLM-as-judge NL comparison

    Usage:
        pack = NL2APIPack()
        for stage in pack.get_stages():
            result = await stage.evaluate(test_case, system_output, context)
    """

    DEFAULT_WEIGHTS = {
        "syntax": 0.1,
        "logic": 0.3,
        "execution": 0.5,  # CRITICAL - most important when implemented
        "semantics": 0.1,  # Optional polish
    }

    def __init__(
        self,
        execution_enabled: bool = False,
        semantics_enabled: bool = False,
        numeric_tolerance: float = 0.0001,
        # Temporal config
        temporal_mode: TemporalValidationMode = TemporalValidationMode.STRUCTURAL,
        evaluation_date: date | None = None,
        relative_date_fields: tuple[str, ...] = ("start", "end", "SDate", "EDate", "Period"),
        fiscal_year_end_month: int = 12,
    ):
        """
        Initialize the NL2API pack.

        Args:
            execution_enabled: Whether to include execution stage (deferred)
            semantics_enabled: Whether to include semantics stage
            numeric_tolerance: Tolerance for numeric comparisons in logic stage
            temporal_mode: Temporal validation mode (BEHAVIORAL, STRUCTURAL, DATA)
            evaluation_date: Reference date for temporal normalization (defaults to today)
            relative_date_fields: Field names that may contain relative date expressions
            fiscal_year_end_month: Month when fiscal year ends (1-12)
        """
        self.execution_enabled = execution_enabled
        self.semantics_enabled = semantics_enabled
        self.numeric_tolerance = numeric_tolerance
        self.temporal_mode = temporal_mode
        self.evaluation_date = evaluation_date
        self.relative_date_fields = relative_date_fields
        self.fiscal_year_end_month = fiscal_year_end_month

        # Create comparator - use TemporalComparator if temporal mode is not DATA (exact match)
        if temporal_mode != TemporalValidationMode.DATA:
            date_resolver = DateResolver(
                reference_date=evaluation_date,
                fiscal_year_end_month=fiscal_year_end_month,
            )
            base_comparator = ASTComparator(numeric_tolerance=numeric_tolerance)
            self._comparator: ASTComparator = TemporalComparator(
                date_resolver=date_resolver,
                validation_mode=temporal_mode,
                relative_date_fields=relative_date_fields,
                base_comparator=base_comparator,
            )
        else:
            self._comparator = ASTComparator(numeric_tolerance=numeric_tolerance)

        # Build stages
        self._stages = [
            SyntaxStage(),
            LogicStage(numeric_tolerance=numeric_tolerance),
        ]
        if execution_enabled:
            self._stages.append(ExecutionStage())
        if semantics_enabled:
            self._stages.append(SemanticsStage())

    @property
    def name(self) -> str:
        return "nl2api"

    def get_stages(self) -> list:
        """Return ordered list of evaluation stages."""
        return list(self._stages)

    def get_default_weights(self) -> dict[str, float]:
        """Return default scoring weights per stage name."""
        return dict(self.DEFAULT_WEIGHTS)

    def validate_test_case(self, test_case: TestCase) -> list[str]:
        """
        Validate test case has required fields for NL2API evaluation.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        # Must have either NL2API-specific or generic input
        has_nl_query = test_case.nl_query is not None
        has_generic_query = test_case.input.get("nl_query") is not None

        if not has_nl_query and not has_generic_query:
            errors.append(
                "Missing nl_query (either test_case.nl_query or test_case.input['nl_query'])"
            )

        # Must have expected tool calls
        has_tool_calls = len(test_case.expected_tool_calls) > 0
        has_generic_tool_calls = test_case.expected.get("tool_calls") is not None

        if not has_tool_calls and not has_generic_tool_calls:
            errors.append(
                "Missing expected_tool_calls (either test_case.expected_tool_calls or test_case.expected['tool_calls'])"
            )

        return errors

    def compute_overall_score(
        self,
        stage_results: dict[str, StageResult],
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Compute weighted overall score from stage results.

        Args:
            stage_results: Results keyed by stage name.
            weights: Optional custom weights. Uses default if None.

        Returns:
            Weighted average score (0.0 to 1.0).
        """
        weights = weights or self.DEFAULT_WEIGHTS

        total_weight = 0.0
        weighted_score = 0.0

        for stage_name, result in stage_results.items():
            w = weights.get(stage_name, 0.25)  # Default weight for unknown stages
            weighted_score += result.score * w
            total_weight += w

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def compute_overall_passed(
        self,
        stage_results: dict[str, StageResult],
    ) -> bool:
        """
        Determine if overall evaluation passed.

        For NL2API, all stages must pass.
        """
        return all(r.passed for r in stage_results.values())

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None = None,
    ) -> Scorecard:
        """
        Run the complete NL2API evaluation pipeline.

        This is a convenience method that runs all stages and creates a Scorecard.

        Args:
            test_case: The test case with expected values
            system_output: Output from target system. Expected keys:
                - raw_output: Raw string output for syntax parsing
                - nl_response: Optional NL response for semantics
            context: Optional evaluation context

        Returns:
            Scorecard with all stage results
        """
        # Create context with comparator if not provided
        if context is None:
            context = EvalContext(config={"comparator": self._comparator})
        elif "comparator" not in context.config:
            # Add comparator to existing context
            context = EvalContext(
                batch_id=context.batch_id,
                worker_id=context.worker_id,
                config={**context.config, "comparator": self._comparator},
            )
        start_time = time.perf_counter()

        stage_results: dict[str, StageResult] = {}

        with tracer.start_as_current_span("nl2api_pack.evaluate") as span:
            span.set_attribute("test_case.id", test_case.id)
            span.set_attribute("pack.name", self.name)

            # Add input context to trace for debugging
            query = test_case.nl_query or test_case.input.get("query", "")
            raw_output = str(system_output.get("raw_output", ""))
            span.set_attribute("input.query", _truncate(query, 500))
            span.set_attribute("input.raw_output_length", len(raw_output))
            span.add_event("input", {"query": _truncate(query, 1000)})
            span.add_event(
                "system_output",
                {"raw_output": _truncate(raw_output, 2000)},
            )

            for stage in self._stages:
                with tracer.start_as_current_span(f"nl2api_pack.{stage.name}") as stage_span:
                    stage_span.set_attribute("stage.is_gate", stage.is_gate)
                    result = await stage.evaluate(test_case, system_output, context)
                    stage_results[stage.name] = result

                    # Core result attributes
                    stage_span.set_attribute("result.passed", result.passed)
                    stage_span.set_attribute("result.score", result.score)
                    stage_span.set_attribute("result.duration_ms", result.duration_ms)

                    # Add reason and error code if present
                    if result.reason:
                        stage_span.set_attribute("result.reason", _truncate(result.reason, 500))
                    if result.error_code:
                        stage_span.set_attribute("result.error_code", result.error_code.value)

                    # Add key metrics as attributes
                    for key, value in result.metrics.items():
                        if isinstance(value, (int, float, bool, str)):
                            stage_span.set_attribute(f"metrics.{key}", value)

                    # Add artifacts as event
                    if result.artifacts:
                        safe_artifacts = {}
                        for k, v in result.artifacts.items():
                            if isinstance(v, (int, float, bool)):
                                safe_artifacts[k] = v
                            elif isinstance(v, str):
                                safe_artifacts[k] = _truncate(v, 500)
                        if safe_artifacts:
                            stage_span.add_event("artifacts", safe_artifacts)

                # Check for gate failure
                if stage.is_gate and not result.passed:
                    span.set_attribute("gate_failed", stage.name)
                    break

            total_latency_ms = int((time.perf_counter() - start_time) * 1000)
            overall_passed = self.compute_overall_passed(stage_results)
            overall_score = self.compute_overall_score(stage_results)

            span.set_attribute("result.overall_passed", overall_passed)
            span.set_attribute("result.overall_score", overall_score)
            span.set_attribute("result.total_latency_ms", total_latency_ms)

        # Build scorecard
        # Filter generated_output to only include JSON-serializable fields
        # (parsed_tool_calls contains ToolCall objects which aren't JSON serializable)
        serializable_output = {k: v for k, v in system_output.items() if k != "parsed_tool_calls"}

        return Scorecard(
            test_case_id=test_case.id,
            batch_id=context.batch_id,
            pack_name=self.name,
            stage_results=stage_results,
            stage_weights=self.get_default_weights(),
            # NL2API-specific fields for backwards compatibility
            syntax_result=stage_results.get("syntax"),
            logic_result=stage_results.get("logic"),
            execution_result=stage_results.get("execution"),
            semantics_result=stage_results.get("semantics"),
            # Captured output
            generated_tool_calls=system_output.get("parsed_tool_calls"),
            generated_nl_response=system_output.get("nl_response"),
            generated_output=serializable_output,
            # Context
            worker_id=context.worker_id,
            total_latency_ms=total_latency_ms,
        )
