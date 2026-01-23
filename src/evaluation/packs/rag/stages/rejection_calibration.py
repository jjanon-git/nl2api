"""
Rejection Calibration Stage

Evaluates whether the system correctly handles queries it should/shouldn't answer.

Detects:
- False positives: Rejecting queries it should answer
- False negatives: Answering queries it should reject
- Training cutoff excuses: Falsely claiming knowledge cutoff
- Hallucinated rejections: Making up reasons to not answer
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.contracts import StageResult, TestCase

if TYPE_CHECKING:
    from src.contracts import EvalContext


@dataclass
class RejectionCalibrationStage:
    """
    Stage 8: Rejection Calibration Evaluation

    Evaluates whether system correctly decides when to answer vs reject.

    Expected behaviors in test case:
    - expected['behavior'] = "answer": Should provide answer
    - expected['behavior'] = "reject": Should reject/refuse
    - expected['behavior'] = "partial": May provide partial answer
    """

    name: str = field(default="rejection_calibration", init=False)
    is_gate: bool = field(default=False, init=False)

    # Rejection detection patterns
    rejection_patterns: tuple[str, ...] = (
        r"i cannot",
        r"i can't",
        r"i'm not able to",
        r"i am not able to",
        r"i don't have",
        r"i do not have",
        r"i'm unable to",
        r"i am unable to",
        r"cannot provide",
        r"cannot answer",
        r"not available",
        r"no information",
        r"outside my scope",
        r"beyond my knowledge",
        r"i apologize",
        r"unfortunately",
        r"i'm sorry",
        r"i am sorry",
    )

    # Training cutoff excuse patterns
    training_cutoff_patterns: tuple[str, ...] = (
        r"my (?:training|knowledge) (?:data |cutoff |was )?(?:only goes|ends|stops|is limited)",
        r"(?:as of |after |before )(?:my |the )?(?:training|cutoff)",
        r"(?:october|november|september|january|april) 20\d\d",
        r"my training (?:data )?(?:only )?(?:goes up to|ends at|is from)",
        r"knowledge cutoff",
        r"don't have (?:access to )?(?:information|data) (?:after|beyond|past)",
    )

    # Hallucinated rejection patterns
    hallucinated_rejection_patterns: tuple[str, ...] = (
        r"that information is (?:classified|confidential|restricted)",
        r"i'm not allowed to",
        r"my programming prevents",
        r"i've been instructed not to",
    )

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None,
    ) -> StageResult:
        """
        Evaluate rejection calibration.

        Args:
            test_case: Test case with expected behavior
            system_output: System output with response
            context: Evaluation context

        Returns:
            StageResult with calibration metrics
        """
        start_time = time.perf_counter()

        # Get expected behavior (None if not specified)
        expected_behavior = test_case.expected.get("behavior")

        # Get response
        response = self._extract_response(system_output)
        if not response:
            # No response is effectively a rejection
            actual_behavior = "reject"
        else:
            # Analyze response behavior
            actual_behavior = self._classify_response(response)

        # Detect specific rejection types
        metrics: dict[str, Any] = {
            "expected_behavior": expected_behavior,
            "actual_behavior": actual_behavior,
        }

        if actual_behavior == "reject":
            metrics["is_training_cutoff_excuse"] = self._is_training_cutoff_excuse(response)
            metrics["is_hallucinated_rejection"] = self._is_hallucinated_rejection(response)
            metrics["rejection_patterns_matched"] = self._get_matched_patterns(response)

        # Determine correctness
        if expected_behavior is None:
            # No expected behavior specified - give neutral score, just analyze
            score = 0.5
            passed = True
            reason = f"No expected behavior specified, actual: {actual_behavior}"

        elif expected_behavior == "answer":
            if actual_behavior == "answer":
                score = 1.0
                passed = True
                reason = "Correctly answered (expected: answer)"
            elif actual_behavior == "partial":
                score = 0.7
                passed = True
                reason = "Partial answer provided (expected: full answer)"
            else:
                score = 0.0
                passed = False
                reason = self._build_false_positive_reason(response, metrics)

        elif expected_behavior == "reject":
            if actual_behavior == "reject":
                score = 1.0
                passed = True
                reason = "Correctly rejected (expected: reject)"
            else:
                score = 0.0
                passed = False
                reason = "False negative: Should have rejected but answered"

        elif expected_behavior == "partial":
            if actual_behavior in ("answer", "partial"):
                score = 1.0
                passed = True
                reason = "Acceptable response for partial expectation"
            else:
                score = 0.3
                passed = False
                reason = "Rejected when partial answer was possible"

        else:
            # Unknown expected behavior value - treat as neutral
            score = 0.5
            passed = True
            reason = f"Unknown expected behavior '{expected_behavior}', actual: {actual_behavior}"

        metrics["score"] = score
        metrics["passed"] = passed

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=score,
            reason=reason,
            metrics=metrics,
            artifacts={
                "response_preview": response[:300] if response else None,
                "query": test_case.input.get("query", "")[:200],
            },
            duration_ms=duration_ms,
        )

    def _extract_response(self, system_output: dict[str, Any]) -> str:
        """Extract response text from system output."""
        for field_name in ["response", "answer", "generated_text", "output"]:
            if field_name in system_output:
                value = system_output[field_name]
                if isinstance(value, str):
                    return value
        return ""

    def _classify_response(self, response: str) -> str:
        """Classify response as answer, reject, or partial."""
        response_lower = response.lower()

        # Check for rejection patterns
        rejection_count = 0
        for pattern in self.rejection_patterns:
            if re.search(pattern, response_lower):
                rejection_count += 1

        # Also check for training cutoff excuses - these are implicit rejections
        is_cutoff_excuse = self._is_training_cutoff_excuse(response)
        if is_cutoff_excuse:
            rejection_count += 1

        # Also check for hallucinated rejections - these are implicit rejections
        is_hallucinated = self._is_hallucinated_rejection(response)
        if is_hallucinated:
            rejection_count += 1

        # Very short response with rejection patterns = rejection
        if rejection_count >= 2 or (rejection_count >= 1 and len(response) < 100):
            return "reject"

        # Some rejection but also substantial content = partial
        if rejection_count >= 1 and len(response) > 200:
            return "partial"

        # No rejection patterns = answer
        return "answer"

    def _is_training_cutoff_excuse(self, response: str) -> bool:
        """Detect training cutoff excuses."""
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in self.training_cutoff_patterns)

    def _is_hallucinated_rejection(self, response: str) -> bool:
        """Detect hallucinated rejection reasons."""
        response_lower = response.lower()
        return any(
            re.search(pattern, response_lower) for pattern in self.hallucinated_rejection_patterns
        )

    def _get_matched_patterns(self, response: str) -> list[str]:
        """Get list of matched rejection patterns."""
        response_lower = response.lower()
        matched = []

        for pattern in self.rejection_patterns:
            match = re.search(pattern, response_lower)
            if match:
                matched.append(match.group())

        return matched[:5]  # Limit

    def _build_false_positive_reason(
        self,
        response: str,
        metrics: dict[str, Any],
    ) -> str:
        """Build explanation for false positive rejection."""
        parts = ["False positive: Should have answered but rejected."]

        if metrics.get("is_training_cutoff_excuse"):
            parts.append("Used training cutoff excuse.")

        if metrics.get("is_hallucinated_rejection"):
            parts.append("Used hallucinated rejection reason.")

        patterns = metrics.get("rejection_patterns_matched", [])
        if patterns:
            parts.append(f"Patterns: {', '.join(patterns[:3])}")

        return " ".join(parts)
