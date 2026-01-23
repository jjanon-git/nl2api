"""
Source Policy Stage (GATE)

Evaluates compliance with source usage policies.
This is a GATE stage - violation stops the pipeline.

Policies:
- quote_only: Only direct quotes allowed (no paraphrasing)
- summarize: Summarization allowed
- no_use: Source cannot be used at all

Critical for legal/licensing compliance.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.contracts import StageResult, TestCase

if TYPE_CHECKING:
    from src.contracts import EvalContext


class SourcePolicy(str, Enum):
    """Source usage policy types."""

    QUOTE_ONLY = "quote_only"  # Only direct quotes allowed
    SUMMARIZE = "summarize"  # Summarization allowed
    NO_USE = "no_use"  # Source cannot be used


@dataclass
class SourcePolicyStage:
    """
    Stage 6: Source Policy Enforcement (GATE)

    Ensures response complies with source usage policies.
    Critical for legal/licensing compliance.

    GATE stage: Failure stops the pipeline.
    """

    name: str = field(default="source_policy", init=False)
    is_gate: bool = field(default=True, init=False)  # GATE stage

    # Minimum quote similarity threshold
    quote_similarity_threshold: float = 0.95

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None,
    ) -> StageResult:
        """
        Evaluate source policy compliance.

        Args:
            test_case: Test case with source policy requirements
            system_output: System output with response and sources
            context: Evaluation context with LLM judge

        Returns:
            StageResult with compliance details
        """
        start_time = time.perf_counter()

        # Get source policies from test case or sources
        source_policies = self._get_source_policies(test_case, system_output)

        if not source_policies:
            return self._skip_result("No source policies defined", start_time)

        # Get response
        response = self._extract_response(system_output)
        if not response:
            return self._skip_result("No response to evaluate", start_time)

        # Get sources
        sources = self._extract_sources(system_output)

        # Check each quote-only source
        violations: list[dict[str, Any]] = []
        checks_performed = 0

        for source_id, policy in source_policies.items():
            if policy == SourcePolicy.NO_USE:
                # Check if source was used at all
                source = sources.get(source_id, {})
                source_text = source.get("text", "") if isinstance(source, dict) else str(source)

                if source_text and self._is_source_used(response, source_text):
                    violations.append(
                        {
                            "source_id": source_id,
                            "policy": "no_use",
                            "violation": "Source used despite no_use policy",
                        }
                    )
                checks_performed += 1

            elif policy == SourcePolicy.QUOTE_ONLY:
                # Check if any paraphrasing occurred
                source = sources.get(source_id, {})
                source_text = source.get("text", "") if isinstance(source, dict) else str(source)

                if source_text:
                    paraphrases = await self._detect_paraphrasing(response, source_text, context)
                    if paraphrases:
                        violations.append(
                            {
                                "source_id": source_id,
                                "policy": "quote_only",
                                "violation": "Paraphrasing detected",
                                "paraphrases": paraphrases[:3],
                            }
                        )
                checks_performed += 1

        # Compute result
        passed = len(violations) == 0
        score = 1.0 if passed else 0.0
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=score,
            reason=self._build_reason(violations, checks_performed, passed),
            metrics={
                "checks_performed": checks_performed,
                "violations_count": len(violations),
                "quote_only_sources": sum(
                    1 for p in source_policies.values() if p == SourcePolicy.QUOTE_ONLY
                ),
                "no_use_sources": sum(
                    1 for p in source_policies.values() if p == SourcePolicy.NO_USE
                ),
            },
            artifacts={
                "violations": violations,
                "source_policies": {k: v.value for k, v in source_policies.items()},
            },
            duration_ms=duration_ms,
        )

    def _get_source_policies(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
    ) -> dict[str, SourcePolicy]:
        """Extract source policies from test case or system output."""
        policies: dict[str, SourcePolicy] = {}

        # From test case expected
        if test_case.expected.get("source_policies"):
            raw_policies = test_case.expected["source_policies"]
            for source_id, policy in raw_policies.items():
                try:
                    policies[source_id] = SourcePolicy(policy)
                except ValueError:
                    pass

        # From sources metadata
        sources = system_output.get("sources", [])
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict) and "policy" in source:
                    source_id = source.get("id", source.get("doc_id", ""))
                    try:
                        policies[str(source_id)] = SourcePolicy(source["policy"])
                    except ValueError:
                        pass

        return policies

    def _extract_response(self, system_output: dict[str, Any]) -> str:
        """Extract response text from system output."""
        for field_name in ["response", "answer", "generated_text", "output"]:
            if field_name in system_output:
                value = system_output[field_name]
                if isinstance(value, str):
                    return value
        return ""

    def _extract_sources(self, system_output: dict[str, Any]) -> dict[str, Any]:
        """Extract sources from system output."""
        sources: dict[str, Any] = {}

        source_data = system_output.get("sources") or system_output.get("retrieved_chunks") or []

        if isinstance(source_data, list):
            for i, source in enumerate(source_data):
                if isinstance(source, dict):
                    source_id = str(source.get("id") or source.get("doc_id") or (i + 1))
                    sources[source_id] = source
                else:
                    sources[str(i + 1)] = {"text": str(source)}

        return sources

    def _is_source_used(self, response: str, source_text: str) -> bool:
        """Check if source content appears in response."""
        # Normalize texts
        response_lower = response.lower()
        source_lower = source_text.lower()

        # Check for substantial overlap
        source_words = set(source_lower.split())
        response_words = set(response_lower.split())

        # Overlap threshold
        overlap = len(source_words & response_words)
        return overlap > len(source_words) * 0.3

    async def _detect_paraphrasing(
        self,
        response: str,
        source_text: str,
        context: EvalContext | None,
    ) -> list[dict[str, str]]:
        """
        Detect paraphrasing of source content.

        Returns list of paraphrased segments.
        """
        paraphrases: list[dict[str, str]] = []

        # Find quoted text in response
        quoted_segments = re.findall(r'"([^"]+)"', response)
        quoted_segments += re.findall(r"'([^']+)'", response)
        quoted_set = set(q.lower() for q in quoted_segments)

        # Extract sentences from response
        response_sentences = re.split(r"[.!?]+", response)

        # Check each sentence for similarity to source without being a quote
        source_lower = source_text.lower()

        for sentence in response_sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            # Strip leading/trailing quote marks from sentence for comparison
            sentence_clean = sentence.strip("\"'")
            sentence_lower = sentence_clean.lower()

            # Skip if it's a direct quote (content matches quoted content)
            if sentence_lower in quoted_set:
                continue

            # Also skip if the original sentence was wrapped in quotes
            if (sentence.startswith('"') and sentence.endswith('"')) or (
                sentence.startswith("'") and sentence.endswith("'")
            ):
                continue

            # Check similarity to source
            sentence_words = set(sentence_lower.split())
            source_words = set(source_lower.split())

            overlap = len(sentence_words & source_words)
            overlap_ratio = overlap / len(sentence_words) if sentence_words else 0

            # High overlap but not a quote = paraphrase
            if 0.4 < overlap_ratio < self.quote_similarity_threshold:
                paraphrases.append(
                    {
                        "sentence": sentence[:100],
                        "overlap_ratio": round(overlap_ratio, 2),
                    }
                )

        return paraphrases

    def _skip_result(self, reason: str, start_time: float) -> StageResult:
        """Create a skip result."""
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return StageResult(
            stage_name=self.name,
            passed=True,
            score=1.0,
            reason=f"Skipped - {reason}",
            metrics={"skipped": True, "reason": reason},
            duration_ms=duration_ms,
        )

    def _build_reason(
        self,
        violations: list,
        checks_performed: int,
        passed: bool,
    ) -> str:
        """Build human-readable explanation."""
        if passed:
            return f"Source policy compliance verified ({checks_performed} checks)"
        else:
            violation_types = set(v["policy"] for v in violations)
            return (
                f"Source policy violation: "
                f"{len(violations)} violations ({', '.join(violation_types)})"
            )
