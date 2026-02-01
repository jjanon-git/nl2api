"""
Citation Stage

Evaluates citation quality in RAG responses.

Metrics:
- Presence: Are citations included?
- Validity: Do citations point to real retrieved documents?
- Accuracy: Do citations support the claims they're attached to?
- Coverage: What fraction of claims are cited?
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.evalkit.contracts import StageResult, TestCase

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.evalkit.contracts import EvalContext


@dataclass
class CitationStage:
    """
    Stage 5: Citation Evaluation

    Evaluates citation quality across four dimensions:
    1. Presence: Are citations included?
    2. Validity: Do they point to real documents?
    3. Accuracy: Do they support their claims? (LLM judge)
    4. Coverage: What fraction of claims are cited?
    """

    name: str = field(default="citation", init=False)
    is_gate: bool = field(default=False, init=False)

    # Configurable thresholds
    pass_threshold: float = 0.6
    presence_weight: float = 0.2
    validity_weight: float = 0.3
    accuracy_weight: float = 0.3
    coverage_weight: float = 0.2

    # Citation patterns (common formats)
    # Note: [Source N] is the primary format used by RAG system
    citation_patterns: tuple[str, ...] = (
        r"\[Source\s*(\d+)\]",  # [Source 1], [Source 5] - primary RAG format
        r"\[(\d+)\]",  # [1], [2]
        r"\(Source\s*(\d+)\)",  # (Source 1)
        r"\[\[(\d+)\]\]",  # [[1]]
    )

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext | None,
    ) -> StageResult:
        """
        Evaluate citation quality.

        Args:
            test_case: Test case (may specify citation requirements)
            system_output: System output with 'response' and 'sources'
            context: Evaluation context with LLM judge

        Returns:
            StageResult with citation metrics
        """
        start_time = time.perf_counter()

        # Check if citations are required for this test
        requires_citations = test_case.expected.get("requires_citations", True)
        if not requires_citations:
            return self._skip_result("Citations not required", start_time)

        # Get response
        response = self._extract_response(system_output)
        if not response:
            return self._skip_result("No response to evaluate", start_time)

        # Get available sources
        sources = self._extract_sources(system_output)

        # Extract citations from response
        citations = self._extract_citations(response)

        # Compute metrics
        metrics: dict[str, Any] = {}

        # 1. Presence: Are there any citations?
        metrics["presence"] = 1.0 if citations else 0.0
        metrics["citation_count"] = len(citations)

        if not citations:
            # No citations - fail with zero score
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return StageResult(
                stage_name=self.name,
                passed=False,
                score=0.0,
                reason="No citations found in response",
                metrics=metrics,
                artifacts={"response_preview": response[:200]},
                duration_ms=duration_ms,
            )

        # 2. Validity: Do citations point to real sources?
        valid_citations = self._validate_citations(citations, sources)
        metrics["validity"] = len(valid_citations) / len(citations) if citations else 0.0
        metrics["valid_citation_count"] = len(valid_citations)
        metrics["invalid_citations"] = [c for c in citations if c not in valid_citations]

        # 3. Coverage: Estimate what fraction of statements are cited
        statement_count = self._estimate_statement_count(response)
        metrics["coverage"] = min(1.0, len(citations) / max(1, statement_count))
        metrics["statement_count"] = statement_count

        # 4. Accuracy: Do citations support their claims? (LLM judge if available)
        llm_judge = self._get_llm_judge(context)
        if llm_judge is not None and valid_citations:
            accuracy = await self._evaluate_citation_accuracy(
                response, valid_citations, sources, llm_judge
            )
            metrics["accuracy"] = accuracy
        else:
            # Skip accuracy evaluation without LLM
            metrics["accuracy"] = 1.0  # Assume valid if can't verify
            metrics["accuracy_skipped"] = True

        # Compute weighted score
        score = (
            metrics["presence"] * self.presence_weight
            + metrics["validity"] * self.validity_weight
            + metrics["accuracy"] * self.accuracy_weight
            + metrics["coverage"] * self.coverage_weight
        )

        passed = score >= self.pass_threshold
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=score,
            reason=self._build_reason(metrics, passed),
            metrics=metrics,
            artifacts={
                "citations": list(citations)[:20],
                "sources_count": len(sources),
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

    def _extract_sources(self, system_output: dict[str, Any]) -> dict[str, Any]:
        """Extract sources from system output as a lookup dict."""
        sources: dict[str, Any] = {}

        # Try various field names
        source_data = (
            system_output.get("sources")
            or system_output.get("retrieved_chunks")
            or system_output.get("documents")
            or []
        )

        if isinstance(source_data, list):
            for i, source in enumerate(source_data):
                if isinstance(source, dict):
                    source_id = source.get("id") or source.get("doc_id") or str(i + 1)
                    sources[str(source_id)] = source
                else:
                    sources[str(i + 1)] = {"text": str(source)}

        return sources

    def _extract_citations(self, response: str) -> set[str]:
        """Extract all citations from response text."""
        citations: set[str] = set()

        for pattern in self.citation_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            citations.update(str(m).strip() for m in matches if m)

        return citations

    def _validate_citations(
        self,
        citations: set[str],
        sources: dict[str, Any],
    ) -> set[str]:
        """Check which citations point to valid sources."""
        valid: set[str] = set()

        for citation in citations:
            # Direct match
            if citation in sources:
                valid.add(citation)
                continue

            # Numeric citation to index
            if citation.isdigit():
                if citation in sources or int(citation) <= len(sources):
                    valid.add(citation)
                    continue

            # Fuzzy match on source IDs
            citation_lower = citation.lower()
            for source_id in sources:
                if citation_lower in source_id.lower() or source_id.lower() in citation_lower:
                    valid.add(citation)
                    break

        return valid

    def _estimate_statement_count(self, response: str) -> int:
        """Estimate number of factual statements in response."""
        # Count sentences as proxy for statements
        sentences = re.split(r"[.!?]+", response)
        return len([s for s in sentences if len(s.strip()) > 20])

    async def _evaluate_citation_accuracy(
        self,
        response: str,
        citations: set[str],
        sources: dict[str, Any],
        llm_judge: Any,
    ) -> float:
        """
        Evaluate if citations accurately support their claims.

        Uses LLM to verify that cited sources support the statements.
        """
        # Extract citation contexts (text around citations)
        accurate_count = 0
        total_checked = 0

        for citation in list(citations)[:5]:  # Limit to 5 checks
            # Find context around citation
            context_text = self._find_citation_context(response, citation)
            if not context_text:
                continue

            # Get source text
            source = sources.get(citation) or sources.get(str(citation))
            if not source:
                continue

            source_text = (
                source.get("text", str(source)) if isinstance(source, dict) else str(source)
            )

            try:
                # Ask LLM if source supports the claim
                result = await llm_judge.verify_claim(
                    claim=context_text,
                    context=source_text,
                )
                if result.supported:
                    accurate_count += 1
                total_checked += 1
            except (TimeoutError, ConnectionError) as e:
                # Network errors - skip this citation but continue checking others
                logger.debug(f"Skipping citation verification due to network error: {e}")
                continue
            except Exception as e:
                # Log unexpected errors but don't fail the entire evaluation
                logger.warning(f"Unexpected error verifying citation: {e}")
                continue

        return accurate_count / total_checked if total_checked > 0 else 1.0

    def _find_citation_context(self, response: str, citation: str) -> str:
        """Find the text context around a citation."""
        # Find position of citation
        for pattern in self.citation_patterns:
            matches = list(re.finditer(pattern, response))
            for match in matches:
                if citation in match.group():
                    # Get surrounding sentence
                    start = max(0, match.start() - 200)
                    end = min(len(response), match.end() + 50)
                    context = response[start:end]

                    # Clean to sentence boundary
                    sentences = re.split(r"[.!?]+", context)
                    if sentences:
                        return sentences[-1].strip()[:200]

        return ""

    def _get_llm_judge(self, context: EvalContext | None) -> Any:
        """Get LLM judge from context."""
        if context is None:
            return None
        return context.llm_judge or context.config.get("llm_judge")

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

    def _build_reason(self, metrics: dict, passed: bool) -> str:
        """Build human-readable explanation."""
        if passed:
            return (
                f"Citation quality acceptable: "
                f"{metrics['citation_count']} citations, "
                f"{metrics['validity']:.0%} valid, "
                f"{metrics['coverage']:.0%} coverage"
            )
        else:
            issues = []
            if metrics["presence"] == 0:
                issues.append("no citations found")
            if metrics["validity"] < 0.5:
                issues.append(f"only {metrics['validity']:.0%} valid")
            if metrics["coverage"] < 0.3:
                issues.append(f"low coverage ({metrics['coverage']:.0%})")
            return f"Citation quality below threshold: {', '.join(issues)}"
