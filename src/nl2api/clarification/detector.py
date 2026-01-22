"""
Ambiguity Detector

Detects ambiguous queries and generates clarifying questions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.nl2api.models import ClarificationQuestion
from src.nl2api.llm.protocols import LLMMessage, LLMProvider, MessageRole

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AmbiguityAnalysis:
    """
    Result of ambiguity analysis for a query.
    """

    is_ambiguous: bool
    ambiguity_types: tuple[str, ...] = ()
    clarification_questions: tuple[ClarificationQuestion, ...] = ()
    confidence: float = 1.0


class AmbiguityDetector:
    """
    Detects ambiguity in natural language queries.

    Identifies various types of ambiguity:
    - Entity ambiguity (which company?)
    - Time period ambiguity (which quarter/year?)
    - Metric ambiguity (which specific metric?)
    - Domain ambiguity (which API?)
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        use_llm_detection: bool = False,
    ):
        """
        Initialize the ambiguity detector.

        Args:
            llm: Optional LLM for advanced ambiguity detection
            use_llm_detection: Whether to use LLM for detection
        """
        self._llm = llm
        self._use_llm_detection = use_llm_detection and llm is not None

        # Ambiguous time references
        self._ambiguous_time_patterns = [
            r'\b(recent|latest|current|now)\b',
            r'\bthis (year|quarter|month)\b',
            r'\blast (year|quarter|month)\b',
            r'\bnext (year|quarter|month)\b',
        ]

        # Vague metric references
        self._vague_metric_patterns = [
            r'\b(performance|results|metrics|numbers)\b',
            r'\b(how is|how\'s|how are|how did)\b',
            r'\b(doing|performing|going)\b',
        ]

    async def analyze(
        self,
        query: str,
        resolved_entities: dict[str, str] | None = None,
    ) -> AmbiguityAnalysis:
        """
        Analyze a query for ambiguity.

        Args:
            query: Natural language query
            resolved_entities: Already resolved entities

        Returns:
            AmbiguityAnalysis with detected ambiguities
        """
        ambiguity_types = []
        questions = []

        # Check for entity ambiguity
        entity_ambiguity = self._check_entity_ambiguity(query, resolved_entities)
        if entity_ambiguity:
            ambiguity_types.append("entity")
            questions.extend(entity_ambiguity)

        # Check for time period ambiguity
        time_ambiguity = self._check_time_ambiguity(query)
        if time_ambiguity:
            ambiguity_types.append("time_period")
            questions.extend(time_ambiguity)

        # Check for vague metric references
        metric_ambiguity = self._check_metric_ambiguity(query)
        if metric_ambiguity:
            ambiguity_types.append("metric")
            questions.extend(metric_ambiguity)

        # Use LLM for additional detection if enabled
        if self._use_llm_detection and self._llm:
            llm_analysis = await self._llm_analyze(query)
            if llm_analysis:
                for q in llm_analysis:
                    if q not in questions:
                        questions.append(q)
                        ambiguity_types.append("llm_detected")

        is_ambiguous = len(questions) > 0

        return AmbiguityAnalysis(
            is_ambiguous=is_ambiguous,
            ambiguity_types=tuple(set(ambiguity_types)),
            clarification_questions=tuple(questions),
            confidence=0.8 if is_ambiguous else 1.0,
        )

    def _check_entity_ambiguity(
        self,
        query: str,
        resolved_entities: dict[str, str] | None,
    ) -> list[ClarificationQuestion]:
        """Check for entity-related ambiguity."""
        questions = []

        # If no entities resolved but query seems to reference a company
        company_indicators = [
            r'\b(company|firm|stock|ticker|corporation)\b',
            r'\b(their|its|the company\'s)\b',
        ]

        has_company_reference = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in company_indicators
        )

        if has_company_reference and not resolved_entities:
            questions.append(ClarificationQuestion(
                question="Which company are you asking about?",
                category="entity",
            ))

        # Check for pronouns without clear antecedent
        pronoun_pattern = r'\b(it|they|them|their|its)\b'
        if re.search(pronoun_pattern, query, re.IGNORECASE) and not resolved_entities:
            questions.append(ClarificationQuestion(
                question="Could you specify which company or entity you're referring to?",
                category="entity",
            ))

        return questions

    def _check_time_ambiguity(self, query: str) -> list[ClarificationQuestion]:
        """Check for time period ambiguity."""
        questions = []

        # Check for ambiguous time references
        for pattern in self._ambiguous_time_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                # Don't flag if there's also a specific date/period
                specific_time_pattern = r'\b(20\d{2}|Q[1-4]|FY\d{2,4}|January|February|March|April|May|June|July|August|September|October|November|December)\b'
                if not re.search(specific_time_pattern, query, re.IGNORECASE):
                    questions.append(ClarificationQuestion(
                        question="Could you specify the time period? (e.g., Q4 2023, FY2024, or a specific date range)",
                        options=("Current quarter", "Current year", "Last year", "Next year"),
                        category="time_period",
                    ))
                    break  # Only ask once

        return questions

    def _check_metric_ambiguity(self, query: str) -> list[ClarificationQuestion]:
        """Check for vague metric references."""
        questions = []

        for pattern in self._vague_metric_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                # Check if there's a specific metric mentioned
                specific_metrics = [
                    r'\b(EPS|earnings per share)\b',
                    r'\b(revenue|sales)\b',
                    r'\b(profit|income|earnings)\b',
                    r'\b(price|stock price|share price)\b',
                    r'\b(P/E|PE ratio|price.to.earnings)\b',
                    r'\b(dividend|yield)\b',
                    r'\b(market cap|capitalization)\b',
                ]

                has_specific = any(
                    re.search(m, query, re.IGNORECASE)
                    for m in specific_metrics
                )

                if not has_specific:
                    questions.append(ClarificationQuestion(
                        question="What specific metrics or data are you looking for?",
                        options=(
                            "Earnings (EPS)",
                            "Revenue",
                            "Stock price",
                            "Analyst estimates",
                        ),
                        category="metric",
                    ))
                    break

        return questions

    async def _llm_analyze(
        self,
        query: str,
    ) -> list[ClarificationQuestion]:
        """Use LLM to detect additional ambiguity."""
        if not self._llm:
            return []

        prompt = f"""Analyze this financial data query for ambiguity.

Query: "{query}"

If the query is clear and unambiguous, respond with: CLEAR

If the query is ambiguous, respond with a clarifying question in this format:
AMBIGUOUS: [Your clarifying question here]

Only flag truly ambiguous queries that would prevent accurate API call generation.
Do not flag queries that are simply broad or general."""

        try:
            response = await self._llm.complete(
                messages=[
                    LLMMessage(role=MessageRole.USER, content=prompt),
                ],
                temperature=0.0,
                max_tokens=100,
            )

            content = response.content.strip()

            if content.startswith("AMBIGUOUS:"):
                question = content.replace("AMBIGUOUS:", "").strip()
                return [ClarificationQuestion(
                    question=question,
                    category="llm_detected",
                )]

        except Exception as e:
            logger.warning(f"LLM ambiguity detection failed: {e}")

        return []
