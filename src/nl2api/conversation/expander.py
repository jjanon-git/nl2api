"""
Query Expander

Expands follow-up queries using conversation context.
Handles references like "their EPS", "now quarterly", "what about Microsoft?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.nl2api.conversation.models import ConversationContext


@dataclass
class ExpansionResult:
    """Result of query expansion."""

    original_query: str
    expanded_query: str
    was_expanded: bool
    expansion_type: str | None = None  # "entity", "metric", "period", "comparison"
    confidence: float = 1.0


class QueryExpander:
    """
    Expands follow-up queries using conversation context.

    Handles common follow-up patterns:
    - Pronoun references: "their EPS", "its revenue"
    - Period changes: "now quarterly", "for next year"
    - Comparisons: "what about Microsoft?", "compare to Google"
    - Metric changes: "what about revenue?", "and EBITDA?"
    """

    # Pronouns that refer to previous entities
    # Sorted by length (longest first) to match longer patterns before shorter ones
    # e.g., "its" before "it" to avoid partial matches
    ENTITY_PRONOUNS = [
        "the company's", "that company's", "the company",
        "that company", "this company", "the stock's",
        "the stock", "their", "they", "its", "it",
    ]

    # Patterns for period changes
    PERIOD_PATTERNS = {
        r"\bnow quarterly\b": "quarterly",
        r"\bfor quarterly\b": "quarterly",
        r"\bquarterly\b": "quarterly",
        r"\bnow annual\b": "annual",
        r"\bfor annual\b": "annual",
        r"\bannual(ly)?\b": "annual",
        r"\bnext year\b": "FY1",
        r"\bthis year\b": "FY0",
        r"\bnext quarter\b": "FQ1",
        r"\bthis quarter\b": "FQ0",
        r"\blast quarter\b": "FQ0",
        r"\blast year\b": "FY0",
    }

    # Patterns for comparison/addition
    COMPARISON_PATTERNS = [
        r"^what about (.+)\??$",
        r"^how about (.+)\??$",
        r"^and (.+)\??$",
        r"^compare (?:to |with )?(.+)$",
        r"^also (.+)$",
        r"^same for (.+)$",
        r"^now (?:for )?(.+)$",
    ]

    # Patterns for metric changes
    METRIC_PATTERNS = [
        r"^what about (.+)\??$",
        r"^and (.+)\??$",
        r"^also (.+)$",
        r"^show (?:me )?(.+)$",
        r"^get (.+)$",
    ]

    # Known metrics for detection
    KNOWN_METRICS = {
        "eps", "earnings", "revenue", "sales", "ebitda", "ebit",
        "profit", "income", "dividend", "dps", "price target",
        "rating", "recommendation", "pe", "p/e", "peg",
        "free cash flow", "fcf", "book value", "roe", "roa",
    }

    def expand(
        self,
        query: str,
        context: ConversationContext,
    ) -> ExpansionResult:
        """
        Expand a follow-up query using conversation context.

        Args:
            query: The user's follow-up query
            context: Context from previous conversation turns

        Returns:
            ExpansionResult with original and expanded query
        """
        if not context.turn_count:
            # No context, return as-is
            return ExpansionResult(
                original_query=query,
                expanded_query=query,
                was_expanded=False,
            )

        query_lower = query.lower().strip()

        # Try different expansion strategies in order
        result = self._try_pronoun_expansion(query, query_lower, context)
        if result:
            return result

        result = self._try_comparison_expansion(query, query_lower, context)
        if result:
            return result

        result = self._try_period_expansion(query, query_lower, context)
        if result:
            return result

        result = self._try_metric_expansion(query, query_lower, context)
        if result:
            return result

        # No expansion needed
        return ExpansionResult(
            original_query=query,
            expanded_query=query,
            was_expanded=False,
        )

    def _try_pronoun_expansion(
        self,
        query: str,
        query_lower: str,
        context: ConversationContext,
    ) -> ExpansionResult | None:
        """Try to expand pronoun references to entities."""
        if not context.entities:
            return None

        # Check for pronoun patterns (list is sorted longest first)
        for pronoun in self.ENTITY_PRONOUNS:
            # Use word boundaries to avoid partial matches (e.g., "it" in "its")
            pattern = re.compile(r'\b' + re.escape(pronoun) + r'\b', re.IGNORECASE)
            if pattern.search(query_lower):
                # Replace pronoun with last entity
                # Use the most recently mentioned entity
                entities = list(context.entities.keys())
                if entities:
                    last_entity = entities[-1]
                    # Smart replacement based on pronoun type
                    if pronoun.endswith("'s") or pronoun == "its":
                        replacement = f"{last_entity}'s"
                    else:
                        replacement = last_entity

                    # Case-insensitive replacement with word boundaries
                    expanded = pattern.sub(replacement, query, count=1)

                    return ExpansionResult(
                        original_query=query,
                        expanded_query=expanded,
                        was_expanded=True,
                        expansion_type="entity",
                        confidence=0.9,
                    )

        return None

    # Keywords that indicate period changes, not entity comparisons
    PERIOD_KEYWORDS = {
        "quarterly", "annual", "annually", "yearly", "monthly",
        "next year", "this year", "last year",
        "next quarter", "this quarter", "last quarter",
    }

    def _try_comparison_expansion(
        self,
        query: str,
        query_lower: str,
        context: ConversationContext,
    ) -> ExpansionResult | None:
        """Try to expand comparison queries."""
        if not context.fields:
            return None

        for pattern in self.COMPARISON_PATTERNS:
            match = re.match(pattern, query_lower)
            if match:
                # Strip trailing punctuation and whitespace
                new_entity = match.group(1).strip().rstrip("?!.,;:")

                # Check if this looks like a period keyword (not an entity)
                if new_entity.lower() in self.PERIOD_KEYWORDS:
                    continue  # Skip, let period expansion handle it

                # Check if this looks like an entity (not a metric)
                if new_entity.lower() not in self.KNOWN_METRICS:
                    # Build query with same metrics but new entity
                    # Get the metric type from context
                    metric = self._infer_metric_from_fields(context.fields)
                    if metric:
                        # Capitalize first letter of entity name
                        entity_display = new_entity[0].upper() + new_entity[1:] if new_entity else new_entity
                        expanded = f"What is {entity_display}'s {metric}?"
                        return ExpansionResult(
                            original_query=query,
                            expanded_query=expanded,
                            was_expanded=True,
                            expansion_type="comparison",
                            confidence=0.85,
                        )

        return None

    def _try_period_expansion(
        self,
        query: str,
        query_lower: str,
        context: ConversationContext,
    ) -> ExpansionResult | None:
        """Try to expand period change queries."""
        if not context.entities or not context.fields:
            return None

        for pattern, period_type in self.PERIOD_PATTERNS.items():
            if re.search(pattern, query_lower):
                # Check if query is just about period change
                # e.g., "now quarterly" or "what about quarterly?"
                if len(query_lower.split()) <= 4:
                    # Build full query with previous entity and metric
                    entity = list(context.entities.keys())[-1]
                    metric = self._infer_metric_from_fields(context.fields)
                    if metric:
                        period_text = "quarterly" if "Q" in period_type or period_type == "quarterly" else "annual"
                        expanded = f"What is {entity}'s {period_text} {metric}?"
                        return ExpansionResult(
                            original_query=query,
                            expanded_query=expanded,
                            was_expanded=True,
                            expansion_type="period",
                            confidence=0.85,
                        )

        return None

    def _try_metric_expansion(
        self,
        query: str,
        query_lower: str,
        context: ConversationContext,
    ) -> ExpansionResult | None:
        """Try to expand metric change queries."""
        if not context.entities:
            return None

        for pattern in self.METRIC_PATTERNS:
            match = re.match(pattern, query_lower)
            if match:
                potential_metric = match.group(1).strip().rstrip("?")

                # Check if this is a known metric
                if potential_metric.lower() in self.KNOWN_METRICS:
                    entity = list(context.entities.keys())[-1]
                    expanded = f"What is {entity}'s {potential_metric}?"
                    return ExpansionResult(
                        original_query=query,
                        expanded_query=expanded,
                        was_expanded=True,
                        expansion_type="metric",
                        confidence=0.85,
                    )

        return None

    def _infer_metric_from_fields(self, fields: list[str]) -> str | None:
        """Infer the metric type from field codes."""
        if not fields:
            return None

        # Look at the most recent field
        field = fields[-1].upper()

        if "EPS" in field:
            return "EPS estimate"
        elif "REVENUE" in field:
            return "revenue estimate"
        elif "EBITDA" in field:
            return "EBITDA estimate"
        elif "EBIT" in field and "EBITDA" not in field:
            return "EBIT estimate"
        elif "NETPROFIT" in field or "PROFIT" in field:
            return "net income estimate"
        elif "DPS" in field or "DIVIDEND" in field:
            return "dividend estimate"
        elif "REC" in field:
            return "analyst rating"
        elif "PRICETARGET" in field:
            return "price target"
        elif "FCF" in field:
            return "free cash flow estimate"
        elif "PE" in field or "PTOEPS" in field:
            return "forward P/E"
        elif "PEG" in field:
            return "PEG ratio"
        elif "LTG" in field:
            return "long-term growth estimate"

        return "estimate"
