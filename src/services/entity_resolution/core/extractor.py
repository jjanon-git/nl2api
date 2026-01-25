"""
Entity Extraction

Extract potential entity names from natural language queries.
"""

from __future__ import annotations

import re


class EntityExtractor:
    """
    Extracts potential company/entity names from queries.

    Uses regex patterns to identify company names, tickers,
    and other financial entity references.
    """

    def __init__(self) -> None:
        # Common words that should not be treated as companies
        self._ignore_words = {
            "what",
            "how",
            "show",
            "get",
            "find",
            "list",
            "who",
            "when",
            "where",
            "why",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "and",
            "or",
            "for",
            "with",
            "forecast",
            "estimate",
            "eps",
            "revenue",
            "price",
            "target",
            "rating",
            "of",
            "in",
            "to",
            "at",
            "by",
            "from",
            "on",
            "about",
            "above",
            "below",
            "best",
            "stock",
            "stocks",
            "market",
            "data",
            "show",
            "me",
            "tell",
            "need",
            "corp",
            "inc",
            "ltd",
            "limited",
            "company",
            "corporation",
        }

    def extract(self, query: str) -> list[str]:
        """
        Extract potential company/entity names from a query.

        Args:
            query: Natural language query

        Returns:
            List of potential entity names
        """
        entities = []

        # Pattern 1: Capitalized words that might be company names
        # Matches: "Apple", "Microsoft Corporation", "JP Morgan"
        cap_pattern = r"\b([A-Z][a-z]+(?:\s+(?:&\s+)?[A-Z][a-z]+)*(?:\s+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|PLC))?)\b"
        matches = re.findall(cap_pattern, query)
        entities.extend(matches)

        # Pattern 2: Ticker-like patterns (all caps 1-5 letters)
        ticker_pattern = r"\b([A-Z]{1,5})\b"
        ticker_matches = re.findall(ticker_pattern, query)
        # Only add if they look like real tickers (not common words)
        common_words = {"I", "A", "THE", "FOR", "AND", "OR", "EPS", "PE", "ROE", "ROA"}
        for ticker in ticker_matches:
            if ticker not in common_words:
                entities.append(ticker)

        # Pattern 3: Possessive forms (case-insensitive) - strong signal of entity
        # Matches: "apple's", "Google's", "microsoft's"
        possessive_pattern = r"\b([a-zA-Z][a-zA-Z]+)(?:'s|'s)\b"
        possessive_matches = re.findall(possessive_pattern, query, re.IGNORECASE)
        for match in possessive_matches:
            # Title-case the match for consistency
            entities.append(match.title())

        # Pattern 4: Words before financial terms (case-insensitive)
        # Matches: "apple revenue", "tesla earnings", "amazon 10-k"
        financial_context_pattern = r"\b([a-zA-Z][a-zA-Z]+)\s+(?:revenue|earnings|income|profit|10-[kq]|filing|stock|shares|price)\b"
        context_matches = re.findall(financial_context_pattern, query, re.IGNORECASE)
        for match in context_matches:
            if match.lower() not in self._ignore_words:
                entities.append(match.title())

        # Deduplicate while preserving order and filter common words/noise
        return self._deduplicate(entities)

    def _deduplicate(self, entities: list[str]) -> list[str]:
        """Remove duplicates and filter noise."""
        seen = set()
        unique_entities = []

        for entity in entities:
            normalized = entity.lower().strip()

            # Skip noise (single chars, common words)
            if len(normalized) < 2 or normalized in self._ignore_words:
                continue

            # Basic normalization for check
            check_name = re.sub(
                r"\s+(inc\.?|corp\.?|ltd\.?|llc|plc)$", "", normalized, flags=re.I
            ).strip()

            if (
                check_name not in seen
                and check_name not in self._ignore_words
                and len(check_name) >= 2
            ):
                seen.add(check_name)
                unique_entities.append(entity)

        return unique_entities
