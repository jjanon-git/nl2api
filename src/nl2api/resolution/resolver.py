"""
Entity Resolver Implementation

Implements entity resolution using external API.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.nl2api.resolution.protocols import ResolvedEntity

logger = logging.getLogger(__name__)


class ExternalEntityResolver:
    """
    Entity resolver using an external API for company/RIC resolution.

    Extracts company mentions from queries and resolves them to RICs
    using a configurable external service.
    """

    def __init__(
        self,
        api_endpoint: str | None = None,
        api_key: str | None = None,
        use_cache: bool = True,
    ):
        """
        Initialize the entity resolver.

        Args:
            api_endpoint: External API endpoint for resolution
            api_key: API key for authentication
            use_cache: Whether to cache resolved entities
        """
        self._api_endpoint = api_endpoint
        self._api_key = api_key
        self._use_cache = use_cache
        self._cache: dict[str, ResolvedEntity] = {}

        # Common company name to RIC mappings (fallback)
        self._common_mappings: dict[str, str] = {
            "apple": "AAPL.O",
            "microsoft": "MSFT.O",
            "google": "GOOGL.O",
            "alphabet": "GOOGL.O",
            "amazon": "AMZN.O",
            "meta": "META.O",
            "facebook": "META.O",
            "tesla": "TSLA.O",
            "nvidia": "NVDA.O",
            "jpmorgan": "JPM.N",
            "jp morgan": "JPM.N",
            "goldman sachs": "GS.N",
            "bank of america": "BAC.N",
            "wells fargo": "WFC.N",
            "exxon": "XOM.N",
            "chevron": "CVX.N",
            "walmart": "WMT.N",
            "johnson & johnson": "JNJ.N",
            "j&j": "JNJ.N",
            "procter & gamble": "PG.N",
            "p&g": "PG.N",
            "coca-cola": "KO.N",
            "pepsi": "PEP.O",
            "pepsico": "PEP.O",
            "disney": "DIS.N",
            "netflix": "NFLX.O",
            "adobe": "ADBE.O",
            "salesforce": "CRM.N",
            "oracle": "ORCL.N",
            "intel": "INTC.O",
            "amd": "AMD.O",
            "cisco": "CSCO.O",
            "ibm": "IBM.N",
        }

    async def resolve(
        self,
        query: str,
    ) -> dict[str, str]:
        """
        Extract and resolve entities from a query.

        Uses regex patterns to identify potential company names,
        then resolves them to RICs.

        Args:
            query: Natural language query

        Returns:
            Dictionary mapping entity names to RICs
        """
        entities = self._extract_entities(query)
        resolved: dict[str, str] = {}

        for entity in entities:
            result = await self.resolve_single(entity)
            if result:
                resolved[result.original] = result.identifier

        return resolved

    async def resolve_single(
        self,
        entity: str,
        entity_type: str | None = None,
    ) -> ResolvedEntity | None:
        """
        Resolve a single entity to its identifier.

        Args:
            entity: Entity name (e.g., "Apple Inc.")
            entity_type: Optional type hint

        Returns:
            ResolvedEntity if found
        """
        # Normalize entity name
        normalized = entity.lower().strip()
        normalized = re.sub(r'\s+(inc\.?|corp\.?|ltd\.?|llc|plc)$', '', normalized, flags=re.I)
        normalized = normalized.strip()

        # Check cache
        if self._use_cache and normalized in self._cache:
            return self._cache[normalized]

        # Check common mappings
        if normalized in self._common_mappings:
            result = ResolvedEntity(
                original=entity,
                identifier=self._common_mappings[normalized],
                entity_type="company",
                confidence=0.95,
            )
            if self._use_cache:
                self._cache[normalized] = result
            return result

        # Try external API if configured
        if self._api_endpoint:
            result = await self._resolve_via_api(entity)
            if result:
                if self._use_cache:
                    self._cache[normalized] = result
                return result

        # No resolution found
        logger.debug(f"Could not resolve entity: {entity}")
        return None

    async def resolve_batch(
        self,
        entities: list[str],
    ) -> list[ResolvedEntity]:
        """
        Resolve multiple entities in batch.

        Args:
            entities: List of entity names

        Returns:
            List of resolved entities (may be shorter than input)
        """
        results = []
        for entity in entities:
            result = await self.resolve_single(entity)
            if result:
                results.append(result)
        return results

    def _extract_entities(self, query: str) -> list[str]:
        """
        Extract potential company/entity names from a query.

        Uses patterns to identify company names.

        Args:
            query: Natural language query

        Returns:
            List of potential entity names
        """
        entities = []

        # Pattern 1: Capitalized words that might be company names
        # Matches: "Apple", "Microsoft Corporation", "JP Morgan"
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+(?:&\s+)?[A-Z][a-z]+)*(?:\s+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|PLC))?)\b'
        matches = re.findall(cap_pattern, query)
        entities.extend(matches)

        # Pattern 2: Known company name patterns
        for company in self._common_mappings.keys():
            if company.lower() in query.lower():
                # Find the actual casing in the query
                pattern = re.compile(re.escape(company), re.IGNORECASE)
                match = pattern.search(query)
                if match:
                    entities.append(match.group())

        # Pattern 3: Ticker-like patterns (all caps 1-5 letters)
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        ticker_matches = re.findall(ticker_pattern, query)
        # Only add if they look like real tickers (not common words)
        common_words = {"I", "A", "THE", "FOR", "AND", "OR", "EPS", "PE", "ROE", "ROA"}
        for ticker in ticker_matches:
            if ticker not in common_words:
                entities.append(ticker)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            normalized = entity.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_entities.append(entity)

        return unique_entities

    async def _resolve_via_api(self, entity: str) -> ResolvedEntity | None:
        """
        Resolve entity using external API.

        Args:
            entity: Entity name to resolve

        Returns:
            ResolvedEntity if found
        """
        if not self._api_endpoint:
            return None

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"

                async with session.get(
                    f"{self._api_endpoint}/resolve",
                    params={"entity": entity},
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("found"):
                            return ResolvedEntity(
                                original=entity,
                                identifier=data["identifier"],
                                entity_type=data.get("type", "company"),
                                confidence=data.get("confidence", 0.8),
                                alternatives=tuple(data.get("alternatives", [])),
                            )
        except ImportError:
            logger.warning("aiohttp not installed, cannot use external API")
        except Exception as e:
            logger.warning(f"Error resolving entity via API: {e}")

        return None
