"""
Mock Entity Resolver

Provides a mock implementation of EntityResolver for testing
without calling external APIs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.nl2api.resolution.protocols import ResolvedEntity

logger = logging.getLogger(__name__)


class MockEntityResolver:
    """
    Mock entity resolver for testing.

    Uses a comprehensive mapping of company names to RIC codes
    without requiring external API calls.
    """

    def __init__(
        self,
        additional_mappings: dict[str, str] | None = None,
    ):
        """
        Initialize the mock resolver.

        Args:
            additional_mappings: Optional additional company -> RIC mappings
        """
        # Comprehensive company name to RIC mappings
        self._mappings: dict[str, str] = {
            # Big Tech
            "apple": "AAPL.O",
            "apple inc": "AAPL.O",
            "microsoft": "MSFT.O",
            "microsoft corp": "MSFT.O",
            "google": "GOOGL.O",
            "alphabet": "GOOGL.O",
            "alphabet inc": "GOOGL.O",
            "amazon": "AMZN.O",
            "amazon.com": "AMZN.O",
            "meta": "META.O",
            "meta platforms": "META.O",
            "facebook": "META.O",
            "tesla": "TSLA.O",
            "tesla inc": "TSLA.O",
            "nvidia": "NVDA.O",
            "netflix": "NFLX.O",
            # Financial
            "jpmorgan": "JPM.N",
            "jp morgan": "JPM.N",
            "jpmorgan chase": "JPM.N",
            "goldman sachs": "GS.N",
            "goldman": "GS.N",
            "bank of america": "BAC.N",
            "bofa": "BAC.N",
            "wells fargo": "WFC.N",
            "citigroup": "C.N",
            "citi": "C.N",
            "morgan stanley": "MS.N",
            "american express": "AXP.N",
            "amex": "AXP.N",
            "visa": "V.N",
            "mastercard": "MA.N",
            "berkshire hathaway": "BRK.A",
            "berkshire": "BRK.A",
            # Energy
            "exxon": "XOM.N",
            "exxonmobil": "XOM.N",
            "exxon mobil": "XOM.N",
            "chevron": "CVX.N",
            "conocophillips": "COP.N",
            "shell": "SHEL.N",
            "bp": "BP.N",
            "british petroleum": "BP.N",
            # Consumer
            "walmart": "WMT.N",
            "wal-mart": "WMT.N",
            "costco": "COST.O",
            "home depot": "HD.N",
            "target": "TGT.N",
            "disney": "DIS.N",
            "walt disney": "DIS.N",
            "nike": "NKE.N",
            "starbucks": "SBUX.O",
            "mcdonald's": "MCD.N",
            "mcdonalds": "MCD.N",
            "coca-cola": "KO.N",
            "coca cola": "KO.N",
            "coke": "KO.N",
            "pepsi": "PEP.O",
            "pepsico": "PEP.O",
            # Healthcare
            "johnson & johnson": "JNJ.N",
            "j&j": "JNJ.N",
            "johnson and johnson": "JNJ.N",
            "pfizer": "PFE.N",
            "merck": "MRK.N",
            "abbvie": "ABBV.N",
            "eli lilly": "LLY.N",
            "lilly": "LLY.N",
            "unitedhealth": "UNH.N",
            "united health": "UNH.N",
            "cvs": "CVS.N",
            "cvs health": "CVS.N",
            # Technology
            "intel": "INTC.O",
            "amd": "AMD.O",
            "advanced micro devices": "AMD.O",
            "cisco": "CSCO.O",
            "cisco systems": "CSCO.O",
            "ibm": "IBM.N",
            "oracle": "ORCL.N",
            "salesforce": "CRM.N",
            "adobe": "ADBE.O",
            "qualcomm": "QCOM.O",
            "broadcom": "AVGO.O",
            "texas instruments": "TXN.O",
            "servicenow": "NOW.N",
            "snowflake": "SNOW.N",
            "palantir": "PLTR.N",
            "crowdstrike": "CRWD.O",
            "zoom": "ZM.O",
            "zoom video": "ZM.O",
            # Industrial
            "procter & gamble": "PG.N",
            "p&g": "PG.N",
            "procter and gamble": "PG.N",
            "3m": "MMM.N",
            "caterpillar": "CAT.N",
            "boeing": "BA.N",
            "lockheed martin": "LMT.N",
            "raytheon": "RTX.N",
            "general electric": "GE.N",
            "ge": "GE.N",
            "honeywell": "HON.O",
            "union pacific": "UNP.N",
            "ups": "UPS.N",
            "fedex": "FDX.N",
            # Automotive
            "ford": "F.N",
            "ford motor": "F.N",
            "general motors": "GM.N",
            "gm": "GM.N",
            "rivian": "RIVN.O",
            "lucid": "LCID.O",
            "lucid motors": "LCID.O",
            # Telecom
            "at&t": "T.N",
            "verizon": "VZ.N",
            "t-mobile": "TMUS.O",
            "comcast": "CMCSA.O",
            # Real Estate
            "american tower": "AMT.N",
            "prologis": "PLD.N",
            "simon property": "SPG.N",
            # UK Companies
            "barclays": "BARC.L",
            "hsbc": "HSBA.L",
            "vodafone": "VOD.L",
            "astrazeneca": "AZN.L",
            "unilever": "ULVR.L",
            "glaxosmithkline": "GSK.L",
            "gsk": "GSK.L",
            "rio tinto": "RIO.L",
            "bp": "BP.L",
            # German Companies
            "volkswagen": "VOWG.DE",
            "vw": "VOWG.DE",
            "bmw": "BMW.DE",
            "siemens": "SIE.DE",
            "sap": "SAP.DE",
            "basf": "BAS.DE",
            "bayer": "BAYN.DE",
            "deutsche bank": "DBK.DE",
            # Japanese Companies
            "toyota": "7203.T",
            "sony": "6758.T",
            "honda": "7267.T",
            "nintendo": "7974.T",
            "softbank": "9984.T",
        }

        # Add any additional mappings
        if additional_mappings:
            self._mappings.update(additional_mappings)

        # Cache for resolved entities
        self._cache: dict[str, ResolvedEntity] = {}

    async def resolve(
        self,
        query: str,
    ) -> dict[str, str]:
        """
        Extract and resolve entities from a query.

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
        normalized = re.sub(r'\s+(inc\.?|corp\.?|corporation|ltd\.?|llc|plc)$', '', normalized, flags=re.I)
        normalized = normalized.strip()

        # Check cache
        if normalized in self._cache:
            return self._cache[normalized]

        # Check mappings
        if normalized in self._mappings:
            result = ResolvedEntity(
                original=entity,
                identifier=self._mappings[normalized],
                entity_type=entity_type or "company",
                confidence=0.95,
            )
            self._cache[normalized] = result
            return result

        # Try partial matches
        for key, ric in self._mappings.items():
            if key in normalized or normalized in key:
                result = ResolvedEntity(
                    original=entity,
                    identifier=ric,
                    entity_type=entity_type or "company",
                    confidence=0.8,
                )
                self._cache[normalized] = result
                return result

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
        Extract potential company names from a query.

        Args:
            query: Natural language query

        Returns:
            List of potential entity names
        """
        entities = []

        # Pattern 1: Known company names
        query_lower = query.lower()
        for company in self._mappings.keys():
            if company in query_lower:
                # Find the actual casing in the query
                pattern = re.compile(re.escape(company), re.IGNORECASE)
                match = pattern.search(query)
                if match:
                    entities.append(match.group())

        # Pattern 2: Capitalized words (potential company names)
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+(?:&\s+)?[A-Z][a-z]+)*(?:\s+(?:Inc\.?|Corp\.?|Ltd\.?))?)\b'
        matches = re.findall(cap_pattern, query)
        for match in matches:
            # Check if this might be a company name
            match_lower = match.lower()
            if match_lower in self._mappings or any(k in match_lower for k in self._mappings.keys()):
                entities.append(match)

        # Pattern 3: Possessive forms (Apple's, Microsoft's)
        possessive_pattern = r"([A-Z][a-z]+(?:'s)?)"
        possessive_matches = re.findall(possessive_pattern, query)
        for match in possessive_matches:
            clean = match.rstrip("'s")
            if clean.lower() in self._mappings:
                entities.append(clean)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            normalized = entity.lower().rstrip("'s")
            if normalized not in seen:
                seen.add(normalized)
                unique_entities.append(entity.rstrip("'s"))

        return unique_entities

    def add_mapping(self, company: str, ric: str) -> None:
        """
        Add a company to RIC mapping.

        Args:
            company: Company name (will be normalized to lowercase)
            ric: RIC code
        """
        self._mappings[company.lower()] = ric

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self._cache.clear()
