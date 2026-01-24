# P0.2: Entity Resolution Coverage

**Priority:** P0 (Critical)
**Effort:** 1 week
**Status:** ✅ Completed (Phases 1-2 implemented)

---

## Problem Statement

The current entity resolution system has critical limitations:

1. **Limited Static Mappings:** Only ~30 companies hardcoded in `_common_mappings`
2. **External API Not Wired:** `api_endpoint` parameter exists but is never configured
3. **No Fuzzy Matching:** "JPMorgan Chase" won't match "JP Morgan"
4. **No Ticker Resolution:** Direct ticker inputs (e.g., "AAPL") not handled well

This directly impacts accuracy - queries mentioning companies outside the top 30 fail to resolve entities.

---

## Current State Analysis

**File:** `src/nl2api/resolution/resolver.py`

```python
# Current mappings (lines 94-128)
self._common_mappings: dict[str, str] = {
    "apple": "AAPL.O",
    "microsoft": "MSFT.O",
    # ... only 28 companies total
}
```

**Coverage Gap:**
- S&P 500 has 500+ companies
- Current coverage: ~6% of S&P 500
- Missing: Most financial companies, healthcare, industrials

---

## Goals

1. Expand static mappings to 500+ companies (S&P 500 + major global)
2. Add fuzzy matching for company name variants
3. Wire external API with OpenFIGI or LSEG PermID
4. Add ticker-to-RIC resolution
5. Improve extraction patterns

---

## Implementation Plan

### Phase 1: Expand Static Mappings (2-3 days)

#### 1.1 Create Company Mappings Data File

**File:** `src/nl2api/resolution/data/company_mappings.json`

```json
{
  "version": "1.0.0",
  "updated": "2026-01-20",
  "mappings": {
    "apple": {"ric": "AAPL.O", "aliases": ["apple inc", "apple computer"]},
    "microsoft": {"ric": "MSFT.O", "aliases": ["microsoft corp", "msft"]},
    "jpmorgan": {"ric": "JPM.N", "aliases": ["jp morgan", "jpmorgan chase", "jpm", "chase"]},
    "berkshire hathaway": {"ric": "BRKa.N", "aliases": ["berkshire", "brk"]},
    ...
  },
  "tickers": {
    "AAPL": "AAPL.O",
    "MSFT": "MSFT.O",
    "JPM": "JPM.N",
    ...
  }
}
```

#### 1.2 Create Mapping Loader

**File:** `src/nl2api/resolution/mappings.py`

```python
"""Company name to RIC mappings loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAPPINGS_FILE = Path(__file__).parent / "data" / "company_mappings.json"
_mappings_cache: dict[str, Any] | None = None


def load_mappings() -> dict[str, Any]:
    """Load company mappings from JSON file."""
    global _mappings_cache

    if _mappings_cache is not None:
        return _mappings_cache

    if not _MAPPINGS_FILE.exists():
        logger.warning(f"Mappings file not found: {_MAPPINGS_FILE}")
        return {"mappings": {}, "tickers": {}}

    with open(_MAPPINGS_FILE) as f:
        _mappings_cache = json.load(f)

    logger.info(f"Loaded {len(_mappings_cache.get('mappings', {}))} company mappings")
    return _mappings_cache


def get_ric_for_company(name: str) -> str | None:
    """
    Get RIC for a company name.

    Checks primary name, then aliases.

    Args:
        name: Normalized company name (lowercase, stripped)

    Returns:
        RIC if found, None otherwise
    """
    mappings = load_mappings()

    # Direct lookup
    if name in mappings["mappings"]:
        return mappings["mappings"][name]["ric"]

    # Check aliases
    for company, data in mappings["mappings"].items():
        if name in data.get("aliases", []):
            return data["ric"]

    return None


def get_ric_for_ticker(ticker: str) -> str | None:
    """
    Get RIC for a ticker symbol.

    Args:
        ticker: Ticker symbol (e.g., "AAPL")

    Returns:
        RIC if found, None otherwise
    """
    mappings = load_mappings()
    return mappings.get("tickers", {}).get(ticker.upper())


def get_all_known_names() -> set[str]:
    """Get all known company names and aliases for pattern matching."""
    mappings = load_mappings()
    names = set()

    for company, data in mappings.get("mappings", {}).items():
        names.add(company)
        names.update(data.get("aliases", []))

    return names
```

#### 1.3 Generate Mappings Data

**Script:** `scripts/generate_company_mappings.py`

```python
"""
Generate company_mappings.json from S&P 500 and major global companies.

Sources:
- S&P 500 constituents
- FTSE 100
- Major tech, finance, healthcare companies

Run: python scripts/generate_company_mappings.py
"""

import json
from pathlib import Path

# S&P 500 + major global companies
# Format: (primary_name, ric, aliases)
COMPANIES = [
    # Technology
    ("apple", "AAPL.O", ["apple inc", "apple computer"]),
    ("microsoft", "MSFT.O", ["microsoft corp", "msft"]),
    ("alphabet", "GOOGL.O", ["google", "goog"]),
    ("amazon", "AMZN.O", ["amazon.com", "amzn"]),
    ("meta", "META.O", ["facebook", "meta platforms", "fb"]),
    ("nvidia", "NVDA.O", ["nvidia corp"]),
    ("tesla", "TSLA.O", ["tesla motors", "tsla"]),
    ("broadcom", "AVGO.O", []),
    ("oracle", "ORCL.N", ["oracle corp"]),
    ("salesforce", "CRM.N", ["salesforce.com"]),
    ("adobe", "ADBE.O", ["adobe systems"]),
    ("cisco", "CSCO.O", ["cisco systems"]),
    ("intel", "INTC.O", ["intel corp"]),
    ("amd", "AMD.O", ["advanced micro devices"]),
    ("ibm", "IBM.N", ["international business machines"]),
    ("qualcomm", "QCOM.O", []),
    ("texas instruments", "TXN.O", ["ti"]),
    ("applied materials", "AMAT.O", []),
    ("intuit", "INTU.O", []),
    ("servicenow", "NOW.N", []),

    # Finance
    ("jpmorgan", "JPM.N", ["jp morgan", "jpmorgan chase", "jpm", "chase"]),
    ("bank of america", "BAC.N", ["bofa", "boa"]),
    ("wells fargo", "WFC.N", []),
    ("goldman sachs", "GS.N", ["goldman"]),
    ("morgan stanley", "MS.N", []),
    ("citigroup", "C.N", ["citi", "citibank"]),
    ("blackrock", "BLK.N", []),
    ("charles schwab", "SCHW.N", ["schwab"]),
    ("american express", "AXP.N", ["amex"]),
    ("visa", "V.N", []),
    ("mastercard", "MA.N", []),
    ("paypal", "PYPL.O", []),
    ("s&p global", "SPGI.N", ["s&p", "standard and poors"]),
    ("moody's", "MCO.N", ["moodys"]),
    ("cme group", "CME.O", ["cme"]),
    ("intercontinental exchange", "ICE.N", ["ice"]),
    ("nasdaq", "NDAQ.O", []),

    # Healthcare
    ("unitedhealth", "UNH.N", ["unitedhealth group", "united health"]),
    ("johnson & johnson", "JNJ.N", ["j&j", "jnj"]),
    ("eli lilly", "LLY.N", ["lilly"]),
    ("pfizer", "PFE.N", []),
    ("abbvie", "ABBV.N", []),
    ("merck", "MRK.N", ["merck & co"]),
    ("thermo fisher", "TMO.N", ["thermo fisher scientific"]),
    ("abbott", "ABT.N", ["abbott laboratories"]),
    ("danaher", "DHR.N", []),
    ("amgen", "AMGN.O", []),
    ("gilead", "GILD.O", ["gilead sciences"]),
    ("regeneron", "REGN.O", []),
    ("vertex", "VRTX.O", ["vertex pharmaceuticals"]),
    ("moderna", "MRNA.O", []),
    ("intuitive surgical", "ISRG.O", []),
    ("boston scientific", "BSX.N", []),
    ("medtronic", "MDT.N", []),
    ("stryker", "SYK.N", []),
    ("cigna", "CI.N", []),
    ("cvs health", "CVS.N", ["cvs"]),

    # Consumer
    ("walmart", "WMT.N", ["wal-mart"]),
    ("amazon", "AMZN.O", ["amazon.com"]),
    ("costco", "COST.O", []),
    ("home depot", "HD.N", []),
    ("procter & gamble", "PG.N", ["p&g", "procter and gamble"]),
    ("coca-cola", "KO.N", ["coke"]),
    ("pepsico", "PEP.O", ["pepsi"]),
    ("nike", "NKE.N", []),
    ("mcdonald's", "MCD.N", ["mcdonalds"]),
    ("starbucks", "SBUX.O", []),
    ("disney", "DIS.N", ["walt disney"]),
    ("netflix", "NFLX.O", []),
    ("booking holdings", "BKNG.O", ["booking.com", "priceline"]),
    ("lowe's", "LOW.N", ["lowes"]),
    ("target", "TGT.N", []),

    # Industrials
    ("boeing", "BA.N", []),
    ("caterpillar", "CAT.N", ["cat"]),
    ("general electric", "GE.N", ["ge"]),
    ("honeywell", "HON.O", []),
    ("3m", "MMM.N", []),
    ("lockheed martin", "LMT.N", ["lockheed"]),
    ("raytheon", "RTX.N", ["rtx"]),
    ("union pacific", "UNP.N", []),
    ("ups", "UPS.N", ["united parcel service"]),
    ("fedex", "FDX.N", []),
    ("deere", "DE.N", ["john deere"]),

    # Energy
    ("exxon", "XOM.N", ["exxon mobil", "exxonmobil"]),
    ("chevron", "CVX.N", []),
    ("conocophillips", "COP.N", []),
    ("schlumberger", "SLB.N", []),
    ("eog resources", "EOG.N", ["eog"]),
    ("pioneer natural resources", "PXD.N", ["pioneer"]),

    # Telecom/Media
    ("at&t", "T.N", ["att"]),
    ("verizon", "VZ.N", []),
    ("t-mobile", "TMUS.O", ["tmobile"]),
    ("comcast", "CMCSA.O", []),
    ("charter", "CHTR.O", ["charter communications"]),

    # Utilities
    ("nextera energy", "NEE.N", ["nextera"]),
    ("duke energy", "DUK.N", ["duke"]),
    ("southern company", "SO.N", ["southern"]),

    # Real Estate
    ("american tower", "AMT.N", []),
    ("prologis", "PLD.N", []),
    ("equinix", "EQIX.O", []),

    # International
    ("berkshire hathaway", "BRKa.N", ["berkshire", "brk"]),
    ("toyota", "7203.T", ["toyota motor"]),
    ("samsung", "005930.KS", ["samsung electronics"]),
    ("nestle", "NESN.S", []),
    ("lvmh", "LVMH.PA", []),
    ("novo nordisk", "NOVOb.CO", []),
    ("asml", "ASML.AS", []),
    ("shell", "SHEL.L", ["royal dutch shell"]),
    ("bp", "BP.L", ["british petroleum"]),
    ("hsbc", "HSBA.L", []),
]


def generate_mappings():
    """Generate the mappings JSON file."""
    mappings = {}
    tickers = {}

    for primary, ric, aliases in COMPANIES:
        # Extract ticker from RIC
        ticker = ric.split(".")[0]

        mappings[primary] = {
            "ric": ric,
            "aliases": aliases,
        }

        # Add ticker mapping
        tickers[ticker] = ric

        # Also add uppercase ticker as alias
        if ticker.upper() not in [a.upper() for a in aliases]:
            mappings[primary]["aliases"].append(ticker.lower())

    data = {
        "version": "1.0.0",
        "updated": "2026-01-20",
        "company_count": len(mappings),
        "mappings": mappings,
        "tickers": tickers,
    }

    output_path = Path("src/nl2api/resolution/data/company_mappings.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(mappings)} company mappings to {output_path}")


if __name__ == "__main__":
    generate_mappings()
```

### Phase 2: Add Fuzzy Matching (1-2 days)

#### 2.1 Add Fuzzy Matching to Resolver

**File:** `src/nl2api/resolution/resolver.py` (modifications)

```python
from rapidfuzz import fuzz, process

class ExternalEntityResolver:
    def __init__(self, ...):
        # ... existing init ...

        # Load mappings from JSON
        from src.nl2api.resolution.mappings import load_mappings, get_all_known_names
        mappings_data = load_mappings()
        self._common_mappings = {
            name: data["ric"]
            for name, data in mappings_data.get("mappings", {}).items()
        }
        # Add aliases
        for name, data in mappings_data.get("mappings", {}).items():
            for alias in data.get("aliases", []):
                self._common_mappings[alias] = data["ric"]

        self._ticker_mappings = mappings_data.get("tickers", {})
        self._known_names = get_all_known_names()

        # Fuzzy matching config
        self._fuzzy_threshold = 85  # Minimum similarity score

    async def resolve_single(self, entity: str, ...) -> ResolvedEntity | None:
        # ... existing cache checks ...

        # Check common mappings (exact)
        if normalized in self._common_mappings:
            result = ResolvedEntity(
                original=entity,
                identifier=self._common_mappings[normalized],
                entity_type="company",
                confidence=0.95,
            )
            await self._cache_result(normalized, result)
            return result

        # Check ticker mappings
        if normalized.upper() in self._ticker_mappings:
            result = ResolvedEntity(
                original=entity,
                identifier=self._ticker_mappings[normalized.upper()],
                entity_type="ticker",
                confidence=0.99,
            )
            await self._cache_result(normalized, result)
            return result

        # Try fuzzy matching
        fuzzy_result = self._fuzzy_match(normalized)
        if fuzzy_result:
            result = ResolvedEntity(
                original=entity,
                identifier=fuzzy_result["ric"],
                entity_type="company",
                confidence=fuzzy_result["score"] / 100,
            )
            await self._cache_result(normalized, result)
            return result

        # Try external API if configured
        # ... existing API code ...

    def _fuzzy_match(self, query: str) -> dict | None:
        """
        Fuzzy match against known company names.

        Args:
            query: Normalized query string

        Returns:
            Dict with 'ric' and 'score' if match found
        """
        if not self._known_names:
            return None

        # Use rapidfuzz for fast fuzzy matching
        result = process.extractOne(
            query,
            self._known_names,
            scorer=fuzz.WRatio,
            score_cutoff=self._fuzzy_threshold,
        )

        if result:
            matched_name, score, _ = result
            ric = self._common_mappings.get(matched_name)
            if ric:
                logger.debug(f"Fuzzy matched '{query}' to '{matched_name}' (score={score})")
                return {"ric": ric, "score": score}

        return None
```

### Phase 3: Wire External API (2-3 days)

#### 3.1 Add OpenFIGI Integration

**File:** `src/nl2api/resolution/openfigi.py`

```python
"""
OpenFIGI API integration for entity resolution.

OpenFIGI is a free API for mapping securities identifiers.
https://www.openfigi.com/api
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"


async def resolve_via_openfigi(
    query: str,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """
    Resolve a company/ticker via OpenFIGI API.

    Args:
        query: Company name or ticker
        api_key: Optional API key (higher rate limits)
        timeout: Request timeout

    Returns:
        Dict with identifier info if found
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key

    # Try as ticker first
    payload = [{"idType": "TICKER", "idValue": query.upper()}]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENFIGI_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and data[0].get("data"):
                        figi_data = data[0]["data"][0]
                        return {
                            "found": True,
                            "identifier": _convert_to_ric(figi_data),
                            "type": "equity",
                            "confidence": 0.9,
                            "source": "openfigi",
                            "raw": figi_data,
                        }
    except Exception as e:
        logger.warning(f"OpenFIGI lookup failed: {e}")

    return None


def _convert_to_ric(figi_data: dict) -> str:
    """Convert OpenFIGI data to Reuters RIC format."""
    ticker = figi_data.get("ticker", "")
    exchange = figi_data.get("exchCode", "")

    # Map exchange codes to RIC suffixes
    exchange_map = {
        "US": ".O",  # NASDAQ
        "UN": ".N",  # NYSE
        "UW": ".O",  # NASDAQ
        "LN": ".L",  # London
        "JP": ".T",  # Tokyo
    }

    suffix = exchange_map.get(exchange, ".O")
    return f"{ticker}{suffix}"
```

#### 3.2 Update Resolver to Use OpenFIGI

**File:** `src/nl2api/resolution/resolver.py` (modifications)

```python
from src.nl2api.resolution.openfigi import resolve_via_openfigi

async def _resolve_via_api(self, entity: str) -> ResolvedEntity | None:
    """Resolve using OpenFIGI or configured external API."""

    # Try OpenFIGI first (free, no API key required)
    result = await resolve_via_openfigi(
        query=entity,
        api_key=self._api_key,
        timeout=self._timeout_seconds,
    )

    if result and result.get("found"):
        return ResolvedEntity(
            original=entity,
            identifier=result["identifier"],
            entity_type=result.get("type", "company"),
            confidence=result.get("confidence", 0.8),
        )

    # Fall back to configured external API if available
    if self._api_endpoint:
        # ... existing external API code ...
        pass

    return None
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/nl2api/resolution/data/company_mappings.json` | 500+ company mappings |
| `src/nl2api/resolution/mappings.py` | Mappings loader utilities |
| `src/nl2api/resolution/openfigi.py` | OpenFIGI API integration |
| `scripts/generate_company_mappings.py` | Script to generate mappings |
| `tests/unit/nl2api/resolution/test_mappings.py` | Mappings tests |
| `tests/unit/nl2api/resolution/test_openfigi.py` | OpenFIGI tests |

## Files to Modify

| File | Changes |
|------|---------|
| `src/nl2api/resolution/resolver.py` | Load from JSON, add fuzzy matching, wire OpenFIGI |
| `src/nl2api/config.py` | Add OpenFIGI config options |
| `pyproject.toml` | Add rapidfuzz dependency |

---

## Dependencies to Add

```toml
[project.dependencies]
rapidfuzz = ">=3.0.0"  # Fast fuzzy string matching
```

---

## Testing Plan

1. **Unit Tests**
   - Test mappings loader
   - Test fuzzy matching accuracy
   - Test ticker resolution
   - Test OpenFIGI integration (mocked)

2. **Integration Tests**
   - Test end-to-end resolution with real queries
   - Test cache behavior
   - Test fallback chain (cache → mappings → fuzzy → API)

3. **Accuracy Measurement**
   - Run against fixture queries
   - Measure resolution rate before/after
   - Track confidence distribution

---

## Success Criteria

- [ ] 500+ companies in static mappings
- [ ] Fuzzy matching with 85% threshold
- [ ] OpenFIGI integration working
- [ ] Resolution rate > 80% on fixture queries
- [ ] P95 resolution latency < 50ms (cached)
- [ ] 90%+ test coverage for resolution module

---

## Rollback Plan

1. Mappings are loaded from JSON - can revert to smaller set
2. Fuzzy matching has configurable threshold
3. OpenFIGI integration is optional (degrades gracefully)
4. Original `_common_mappings` dict preserved as fallback
