# Entity Resolution Improvement Plan

**Status:** Draft - Pending Review
**Priority:** P0.2 (Critical)
**Author:** AI Engineering Review

---

## Executive Summary

The entity resolution system has a solid foundation with multi-level caching, fuzzy matching, and external API integration. However, analysis reveals significant gaps in **coverage** (20% of S&P 500), **accuracy** (false positives from extraction), and **latency** (250ms+ for API fallback). This plan proposes targeted improvements to achieve 95%+ resolution accuracy for common financial queries.

---

## Current State Analysis

### What Exists (Implemented)

| Component | Status | Notes |
|-----------|--------|-------|
| Static mappings | 109 companies | ~20% S&P 500 coverage |
| Ticker resolution | 109 tickers | Direct ticker → RIC |
| Fuzzy matching | rapidfuzz | 75-85 threshold, handles typos |
| Multi-level caching | L1 (memory) + L2 (Redis) | 0.03ms cached, 0.3ms cold |
| Circuit breaker | Implemented | Protects against API failures |
| OpenFIGI integration | Working | 250ms latency per call |
| Test coverage | 42 tests | Normalization, fuzzy, extraction |

### Coverage Gaps

```
Current Coverage:
├── US Companies (.O, .N): 100 companies
│   ├── S&P 500: ~20% coverage (100/500)
│   ├── NASDAQ 100: ~60% coverage (est. 60/100)
│   └── Russell 2000: <1% coverage
├── International: 9 companies
│   ├── FTSE 100: ~3% coverage
│   ├── DAX 40: 0% coverage
│   └── Nikkei 225: ~1% coverage
└── Total known entities: 280 (including aliases)
```

### Accuracy Issues Identified

| Issue | Example | Impact |
|-------|---------|--------|
| False positive extraction | "price" → ICE, "so" → SO | Medium |
| Over-extraction | "Compare Microsoft" extracted as entity | Low |
| Ambiguous short names | "GE" could be General Electric or false positive | Medium |
| API latency on miss | 250ms+ for OpenFIGI calls | High |
| No negative cache | Repeated misses hit API every time | Medium |

### Performance Profile

```
Latency Breakdown:
├── L1 Cache Hit: 0.03ms (excellent)
├── L2 Cache Hit: ~1ms (good, estimated)
├── Static Mapping: 0.3ms (good)
├── Fuzzy Match: 1-2ms (acceptable)
└── OpenFIGI API: 250ms+ (bottleneck)

Throughput Impact:
- Cached queries: ~30,000 req/sec theoretical
- API fallback: ~4 req/sec per entity
```

---

## Goals

| Goal | Metric | Current | Target |
|------|--------|---------|--------|
| Coverage | Companies in mappings | 109 | 500+ |
| Resolution rate | % of queries with entities resolved | ~70% | 95%+ |
| False positive rate | Incorrect extractions | ~10% | <2% |
| P95 latency (cached) | Resolution time | 2ms | <5ms |
| P95 latency (API) | Resolution time | 300ms | <100ms |
| Test coverage | Lines covered | ~85% | 95%+ |

---

## Improvement Plan

### Phase 1: Reduce False Positives (3-4 days)

**Problem:** Entity extraction is too aggressive, matching common words like "ice", "so", "now" to tickers.

#### 1.1 Improve Extraction Patterns

**File:** `src/nl2api/resolution/resolver.py`

```python
# Expand ignore words to include common financial terms and short words
self._ignore_words = {
    # Existing words...
    # Add: short words that are also tickers
    "ice", "so", "now", "it", "on", "or", "an", "as", "at", "by",
    "be", "do", "go", "if", "in", "is", "me", "my", "no", "of",
    # Common financial terms that happen to be tickers
    "all", "any", "can", "has", "new", "own", "per", "see", "two",
}

# Add context-aware extraction
def _extract_entities(self, query: str) -> list[str]:
    # Don't extract ticker-like patterns from within words
    # e.g., "price" should not match "ICE"
    ...
```

#### 1.2 Add Contextual Validation

```python
def _validate_extraction(self, entity: str, query: str) -> bool:
    """Validate that an extracted entity is likely a real company reference."""
    normalized = entity.lower()

    # Short entities (2-3 chars) need stronger evidence
    if len(normalized) <= 3:
        # Must be uppercase in original query (likely ticker)
        if entity != entity.upper():
            return False
        # Must not be part of a larger word
        pattern = rf'\b{re.escape(entity)}\b'
        if not re.search(pattern, query):
            return False

    return True
```

#### 1.3 Add Negative Examples to Tests

```python
# tests/unit/nl2api/resolution/test_false_positives.py
@pytest.mark.parametrize("query,should_not_extract", [
    ("Show me the price", ["ICE"]),
    ("What is the best stock", ["IS", "BE"]),
    ("Show revenue for company", ["SO", "FOR"]),
    ("Compare these two stocks", ["TWO"]),
])
async def test_no_false_positive_extraction(query, should_not_extract):
    resolver = ExternalEntityResolver()
    result = await resolver.resolve(query)
    for entity in should_not_extract:
        assert entity not in result.values()
```

### Phase 2: Expand Coverage to 500+ Companies (5-7 days)

**Problem:** Only 109 companies, missing 80% of S&P 500.

#### 2.1 Data Sources for Expansion

| Source | Companies | Format | Effort |
|--------|-----------|--------|--------|
| S&P 500 list | 500 | CSV/JSON | 2 days |
| NASDAQ 100 | 100 | CSV/JSON | 0.5 days |
| FTSE 100 | 100 | CSV/JSON | 0.5 days |
| DAX 40 | 40 | CSV/JSON | 0.5 days |
| Top 50 by market cap (global) | 50 | Manual | 1 day |

#### 2.2 Create Data Generation Pipeline

**File:** `scripts/generate_company_mappings.py`

```python
"""
Generate comprehensive company mappings from multiple sources.

Sources:
1. S&P 500 constituents (Wikipedia/official list)
2. NASDAQ 100
3. FTSE 100
4. Manual additions for common aliases

Output: src/nl2api/resolution/data/company_mappings.json
"""

import json
import csv
from pathlib import Path

# S&P 500 data (would be loaded from CSV)
SP500_DATA = [
    # (name, ticker, exchange, aliases)
    ("3M Company", "MMM", "N", ["3m", "mmm"]),
    ("A.O. Smith", "AOS", "N", ["ao smith", "aos"]),
    ("Abbott Laboratories", "ABT", "N", ["abbott", "abt"]),
    # ... 497 more entries
]

def generate_ric(ticker: str, exchange: str) -> str:
    """Generate RIC from ticker and exchange."""
    suffix_map = {
        "N": ".N",   # NYSE
        "O": ".O",   # NASDAQ
        "A": ".A",   # AMEX
        "L": ".L",   # London
        "T": ".T",   # Tokyo
        "PA": ".PA", # Paris
        "DE": ".DE", # Frankfurt
    }
    return f"{ticker}{suffix_map.get(exchange, '.O')}"
```

#### 2.3 Add Common Aliases Automatically

```python
def generate_aliases(name: str, ticker: str) -> list[str]:
    """Generate common aliases for a company."""
    aliases = [ticker.lower()]

    # Remove common suffixes
    for suffix in [" Inc", " Corp", " Co", " Ltd", " LLC", " PLC",
                   " Holdings", " Group", " Company"]:
        if name.endswith(suffix):
            aliases.append(name[:-len(suffix)].lower())

    # Add without punctuation
    clean_name = re.sub(r'[^\w\s]', '', name.lower())
    if clean_name not in aliases:
        aliases.append(clean_name)

    return list(set(aliases))
```

### Phase 3: Optimize API Fallback (2-3 days)

**Problem:** OpenFIGI calls add 250ms+ latency.

#### 3.1 Add Negative Caching

```python
# Cache entities that we know don't exist
self._negative_cache: dict[str, float] = {}  # entity -> timestamp
NEGATIVE_CACHE_TTL = 3600  # 1 hour

async def resolve_single(self, entity: str, ...) -> ResolvedEntity | None:
    # Check negative cache first
    if normalized in self._negative_cache:
        if time.time() - self._negative_cache[normalized] < NEGATIVE_CACHE_TTL:
            return None  # Known non-entity

    # ... existing resolution logic ...

    # If not found, add to negative cache
    if result is None:
        self._negative_cache[normalized] = time.time()
```

#### 3.2 Batch API Requests

```python
async def resolve_batch(self, entities: list[str]) -> list[ResolvedEntity]:
    """Resolve multiple entities in batch, using single API call where possible."""
    # Separate into cached vs needs-API
    cached_results = []
    needs_api = []

    for entity in entities:
        cached = await self._check_cache(entity)
        if cached:
            cached_results.append(cached)
        else:
            needs_api.append(entity)

    # Batch API call for unknowns
    if needs_api:
        api_results = await self._batch_api_resolve(needs_api)
        cached_results.extend(api_results)

    return cached_results
```

#### 3.3 Add Prefetch for Common Entities

```python
# Prefetch most common entities on startup
PREFETCH_ENTITIES = [
    "Apple", "Microsoft", "Google", "Amazon", "Tesla",
    "Meta", "Nvidia", "JPMorgan", "Goldman Sachs", ...
]

async def prefetch_common_entities(self):
    """Warm cache with most commonly queried entities."""
    for entity in PREFETCH_ENTITIES:
        await self.resolve_single(entity)
```

### Phase 4: Accuracy Testing Framework (2-3 days)

**Problem:** No systematic way to measure and track resolution accuracy.

#### 4.1 Create Entity Resolution Benchmark

**File:** `tests/accuracy/resolution/test_entity_resolution_accuracy.py`

```python
"""
Accuracy tests for entity resolution.

Measures:
- Resolution rate (% of entities correctly resolved)
- False positive rate (incorrect extractions)
- Latency distribution
"""

BENCHMARK_QUERIES = [
    # (query, expected_entities)
    ("What is Apple's EPS?", {"Apple": "AAPL.O"}),
    ("Compare MSFT and GOOGL", {"MSFT": "MSFT.O", "GOOGL": "GOOGL.O"}),
    ("JP Morgan earnings forecast", {"JP Morgan": "JPM.N"}),
    ("Show me the stock price", {}),  # No entities expected
    ...
]

@pytest.mark.accuracy
async def test_resolution_accuracy():
    resolver = ExternalEntityResolver()

    correct = 0
    total = len(BENCHMARK_QUERIES)
    false_positives = 0

    for query, expected in BENCHMARK_QUERIES:
        result = await resolver.resolve(query)

        # Check if expected entities were found
        for name, ric in expected.items():
            if name in result and result[name] == ric:
                correct += 1

        # Check for false positives
        for found_name in result:
            if found_name not in expected:
                false_positives += 1

    accuracy = correct / total
    fpr = false_positives / total

    assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} below 95% threshold"
    assert fpr <= 0.02, f"False positive rate {fpr:.2%} above 2% threshold"
```

#### 4.2 Add Coverage Metrics

```python
def measure_coverage():
    """Measure entity resolution coverage against known datasets."""
    from src.nl2api.resolution.mappings import load_mappings

    mappings = load_mappings()

    # S&P 500 coverage
    sp500_tickers = load_sp500_tickers()  # From reference file
    covered = sum(1 for t in sp500_tickers if t in mappings["tickers"])

    print(f"S&P 500 coverage: {covered}/500 ({covered/5:.1f}%)")
```

---

## Implementation Timeline

```
Week 1:
├── Day 1-2: Phase 1.1-1.2 (False positive reduction)
├── Day 3-4: Phase 1.3 (Negative tests)
└── Day 5: Review and merge Phase 1

Week 2:
├── Day 1-3: Phase 2.1-2.2 (Data collection, script)
├── Day 4-5: Phase 2.3 (Alias generation)
└── Merge expanded mappings

Week 3:
├── Day 1-2: Phase 3.1-3.2 (Negative cache, batching)
├── Day 3: Phase 3.3 (Prefetch)
├── Day 4-5: Phase 4 (Accuracy tests)
└── Final review and documentation
```

---

## Success Criteria

| Criterion | Measurement | Target |
|-----------|-------------|--------|
| Companies in mappings | Count | 500+ |
| S&P 500 coverage | % of constituents | 95%+ |
| Resolution accuracy | Benchmark test | 95%+ |
| False positive rate | Benchmark test | <2% |
| P95 latency (cached) | Performance test | <5ms |
| Test coverage | pytest-cov | 95%+ |
| No regressions | Existing tests pass | 100% |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data quality issues | Medium | Validate against known RICs |
| OpenFIGI rate limits | Low | Caching + circuit breaker |
| Fuzzy match false positives | Medium | Configurable threshold + validation |
| Breaking existing tests | High | Run full test suite before merge |

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `scripts/generate_company_mappings.py` | Data generation pipeline |
| `data/sp500_constituents.csv` | S&P 500 reference data |
| `tests/accuracy/resolution/` | Accuracy test suite |
| `tests/unit/nl2api/resolution/test_false_positives.py` | False positive tests |

### Modified Files

| File | Changes |
|------|---------|
| `src/nl2api/resolution/resolver.py` | Improved extraction, negative cache |
| `src/nl2api/resolution/data/company_mappings.json` | Expanded to 500+ |
| `tests/unit/nl2api/resolution/test_resolver_fuzzy.py` | Additional edge cases |

---

## Appendix: Test Coverage Checklist

### Unit Tests Required

- [ ] Normalization for all suffix types (Inc, Corp, Ltd, LLC, PLC, & Co)
- [ ] Fuzzy matching edge cases (typos, partial names)
- [ ] Ticker resolution (uppercase, lowercase, mixed)
- [ ] Negative cache expiration
- [ ] Batch resolution
- [ ] Circuit breaker behavior
- [ ] Redis cache integration (mocked)
- [ ] OpenFIGI integration (mocked)

### Integration Tests Required

- [ ] End-to-end resolution with real database
- [ ] Cache warming / prefetch
- [ ] Multi-level cache fallback

### Accuracy Tests Required

- [ ] S&P 500 resolution benchmark
- [ ] False positive detection benchmark
- [ ] Latency distribution benchmark
