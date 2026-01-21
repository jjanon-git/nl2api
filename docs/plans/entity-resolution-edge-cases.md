# Entity Resolution Edge Cases (P3 - Low Priority)

**Status:** Backlog
**Created:** 2026-01-21
**Context:** Findings from entity resolution evaluation improvements

---

## Summary

After implementing database-backed entity resolution with fuzzy matching, pass rate improved from **0.1% to 99.5%**. The remaining 14 failures (0.5%) are edge cases documented below for future consideration.

## Current Performance

| Stage | Pass Rate | Implementation |
|-------|-----------|----------------|
| Regex extraction only | 0.1% | Original baseline |
| + Database aliases lookup | 56.1% | `entity_aliases` table (3.7M entries) |
| + Primary name fallback | 69.9% | Direct `entities.primary_name` query |
| + Ticker direct lookup | 93.7% | `entities.ticker` column |
| + Fuzzy matching (pg_trgm) | **99.5%** | Trigram similarity > 0.3 |

## Remaining Edge Cases (14 failures)

### Category 1: Multiple Share Classes / Exchange Listings

| Query | Expected | Got | Issue |
|-------|----------|-----|-------|
| CARNIVAL | CUKPF.O | CCL.N | UK ADR vs US listing |
| RIO TINTO PLC | RTPPF.O | RTNTF.O | Different share class |
| 3M | MMM.N | 3M.O | NYSE vs OTC |
| UTG | UTGN.O | UTG.N | Different exchange suffix |
| QVC | QVCC.N | QVC.O | Different exchange |

**Root Cause:** Companies have multiple RICs across exchanges. Resolver returns first match.

**Potential Solutions:**
- Add exchange preference logic (prefer NYSE > NASDAQ > OTC)
- Support multi-RIC responses with disambiguation
- Add exchange context to test cases

**Effort:** Medium (2-3 days)
**Impact:** Low (affects <0.2% of queries)

---

### Category 2: Ambiguous Short Queries (1-2 chars)

| Query | Expected | Got | Issue |
|-------|----------|-----|-------|
| OR | OR.N | NONE | Common word, no match |
| A | A.N | NONE | Single letter, too ambiguous |
| BY | BY.N | NONE | Common word |
| ON | ON.O | NONE | Common word |
| AN | AN.N | NONE | Common word |

**Root Cause:** Very short queries are filtered out as common words or don't meet minimum length requirements.

**Potential Solutions:**
- Whitelist known short tickers (A, V, T, X, etc.)
- Add explicit 1-2 char ticker lookup before length filter
- Context-aware resolution (if query mentions "stock" or "company")

**Effort:** Low (1 day)
**Impact:** Low (affects <0.2% of queries)

---

### Category 3: Fuzzy Match False Positives

| Query | Expected | Got | Issue |
|-------|----------|-----|-------|
| INTEAT CORP | INTT.N | INTC.O | Matched Intel instead |
| GREENLIT EVENTURES INC | GRNLD.O | GLVT.O | Wrong fuzzy match |
| UNIT CORP | UNTCW.O | UNIT.O | Wrong entity |
| HP | HPQ.N | HP.N | Test expects HP Inc, got Helmerich & Payne |

**Root Cause:** Fuzzy matching can return wrong entities when:
- Query has typos that match a more common company
- Multiple similar company names exist
- Test data may have incorrect expected RICs

**Potential Solutions:**
- Increase fuzzy threshold for ambiguous cases
- Add company popularity/market cap weighting
- Review and fix test data where appropriate
- Consider LLM-based disambiguation for low-confidence matches

**Effort:** Medium-High (3-5 days)
**Impact:** Low (affects <0.2% of queries)

---

## Recommendations

1. **Do Not Fix Now** - 99.5% pass rate is sufficient for production use
2. **Review Test Data** - Some expected RICs may be incorrect (HP case)
3. **Consider for P2** - If entity resolution accuracy becomes critical, address Category 1 first (exchange preference)
4. **Defer Category 3** - LLM disambiguation is overkill for 0.1% of queries

## Related Files

- `src/nl2api/resolution/resolver.py` - Entity resolver implementation
- `tests/fixtures/lseg/generated/entity_resolution/` - Test fixtures
- `scripts/generators/entity_resolution_generator.py` - Fixture generator

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-21 | Accept 99.5% baseline | Edge cases are rare and complex to fix |
| 2026-01-21 | Defer to P3 backlog | Focus on higher-impact improvements |
