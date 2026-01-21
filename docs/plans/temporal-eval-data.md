# Temporal Evaluation Data Handling

**Status:** Planning Required
**Priority:** HIGH
**Created:** 2026-01-21

## Problem Statement

Financial data is temporal - stock prices, market caps, and other values change constantly. This creates challenges for evaluation:

1. **Stale expected values**: A test case expecting Apple's price to be $187.50 will fail when the price changes
2. **Point-in-time accuracy**: What was "correct" yesterday may be "wrong" today
3. **Historical vs current**: Some queries ask for historical data (stable), others for current data (changes)

## Scenarios to Handle

| Query Type | Example | Temporal Behavior |
|------------|---------|-------------------|
| Current price | "What is Apple's stock price?" | Changes constantly |
| Historical price | "What was Apple's price on Jan 1, 2024?" | Stable |
| Time series | "Show Apple's price for the last month" | End date changes |
| Fundamentals | "What is Apple's PE ratio?" | Changes quarterly |
| Static data | "What sector is Apple in?" | Rarely changes |

## Potential Solutions

### Option A: Snapshot-Based Evaluation
- Capture expected values at a specific point in time
- Run evaluations against that same data snapshot
- Pro: Deterministic, reproducible
- Con: Requires data snapshot infrastructure

### Option B: Range-Based Validation
- Instead of exact match, validate value is within expected range
- e.g., Apple's price should be $150-$250 (reasonable range)
- Pro: Tolerant of normal fluctuations
- Con: Less precise, might miss real errors

### Option C: Semantic Validation Only
- Don't validate exact values
- Only validate that the response is structurally correct and mentions the right entities
- Pro: Simple, robust to temporal changes
- Con: Misses data accuracy issues

### Option D: Live Validation Mode
- At evaluation time, fetch current data from live APIs
- Compare agent's response against fresh ground truth
- Pro: Always current
- Con: Requires API access during evaluation, non-deterministic

### Option E: Hybrid Approach
- Use query type to determine validation strategy:
  - Historical queries → exact match (stable data)
  - Current queries → range-based or live validation
  - Structural queries → semantic validation only

## Recommended Approach

**Phase 1 (Current):** Use synthetic data for platform testing. Mark all temporal data as synthetic.

**Phase 2 (Live Integration):** Implement Option E (Hybrid):
1. Tag test cases with temporal_type: `historical`, `current`, `structural`
2. Historical → exact match against snapshot
3. Current → range validation OR live fetch
4. Structural → semantic validation

## Schema Implications

May need to add to TestCase:
```python
temporal_type: Literal["historical", "current", "structural"] = "current"
valid_as_of: datetime | None = None  # When expected_response was captured
value_tolerance: float | None = None  # For range-based validation (e.g., 0.1 = 10%)
```

## Open Questions

1. How often should snapshots be refreshed?
2. Should we store multiple snapshots for historical comparison?
3. How do we handle corporate actions (splits, mergers) that change historical data?
4. What's the acceptable staleness for "current" data in evaluations?

## Next Steps

- [ ] Decide on approach before live API integration
- [ ] Design schema changes if needed
- [ ] Implement temporal-aware validation logic
- [ ] Create snapshot management tooling (if Option A/E)
