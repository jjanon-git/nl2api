# Project Backlog

This file tracks high-priority items and technical debt for the NL2API project.

---

## High Priority

### [P0] Temporal Evaluation Data Handling
**Created:** 2026-01-21
**Status:** Planning Required
**Docs:** [docs/plans/temporal-eval-data.md](docs/plans/temporal-eval-data.md)

Financial data changes constantly. Before live API integration, we need a strategy for:
- Handling stale expected values (stock prices change)
- Point-in-time vs current data queries
- Validation approaches (exact match, range-based, semantic)

**Blocking:** Live API integration for evaluation

**Next steps:**
1. Decide on validation approach (see options in plan doc)
2. Design schema changes if needed (temporal_type, valid_as_of, tolerance fields)
3. Implement temporal-aware validation logic

---

### [P0] Live API Integration for Evaluation
**Created:** 2026-01-21
**Status:** Blocked by temporal data handling
**Depends on:** Temporal Evaluation Data Handling

Connect evaluation pipeline to real LSEG APIs to:
- Populate `expected_response` with real data
- Enable Stage 3 (Execution) evaluation
- Generate accurate `expected_nl_response` values

---

## Medium Priority

### [P1] Full Synthetic Dataset for Pipeline Testing
**Created:** 2026-01-21
**Status:** Planned

Create small (100-500 cases) fully-hydrated synthetic dataset:
- Populated `expected_response` (synthetic but structurally valid)
- Populated `expected_nl_response` (LLM-generated)
- Used to pressure-test full evaluation pipeline (all 4 stages)

**Not for accuracy tracking** - only for infrastructure validation.

---

## Low Priority / Future

### [P2] Entity Resolution Test Coverage
**Created:** 2026-01-21
**Status:** Known gap

`tests/unit/nl2api/test_entity_resolution_fixtures.py` failing due to coverage threshold.
Need to either:
- Generate more entity resolution test cases
- Adjust coverage thresholds

---

## Completed

- [x] Phase 1: Contract & Schema Updates (2026-01-21)
- [x] Phase 2: Generator Alignment (2026-01-21)
- [x] Phase 3: Documentation Updates (2026-01-21)
