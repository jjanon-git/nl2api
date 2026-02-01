# RAG Evaluation Pipeline Performance Optimization

**Status:** Complete (Phase 3 rejected after testing)
**Created:** 2026-01-31
**Updated:** 2026-01-31
**Priority:** High

## Problem Statement

RAG generation evaluation takes ~30-40 seconds per test case due to sequential LLM calls across multiple stages. A 400-test batch takes ~2 hours.

### Root Cause Analysis

4-26 LLM calls per test case, all **sequential within stages**:

| Stage | LLM Calls | Time (sequential) |
|-------|-----------|-------------------|
| Generation | 1 | 1-5s |
| Context Relevance | 1-5 (per chunk) | 5-10s |
| Faithfulness | 1 + N (extract + verify) | 6-15s |
| Answer Relevance | 1 | 1-2s |
| Citation | 0-5 (per citation) | 0-10s |
| **Total** | **4-26 calls** | **13-42s** |

## Implementation Plan

### Phase 1: Parallelize Within-Stage LLM Calls ✅ COMPLETED

**Status:** Implemented and verified
**Measured Impact:** ~25% speedup (24s → 18-19s per test)

#### 1.1 Context Relevance Stage

**File:** `src/rag/evaluation/stages/context_relevance.py`

Change the sequential chunk evaluation loop (lines ~83-109) to use `asyncio.gather()`:

```python
# Current: sequential
for i, chunk in enumerate(chunks_to_evaluate):
    result = await llm_judge.evaluate_relevance(query, chunk, "context")

# New: parallel
tasks = [
    self._evaluate_chunk(llm_judge, query, i, chunk)
    for i, chunk in enumerate(chunks_to_evaluate)
]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

#### 1.2 Faithfulness Stage - Claim Verification

**File:** `src/rag/evaluation/llm_judge.py`

In `evaluate_faithfulness()` method (lines ~228-231), parallelize claim verification:

```python
# Current: sequential
for claim in claims:
    result = await self.verify_claim(claim, context)

# New: parallel
tasks = [self.verify_claim(claim, context) for claim in claims]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

#### 1.3 Citation Stage - Verification Loop

**File:** `src/rag/evaluation/stages/citation.py`

Parallelize the citation verification loop (lines ~251-278).

---

### Phase 2: Evaluation Mode Simplification ✅ COMPLETED

**Status:** Implemented and verified
**Impact:** `--mode retrieval` runs in ~2s per test (IR metrics only)

#### Implementation Summary

Replaced the originally planned `--quick` flag with a unified `--mode` approach:

```bash
# Full generation evaluation (default)
batch run --pack rag --mode generation --tag rag --label test

# Fast retrieval-only evaluation
batch run --pack rag --mode retrieval --tag rag --label test
```

#### Changes Made

**File:** `src/rag/evaluation/pack.py`
```python
@dataclass
class RAGPackConfig:
    eval_mode: str = "generation"  # "retrieval" or "generation"

def _build_stages(self) -> list[Any]:
    retrieval_only = self.config.eval_mode == "retrieval"
    # Skip LLM-heavy stages when retrieval_only is True
```

**File:** `src/evalkit/cli/commands/batch.py`
- Mode options: `resolver`, `orchestrator`, `simulated`, `retrieval`, `generation`
- Removed `rag-retrieval` pack (consolidated into single `rag` pack with mode flag)

---

## Testing Requirements

### Unit Tests (Required)

#### Test 1: Parallel Context Relevance
**File:** `tests/unit/rag/evaluation/test_context_relevance.py`

```python
@pytest.mark.asyncio
async def test_context_relevance_parallel_execution():
    """Verify chunks are evaluated in parallel, not sequentially."""
    # Mock llm_judge with delays to prove parallelism
    # 5 chunks × 100ms should complete in ~100-200ms, not 500ms

@pytest.mark.asyncio
async def test_context_relevance_handles_partial_failures():
    """Verify evaluation continues if one chunk fails."""
    # Mock one chunk to raise exception
    # Other chunks should still be evaluated
```

#### Test 2: Parallel Faithfulness Verification
**File:** `tests/unit/rag/evaluation/test_faithfulness.py`

```python
@pytest.mark.asyncio
async def test_claim_verification_parallel():
    """Verify claims are verified in parallel."""

@pytest.mark.asyncio
async def test_claim_verification_error_isolation():
    """Verify one failed claim doesn't break others."""
```

#### Test 3: Parallel Citation Verification
**File:** `tests/unit/rag/evaluation/test_citation.py`

```python
@pytest.mark.asyncio
async def test_citation_verification_parallel():
    """Verify citations are checked in parallel."""
```

#### Test 4: Quick Mode Configuration
**File:** `tests/unit/rag/evaluation/test_pack.py`

```python
def test_quick_mode_skips_llm_stages():
    """Verify quick_mode=True excludes expensive stages."""
    config = RAGPackConfig(quick_mode=True)
    pack = RAGPack(config)
    stage_names = [s.name for s in pack.get_stages()]
    assert "faithfulness" not in stage_names
    assert "context_relevance" not in stage_names
    assert "retrieval" in stage_names  # Should still run

def test_quick_mode_default_is_false():
    """Verify quick_mode defaults to False (full evaluation)."""
    config = RAGPackConfig()
    assert config.quick_mode is False
```

### Integration Tests (Required)

#### Test 5: End-to-End Performance
**File:** `tests/integration/rag/test_eval_performance.py`

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_evaluation_faster_than_sequential():
    """Verify parallel implementation is measurably faster."""
    # Run 3 test cases and verify total time < 3 × single-case time
```

### Manual Verification

1. **Before/After Timing:**
   ```bash
   # Before changes
   time .venv/bin/python -m src.evalkit.cli.main batch run \
     --pack rag --mode generation --tag rag --label before-opt --limit 5

   # After changes
   time .venv/bin/python -m src.evalkit.cli.main batch run \
     --pack rag --mode generation --tag rag --label after-opt --limit 5
   ```

2. **Quick Mode Test:**
   ```bash
   .venv/bin/python -m src.evalkit.cli.main batch run \
     --pack rag --mode generation --tag rag --label quick-test --quick --limit 10
   ```

3. **Verify Metrics Flow:**
   - Check Grafana RAG Evaluation dashboard shows data
   - Verify pass rates are reasonable (not all 0% or 100%)

---

## Files to Modify

| File | Change |
|------|--------|
| `src/rag/evaluation/stages/context_relevance.py` | Parallelize chunk evaluation |
| `src/rag/evaluation/llm_judge.py` | Parallelize claim verification |
| `src/rag/evaluation/stages/citation.py` | Parallelize citation checks |
| `src/rag/evaluation/pack.py` | Add quick_mode config |
| `src/evalkit/cli/commands/batch.py` | Add --quick flag |
| `docs/rag-evaluation-guide.md` | Document quick mode |

## Success Criteria

1. **Performance:** 5-item batch completes in <60s (vs ~180s before)
2. **Quality:** Pass rates unchanged (within 5% variance)
3. **Tests:** All unit tests pass, no regressions
4. **Documentation:** rag-evaluation-guide.md updated

---

### Phase 3: Cross-Stage Parallelization ❌ REJECTED

**Status:** Tested and rejected
**Decision Date:** 2026-01-31
**Reason:** No throughput improvement; rate limits are the bottleneck

#### Testing Results (2026-01-31)

We implemented and tested cross-stage parallelization. Key findings:

| Test | Parallel Stages | Sequential Stages |
|------|-----------------|-------------------|
| 50 tests, concurrency=5 | 21/50 in ~10 min | N/A (baseline) |
| Rate limit errors | Frequent (429s) | Less frequent |
| Tests/minute | ~2-3 | ~2-3 (same) |

**Conclusion:** Parallelization creates bursty traffic that triggers more rate limit errors without improving throughput. The bottleneck is Anthropic API rate limits, not stage execution speed.

**Decision:** Keep sequential stage execution. The simpler approach:
1. Creates smoother traffic patterns
2. Avoids rate limit spikes
3. Same effective throughput
4. Preserves fail-fast GATE semantics

If rate limits increase in the future, parallelization can be reconsidered.

---

#### Original Analysis (for reference)

##### Problem

Currently, even with within-stage parallelization, stages run sequentially:

```
Retrieval → Context Relevance → Faithfulness → Answer Relevance → Citation → Source Policy → Policy Compliance → Rejection Cal
   10ms         1500ms            1200ms           800ms           1000ms        200ms           100ms              30ms
```

Total time is sum of all stages: ~4.8s (LLM-dependent stages dominate).

#### Key Insight: All Stages Are Independent

**Each stage only reads from `test_case` and `system_output`.** No stage produces output consumed by another stage:

```
                    ┌─── Retrieval          (~10ms, IR metrics only)
                    ├─── Context Relevance  (~1500ms, LLM)
test_case ──────────┼─── Faithfulness       (~1200ms, LLM)
system_output       ├─── Answer Relevance   (~800ms, LLM)
                    ├─── Citation           (~1000ms, LLM)
                    ├─── Source Policy      (~200ms, GATE)
                    ├─── Policy Compliance  (~100ms, GATE)
                    └─── Rejection Cal      (~30ms, pattern)

NO cross-stage dependencies. All can run in parallel.
```

#### Design: All-Parallel with Post-Hoc GATE Check

Run all 8 stages in parallel via `asyncio.gather`, check GATE results after:

**File:** `src/rag/evaluation/pack.py`

```python
# Current (sequential)
for stage in self._stages:
    result = await stage.evaluate(test_case, system_output, context)
    stage_results[stage.name] = result
    if stage.is_gate and not result.passed:
        break

# New (parallel with post-hoc GATE check)
import asyncio

tasks = [
    stage.evaluate(test_case, system_output, context)
    for stage in self._stages
]
results = await asyncio.gather(*tasks)
stage_results = {
    stage.name: result
    for stage, result in zip(self._stages, results)
}

# Check GATE stages after all complete
# (Mark as failed if any GATE failed, but all stages have results)
```

#### GATE Semantics

**GATE stages:** `source_policy`, `policy_compliance`

Current behavior: GATE failure stops pipeline (subsequent stages don't run).

New behavior: GATE failure is still recorded, but all stages run. Semantics preserved:
- Overall scorecard shows GATE failure
- Metrics/dashboard correctly show failure
- But we have results from all stages (useful for debugging)

This is acceptable because:
1. GATE stages are fast (~300ms combined, no LLM calls)
2. GATE failures are rare in practice
3. For batch eval, having all results is more useful than fail-fast

#### Expected Performance

| Scenario | Current | Parallel | Speedup |
|----------|---------|----------|---------|
| Per test (generation mode) | ~18-19s | ~6-8s | 2.5-3x |
| 400 tests (concurrency=10) | ~60 min | ~20 min | 3x |

Parallel time = max(stage times) instead of sum(stage times).

---

## Validation Strategy

### Phase 3 Correctness Verification

#### 1. Unit Test: Result Equivalence

```python
# tests/unit/rag/evaluation/test_pack.py

@pytest.mark.asyncio
async def test_parallel_vs_sequential_results_match():
    """Verify parallel execution produces identical results to sequential."""
    pack_sequential = RAGPack(RAGPackConfig(parallel_stages=False))
    pack_parallel = RAGPack(RAGPackConfig(parallel_stages=True))

    test_case = create_test_case()
    system_output = create_mock_system_output()
    context = create_mock_context()

    result_seq = await pack_sequential.evaluate(test_case, system_output, context)
    result_par = await pack_parallel.evaluate(test_case, system_output, context)

    # Compare all stage results
    assert result_seq.passed == result_par.passed
    for stage_name in result_seq.stage_results:
        assert result_seq.stage_results[stage_name].score == \
               result_par.stage_results[stage_name].score
        assert result_seq.stage_results[stage_name].passed == \
               result_par.stage_results[stage_name].passed
```

#### 2. Unit Test: GATE Semantics Preserved

```python
@pytest.mark.asyncio
async def test_gate_failure_marks_overall_as_failed():
    """Verify GATE failure correctly marks overall scorecard as failed."""
    # Mock source_policy stage to fail
    pack = RAGPack(RAGPackConfig())
    result = await pack.evaluate(test_case_with_gate_failure, system_output, context)

    assert result.passed is False
    assert result.stage_results["source_policy"].passed is False
    # But all other stages still have results
    assert "faithfulness" in result.stage_results
    assert "context_relevance" in result.stage_results
```

#### 3. Unit Test: Parallelism Actually Occurs

```python
@pytest.mark.asyncio
async def test_stages_run_in_parallel():
    """Verify stages actually execute concurrently, not sequentially."""
    # Mock each stage to take 100ms
    # With 8 stages: sequential = 800ms, parallel = ~150ms

    pack = RAGPack(RAGPackConfig())
    start = time.perf_counter()
    await pack.evaluate(test_case, system_output, context)
    elapsed = time.perf_counter() - start

    # If truly parallel, should complete in ~150-200ms (max stage time + overhead)
    # If sequential, would take ~800ms
    assert elapsed < 0.4  # Conservative threshold
```

#### 4. Integration Test: Before/After Comparison

```bash
# Run same test cases with sequential and parallel, compare results

# Sequential baseline (before changes)
.venv/bin/python -m src.evalkit.cli.main batch run \
  --pack rag --mode generation --tag rag --label before-parallel --limit 20

# Parallel (after changes)
.venv/bin/python -m src.evalkit.cli.main batch run \
  --pack rag --mode generation --tag rag --label after-parallel --limit 20

# Compare: pass rates should match within 1% (LLM variance)
```

#### 5. Verification Checklist

Before merging Phase 3:

- [ ] Unit tests pass: `pytest tests/unit/rag/evaluation/test_pack.py -v`
- [ ] Result equivalence test confirms identical scores
- [ ] GATE semantics test confirms failure handling
- [ ] Parallelism test confirms < 400ms for 8 mocked stages
- [ ] Integration test: pass rates match before/after (±1%)
- [ ] Timing improvement: >2x speedup measured on 10+ test cases
- [ ] Grafana dashboards still show correct metrics

---

## Files to Modify

| File | Change | Phase |
|------|--------|-------|
| `src/rag/evaluation/stages/context_relevance.py` | Parallelize chunk evaluation | ✅ 1 |
| `src/rag/evaluation/llm_judge.py` | Parallelize claim verification | ✅ 1 |
| `src/rag/evaluation/stages/citation.py` | Parallelize citation checks | ✅ 1 |
| `src/rag/evaluation/pack.py` | Add eval_mode config | ✅ 2 |
| `src/evalkit/cli/commands/batch.py` | Unified --mode flag | ✅ 2 |
| `docs/rag-evaluation-guide.md` | Document modes | ✅ 2 |
| `src/rag/evaluation/pack.py` | Parallel stage execution | ❌ Rejected |
| `tests/unit/rag/evaluation/test_pack.py` | Validation tests | ❌ Rejected |

---

## Future Improvements (Phase 4+)

- Anthropic Batch API for judge calls (50% cost reduction)
- Cross-test-case batching for bulk evaluations
