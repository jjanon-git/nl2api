# HyDE A/B Experiment Plan

**Status:** In Progress
**Date:** 2026-02-03
**Goal:** Evaluate whether HyDE (Hypothetical Document Embeddings) improves RAG retrieval quality

---

## Executive Summary

Test whether adding HyDE query expansion improves retrieval quality on SEC filing queries.

| Aspect | Baseline | Variant |
|--------|----------|---------|
| Method | Direct query embedding | Hypothetical answer embedding |
| Latency | ~100ms | ~500-800ms (+LLM call) |
| Expected improvement | - | +5-10% on analytical queries |

---

## Background

### What is HyDE?

HyDE (Hypothetical Document Embeddings) improves retrieval by:
1. Generating a hypothetical answer to the query
2. Embedding the hypothetical answer (not the original query)
3. Using that embedding for vector search

### Why it might help

Queries and answers often use different vocabulary:
- Query: "What risks does Apple face?"
- Answer: "Apple faces risks including supply chain disruption, regulatory changes, currency fluctuations..."

The hypothetical answer's embedding is closer to actual document embeddings than the query's embedding.

### Research Support
- [HyDE Paper](https://arxiv.org/abs/2212.10496) - Shows HyDE can match fine-tuned retrievers zero-shot
- Most effective for analytical/open-ended queries
- Less effective for factual lookups where query ≈ answer

---

## Phase 1: Implementation

### Step 1: Create HyDE Expander

```python
# src/rag/retriever/hyde.py

class HyDEExpander:
    """Generate hypothetical document for query expansion."""

    def __init__(self, llm_client, model: str = "claude-3-5-haiku"):
        self._client = llm_client
        self._model = model

    async def expand(self, query: str) -> str:
        """Generate hypothetical answer to use for embedding."""
        prompt = f"""Generate a brief, factual answer to this SEC filing question.
Write as if quoting from an actual SEC filing (10-K, 10-Q).
Be specific and use financial terminology.

Question: {query}

Hypothetical answer from SEC filing:"""

        response = await self._client.generate(prompt, max_tokens=200)
        return response
```

### Step 2: Integrate with Retriever

Add optional HyDE expansion in `HybridRAGRetriever`:

```python
def set_hyde_expander(self, expander: HyDEExpander) -> None:
    """Set HyDE expander for query expansion."""
    self._hyde_expander = expander

async def _retrieve_impl(self, ...):
    # Generate hypothetical answer if HyDE is enabled
    if self._hyde_expander:
        with tracer.start_span("rag.hyde_expand") as hyde_span:
            hypothetical = await self._hyde_expander.expand(query)
            hyde_span.set_attribute("hyde.hypothetical_length", len(hypothetical))
            # Embed hypothetical answer instead of query
            query_embedding_list = await self._embedder.embed(hypothetical)
    else:
        query_embedding_list = await self._embedder.embed(query)
```

### Step 3: Add CLI Option

```bash
# Enable HyDE for batch evaluation
python -m src.evalkit.cli batch run --pack rag-retrieval --hyde --limit 100
```

---

## Phase 2: Experiment Design

### Hypothesis
HyDE will improve retrieval quality by 5-10% on analytical queries due to:
- Better vocabulary alignment between hypothetical and actual documents
- More context in embedding (hypothetical is longer than query)

### Test Cases (Canonical Set)

| Category | Count | Expected HyDE Impact |
|----------|-------|---------------------|
| Analytical queries | ~200 | HIGH - vocabulary mismatch |
| Factual lookups | ~200 | LOW - query ≈ answer |
| Multi-part questions | ~100 | MEDIUM |

### Metrics

**Primary (same as reranker A/B test):**
- `context_relevance_score` - LLM-judged relevance (0-1)
- `context_relevance_pass_rate` - % queries with acceptable context

**Secondary:**
- `recall@5` - Expected docs in top 5
- `latency_p50`, `latency_p95` - Including HyDE generation time

### Experimental Conditions

| Run | Label | HyDE | Reranker |
|-----|-------|------|----------|
| A | `baseline-no-hyde` | Disabled | msmarco |
| B | `variant-hyde` | Enabled | msmarco |

---

## Phase 3: Execution

### Commands

```bash
# Baseline (no HyDE, with msmarco reranker)
python -m src.evalkit.cli.main batch run \
    --pack rag-retrieval \
    --limit 100 \
    --label baseline-no-hyde \
    --reranker msmarco

# Variant (HyDE enabled, with msmarco reranker)
python -m src.evalkit.cli.main batch run \
    --pack rag-retrieval \
    --limit 100 \
    --label variant-hyde \
    --reranker msmarco \
    --hyde
```

### Success Criteria

| Metric | Baseline Target | Success Threshold |
|--------|-----------------|-------------------|
| Context Relevance Pass | 92% (current) | +3% improvement |
| Latency P95 | Measure | < 2x baseline |

**Decision Framework:**
- If context_relevance improves >5% → Ship HyDE (optional flag)
- If improvement <3% → Don't ship (latency not worth it)
- If latency >3x → Consider caching hypothetical answers

---

## Phase 4: Results

### A/B Test Results (100 cases each)

| Condition | Pass Rate | Avg Score | Duration |
|-----------|-----------|-----------|----------|
| **No HyDE (baseline)** | **62%** | **0.56** | 201.6s |
| **HyDE** | **36%** | **0.40** | 241.6s |

**Batch IDs:**
- Baseline (no HyDE): `2a29bae6-96c1-411f-9b1a-a89b90a2caf4`
- HyDE variant: `84050f06-1c24-454e-8767-8c5211a6723a`

### Key Findings

1. **HyDE significantly DECREASED retrieval quality:**
   - -26% absolute drop in pass rate (62% → 36%)
   - -0.16 drop in average score (0.56 → 0.40)
   - HyDE made retrieval WORSE, not better

2. **HyDE adds latency without benefit:**
   - +40s for 100 cases (~20% slower)
   - LLM call for hypothetical generation adds overhead

3. **Hypothesis for poor performance:**
   - Canonical test set is mostly **factual queries** ("What was X's revenue?")
   - HyDE works best for **analytical queries** where query ≠ answer vocabulary
   - The hypothetical answers may introduce **noise or incorrect information**
   - SEC filings have **structured content** that already retrieves well

### Recommendation

**Do NOT ship HyDE as a default or recommended option.**

HyDE may still help for specific query types (analytical, open-ended), but the canonical SEC filing dataset consists mostly of factual lookups where HyDE hurts performance.

If HyDE is desired in the future:
1. Add query classification to identify analytical vs factual queries
2. Only apply HyDE to analytical queries
3. Test on a curated analytical-query subset

### Prompt Improvement Attempts

Tested three prompt variants to see if better prompts could salvage HyDE:

| Prompt | Strategy | Pass Rate (20 cases) |
|--------|----------|---------------------|
| V1 (original) | "Be specific, use financial terminology" | ~36% |
| V2 | "Focus on TOPICS, no specific numbers" | 45% |
| V3 | "List 5-10 key terms only" | 40% |
| **Baseline (no HyDE)** | Direct query embedding | **70%** |

**Finding:** Even with improved prompts that avoid hallucinating numbers, HyDE still significantly hurts retrieval. The problem isn't the prompt—it's that HyDE fundamentally doesn't help for factual queries.

### Lessons Learned

1. **Research claims don't always transfer** - HyDE papers show improvement on general datasets, but SEC filing queries are mostly factual
2. **Test before shipping** - The quick 10-case test already showed HyDE underperforming (30% vs 60%)
3. **Query type matters** - Different retrieval strategies work for different query types
4. **Prompt tuning has limits** - When the fundamental approach doesn't fit the data, no prompt will fix it

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM latency adds 500ms+ | Slower retrieval | Cache hypothetical answers |
| LLM hallucination | Wrong vocabulary in hypothetical | Use constrained prompt |
| API costs | Higher evaluation cost | Limit test size first |
| Worse on factual queries | Overall regression | Analyze by query type |

---

## References

- [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- [rag-00-improvements.md](rag-00-improvements.md) - Canonical experiment store
- [rag-10-retrieval-faithfulness-improvements.md](rag-10-retrieval-faithfulness-improvements.md) - HyDE design notes
