# RAG Retrieval & Faithfulness Improvements Plan

**Status:** Draft
**Date:** 2026-02-03
**Context:** Full RAG evaluation showed 37.6% pass rate with two major issues

---

## Executive Summary

Evaluation of 548 RAG tests revealed two primary failure modes:

| Issue | Failures | % of Total Failures |
|-------|----------|---------------------|
| **Retrieval** - Ground truth not in top-k | 206 | 60% |
| **Faithfulness** - False positives on refusals/hedging | 77 | 23% |

This document outlines research-backed approaches to address both issues.

---

## Issue 1: Retrieval Quality

### Current State
- **Embedding Model:** OpenAI text-embedding-3-small (1536d)
- **Retrieval Strategy:** Small-to-big with parent-child hierarchy
- **Reranker:** ms-marco-MiniLM-L-6-v2 (basic cross-encoder)
- **Hybrid Search:** 70% vector + 30% keyword
- **Result:** 206 failures where ground truth chunk exists but isn't retrieved

### Root Cause Analysis

**Detailed breakdown of 206 "retrieval failures":**

| Category | Count | % | Interpretation |
|----------|-------|---|----------------|
| **Ground truth wrong** | 84 | 41% | Retrieved chunks WERE relevant; answer was correct |
| **True retrieval gap** | 106 | 51% | Needed info not retrieved; model correctly refused |
| **True failure** | 17 | 8% | Wrong chunks → wrong answer |

**Key findings:**
1. **41% are ground truth errors**: The test fixtures have incorrect `relevant_chunk_ids`. The retriever found useful chunks that answered the query, just not the specific IDs marked as "correct."

2. **51% are legitimate retrieval gaps**: The retriever genuinely didn't find chunks containing the answer. The model appropriately refused rather than hallucinating - this is correct behavior, but retrieval should be improved.

3. **8% are true failures**: Wrong chunks retrieved, leading to wrong answers.

**Implication:** The retrieval system is better than the 62.4% pass rate suggests. However, ~106 cases still need retrieval improvement, and ground truth fixtures need auditing.

### Improvement Approaches

#### Approach 1A: Upgrade Embedding Model (HIGH IMPACT)

**Option: voyage-finance-2 (Recommended)**
- Domain-specific financial embeddings
- [+7% over OpenAI](https://blog.voyageai.com/2024/06/03/domain-specific-embeddings-finance-edition-voyage-finance-2/) on financial retrieval datasets
- [54% vs 38.5%](https://www.tigerdata.com/blog/general-purpose-vs-domain-specific-embedding-models) accuracy on SEC filings specifically
- 32K context length (vs 8K for OpenAI)
- Cost: $0.22 per 1M tokens

**Option: text-embedding-3-large**
- Same provider (simpler integration)
- 3072 dimensions (vs 1536)
- ~20% better quality than small
- Cost: $0.13 per 1M tokens

**Recommendation:** A/B test voyage-finance-2 vs current. The SEC filing-specific benchmarks strongly favor domain embeddings.

| Model | SEC Filing Accuracy | Cost/1M tokens |
|-------|---------------------|----------------|
| text-embedding-3-small (current) | 38.5% | $0.02 |
| text-embedding-3-large | ~46%* | $0.13 |
| voyage-finance-2 | 54% | $0.22 |

*Estimated based on general benchmark improvements

**Effort:** Medium (requires re-embedding 1.67M documents)

---

#### Approach 1B: Upgrade Reranker (HIGH IMPACT, LOW EFFORT)

Current reranker (ms-marco-MiniLM-L-6-v2) provides only +2-3% improvement. Modern rerankers achieve [+15-30% improvement](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025).

**Option: BAAI/bge-reranker-v2-m3 (Recommended)**
- [State-of-the-art open source](https://huggingface.co/BAAI/bge-reranker-base)
- Multi-lingual, larger input support
- Free, self-hosted
- Drop-in replacement

**Option: Cohere Rerank v3.5**
- Best commercial option
- 100+ language support
- 4096 token context
- Cost: ~$0.001 per query

**Option: Jina-ColBERT v3**
- [Late interaction architecture](https://arxiv.org/html/2509.25085v2)
- 8K token context (good for long SEC sections)
- Novel "last but not late interaction" approach

**Recommendation:** Start with bge-reranker-v2-m3 (free, significant upgrade). If insufficient, try Cohere Rerank.

**Effort:** Low (swap model in `src/rag/retriever/reranker.py`)

---

#### Approach 1C: Multi-Stage Retrieval Pipeline (MEDIUM IMPACT)

Current: Single hybrid search → rerank → top-k

Proposed [3-stage pipeline](https://www.genzeon.com/hybrid-retrieval-deranking-in-rag-recall-precision/):
1. **Stage 1:** BM25 retrieves 200 keyword-matched candidates
2. **Stage 2:** Dense retrieval adds 100 semantically similar documents
3. **Stage 3:** Reranker processes combined 300 candidates → top-k

[Pinecone analysis](https://neo4j.com/blog/genai/advanced-rag-techniques/) shows **+48% improvement** with this architecture.

**Why it helps:** Our current pipeline may over-weight semantic similarity, missing keyword-matched relevant chunks.

**Effort:** Medium (modify retriever to use RRF fusion)

---

#### Approach 1D: Implement HyDE (LOW-MEDIUM IMPACT)

Already designed in rag-06-improvements.md but not implemented.

**How it works:**
1. Generate hypothetical answer to query
2. Embed the hypothetical answer
3. Search using that embedding

**When it helps:** Analytical queries where query ≠ answer vocabulary.

Example:
- Query: "What risks does Apple face?"
- HyDE generates: "Apple faces risks including supply chain disruption, regulatory changes..."
- The hypothetical answer's embedding better matches actual risk factor text

**Effort:** Low (design exists, just needs implementation)

---

### Retrieval Improvement Priority

| Approach | Expected Impact | Effort | Priority |
|----------|-----------------|--------|----------|
| 1B: Upgrade reranker | +15-30% | Low | **P0** |
| 1A: voyage-finance-2 | +16% (54% vs 38%) | Medium | **P1** |
| 1C: Multi-stage pipeline | +20-48% | Medium | **P1** |
| 1D: HyDE | +5-10% (query-dependent) | Low | **P2** |

**Recommended sequence:**
1. Upgrade reranker (quick win, low risk)
2. A/B test voyage-finance-2 on subset
3. If still insufficient, implement multi-stage pipeline

---

## Issue 2: Faithfulness Evaluation

### Current State
- **Approach:** RAGAS-style claim extraction + verification
- **Problem:** 77 false positives (67 refusals + 10 hedging)
- Appropriate "I cannot determine X" responses marked as failures

### Root Cause Analysis

[RAGAS assigns refusals a score of 0](https://www.cohorte.co/blog/evaluating-rag-systems-in-2025-ragas-deep-dive-giskard-showdown-and-the-future-of-context), which is inappropriate when:
1. The information genuinely isn't in the retrieved context
2. The model correctly refuses rather than hallucinating

**Example failure:**
```
Query: "What is the CEO's compensation at NVIDIA?"
Response: "I cannot determine the CEO's compensation from these excerpts."
Context: (Contains only risk factors, not compensation data)

Current: FAIL (claim "I cannot determine" not supported by context)
Should be: PASS (correct refusal - info not in context)
```

### Improvement Approaches

#### Approach 2A: Refusal-Aware Faithfulness (HIGH IMPACT, LOW EFFORT) ✅ IMPLEMENTED

**Status:** Completed 2026-02-03

**Implementation:** Instead of regex patterns (which can miss nuanced refusals), we augmented the LLM judge to detect refusals intelligently. The `_detect_refusal` method in `src/rag/evaluation/llm_judge.py`:

1. Uses the LLM to determine if the response is a refusal/abstention
2. If refusal, checks whether it's appropriate (context lacks the information)
3. Returns score=1.0 for appropriate refusals, score=0.0 for inappropriate ones
4. Returns None for non-refusals (proceeds with standard claim verification)

```python
# In llm_judge.py
async def _detect_refusal(self, response: str, context: str, query: str | None) -> JudgeResult | None:
    """Detect if response is a refusal and evaluate appropriateness using LLM."""
    # LLM determines: is_refusal, refusal_appropriate, reasoning
    # Returns JudgeResult for refusals, None for non-refusals
```

**Benefits over regex approach:**
- Understands nuanced refusal language (not just keyword patterns)
- Can determine if refusal is appropriate by analyzing the context
- More robust to variations in phrasing
- Handles edge cases that regex would miss

This approach is inspired by [ChainPoll](https://www.cohorte.co/blog/evaluating-rag-systems-in-2025-ragas-deep-dive-giskard-showdown-and-the-future-of-context).

**Files modified:**
- `src/rag/evaluation/llm_judge.py` - Added `_detect_refusal` method
- `src/rag/evaluation/stages/faithfulness.py` - Removed regex patterns, passes query to LLM judge
- `tests/unit/evalkit/packs/rag/test_llm_judge.py` - Added 6 tests for refusal detection
- `tests/unit/evalkit/packs/rag/test_faithfulness_stage.py` - Updated tests for LLM-based approach

---

#### Approach 2B: Abstention-Aware Evaluation Framework (MEDIUM IMPACT)

Extend evaluation to explicitly handle three cases:

| Response Type | Context Has Answer | Expected Score |
|---------------|-------------------|----------------|
| Grounded answer | Yes | Score based on claim verification |
| Correct abstention | No | 1.0 (faithful to limitations) |
| Hallucination | No | 0.0 (unfaithful - made up answer) |
| Incorrect refusal | Yes | 0.0 (unfaithful - refused when could answer) |

This requires:
1. Determining if context can answer the query (context sufficiency check)
2. Classifying response as answer vs refusal
3. Applying appropriate scoring

**Effort:** Medium (requires context sufficiency classifier)

---

#### Approach 2C: Fix Claim Extraction (LOW IMPACT)

One case showed numeric sign loss:
- Response: "-4 million dollars"
- Extracted claim: "4 million dollars" (lost negative)

Fix: Update claim extraction prompt to preserve numeric signs:

```
Extract claims preserving exact numeric values including signs.
"-4 million" should become "-4 million", not "4 million".
```

**Effort:** Low (prompt update in `src/rag/evaluation/llm_judge.py`)

---

### Faithfulness Improvement Priority

| Approach | Expected Impact | Effort | Priority | Status |
|----------|-----------------|--------|----------|--------|
| 2A: Refusal-aware evaluation | Fixes 77 false positives | Low | **P0** | ✅ Done |
| 2C: Claim extraction fix | Fixes edge cases | Low | **P1** | Pending |
| 2B: Full abstention framework | More robust long-term | Medium | **P2** | Pending |

---

## Implementation Plan

### Phase 0: Ground Truth Audit (CRITICAL)
Before investing in retrieval improvements, fix the evaluation data:

1. **Audit 84 false negative cases** - where retrieval "failed" but answer was correct
2. **Update `relevant_chunk_ids`** to include actually-relevant chunks
3. **Consider multi-chunk ground truth** - many queries can be answered by multiple chunks

Without this, retrieval metrics will remain misleading.

### Phase 1: Quick Wins (1-2 days)
1. **Upgrade reranker** to bge-reranker-v2-m3
2. ✅ **Add refusal detection** to faithfulness stage (LLM-based, completed 2026-02-03)
3. **Fix claim extraction** prompt for numeric signs

### Phase 2: Embedding Upgrade (3-5 days)
1. Set up voyage-finance-2 embedder
2. Re-embed subset (10K documents) for A/B test
3. Compare retrieval metrics (after ground truth fix)
4. If positive, full re-embedding (~8 hours based on previous)

### Phase 3: Advanced Retrieval (if needed)
1. Implement multi-stage RRF pipeline
2. Add HyDE for analytical queries
3. Query-adaptive hybrid weights

---

## Success Metrics

| Metric | Current | Target (Phase 1) | Target (Phase 2) |
|--------|---------|------------------|------------------|
| Retrieval Pass Rate | 62.4% | 70% | 80% |
| Faithfulness Pass Rate | 72.6% | 85% | 85% |
| Overall Pass Rate | 37.6% | 50% | 65% |

---

## References

### Retrieval
- [VectorHub: Optimizing RAG with Hybrid Search & Reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [ZeroEntropy: Best Reranking Models 2025](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025)
- [Voyage AI: Finance Embeddings](https://blog.voyageai.com/2024/06/03/domain-specific-embeddings-finance-edition-voyage-finance-2/)
- [FinMTEB Benchmark](https://arxiv.org/html/2502.10990v1)
- [Neo4j: Advanced RAG Techniques](https://neo4j.com/blog/genai/advanced-rag-techniques/)

### Faithfulness
- [RAGAS Documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [Cohorte: Evaluating RAG Systems 2025](https://www.cohorte.co/blog/evaluating-rag-systems-in-2025-ragas-deep-dive-giskard-showdown-and-the-future-of-context)
- [Galileo: How to Evaluate LLMs for RAG](https://galileo.ai/blog/how-to-evaluate-llms-for-rag)
