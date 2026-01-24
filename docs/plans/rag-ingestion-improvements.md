# RAG Ingestion Improvements Design Document

**Author:** Principal Data Scientist Review
**Date:** 2026-01-23
**Status:** Draft - Pending Review

---

## Executive Summary

The current RAG ingestion pipeline is functional but employs relatively naive techniques that limit retrieval performance. This document analyzes the current state, identifies gaps relative to state-of-the-art practices, and proposes targeted improvements prioritized by impact and implementation complexity.

**Key Finding:** The current implementation uses fixed-size character-based chunking with general-purpose embeddings and no reranking—a pattern that research shows can result in 20-40% lower retrieval accuracy compared to modern approaches.

**Recommended Priority Improvements:**
1. **Add Cross-Encoder Reranking** (High impact, Medium effort) - +20-35% accuracy
2. **Implement Contextual Chunking** (High impact, Medium effort) - Better semantic boundaries
3. **Small-to-Big Retrieval** (Medium impact, Medium effort) - Better precision + context
4. **Domain-Adapted Embeddings** (Medium impact, High effort) - Financial domain optimization

---

## 1. Current State Analysis

### 1.1 Chunking Strategy

**Location:** `src/nl2api/ingestion/sec_filings/chunker.py`

**Current Approach:**
```python
chunk_size: 4000 characters
chunk_overlap: 800 characters
min_chunk_size: 200 characters
```

**Strategy:** 3-level fallback
1. Paragraph-aware splitting (`\n\s*\n`)
2. Sentence-aware splitting (`(?<=[.!?])\s+(?=[A-Z])`)
3. Character-based with word boundaries

**Issues Identified:**

| Issue | Impact | Evidence |
|-------|--------|----------|
| Fixed character size ignores semantic boundaries | Chunks split mid-topic, diluting vector representation | Research shows 9% recall gap between worst/best strategies |
| Overlap is character-based, not semantic | Adjacent chunks may split related concepts | Late chunking research shows significant improvement |
| No document hierarchy awareness | Loses parent-child relationships | Small-to-big retrieval achieves 65% win rate over baseline |
| No section-aware summarization | Each chunk lacks document context | Contextual chunking shows better coherence |

### 1.2 Embedding Generation

**Location:** `src/nl2api/rag/embedders.py`

**Current Options:**
| Embedder | Model | Dimensions | Notes |
|----------|-------|------------|-------|
| LocalEmbedder | all-MiniLM-L6-v2 | 384 | General purpose, fast |
| OpenAIEmbedder | text-embedding-3-small | 1536 | Better quality, API cost |

**Issues Identified:**

| Issue | Impact | Evidence |
|-------|--------|----------|
| General-purpose embeddings | Under-represents financial terminology | FinSage paper shows domain adaptation improves accuracy |
| Same embedding for query and document | Query-document asymmetry not addressed | HyDE shows document-style queries retrieve better |
| Single vector per chunk | Loses token-level nuance | ColBERT shows multi-vector improves complex queries |
| No query expansion | Short queries miss semantic matches | HyDE achieves performance similar to fine-tuned retrievers |

### 1.3 Retrieval Pipeline

**Location:** `src/nl2api/rag/retriever.py`

**Current Approach:**
- Hybrid search (70% vector + 30% keyword)
- HNSW index (m=16, ef_construction=64)
- Single-stage retrieval with threshold filtering
- Redis caching for repeated queries

**Issues Identified:**

| Issue | Impact | Evidence |
|-------|--------|----------|
| No reranking stage | First-stage recall is limited | Cross-encoder adds +20-35% accuracy |
| Single retrieval depth | Can't balance precision vs. context | Small-to-big achieves +65% win rate |
| Fixed hybrid weights | Not query-adaptive | Query type affects optimal weights |
| No recency boosting in vectors | Temporal queries underperform | Financial queries often need latest data |

### 1.4 Indexing Strategy

**Location:** `migrations/002_rag_documents.sql`

**Current Indexes:**
- HNSW on embeddings (vector_cosine_ops, m=16, ef_construction=64)
- GIN on search_vector (full-text)
- B-tree on document_type, domain, field_code

**Issues Identified:**

| Issue | Impact | Evidence |
|-------|--------|----------|
| HNSW parameters not tuned for recall | May sacrifice recall for speed | Higher ef_construction improves recall |
| No partial indexes for common queries | Slower filtered searches | SEC filings are most common doc type |
| No parent-child document linking | Can't do hierarchical retrieval | Requires schema change |

---

## 2. State-of-the-Art Techniques

Based on research from 2025-2026, the following techniques represent significant improvements:

### 2.1 Chunking Advances

#### Late Chunking
**Source:** [arXiv:2409.04701](https://arxiv.org/pdf/2409.04701)

Process entire document through embedding model at token level, then chunk and pool. Preserves full document context in each chunk's embedding.

**Benchmark:** Significantly higher similarity scores for context-dependent text (e.g., pronoun resolution).

**Trade-off:** Requires long-context embedding model, higher compute cost.

#### Contextual Chunking
**Source:** [Anthropic's Contextual Retrieval](https://arxiv.org/abs/2504.19754)

Prepend each chunk with document-level context generated by LLM:
```
"This chunk is from Apple Inc.'s 2024 10-K filing, Item 7
(Management Discussion), discussing revenue trends."

[Original chunk content...]
```

**Benchmark:** Preserves semantic coherence more effectively than late chunking.

**Trade-off:** Requires LLM call per chunk (can batch), increases token count.

#### Proposition Chunking
Break content into atomic fact-based units, each self-contained.

**Benchmark:** Research shows significant retrieval accuracy improvement.

**Trade-off:** Increases chunk count, may lose narrative flow.

### 2.2 Embedding Advances

#### HyDE (Hypothetical Document Embeddings)
**Source:** [HyDE Paper](https://arxiv.org/abs/2212.10496)

Generate hypothetical answer document from query, embed that instead of raw query.

```
Query: "What is Apple's revenue breakdown by segment?"
→ LLM generates hypothetical answer
→ Embed hypothetical document
→ Search for similar real documents
```

**Benchmark:** Reaches performance similar to fine-tuned retrievers, zero-shot.

**Trade-off:** Adds LLM call per query (~100-200ms), LLM knowledge dependency.

#### Domain-Adapted Embeddings
**Source:** [arXiv:2512.08088](https://arxiv.org/pdf/2512.08088)

Distill domain knowledge from LLM into compact retriever via iterative hard negative mining.

**Benchmark:** Cost-effective bridging of general-purpose to specialized domain.

**Trade-off:** Requires training infrastructure and labeled data.

#### ColBERT Multi-Vector
**Source:** [ColBERT Paper](https://arxiv.org/abs/2004.12832)

Encode query and document as token-level vectors, compute fine-grained similarity.

**Benchmark:** +4.2pp Recall@3 on PubMedQA with ColBERTv2.

**Trade-off:** 10-50x storage increase, more complex retrieval.

### 2.3 Retrieval Advances

#### Two-Stage Retrieval with Cross-Encoder Reranking
**Source:** [Pinecone Research](https://www.pinecone.io/learn/series/rag/rerankers/)

1. First stage: Fast bi-encoder retrieval (50-100 candidates)
2. Second stage: Cross-encoder reranking to top 10

**Benchmark:** +20-35% accuracy, +40% in some studies.

**Trade-off:** Adds 200-500ms latency per query.

**Recommended Models (2025):**
- Cohere Rerank (enterprise SLA)
- ms-marco-MiniLM-L-6-v2 (open source, fast)
- BGE-reranker-large (multilingual)
- Flash Rerank (5x faster, 95% accuracy)

#### Small-to-Big / Parent Document Retrieval
**Source:** [LlamaIndex Docs](https://docs.llamaindex.ai/en/stable/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval/)

Index small chunks for precise retrieval, return parent (larger) chunks for context.

```
Search Index: Small chunks (512 chars) for precision
Retrieved: Parent chunks (4000 chars) for context
```

**Benchmark:** 65% win rate over baseline chunking.

**Trade-off:** Requires parent-child linking in schema.

#### RAPTOR (Recursive Summarization)
Hierarchical tree of summaries, retrieve at multiple granularities.

**Benchmark:** Strong for long documents and multi-hop questions.

**Trade-off:** High preprocessing cost, complex indexing.

### 2.4 Financial Domain Findings

**Source:** [FinSage Paper](https://arxiv.org/html/2504.14493v3), [Financial RAG Study](https://arxiv.org/html/2511.18177)

| Finding | Metric | Implication |
|---------|--------|-------------|
| RAG improves GPT-4 from 19% to 56% on SEC questions | +37pp | RAG is essential for financial QA |
| Vector-based agentic RAG: 68% win rate | vs. node-based | Semantic retrieval > structured traversal |
| Cross-encoder reranking: +59% MRR@5 | absolute | Must-have for financial docs |
| Small-to-big: 65% win rate | +0.2s latency | Excellent precision-context trade-off |
| Hybrid search crucial | vs. vector-only | Domain jargon, tickers need keyword match |

---

## 3. Proposed Improvements

### Priority Matrix

| Improvement | Impact | Effort | Dependencies | Priority |
|-------------|--------|--------|--------------|----------|
| Cross-encoder reranking | High (+20-35%) | Medium | None | **P0** |
| Contextual chunking | High | Medium | LLM calls | **P1** |
| Small-to-big retrieval | Medium (+65% win) | Medium | Schema change | **P1** |
| HyDE query expansion | Medium | Low | LLM calls | **P2** |
| Late chunking | Medium | High | Long-context embedder | **P2** |
| Domain-adapted embeddings | Medium | High | Training infra | **P3** |
| ColBERT multi-vector | Medium | Very High | Storage, new index | **P3** |

### 3.1 P0: Cross-Encoder Reranking

**Goal:** Add two-stage retrieval with cross-encoder reranking.

**Implementation:**

```python
# src/nl2api/rag/reranker.py

from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Reranks retrieval results using cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        self._model = CrossEncoder(model_name, device=device)

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Rerank results using cross-encoder scores."""
        if not results:
            return results

        pairs = [(query, r.content) for r in results]

        # Run in thread pool
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, self._model.predict, pairs
        )

        # Sort by cross-encoder score
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            r._replace(score=float(s))  # Update score
            for r, s in scored[:top_k]
        ]
```

**Retriever Integration:**

```python
# Update HybridRAGRetriever

class HybridRAGRetriever:
    def __init__(
        self,
        ...
        reranker: CrossEncoderReranker | None = None,
        first_stage_limit: int = 50,  # Retrieve more candidates
    ):
        self._reranker = reranker
        self._first_stage_limit = first_stage_limit

    async def retrieve(self, query: str, ..., limit: int = 10):
        # First stage: get more candidates
        candidates = await self._first_stage_retrieve(
            query, limit=self._first_stage_limit
        )

        # Second stage: rerank to final limit
        if self._reranker and len(candidates) > limit:
            return await self._reranker.rerank(query, candidates, top_k=limit)

        return candidates[:limit]
```

**Expected Impact:** +20-35% retrieval accuracy

**Latency:** +200-300ms (can be parallelized with caching)

**Model Options:**
| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| ms-marco-MiniLM-L-6-v2 | Fast | Good | Recommended default |
| BAAI/bge-reranker-large | Medium | Better | Multilingual |
| Cohere Rerank | API | Best | Enterprise SLA |

### 3.2 P1: Contextual Chunking

**Goal:** Prepend document context to each chunk before embedding.

**Implementation:**

```python
# src/nl2api/ingestion/sec_filings/contextual_chunker.py

class ContextualChunker:
    """Adds document context to chunks for better embeddings."""

    CONTEXT_PROMPT = """
    Provide a brief (1-2 sentence) context for this chunk.
    Document: {company} {filing_type} ({filing_date})
    Section: {section}

    Chunk content:
    {chunk_content}

    Context (describe what this chunk discusses and how it fits in the document):
    """

    def __init__(self, llm_client, base_chunker: DocumentChunker):
        self._llm = llm_client
        self._base_chunker = base_chunker

    async def chunk_with_context(
        self,
        sections: dict[str, str],
        filing: Filing,
    ) -> list[FilingChunk]:
        # First, create base chunks
        base_chunks = self._base_chunker.chunk_filing(sections, filing)

        # Batch generate contexts (parallel LLM calls)
        contexts = await self._generate_contexts_batch(base_chunks, filing)

        # Prepend context to each chunk
        return [
            chunk._replace(
                content=f"{ctx}\n\n---\n\n{chunk.content}"
            )
            for chunk, ctx in zip(base_chunks, contexts)
        ]

    async def _generate_contexts_batch(
        self,
        chunks: list[FilingChunk],
        filing: Filing,
        batch_size: int = 20,
    ) -> list[str]:
        """Generate contexts for chunks in batches."""
        contexts = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_contexts = await asyncio.gather(*[
                self._generate_context(chunk, filing)
                for chunk in batch
            ])
            contexts.extend(batch_contexts)
        return contexts
```

**Alternative: Template-Based Context (No LLM)**

For cost efficiency, use template-based context:

```python
def add_template_context(chunk: FilingChunk, filing: Filing) -> str:
    """Add structured context without LLM."""
    context = (
        f"Company: {filing.company_name} ({filing.ticker})\n"
        f"Filing: {filing.filing_type} for fiscal year {filing.period_of_report.year}\n"
        f"Section: {chunk.section}\n"
        f"---\n\n"
    )
    return context + chunk.content
```

**Expected Impact:** Improved semantic coherence in embeddings

**Cost:** ~$0.001 per chunk with Claude Haiku (or free with templates)

### 3.3 P1: Small-to-Big Retrieval

**Goal:** Search on small chunks, return larger parent chunks for context.

**Schema Change:**

```sql
-- Add parent-child relationship
ALTER TABLE rag_documents ADD COLUMN parent_id UUID REFERENCES rag_documents(id);
ALTER TABLE rag_documents ADD COLUMN chunk_level VARCHAR(20);  -- 'child' or 'parent'

-- Index for parent lookups
CREATE INDEX idx_rag_documents_parent ON rag_documents(parent_id) WHERE parent_id IS NOT NULL;
```

**Implementation:**

```python
# src/nl2api/ingestion/sec_filings/hierarchical_chunker.py

class HierarchicalChunker:
    """Creates parent and child chunks for small-to-big retrieval."""

    def __init__(
        self,
        parent_chunk_size: int = 4000,
        child_chunk_size: int = 512,
        child_overlap: int = 50,
    ):
        self._parent_size = parent_chunk_size
        self._child_size = child_chunk_size
        self._child_overlap = child_overlap

    def chunk_hierarchical(
        self,
        sections: dict[str, str],
        filing: Filing,
    ) -> tuple[list[FilingChunk], list[FilingChunk]]:
        """
        Returns (parent_chunks, child_chunks).
        Child chunks reference their parent via parent_id.
        """
        parent_chunks = []
        child_chunks = []

        for section_name, section_text in sections.items():
            # Create parent chunks
            parents = self._create_parent_chunks(section_text, filing, section_name)
            parent_chunks.extend(parents)

            # Create child chunks for each parent
            for parent in parents:
                children = self._create_child_chunks(parent)
                child_chunks.extend(children)

        return parent_chunks, child_chunks
```

**Retriever Change:**

```python
async def retrieve_with_parents(
    self,
    query: str,
    limit: int = 5,
) -> list[RetrievalResult]:
    """Search child chunks, return parent chunks."""
    # Search on child chunks (small, precise)
    child_results = await self.retrieve(
        query,
        document_types=[DocumentType.SEC_FILING],
        limit=limit * 3,  # Over-retrieve
        chunk_level="child",
    )

    # Get unique parent IDs
    parent_ids = list(set(r.metadata.get("parent_id") for r in child_results))

    # Fetch parent chunks
    parent_chunks = await self._fetch_parents(parent_ids[:limit])

    return parent_chunks
```

**Expected Impact:** +65% win rate over baseline (per research)

**Latency:** +0.2s for parent lookup (can be optimized with join)

### 3.4 P2: HyDE Query Expansion

**Goal:** Transform queries into hypothetical documents for better embedding match.

**Implementation:**

```python
# src/nl2api/rag/hyde.py

class HyDEQueryExpander:
    """Generates hypothetical documents for query expansion."""

    HYDE_PROMPT = """
    You are a financial analyst. Given the following question about a company's
    SEC filing, write a short paragraph (3-4 sentences) that would directly
    answer this question. Write as if you are quoting from a 10-K or 10-Q filing.

    Question: {query}

    Hypothetical answer from SEC filing:
    """

    def __init__(self, llm_client, num_hypotheticals: int = 3):
        self._llm = llm_client
        self._num_hypotheticals = num_hypotheticals

    async def expand_query(self, query: str) -> list[str]:
        """Generate hypothetical documents for the query."""
        hypotheticals = await asyncio.gather(*[
            self._generate_hypothetical(query)
            for _ in range(self._num_hypotheticals)
        ])
        return hypotheticals

    async def get_hyde_embedding(
        self,
        query: str,
        embedder: Embedder,
    ) -> list[float]:
        """Get averaged embedding from hypothetical documents."""
        hypotheticals = await self.expand_query(query)

        embeddings = await embedder.embed_batch(hypotheticals)

        # Average the embeddings
        avg_embedding = [
            sum(e[i] for e in embeddings) / len(embeddings)
            for i in range(len(embeddings[0]))
        ]

        # Normalize
        norm = sum(x**2 for x in avg_embedding) ** 0.5
        return [x / norm for x in avg_embedding]
```

**Expected Impact:** Comparable to fine-tuned retrievers (zero-shot)

**Latency:** +300-500ms for LLM calls

**Best for:** Complex analytical queries, domain-specific terminology

### 3.5 P2: Late Chunking (Advanced)

**Goal:** Embed full document first, then chunk embeddings.

**Requirements:**
- Long-context embedding model (8k+ tokens)
- Token-level embedding extraction

**Implementation Sketch:**

```python
class LateChunker:
    """Implements late chunking for context-preserving embeddings."""

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-en"):
        # Jina v2 supports 8k context
        self._model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_with_late_chunking(
        self,
        document: str,
        chunk_boundaries: list[tuple[int, int]],
    ) -> list[list[float]]:
        """
        1. Get token-level embeddings for full document
        2. Pool embeddings for each chunk span
        """
        # Tokenize full document
        encoded = self._model.tokenize([document])

        # Get token embeddings (not pooled)
        with torch.no_grad():
            token_embeddings = self._model[0].auto_model(**encoded).last_hidden_state[0]

        # Map character spans to token spans and pool
        chunk_embeddings = []
        for char_start, char_end in chunk_boundaries:
            token_start, token_end = self._char_to_token_span(
                document, char_start, char_end, encoded
            )
            chunk_emb = token_embeddings[token_start:token_end].mean(dim=0)
            chunk_embeddings.append(chunk_emb.tolist())

        return chunk_embeddings
```

**Expected Impact:** Better context preservation for pronoun resolution, multi-sentence references

**Complexity:** High - requires different embedding workflow

### 3.6 P3: Domain-Adapted Embeddings

**Goal:** Fine-tune embeddings for financial domain.

**Approach (from FinSage paper):**

1. Generate synthetic QA pairs from SEC filings using LLM
2. Use LLM to judge relevance (creates training signal)
3. Iteratively mine hard negatives from corpus
4. Train contrastive loss on (query, positive, negatives)

**Training Pipeline:**

```python
# scripts/train_financial_embeddings.py

class FinancialEmbeddingTrainer:
    def __init__(self, base_model: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(base_model)

    async def generate_training_data(
        self,
        chunks: list[FilingChunk],
        llm_client,
    ) -> list[tuple[str, str, list[str]]]:
        """Generate (query, positive, negatives) tuples."""
        training_data = []

        for chunk in chunks:
            # Generate questions this chunk answers
            questions = await self._generate_questions(chunk, llm_client)

            # Mine hard negatives
            negatives = await self._mine_hard_negatives(questions[0], chunk)

            training_data.append((questions[0], chunk.content, negatives))

        return training_data

    def train(self, training_data, epochs: int = 3):
        """Fine-tune with contrastive loss."""
        train_examples = [
            InputExample(texts=[q, pos, neg])
            for q, pos, negs in training_data
            for neg in negs
        ]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.TripletLoss(model=self._model)

        self._model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
        )
```

**Expected Impact:** Better domain-specific retrieval

**Effort:** High - requires training infrastructure, evaluation pipeline

---

## 4. Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

| Task | Priority | Effort |
|------|----------|--------|
| Add CrossEncoderReranker class | P0 | 2 days |
| Integrate reranker into HybridRAGRetriever | P0 | 1 day |
| Add reranker configuration options | P0 | 1 day |
| Write unit tests for reranker | P0 | 1 day |
| Add evaluation metrics for reranking impact | P0 | 2 days |

### Phase 2: Chunking Improvements (2-3 weeks)

| Task | Priority | Effort |
|------|----------|--------|
| Implement template-based contextual chunking | P1 | 2 days |
| Add LLM-based contextual chunking (optional) | P1 | 3 days |
| Design parent-child schema | P1 | 1 day |
| Implement HierarchicalChunker | P1 | 3 days |
| Update indexer for parent-child docs | P1 | 2 days |
| Add small-to-big retrieval mode | P1 | 2 days |
| Re-index existing SEC filings | P1 | 1 day |

### Phase 3: Query Enhancement (1-2 weeks)

| Task | Priority | Effort |
|------|----------|--------|
| Implement HyDEQueryExpander | P2 | 2 days |
| Add HyDE mode to retriever | P2 | 1 day |
| Benchmark HyDE vs. direct query | P2 | 2 days |
| Add query type detection (when to use HyDE) | P2 | 2 days |

### Phase 4: Advanced (4-6 weeks)

| Task | Priority | Effort |
|------|----------|--------|
| Evaluate late chunking feasibility | P2 | 1 week |
| Implement late chunking prototype | P2 | 2 weeks |
| Generate training data for domain embeddings | P3 | 1 week |
| Train and evaluate domain embeddings | P3 | 2 weeks |

---

## 5. Evaluation Plan

### Metrics to Track

| Metric | Current Baseline | Target |
|--------|------------------|--------|
| Recall@5 | TBD (establish baseline) | +20% |
| MRR@10 | TBD | +25% |
| NDCG@10 | TBD | +20% |
| Answer accuracy (end-to-end) | ~56% (estimated) | 70% |

### Evaluation Dataset

Create a held-out evaluation set:
- 100 questions across SEC filing types (10-K, 10-Q)
- 50 simple factual queries
- 30 complex analytical queries
- 20 temporal/comparative queries

### A/B Testing

For each improvement, compare:
1. Baseline retriever
2. Baseline + improvement
3. Combined improvements

Use paired statistical tests to validate significance.

---

## 6. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Reranking latency too high | Use Flash Rerank or batch requests |
| Context generation costs | Start with template-based, upgrade to LLM if needed |
| Schema migration complexity | Use backwards-compatible column additions |
| Re-indexing existing data | Implement incremental migration |
| HyDE generates hallucinated context | Only use for complex queries, validate with reranker |

---

## 7. Open Questions for Review

1. **Embedding model choice:** Should we prioritize local (cost) or OpenAI (quality) for production?

2. **Reranker latency budget:** What's acceptable latency increase for accuracy gain?

3. **Re-indexing strategy:** Full re-index vs. incremental for existing SEC filings?

4. **HyDE adoption:** Query-type specific or universal?

5. **Domain embeddings:** Worth the training investment given our scale?

---

## 8. References

### Research Papers
- [Late Chunking](https://arxiv.org/pdf/2409.04701) - Jina AI, 2024
- [Reconstructing Context: Chunking Strategies](https://arxiv.org/abs/2504.19754) - arXiv, 2025
- [ColBERT](https://arxiv.org/abs/2004.12832) - Stanford, 2020
- [FinSage RAG](https://arxiv.org/html/2504.14493v3) - arXiv, 2025
- [Financial RAG Study](https://arxiv.org/html/2511.18177) - arXiv, 2025
- [Domain Embedding Adaptation](https://arxiv.org/pdf/2512.08088) - arXiv, 2025

### Industry Resources
- [Pinecone Rerankers Guide](https://www.pinecone.io/learn/series/rag/rerankers/)
- [Weaviate Chunking Strategies](https://weaviate.io/blog/chunking-strategies-for-rag)
- [LangWatch RAG Blueprint](https://langwatch.ai/blog/the-ultimate-rag-blueprint-everything-you-need-to-know-about-rag-in-2025-2026)
- [Best Chunking Strategies 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Zeroentropy Reranking Guide](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025)

---

## Appendix A: Current Code Locations

| Component | File |
|-----------|------|
| Document Chunker | `src/nl2api/ingestion/sec_filings/chunker.py` |
| Embedders | `src/nl2api/rag/embedders.py` |
| Hybrid Retriever | `src/nl2api/rag/retriever.py` |
| RAG Indexer | `src/nl2api/rag/indexer.py` |
| Query Handler | `src/rag_ui/query_handler.py` |
| DB Schema | `migrations/002_rag_documents.sql` |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Bi-encoder** | Encodes query and document independently into single vectors |
| **Cross-encoder** | Jointly encodes query-document pair for relevance score |
| **HyDE** | Hypothetical Document Embeddings - generate fake answer to query |
| **Late chunking** | Embed full document, then split into chunks |
| **Small-to-big** | Search small chunks, return larger parent context |
| **HNSW** | Hierarchical Navigable Small World - ANN index algorithm |
| **MRR** | Mean Reciprocal Rank - position of first relevant result |
| **NDCG** | Normalized Discounted Cumulative Gain - ranking quality metric |
