# NL2API Improvement Proposal

**Status:** Approved - Implementation In Progress

---

## Executive Summary

After comprehensive review of the NL2API system, the current implementation is architecturally sound with strong foundations (protocol-based design, FM-first routing, 12,887 test fixtures, dual-mode MCP support). However, there are significant opportunities for improvement across observability, accuracy, and performance.

**Key Metrics:**
- Current test coverage: 82% (812 tests)
- Fixture count: 12,887 across 6 categories
- Estimated P95 latency: ~850ms
- Estimated tokens per request: ~1100

---

## Priority Matrix

### P0: Critical (Do Now)

| # | Recommendation | Problem | Impact | Effort | Status |
|---|---------------|---------|--------|--------|--------|
| P0.1 | End-to-End Observability | No visibility into production performance | High | 1-2 weeks | ✅ Complete |
| P0.2 | Entity Resolution Coverage | Only ~30 companies in static mappings | High | 1 week | 🔲 Not Started |
| P0.3 | Request-Level Metrics | No accuracy measurement in production | High | 3-5 days | ✅ Complete |

### P1: High Priority (Next Sprint)

| # | Recommendation | Problem | Impact | Effort | Status |
|---|---------------|---------|--------|--------|--------|
| P1.1 | Token Optimization | Static prompts waste 800-1200 tokens | Medium | 2 weeks | 🔲 Not Started |
| P1.2 | Smart RAG Context Selection | Fixed retrieval parameters | Medium | 2 weeks | 🔲 Not Started |
| P1.3 | Rule-Based Coverage Expansion | Only 15-50% rule coverage | Medium | 2-3 weeks | 🔲 Not Started |
| P1.4 | Routing Validation Benchmark | FM-first router not validated | High | 1 week | 🔲 Not Started |

### P2: Medium Priority (Backlog)

| # | Recommendation | Problem | Impact | Effort | Status |
|---|---------------|---------|--------|--------|--------|
| P2.1 | Streaming Support | Users wait for full response | Medium | 2 weeks | 🔲 Not Started |
| P2.2 | A/B Testing Infrastructure | No variant comparison | Medium | 2-3 weeks | 🔲 Not Started |
| P2.3 | User Feedback Loop | No accuracy feedback mechanism | High | 3-4 weeks | 🔲 Not Started |
| P2.4 | Query Expansion/Rewriting | Ambiguous queries reduce accuracy | Medium | 2 weeks | 🔲 Not Started |

### P3: Research Spikes

| Spike | Duration | Hypothesis | Status |
|-------|----------|------------|--------|
| Routing Model Comparison | 1 week | Haiku may be sufficient at 1/10th cost | 🔲 Not Started |
| Fine-Tuning Feasibility | 2-3 weeks | Domain-specific model outperforms general LLM | 🔲 Not Started |
| Prompt Optimization (DSPy) | 2 weeks | Automated optimization improves accuracy | 🔲 Not Started |
| Embedding Model Comparison | 1 week | Better embeddings improve RAG retrieval | 🔲 Not Started |

---

## Architecture Analysis

### Current Request Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│  Entity Resolution  │ ──► Static mappings (~30 companies)
│      ~10ms          │     + External API (not wired)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   FM-First Router   │ ──► LLM call with agents as tools
│     ~300ms          │     Cache: Redis L1 + pgvector L2
│    ~100 tokens      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Ambiguity Detection │ ──► Rule-based patterns
│      ~5ms           │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Context Retrieval  │ ──► RAG (pgvector) or MCP
│     ~50ms           │     Dual-mode: local/mcp/hybrid
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│    Domain Agent     │ ──► Rule-based OR LLM tool-calling
│    ~500ms           │     5 agents: datastream, estimates,
│   ~1000 tokens      │     fundamentals, officers, screening
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│     Tool Calls      │ ──► Returned to caller
└─────────────────────┘

Total: ~850ms, ~1100 tokens per request
```

### Strengths

1. **Protocol-Based Design** - Clean separation, easy swapping of implementations
2. **Tiered Caching** - Redis L1 (~1ms) + pgvector L2 (~20ms)
3. **Resilience Patterns** - Circuit breaker, retry with backoff
4. **Comprehensive Tests** - 12,887 fixtures, 82% coverage

### Weaknesses

1. **Multiple LLM Calls** - Routing + Agent = 2 calls per request
2. **Static Prompts** - 800-1200 tokens of fixed content per agent
3. **Limited Entity Resolution** - ~30 companies hardcoded
4. ~~**No Observability** - Basic logging only~~ ✅ OTEL stack implemented

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| P95 Latency | ~850ms | <500ms | OpenTelemetry traces |
| Tokens per Request | ~1100 | <600 | Request metrics |
| Routing Accuracy | Unknown | 95%+ | Benchmark against fixtures |
| Rule-Based Coverage | 15-50% | 60%+ | Fixture coverage tests |
| Entity Resolution | ~30 companies | 500+ | Static mapping count |
| Test Coverage | 82% | 85%+ | pytest-cov |

---

## Implementation Timeline

```
Week 1-2:  P0.1 Observability + P0.3 Request Metrics
Week 2-3:  P0.2 Entity Resolution + P1.4 Routing Validation
Week 3-4:  Spike: Routing Model Comparison
Week 4-6:  P1.1 Token Optimization + P1.3 Rule Coverage
Week 6-8:  P1.2 Smart RAG + Spike: Embedding Comparison
Week 8+:   P2.x based on metrics and business priority
```

---

## Detailed Implementation Plans

See individual implementation plan files:
- [P0.1 Observability](./observability.md)
- [P0.2 Entity Resolution](./entity-resolution-p0.md)
- [P0.3 Request Metrics](./request-metrics.md)

Related plans:
- [Entity Resolution Expansion](./entity-resolution-expansion.md)
- [MCP Migration](./mcp-migration.md)
- [RAG Implementation](./rag-implementation.md)

---

## Critical Files

| File | Priority Changes |
|------|------------------|
| `src/nl2api/orchestrator.py` | Observability spans, metrics emission |
| `src/nl2api/resolution/resolver.py` | Entity mapping expansion, API wiring |
| `src/nl2api/agents/datastream.py` | Rule patterns, prompt compression |
| `src/nl2api/routing/llm_router.py` | Benchmarking, model switching |
| `src/nl2api/rag/retriever.py` | Query-type weighting, reranking |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-20 | Initial proposal created |
| 2026-01-20 | P0.3 Complete: RequestMetrics, emitters, OTEL integration |
| 2026-01-20 | P0.1 80%: OTEL stack, Grafana dashboards, accuracy metrics. Remaining: tracing spans |
| 2026-01-21 | P0.1 Complete: Added tracing spans to orchestrator and router |
