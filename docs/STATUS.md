# Project Status

**Last Updated:** 2026-01-25

## Overview

**Evalkit** is a general-purpose ML evaluation framework. **NL2API** is a reference application demonstrating evalkit's capabilities for financial NL-to-API translation.

---

## Current State

### Evalkit Framework (Complete)

| Component | Status | Description |
|-----------|--------|-------------|
| **Contracts** | ✅ | Generic TestCase, Scorecard, EvaluationPack protocol |
| **Batch Runner** | ✅ | Concurrent execution, checkpointing, resume |
| **Core Evaluators** | ✅ | AST comparator, temporal handling, semantics |
| **Storage Layer** | ✅ | Protocol-based with PostgreSQL/Memory backends |
| **Telemetry** | ✅ | OTEL tracing and Prometheus metrics |
| **CLI** | ✅ | batch, continuous, matrix commands |
| **Distributed** | ✅ | Redis queues, worker coordination (Phases 1-3) |
| **Pack Registry** | ✅ | NL2API and RAG packs registered |

### NL2API Application (Complete)

| Component | Status | Accuracy |
|-----------|--------|----------|
| **Entity Resolution** | ✅ | 99.5% (2.9M entities) |
| **Query Routing** | ✅ | 94.1% (Claude Haiku) |
| **DatastreamAgent** | ✅ | — |
| **EstimatesAgent** | ✅ | — |
| **FundamentalsAgent** | ✅ | — |
| **OfficersAgent** | ✅ | — |
| **ScreeningAgent** | ✅ | — |
| **NL2API Pack** | ✅ | 4-stage evaluation |

### RAG Application (Complete)

| Component | Status | Description |
|-----------|--------|-------------|
| **Hybrid Retriever** | ✅ | Vector + keyword search (pgvector) |
| **SEC EDGAR Ingestion** | ✅ | 10-K, 10-Q filing ingestion |
| **Small-to-Big Retrieval** | ✅ | Hierarchical chunk retrieval |
| **RAG Pack** | ✅ | 8-stage evaluation (466+ fixtures) |

---

## Test Coverage

```
Total Unit Tests:     2,875
Total Fixtures:       19,000+

By Category:
├── entity_resolution/   3,109 cases
├── lookups/             3,745 cases
├── temporal/            2,727 cases
├── comparisons/         3,658 cases
├── screening/             274 cases
├── complex/             2,288 cases
├── routing/               270 cases
└── rag/                   466 cases
```

---

## Directory Structure

```
nl2api/
├── src/
│   ├── evalkit/                # Evaluation Framework
│   │   ├── contracts/          # Data models (TestCase, Scorecard, etc.)
│   │   ├── batch/              # Batch runner, metrics, checkpointing
│   │   ├── core/               # Evaluators (AST, temporal, semantics)
│   │   ├── common/             # Storage, telemetry, cache, resilience
│   │   ├── distributed/        # Redis queues, worker coordination
│   │   ├── continuous/         # Scheduled evaluation, alerts
│   │   ├── packs/              # Pack registry and factory
│   │   └── cli/                # CLI commands
│   │
│   ├── nl2api/                 # NL2API Application
│   │   ├── orchestrator.py     # Query routing + agent dispatch
│   │   ├── agents/             # 5 domain agents
│   │   ├── resolution/         # Entity resolution (2.9M entities)
│   │   ├── llm/                # Claude + OpenAI providers
│   │   ├── conversation/       # Multi-turn support
│   │   └── evaluation/         # NL2API evaluation pack
│   │
│   ├── rag/                    # RAG Application
│   │   ├── retriever/          # Hybrid vector + keyword search
│   │   ├── ingestion/          # SEC EDGAR ingestion
│   │   └── evaluation/         # RAG evaluation pack (8 stages)
│   │
│   └── mcp_servers/            # MCP server implementations
│
├── tests/
│   ├── unit/                   # 2,875 unit tests
│   ├── integration/            # Database + multi-component tests
│   ├── accuracy/               # Real LLM accuracy tests
│   └── fixtures/               # 19,000+ test fixtures
│
├── config/
│   ├── grafana/                # Dashboards and datasources
│   └── otel-collector-config.yaml
│
└── docker-compose.yml          # PostgreSQL + Redis + OTEL stack
```

---

## Running Commands

```bash
# Unit tests
.venv/bin/python -m pytest tests/unit/ -v

# Integration tests
.venv/bin/python -m pytest tests/integration/ -v

# Accuracy tests (requires ANTHROPIC_API_KEY)
.venv/bin/python -m pytest tests/accuracy/ -m tier1 -v

# Batch evaluation
.venv/bin/python -m src.evalkit.cli.main batch run --pack nl2api --tag entity_resolution --limit 100
.venv/bin/python -m src.evalkit.cli.main batch run --pack rag --tag rag --label my-test

# View results
.venv/bin/python -m src.evalkit.cli.main batch list
```

---

## Recent Milestones

### January 2026

- ✅ **Evalkit extraction complete** - Framework separated from applications
- ✅ **RAG evaluation pack** - 8 stages with 466+ test cases
- ✅ **Small-to-big retrieval** - Hierarchical chunk retrieval for RAG
- ✅ **Entity resolution** - 99.5% accuracy on 3,109 test cases
- ✅ **Routing evaluation** - 94.1% accuracy with Haiku (10x cheaper than Sonnet)
- ✅ **Distributed workers** - Phases 1-3 complete (coordinator pending)
- ✅ **Production readiness** - Security, health checks, log sanitization

---

## Known Limitations

1. **Execution Stage (Stage 3)**: Not connected to live LSEG APIs - returns pass-through
2. **Distributed Workers Phase 4+**: Coordinator and batch API not yet implemented
3. **Agent-level accuracy**: Not yet measured for individual domain agents
4. **Test quality**: Tests don't fully validate agent output correctness (see BACKLOG.md)

---

## Next Steps

See [BACKLOG.md](../BACKLOG.md) for prioritized work items:

1. **Test Quality Improvements** (P0) - Validate agent output correctness
2. **Distributed Workers Phase 4+** - Coordinator and batch API
3. **Live API Integration** - Connect to real LSEG APIs
4. **Package publishing** - Publish evalkit as standalone package
