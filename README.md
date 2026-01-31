# Evalkit + NL2API

A **general-purpose ML evaluation framework** with a reference implementation for financial NL-to-API translation.

## What is Evalkit?

**Evalkit** is a flexible evaluation framework for measuring ML system quality at scale. It provides:

- **Multi-stage evaluation pipelines** with configurable gates and soft stops
- **Batch processing** with concurrent execution, checkpointing, and resume
- **Pack-based architecture** - plug in domain-specific evaluation logic
- **Full observability** - OpenTelemetry tracing, Prometheus metrics, Grafana dashboards
- **Distributed execution** - Redis-backed task queues and worker coordination

## Reference Applications

### NL2API

Translates natural language queries into structured API calls for LSEG financial data services:

- **Entity resolution** at 99.5% accuracy (2.9M entities)
- **Query routing** at 94.1% accuracy (Claude Haiku)
- **5 domain agents** for Datastream, Estimates, Fundamentals, Officers, and Screening
- **16,000+ test fixtures** for comprehensive evaluation

### RAG (Retrieval-Augmented Generation)

Financial document Q&A system with SEC EDGAR filings:

- **Hybrid retrieval** with vector + keyword search (pgvector)
- **Small-to-big** hierarchical chunk retrieval
- **8-stage evaluation** including faithfulness, citation accuracy, rejection calibration
- **466+ test fixtures** for SEC filing evaluation

---

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Start infrastructure (PostgreSQL + Redis + OTEL stack)
podman-compose up -d    # or: docker compose up -d

# Run unit tests (2,875 tests)
.venv/bin/python -m pytest tests/unit/ -v

# Run batch evaluation (requires fixtures in DB)
.venv/bin/python scripts/load_fixtures_to_db.py --all
.venv/bin/python -m src.evalkit.cli.main batch run --pack nl2api --tag entity_resolution --limit 100

# View results in Grafana
open http://localhost:3000  # admin/admin
```

---

## Evaluation Packs

Evalkit supports multiple evaluation packs for different ML systems:

### NL2API Pack (4 stages)

Evaluates natural language to API translation:

| Stage | Purpose | Type |
|-------|---------|------|
| **Syntax** | Valid JSON/schema | GATE (hard stop) |
| **Logic** | Correct tool calls (AST comparison) | Scored |
| **Execution** | Live API verification | Configurable |
| **Semantics** | LLM-as-Judge NL comparison | Configurable |

```bash
# Run NL2API evaluation
batch run --pack nl2api --tag entity_resolution --mode resolver
batch run --pack nl2api --tag lookups --mode orchestrator
```

### RAG Pack (8 stages)

Evaluates Retrieval-Augmented Generation systems:

| Stage | Purpose | Type |
|-------|---------|------|
| **Retrieval** | IR metrics (recall@k, precision@k, MRR) | Scored |
| **Context Relevance** | Retrieved context relevance | Scored |
| **Faithfulness** | Response grounded in context | Scored |
| **Answer Relevance** | Response answers the question | Scored |
| **Citation** | Citation presence and accuracy | Scored |
| **Source Policy** | Quote-only vs summarize enforcement | GATE |
| **Policy Compliance** | Content policy violations | GATE |
| **Rejection Calibration** | False positive/negative detection | Scored |

```bash
# Run RAG evaluation (466+ test cases)
batch run --pack rag --tag rag --label my-experiment
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Evalkit Framework                        │
├─────────────────────────────────────────────────────────────────┤
│  src/evalkit/                                                    │
│  ├─ contracts/     Data models (TestCase, Scorecard, etc.)      │
│  ├─ batch/         Batch runner, checkpointing, metrics         │
│  ├─ core/          Evaluators (AST, temporal, semantics)        │
│  ├─ common/        Storage, telemetry, cache, resilience        │
│  ├─ distributed/   Redis queues, worker coordination            │
│  ├─ continuous/    Scheduled evaluation, alerts                 │
│  ├─ packs/         Pack registry and factory                    │
│  └─ cli/           CLI commands (batch, continuous, matrix)     │
├─────────────────────────────────────────────────────────────────┤
│                         Applications                             │
├─────────────────────────────────────────────────────────────────┤
│  src/nl2api/       NL2API Translation System                    │
│  ├─ orchestrator   Query routing + agent dispatch               │
│  ├─ agents/        5 domain agents (datastream, estimates, etc) │
│  ├─ resolution/    Entity resolution (2.9M entities)            │
│  └─ evaluation/    NL2API evaluation pack                       │
├─────────────────────────────────────────────────────────────────┤
│  src/rag/          RAG System                                   │
│  ├─ retriever/     Hybrid vector + keyword search               │
│  ├─ ingestion/     SEC EDGAR filing ingestion                   │
│  └─ evaluation/    RAG evaluation pack (8 stages)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Test Coverage

```
Total Unit Tests:     2,875
Total Test Fixtures:  19,000+

Fixture Categories:
├── entity_resolution/   3,109 cases (99.5% baseline)
├── lookups/             3,745 cases (single/multi-field queries)
├── temporal/            2,727 cases (time series, date ranges)
├── comparisons/         3,658 cases (multi-stock comparisons)
├── screening/             274 cases (SCREEN expressions)
├── complex/             2,288 cases (multi-step queries)
├── routing/               270 cases (94.1% baseline with Haiku)
└── rag/                   466 cases (SEC filings evaluation)
```

### Running Tests

```bash
# Unit tests (fast, mocked dependencies)
pytest tests/unit/ -v

# Integration tests (requires podman-compose up -d)
pytest tests/integration/ -v

# Accuracy tests (real LLM calls, requires API key)
pytest tests/accuracy/ -m tier1 -v   # Quick (~50 samples)
pytest tests/accuracy/ -m tier2 -v   # Standard (~200 samples)
pytest tests/accuracy/ -m tier3 -v   # Comprehensive (all)
```

---

## Observability

The observability stack runs via `podman-compose up -d` (or `docker compose up -d`):

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Primary database with pgvector |
| Redis | 6379 | Caching and task queues |
| OTEL Collector | 4317, 4318 | Telemetry collection |
| Prometheus | 9090 | Metrics storage |
| Grafana | 3000 | Dashboards (admin/admin) |
| Jaeger | 16686 | Distributed tracing |

### Dashboards

- **NL2API Evaluation & Accuracy** - Pass rates, stage breakdowns, trends
- **RAG Evaluation** - Retrieval metrics, faithfulness scores
- **Entity Resolution** - Resolution accuracy, confidence distribution

---

## CLI Commands

```bash
# Batch evaluation
batch run --pack nl2api --tag <tag> --label <label>
batch run --pack rag --tag rag --label <label>
batch list
batch results <batch-id>

# Continuous evaluation
continuous start --pack nl2api --interval 1h
continuous status
continuous stop

# Matrix comparison (A/B testing)
matrix run --component datastream --llm claude-3-5-haiku
matrix compare --runs <id1>,<id2>
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Pack Architecture** | Plug-in evaluation logic for different ML systems |
| **AST Comparison** | Order-independent, type-aware tool call comparison |
| **Temporal Handling** | Normalize relative dates across test cases |
| **Checkpoint/Resume** | Resume interrupted batch runs |
| **Circuit Breaker** | Fail-fast for external service failures |
| **Redis Caching** | L1/L2 caching with in-memory fallback |
| **OTEL Integration** | Full tracing and metrics for all operations |

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Development guide and coding standards |
| [BACKLOG.md](BACKLOG.md) | Project backlog and capability matrix |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and data flow |
| [docs/accuracy-testing.md](docs/accuracy-testing.md) | Accuracy testing patterns |
| [docs/evaluation-data.md](docs/evaluation-data.md) | Test fixture documentation |
| [docs/STATUS.md](docs/STATUS.md) | Implementation status |

---

## Environment Variables

```bash
# Required for LLM operations
NL2API_ANTHROPIC_API_KEY=sk-ant-...
# OR
NL2API_OPENAI_API_KEY=sk-...

# Storage backend
EVAL_BACKEND=postgres  # postgres | memory

# Telemetry
NL2API_TELEMETRY_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## License

[Add license information]
