# EvalPlatform Architecture & Design Contract

> **Version:** 1.0.0
> **Status:** Draft - Pending Review
> **Last Updated:** 2026-01-19

## Executive Summary

EvalPlatform is a distributed evaluation framework for testing 40+ APIs via natural language queries at scale (~400k test cases). It prioritizes time-to-market while maintaining clean interfaces to avoid unrecoverable technical debt.

---

## 1. System Overview

### 1.1 Core Objectives
- **Scale:** Handle ~400k test cases with high concurrency
- **Reliability:** Fault-tolerant, idempotent workers with automatic retry
- **Observability:** Full tracing and metrics for debugging at scale
- **Extensibility:** Clean interfaces for adding new evaluation strategies

### 1.2 Technology Stack
| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Schemas | Pydantic v2 |
| API Framework | FastAPI |
| Queue | Azure Service Bus |
| Workers | Azure Container Apps (ACA) |
| Results Store | Azure Table Storage |
| Gold Store | Azure AI Search (Vector) |
| LLM Judge | Azure OpenAI (GPT-4o) |
| Observability | OpenTelemetry → Azure Monitor |
| CLI | Typer |
| Local Dev | Docker + Azurite |

---

## 2. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MANAGEMENT PLANE (REST API)                            │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐      ┌─────────────────────────────────────────────────────────┐
  │  Clients    │─────▶│  Management API (ACA / Azure Functions)                 │
  │  - Web UI   │      │  ┌─────────────┬──────────────┬────────────────────┐    │
  │  - CI/CD    │      │  │ /test-cases │ /test-suites │ /target-systems    │    │
  │  - Scripts  │      │  │ /clients    │ /runs        │                    │    │
  └─────────────┘      │  └─────────────┴──────────────┴────────────────────┘    │
                       └──────────┬──────────────────────────────┬───────────────┘
                                  │                              │
                                  ▼                              ▼
                       ┌──────────────────┐           ┌─────────────────┐
                       │  Azure AI Search │           │  Service Bus    │
                       │  (Gold Store)    │           │  Queue          │
                       │  - Test Cases    │           └────────┬────────┘
                       │  - Test Suites   │                    │
                       │  - Configs       │                    │
                       └──────────────────┘                    │
                                                               │
┌──────────────────────────────────────────────────────────────┼──────────────────┐
│                           EXECUTION PLANE (Workers)          │                   │
└──────────────────────────────────────────────────────────────┼──────────────────┘
                                                               │
                                                               ▼
                       ┌──────────────────┐           ┌──────────────────┐
                       │  Azure AI Search │◀── fetch ─│  Worker (ACA)    │
                       │  (Gold Store)    │           │  - Stateless     │
                       └──────────────────┘           │  - Idempotent    │
                                                      └────────┬─────────┘
                                                               │
                                                               ▼
                                                      ┌─────────────────────────┐
                                                      │  Target System (LLM)    │
                                                      │  - GPT-4o / Claude / etc│
                                                      │  - Tool Calling         │
                                                      └────────┬────────────────┘
                                                               │
                                                               ▼
                                                      ┌─────────────────────────┐
                                                      │  Evaluation Pipeline    │
                                                      │  ┌───────────────────┐  │
                                                      │  │ Stage 1: Syntax   │  │
                                                      │  │ Stage 2: Logic    │  │
                                                      │  │ Stage 3: Exec     │  │
                                                      │  │ Stage 4: Semantic │  │
                                                      │  └───────────────────┘  │
                                                      └───────────┬─────────────┘
                                                                  │
                                                                  ▼
                                                      ┌─────────────────────────┐
                                                      │  Azure Table Storage    │
                                                      │  - Scorecards           │
                                                      │  - Run Results          │
                                                      └─────────────────────────┘
```

---

## 3. The "Gold Standard" Test Case Schema

### 3.1 Core Model: `TestCase`

The fundamental unit of evaluation is a 4-tuple (plus metadata):

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier (UUID4) |
| `nl_query` | `str` | Natural language input query |
| `expected_tool_calls` | `List[ToolCall]` | Expected API calls (order-independent) |
| `expected_raw_data` | `Dict \| None` | Mock return data (optional) |
| `expected_nl_response` | `str` | Expected semantic summary |
| `metadata` | `TestCaseMetadata` | Version, complexity, tags |
| `embedding` | `List[float]` | Vector embedding for similarity search |

### 3.2 ToolCall Model

```python
class ToolCall:
    tool_name: str           # e.g., "search_products"
    arguments: FrozenDict    # Immutable for hashing/set operations
```

**Critical Design Decision:** Arguments use `FrozenDict` (immutable) to enable:
- Set-based comparison (order independence)
- Proper hashing for deduplication
- Thread-safe operations

### 3.3 Metadata Schema

```python
class TestCaseMetadata:
    api_version: str              # e.g., "v2.1.0"
    complexity_level: int         # 1 (trivial) to 5 (complex)
    tags: List[str]               # e.g., ["search", "multi-tool", "edge-case"]
    created_at: datetime
    updated_at: datetime
    author: str | None
    source: str | None            # Origin of test case
```

---

## 4. Evaluation Pipeline (Waterfall)

### 4.1 Pipeline Philosophy: "Fail Fast, Log Everything"

The pipeline follows a **waterfall with soft stops**:
- **Hard Stop:** Stage 1 (Syntax) failure halts the pipeline
- **Soft Continue:** Stage 2 (Logic) failures are logged but pipeline continues
- **Rationale:** Maximize diagnostic data even on failures

### 4.2 Stage Criticality

| Stage | Purpose | Criticality | Configurable? | Why |
|-------|---------|-------------|---------------|-----|
| 1. Syntax | Valid JSON/schema | Gate | No | Required for parsing |
| 2. Logic | Correct tool calls | High | No | Core correctness check |
| 3. Execution | Actually works | **CRITICAL** | Yes | Cost/rate limits/side effects |
| 4. Semantics | Good explanation | Low | Yes | Polish, not correctness |

**Key Insight:** Stage 3 (Execution) is the most important test—it answers "does this actually work?" However, it's configurable because:
- **Cost:** 400k API calls adds up significantly
- **Rate limits:** Live APIs will throttle at scale
- **Side effects:** Some APIs mutate state (create orders, send emails)
- **Flakiness:** Live APIs have downtime, latency variance, data drift
- **Environment:** Can't always hit production APIs from local/CI

**Recommendation:** Enable Stage 3 for nightly/release runs against live APIs. Disable for rapid iteration in CI where Stages 1-2 provide fast feedback.

### 4.3 Stage Definitions

#### Stage 1: Syntax Validation (GATE)
- **Purpose:** Validate generated output is parseable JSON conforming to schema
- **Input:** Raw string output from target system
- **Output:** Parsed `List[ToolCall]` or error
- **Failure:** Pipeline halts; Scorecard marked as syntax failure
- **Criticality:** Required—cannot proceed without valid structure

#### Stage 2: Logic Comparison (AST-Based)
- **Purpose:** Compare actual vs expected tool calls semantically
- **Method:** Abstract Syntax Tree comparison (NOT string matching)
- **Features:**
  - Order-independent comparison (set semantics)
  - Type-aware argument comparison (`"5"` vs `5` handled correctly)
  - Nested object deep comparison
  - Argument permutation tolerance
- **Output:** Match score (0.0 - 1.0) + detailed diff
- **Criticality:** High—validates the system chose correct APIs with correct arguments

#### Stage 3: Execution Verification (CRITICAL - Configurable)
- **Purpose:** Execute tool calls against live/mock API and verify results
- **Tolerance:** 0.01% for numerical comparisons
- **Configuration:** Enabled/disabled via `EvaluationConfig.execution_stage_enabled`
- **Timeout:** Configurable per-API (default: 30s)
- **Criticality:** **MOST CRITICAL**—this is the ultimate test of correctness
- **Why Configurable:** Cost, rate limits, side effects, environment constraints (see 4.2)

#### Stage 4: Semantic Comparison (OPTIONAL - LLM-as-Judge)
- **Purpose:** Compare generated NL response vs expected using LLM
- **Engine:** Azure OpenAI GPT-4o
- **Method:** Pairwise comparison with structured rubric
- **Output:**
  - Score (0.0 - 1.0)
  - Reasoning text
  - Confidence level
- **Criticality:** Low—evaluates presentation/UX, not correctness
- **Why Optional:** If Stages 1-3 pass, the system works correctly; Stage 4 just assesses how well it explains results

### 4.4 Scorecard Output

```python
class Scorecard:
    test_case_id: str
    batch_id: str | None
    timestamp: datetime

    # Per-stage results
    syntax_result: StageResult      # Always present
    logic_result: StageResult | None
    execution_result: StageResult | None
    semantics_result: StageResult | None

    # Captured outputs
    generated_tool_calls: List[ToolCall] | None
    generated_nl_response: str | None

    # Metrics
    latency_ms: int
    worker_id: str
    attempt_number: int

    # Computed
    overall_passed: bool
    overall_score: float  # Weighted average
```

---

## 5. Distributed Architecture

### 5.1 Component: Producer CLI

**Technology:** Typer (Python)

**Commands:**
```bash
# Submit a batch of test IDs
eval submit --batch-file tests.json --priority high

# Submit by query (fetch from Gold Store)
eval submit --query "complexity_level:5 AND tags:search"

# Check batch status
eval status --batch-id <uuid>

# Replay failed tests
eval replay --batch-id <uuid> --only-failed
```

**Responsibilities:**
- Parse and validate input
- Create `BatchJob` record
- Push `WorkerTask` messages to Service Bus
- Support priority queues (high/normal/low)

### 5.2 Component: Worker (Azure Container Apps)

**Design Principles:**
1. **Stateless:** No local state; all state in Azure services
2. **Idempotent:** Same task can be processed multiple times safely
3. **Graceful Shutdown:** Complete current task before terminating

**Processing Loop:**
```
while running:
    1. Pull message from queue (visibility timeout: 5 min)
    2. Check idempotency key in Table Storage
    3. If already processed → ack message, continue
    4. Fetch TestCase from Gold Store
    5. Invoke Target System (with circuit breaker)
    6. Run Evaluation Pipeline
    7. Write Scorecard to Table Storage (with idempotency key)
    8. Ack message
    9. On failure: message returns to queue (auto-retry)
```

**Scaling:**
- Min replicas: 1
- Max replicas: 50 (configurable)
- Scale trigger: Queue depth > 100 messages

### 5.3 Component: Gold Store (Azure AI Search)

**Index Schema:**
| Field | Type | Features |
|-------|------|----------|
| `id` | `string` | Key |
| `nl_query` | `string` | Searchable |
| `embedding` | `vector(1536)` | Vector search |
| `complexity_level` | `int` | Filterable |
| `tags` | `string[]` | Filterable |
| `api_version` | `string` | Filterable |

**Use Cases:**
1. Retrieve test cases by ID (batch fetch)
2. Find similar test cases (vector similarity) for deduplication
3. Filter by metadata (complexity, tags, version)

### 5.4 Component: Results Store (Azure Table Storage)

**Partition Strategy:**
```
PartitionKey: batch_id (or "UNBATCHED" for single tests)
RowKey: test_case_id
```

**Rationale:**
- Queries typically retrieve all results for a batch
- Partition key groups related scorecards for efficient retrieval
- Supports point queries by test_case_id

---

## 6. Management API

The Management API provides CRUD operations for all platform entities. Deployed as a separate Azure Container App (or Azure Functions) with REST endpoints.

### 6.1 API Overview

```
Management Plane (REST API)
├── /api/v1/test-cases       # Gold Standard CRUD
├── /api/v1/test-suites      # Test collections
├── /api/v1/target-systems   # LLM orchestrator configs
├── /api/v1/clients          # Tenant management
└── /api/v1/runs             # Evaluation execution

Execution Plane (Service Bus + Workers)
├── Queue: eval-tasks        # Worker tasks
└── Workers (ACA)            # Stateless processors
```

### 6.2 Test Case Management

```
POST   /api/v1/test-cases              # Create test case
GET    /api/v1/test-cases              # List/search test cases
GET    /api/v1/test-cases/{id}         # Get single test case
PUT    /api/v1/test-cases/{id}         # Update test case
DELETE /api/v1/test-cases/{id}         # Delete test case
POST   /api/v1/test-cases/import       # Bulk import (JSON/CSV)
GET    /api/v1/test-cases/export       # Bulk export
POST   /api/v1/test-cases/{id}/duplicate  # Clone with new ID
```

**Query Parameters for List:**
```
GET /api/v1/test-cases?tags=search,checkout&complexity_min=3&api_version=v2.0&limit=100&offset=0
```

### 6.3 Test Suite Management

```
POST   /api/v1/test-suites                          # Create suite
GET    /api/v1/test-suites                          # List suites
GET    /api/v1/test-suites/{id}                     # Get suite
PUT    /api/v1/test-suites/{id}                     # Update suite
DELETE /api/v1/test-suites/{id}                     # Delete suite
GET    /api/v1/test-suites/{id}/test-cases          # Get resolved test cases
POST   /api/v1/test-suites/{id}/test-cases          # Add test cases to suite
DELETE /api/v1/test-suites/{id}/test-cases/{tc_id}  # Remove test case
```

**Dynamic vs Static Suites:**
- **Static:** Explicit `test_case_ids` list
- **Dynamic:** Filter criteria (`filter_tags`, `filter_api_version`, etc.) resolved at runtime

### 6.4 Target System Management

```
POST   /api/v1/target-systems           # Register LLM config
GET    /api/v1/target-systems           # List configs
GET    /api/v1/target-systems/{id}      # Get config
PUT    /api/v1/target-systems/{id}      # Update config
DELETE /api/v1/target-systems/{id}      # Delete config
POST   /api/v1/target-systems/{id}/test # Test connectivity
```

**Example Target System:**
```json
{
  "name": "GPT-4o Production",
  "provider": "azure_openai",
  "model": "gpt-4o",
  "endpoint": "https://myorg.openai.azure.com",
  "api_version": "2024-02-15-preview",
  "temperature": 0.0,
  "available_tools": ["search_products", "get_order", "create_ticket"],
  "tool_schema_version": "v2.1"
}
```

### 6.5 Client Management

```
POST   /api/v1/clients              # Register client/tenant
GET    /api/v1/clients              # List clients
GET    /api/v1/clients/{id}         # Get client
PUT    /api/v1/clients/{id}         # Update client
DELETE /api/v1/clients/{id}         # Delete client
GET    /api/v1/clients/{id}/runs    # Get client's run history
```

**Example Client:**
```json
{
  "name": "Search Team",
  "test_suite_ids": ["suite-search-v2", "suite-regression"],
  "target_system_id": "gpt4o-prod",
  "schedule_cron": "0 2 * * *",
  "notify_emails": ["search-team@company.com"],
  "evaluation_config_overrides": {
    "execution_stage_enabled": true,
    "semantics_stage_enabled": false
  }
}
```

### 6.6 Evaluation Run Management

```
POST   /api/v1/runs                     # Trigger new run
GET    /api/v1/runs                     # List runs (filterable)
GET    /api/v1/runs/{id}                # Get run status/summary
DELETE /api/v1/runs/{id}                # Cancel running eval
GET    /api/v1/runs/{id}/scorecards     # Get all scorecards for run
GET    /api/v1/runs/{id}/failures       # Get failed test details
POST   /api/v1/runs/{id}/retry-failures # Retry only failed tests
```

**Trigger Run Request:**
```json
{
  "client_id": "client-search-team",
  "test_suite_id": "suite-search-v2",
  "target_system_id": "gpt4o-prod",
  "triggered_by": "api",
  "trigger_metadata": {
    "commit_sha": "abc123",
    "pr_number": 456
  }
}
```

**Run Response:**
```json
{
  "id": "run-uuid",
  "status": "running",
  "progress_pct": 45.2,
  "total_tests": 1000,
  "completed_tests": 452,
  "passed_tests": 440,
  "failed_tests": 12,
  "stage_pass_rates": {
    "syntax": 1.0,
    "logic": 0.98,
    "execution": 0.95
  }
}
```

### 6.7 Authentication & Authorization

| Endpoint Pattern | Required Role |
|-----------------|---------------|
| `GET /api/v1/*` | `reader` |
| `POST/PUT/DELETE /api/v1/test-cases/*` | `test-admin` |
| `POST/PUT/DELETE /api/v1/test-suites/*` | `suite-admin` |
| `POST/PUT/DELETE /api/v1/target-systems/*` | `platform-admin` |
| `POST/PUT/DELETE /api/v1/clients/*` | `platform-admin` |
| `POST /api/v1/runs` | `runner` |
| `DELETE /api/v1/runs/{id}` | `platform-admin` |

**Authentication:** Azure AD / Entra ID with OAuth 2.0 bearer tokens.

---

## 7. Reliability & Error Handling

### 7.1 Retry Strategy

| Component | Strategy | Max Attempts | Backoff |
|-----------|----------|--------------|---------|
| Service Bus | Built-in | 10 | Exponential |
| Target API | Circuit Breaker | 3 | Exponential (1s, 2s, 4s) |
| Azure OpenAI | Retry with jitter | 5 | Exponential + jitter |
| Table Storage | Retry | 3 | Linear (1s) |

### 7.2 Dead Letter Queue (DLQ)

Messages move to DLQ after max delivery attempts. DLQ handling:
1. Alert on DLQ depth > threshold
2. CLI command to inspect DLQ: `eval dlq list`
3. CLI command to replay: `eval dlq replay --all`

### 7.3 Circuit Breaker

Target systems are wrapped in circuit breakers:
```
States: CLOSED → OPEN → HALF_OPEN → CLOSED
Thresholds:
  - Failure threshold: 5 consecutive failures
  - Recovery timeout: 30 seconds
  - Half-open test: 1 request
```

### 7.4 Idempotency

**Implementation:**
```python
idempotency_key = f"{test_case_id}:{message_id}"
```

Before processing, check if key exists in Table Storage. If exists, skip processing and ack message.

---

## 8. Observability (OpenTelemetry)

All observability is built on **OpenTelemetry (OTel)**, exported to **Azure Monitor Application Insights**.

### 8.1 OpenTelemetry Setup

```python
# Instrumentation stack
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from azure.monitor.opentelemetry.exporter import (
    AzureMonitorTraceExporter,
    AzureMonitorMetricExporter,
    AzureMonitorLogExporter,
)
```

**Auto-instrumented:**
- FastAPI (API requests)
- HTTPX (outbound HTTP to LLMs)
- Azure SDK (Service Bus, Table Storage, AI Search)

### 8.2 Structured Logging

All logs include trace context for correlation:
```json
{
  "timestamp": "2026-01-19T10:30:00Z",
  "level": "INFO",
  "message": "Evaluation completed",
  "trace_id": "abc123...",
  "span_id": "def456...",
  "attributes": {
    "batch_id": "uuid",
    "test_case_id": "uuid",
    "worker_id": "worker-abc123",
    "stage": "logic",
    "duration_ms": 150
  }
}
```

### 8.3 Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `eval.tasks.processed` | Counter | Total tasks completed |
| `eval.tasks.failed` | Counter | Tasks failed |
| `eval.stage.duration_ms` | Histogram | Per-stage latency |
| `eval.queue.depth` | Gauge | Current queue size |
| `eval.circuit.state` | Gauge | Circuit breaker state (0=closed, 1=open, 2=half-open) |
| `eval.llm.tokens` | Counter | Tokens consumed by target LLM |
| `eval.llm.latency_ms` | Histogram | LLM response latency |

### 8.4 Distributed Tracing

Trace propagation across all components:
```
CLI/API → Service Bus → Worker → Target LLM → Evaluation Pipeline → Table Storage
   │          │            │          │              │                  │
   └──────────┴────────────┴──────────┴──────────────┴──────────────────┘
                        Single trace_id throughout
```

**Span hierarchy:**
```
eval.run (root)
├── eval.fetch_test_case
├── eval.invoke_target_system
│   └── llm.completion (auto-instrumented)
├── eval.stage.syntax
├── eval.stage.logic
├── eval.stage.execution (if enabled)
│   └── api.call (to actual APIs)
├── eval.stage.semantics (if enabled)
│   └── llm.completion (judge)
└── eval.write_scorecard
```

**Span attributes:**
- `eval.test_case_id`
- `eval.batch_id`
- `eval.run_id`
- `eval.stage`
- `eval.passed`
- `eval.score`

---

## 9. Security

### 9.1 Authentication

| Component | Method |
|-----------|--------|
| Service Bus | Managed Identity |
| Table Storage | Managed Identity |
| AI Search | API Key (rotated) |
| Azure OpenAI | Managed Identity |

### 9.2 Secrets Management

- All secrets in Azure Key Vault
- Workers fetch secrets at startup
- No secrets in code or config files

### 9.3 Network Security

- Workers in private VNet
- Service Bus with private endpoint
- No public internet access from workers

---

## 10. Configuration Management

### 10.1 Hierarchical Configuration

```
Priority (highest to lowest):
1. Environment variables
2. CLI arguments
3. Config file (config.yaml)
4. Defaults in code
```

### 10.2 Key Configuration Options

```yaml
evaluation:
  # Stage 3: CRITICAL - enable for production/nightly runs
  # Disable only for rapid CI iteration where cost/speed matters
  execution_stage_enabled: true

  # Stage 4: OPTIONAL - nice-to-have, not correctness
  # Enable when you want to assess NL response quality
  semantics_stage_enabled: false

  numeric_tolerance: 0.0001  # 0.01% for execution comparison
  llm_judge_model: "gpt-4o"
  llm_judge_temperature: 0.0

worker:
  max_concurrent_tasks: 10
  task_timeout_seconds: 300
  circuit_breaker_threshold: 5

queue:
  visibility_timeout_seconds: 300
  max_delivery_count: 10
  prefetch_count: 10
```

**Environment Profiles:**
```yaml
# config.ci.yaml - Fast feedback, no API costs
evaluation:
  execution_stage_enabled: false
  semantics_stage_enabled: false

# config.nightly.yaml - Full validation
evaluation:
  execution_stage_enabled: true
  semantics_stage_enabled: true

# config.release.yaml - Production gate
evaluation:
  execution_stage_enabled: true
  semantics_stage_enabled: false  # Correctness only
```

---

## 11. Local Development

### 11.1 Prerequisites

- Docker Desktop
- Python 3.11+
- Make

### 11.2 Quick Start

```bash
# Start local Azure emulators
make docker-up

# Run tests
make test

# Submit a test batch locally
make run-cli -- submit --batch-file sample_tests.json
```

### 11.3 Docker Compose Services

| Service | Port | Description |
|---------|------|-------------|
| azurite | 10000-10002 | Azure Storage emulator |
| service-bus-emulator | 5672 | Service Bus emulator |
| worker | - | Eval worker |

---

## 12. Directory Structure

```
eval-platform/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schemas.py          # Pydantic models (TestCase, Scorecard, etc.)
│   │   ├── evaluators.py       # Evaluator ABC + implementations
│   │   ├── ast_comparator.py   # AST-based tool call comparison
│   │   └── config.py           # Configuration management
│   ├── api/                    # Management API (REST)
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app entry
│   │   ├── dependencies.py     # DI, auth, etc.
│   │   └── routers/
│   │       ├── test_cases.py   # /api/v1/test-cases
│   │       ├── test_suites.py  # /api/v1/test-suites
│   │       ├── target_systems.py # /api/v1/target-systems
│   │       ├── clients.py      # /api/v1/clients
│   │       └── runs.py         # /api/v1/runs
│   ├── worker/                 # Execution plane
│   │   ├── __init__.py
│   │   ├── main.py             # Worker entry point
│   │   ├── consumer.py         # Service Bus consumer
│   │   ├── pipeline.py         # Evaluation pipeline orchestration
│   │   └── clients/
│   │       ├── gold_store.py   # Azure AI Search client
│   │       ├── results_store.py # Table Storage client
│   │       ├── target_invoker.py # LLM orchestrator client
│   │       └── llm_judge.py    # Azure OpenAI client (Stage 4)
│   └── shared/
│       ├── __init__.py
│       ├── circuit_breaker.py
│       ├── retry.py
│       └── telemetry.py
├── cli/
│   ├── __init__.py
│   ├── main.py                 # Typer CLI entry
│   └── commands/
│       ├── submit.py
│       ├── status.py
│       └── replay.py
├── infra/
│   ├── main.bicep              # Azure infrastructure
│   ├── modules/
│   │   ├── api.bicep           # Management API container app
│   │   ├── worker.bicep        # Worker container app
│   │   ├── search.bicep        # AI Search
│   │   ├── servicebus.bicep    # Service Bus
│   │   └── storage.bicep       # Table Storage
│   └── parameters/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 13. Implementation Roadmap (Phase 2)

### Sprint 1: Foundation
1. Scaffold directory structure
2. Implement core schemas (CONTRACTS.py)
3. Set up pyproject.toml with dependencies

### Sprint 2: Evaluation Logic
4. Implement `ASTComparator` for tool call comparison
5. Implement `SyntaxEvaluator`
6. Implement `LogicEvaluator`
7. Unit tests for evaluators

### Sprint 3: Infrastructure
8. Create Bicep templates for Azure resources
9. Set up docker-compose.yml with Azurite
10. Implement Service Bus consumer

### Sprint 4: Integration
11. Implement Gold Store client
12. Implement Results Store client
13. Implement LLM Judge client
14. Integration tests

### Sprint 5: CLI & Polish
15. Implement Typer CLI
16. Add observability (logging, metrics)
17. Documentation
18. End-to-end tests

---

## Appendix A: Decision Log

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| Queue | Kafka, RabbitMQ, Service Bus | Service Bus | Native Azure, dead-letter support, sessions |
| Results Store | Cosmos DB, Table Storage | Table Storage | Cost-effective for write-heavy, simple queries |
| Gold Store | Blob + Cosmos, AI Search | AI Search | Vector search for deduplication, filtering |
| AST Comparison | String diff, jsondiff, custom | Custom AST | Full control over type coercion, permutation |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| Gold Standard | The authoritative test case with expected outputs |
| Scorecard | Evaluation result for a single test execution |
| Tool Call | A function/API invocation with name and arguments |
| Waterfall | Sequential evaluation stages with dependencies |
| DLQ | Dead Letter Queue for failed messages |
