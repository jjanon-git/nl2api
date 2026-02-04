# Evalkit + NL2API + RAG

A **general-purpose ML evaluation framework** with two complete reference implementations:
- **NL2API**: Natural language to structured API translation for financial data
- **RAG**: Retrieval-Augmented Generation system for SEC EDGAR filings

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

**Complete RAG implementation** for financial document Q&A with SEC EDGAR filings:

- **Hybrid retrieval** with vector + keyword search (pgvector)
- **Small-to-big** hierarchical chunk retrieval (1.2M chunks, 246 companies)
- **LLM generation** with Claude or GPT models
- **8-stage evaluation** including faithfulness, citation accuracy, rejection calibration
- **560 test fixtures** for SEC filing evaluation
- **30% end-to-end pass rate** (10x improvement from baseline)

---

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Start infrastructure (PostgreSQL + Redis + OTEL stack)
docker compose up -d

# Run unit tests (2,875 tests)
.venv/bin/python -m pytest tests/unit/ -v

# Run batch evaluation (requires fixtures in DB)
.venv/bin/python scripts/load-nl2api-fixtures.py --all
.venv/bin/python -m src.evalkit.cli.main batch run --pack nl2api --tag entity_resolution --limit 100

# View results in Grafana
open http://localhost:3000  # admin/admin
```

### Database Replication

To replicate the database on another machine (avoids re-running slow ingestion scripts):

```bash
# On source machine: export and upload to GitHub Release
python scripts/db-export.py                    # Export all tables (~2-5 GB)
python scripts/db-export.py --tables rag       # Or just RAG data (~300-500 MB)
python scripts/db-upload.py exports/evalkit_*.dump.gz

# On target machine: download and restore
docker compose up -d
python scripts/db-restore.py --download data-all-20260203
```

See [docs/database-replication.md](docs/database-replication.md) for table groups and detailed workflows.

---

## Integrating with Your RAG System

This section explains how to integrate Evalkit with an existing RAG implementation.

### Step 1: Format System Output

Your RAG system must produce output matching this structure:

```python
system_output = {
    # Required fields
    "response": "The generated answer text from your RAG system",
    "retrieved_chunks": [
        {
            "id": "chunk-123",           # Unique chunk identifier
            "text": "The actual chunk content...",
            "score": 0.85,               # Optional: retrieval score
            "metadata": {...}            # Optional: source info
        },
        # ... more chunks
    ],

    # Optional fields
    "retrieved_doc_ids": ["doc-1", "doc-2"],  # Alternative to extracted from chunks
    "sources": [...],                          # Source attribution metadata
}
```

### Step 2: Create Test Cases

Test cases define expected behavior. Use the `TestCase` model:

```python
from src.evalkit.contracts import TestCase

test_case = TestCase(
    id="rag-revenue-001",
    input={
        "query": "What was Apple's total revenue in FY2024?",
        "ticker": "AAPL",  # Optional: for entity filtering
    },
    expected={
        # For retrieval evaluation (which docs should be found)
        "relevant_docs": ["aapl-10k-2024-item7", "aapl-10k-2024-item8"],

        # For answer evaluation
        "answer": "Apple's total revenue in FY2024 was $383.3 billion",
        "answer_keywords": ["revenue", "383", "billion", "2024"],

        # For behavior testing
        "behavior": "answer",  # or "reject" for unanswerable queries

        # For citation evaluation
        "source_policy": "summarize",  # or "quote_only"
    },
    metadata={
        "api_version": "1.0.0",
        "complexity_level": 2,
        "tags": ("rag", "sec_filing", "revenue"),
    }
)
```

### Step 3: Run Evaluation

```python
from src.rag.evaluation import RAGPack, RAGPackConfig
from src.evalkit.contracts import EvalContext

# Configure the pack
config = RAGPackConfig(
    llm_provider="openai",           # or "anthropic" - tunes thresholds
    parallel_stages=True,            # Faster for OpenAI (token-based limits)
    context_relevance_threshold=0.35,
)

pack = RAGPack(config=config)

# Create evaluation context (optional, provides LLM judge)
context = EvalContext(
    llm_judge=your_llm_client,       # For faithfulness/relevance stages
    batch_id="my-batch-001",
)

# Run evaluation
scorecard = await pack.evaluate(test_case, system_output, context)

# Check results
print(f"Overall passed: {scorecard.overall_passed}")
print(f"Overall score: {scorecard.overall_score:.2f}")

for stage_name, result in scorecard.stage_results.items():
    print(f"  {stage_name}: {'PASS' if result.passed else 'FAIL'} ({result.score:.2f})")
```

### Step 4: Run Batch Evaluation

For evaluating many test cases at once:

```bash
# Load test fixtures into database
python scripts/load-rag-fixtures.py

# Run batch evaluation
python -m src.evalkit.cli.main batch run \
  --pack rag \
  --tag rag \
  --label "my-rag-baseline" \
  --mode generation \
  --limit 100
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
batch run --pack nl2api --tag entity_resolution --mode resolver --label baseline
batch run --pack nl2api --tag lookups --mode orchestrator --label baseline
```

### RAG Pack (8 stages)

Evaluates Retrieval-Augmented Generation systems:

| Stage | Purpose | Type | Default Weight |
|-------|---------|------|----------------|
| **Retrieval** | IR metrics (recall@k, precision@k, MRR) | Scored | 0.15 |
| **Context Relevance** | Retrieved context relevance | Scored | 0.15 |
| **Faithfulness** | Response grounded in context | Scored | 0.25 |
| **Answer Relevance** | Response answers the question | Scored | 0.15 |
| **Citation** | Citation presence and accuracy | Scored | 0.10 |
| **Source Policy** | Quote-only vs summarize enforcement | GATE | 0.05 |
| **Policy Compliance** | Content policy violations | GATE | 0.05 |
| **Rejection Calibration** | False positive/negative detection | Scored | 0.10 |

```bash
# Run RAG evaluation (560 test cases)
batch run --pack rag --tag rag --label my-experiment --mode generation
```

### RAG Retrieval Pack (2 stages)

Retrieval-only evaluation for fast iteration without LLM generation costs:

| Stage | Purpose | Default Weight |
|-------|---------|----------------|
| **Retrieval** | IR metrics | 0.60 |
| **Context Relevance** | Retrieved context quality | 0.40 |

```bash
# Run retrieval-only evaluation (no LLM generation)
batch run --pack rag-retrieval --tag rag --label retrieval-baseline --mode resolver
```

### Configuring RAG Pack

```python
from src.rag.evaluation import RAGPack, RAGPackConfig

config = RAGPackConfig(
    # Stage enablement
    retrieval_enabled=True,
    context_relevance_enabled=True,
    faithfulness_enabled=True,
    answer_relevance_enabled=True,
    citation_enabled=True,
    source_policy_enabled=True,
    policy_compliance_enabled=True,
    rejection_calibration_enabled=True,

    # LLM provider for automatic threshold tuning
    # OpenAI stack uses lower thresholds (gpt-5-nano with reasoning=minimal scores stricter)
    llm_provider="openai",  # or "anthropic"

    # Manual threshold overrides (used when llm_provider=None)
    retrieval_threshold=0.5,
    context_relevance_threshold=0.35,  # 0.25 for OpenAI
    faithfulness_threshold=0.4,
    answer_relevance_threshold=0.5,
    citation_threshold=0.6,

    # Custom weights (override defaults)
    custom_weights={
        "faithfulness": 0.30,  # Increase weight for faithfulness
        "retrieval": 0.20,
    },

    # Execution mode
    parallel_stages=True,  # True for OpenAI, False for Anthropic
)

pack = RAGPack(config=config)
```

### Provider-Specific Thresholds

Different LLM judge models score differently. Evalkit auto-tunes thresholds based on the judge model used for evaluation stages (faithfulness, context_relevance, answer_relevance):

| Provider | Judge Model | Context Relevance | Faithfulness | Answer Relevance |
|----------|-------------|------------------|--------------|------------------|
| **OpenAI** | gpt-5-nano | 0.25 | 0.40 | 0.50 |
| **Anthropic** | claude-3-5-haiku | 0.35 | 0.40 | 0.50 |

**Note:** GPT-5 models use `reasoning_effort="minimal"` for deterministic output (instead of `temperature=0` which they don't support).

Set `llm_provider` in config to auto-select appropriate thresholds.

---

## Adding Custom Evaluation Stages

Create custom stages to evaluate domain-specific aspects of your system.

### Stage Protocol

Stages must implement the `Stage` protocol:

```python
from dataclasses import dataclass, field
from typing import Any
from src.evalkit.contracts import StageResult, TestCase, EvalContext

@dataclass
class MyCustomStage:
    """Custom evaluation stage example."""

    # Required: stage name (must be unique within pack)
    name: str = field(default="my_custom_stage", init=False)

    # Required: is this a gate (hard stop on failure)?
    is_gate: bool = field(default=False, init=False)

    # Configurable threshold
    pass_threshold: float = 0.5

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext,
    ) -> StageResult:
        """
        Evaluate the system output against the test case.

        Args:
            test_case: Test case with expected values in test_case.expected
            system_output: Output from the target system
            context: Evaluation context with config and LLM judge

        Returns:
            StageResult with pass/fail, score, and metrics
        """
        # Extract what you need
        response = system_output.get("response", "")
        expected = test_case.expected.get("my_expected_field", "")

        # Compute your score (0.0 to 1.0)
        score = self._compute_score(response, expected)

        # Determine pass/fail
        passed = score >= self.pass_threshold

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=score,
            reason=f"Score {score:.2f} {'meets' if passed else 'below'} threshold {self.pass_threshold}",
            metrics={
                "my_metric_1": some_value,
                "my_metric_2": another_value,
            },
            artifacts={
                "debug_info": {"expected": expected, "actual": response[:100]},
            },
        )

    def _compute_score(self, response: str, expected: str) -> float:
        # Your scoring logic here
        ...
```

### LLM-as-Judge Stage Example

For stages that use LLM evaluation:

```python
@dataclass
class CustomLLMJudgeStage:
    """Stage that uses LLM-as-Judge for evaluation."""

    name: str = field(default="custom_llm_judge", init=False)
    is_gate: bool = field(default=False, init=False)
    pass_threshold: float = 0.6

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict[str, Any],
        context: EvalContext,
    ) -> StageResult:
        # Get LLM judge from context
        llm_judge = context.llm_judge
        if not llm_judge:
            return StageResult(
                stage_name=self.name,
                passed=True,
                score=1.0,
                reason="Skipped - no LLM judge configured",
            )

        # Build your prompt
        prompt = f"""
        Evaluate the following response for [your criteria].

        Query: {test_case.input.get('query')}
        Response: {system_output.get('response')}

        Score from 0.0 to 1.0 based on [criteria].
        Return JSON: {{"score": float, "reason": string}}
        """

        # Call LLM judge
        result = await llm_judge.evaluate(prompt)
        score = result.get("score", 0.0)

        return StageResult(
            stage_name=self.name,
            passed=score >= self.pass_threshold,
            score=score,
            reason=result.get("reason", ""),
        )
```

### Registering Custom Stages in a Pack

1. Create your stage in `src/rag/evaluation/stages/my_stage.py`
2. Export from `src/rag/evaluation/stages/__init__.py`
3. Add to pack's `_build_stages()` method:

```python
# In src/rag/evaluation/pack.py

class RAGPack:
    DEFAULT_WEIGHTS: dict[str, float] = {
        # ... existing weights ...
        "my_custom_stage": 0.10,  # Add weight for new stage
    }

    def _build_stages(self) -> list[Any]:
        stages = []
        # ... existing stages ...

        if self.config.my_custom_stage_enabled:
            stages.append(MyCustomStage(pass_threshold=0.5))

        return stages
```

---

## Batch Evaluation CLI Reference

### Basic Usage

```bash
python -m src.evalkit.cli.main batch run \
  --pack <pack_name> \
  --tag <tag> \
  --label <run_label> \
  [options]
```

### Required Options

| Option | Description |
|--------|-------------|
| `--pack`, `-p` | Evaluation pack: `nl2api`, `rag`, `rag-retrieval` |
| `--tag`, `-t` | Filter by tag (repeatable for OR logic) |
| `--label`, `-l` | Run label for tracking (e.g., "baseline-v1", "new-embedder") |

### Filtering Options

| Option | Description | Default |
|--------|-------------|---------|
| `--limit`, `-n` | Maximum test cases to run | All |
| `--min-complexity` | Minimum complexity level (1-5) | None |
| `--max-complexity` | Maximum complexity level (1-5) | None |
| `--source-type` | Filter by source: `customer`, `sme`, `synthetic`, `hybrid` | None |
| `--review-status` | Filter by status: `pending`, `approved`, `rejected` | None |

### Execution Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode`, `-m` | Response mode (see below) | `resolver` |
| `--concurrency`, `-c` | Concurrent evaluations | 10 |
| `--parallel-stages` | Run stages in parallel (OpenAI) | False |
| `--sequential-stages` | Run stages sequentially (Anthropic) | True |
| `--resume`, `-r` | Resume interrupted batch by ID | None |

### Response Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `resolver` | Entity resolution only | Fast iteration on retrieval |
| `orchestrator` | Full NL2API pipeline | End-to-end accuracy |
| `generation` | Full RAG pipeline (retrieval + LLM) | RAG system evaluation |
| `routing` | Query routing only | Router accuracy |
| `tool_only` | Single agent (requires `--agent`) | Agent comparison |

### Semantic Evaluation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--semantics` | Enable LLM-as-Judge (Stage 4) | False |
| `--semantics-model` | Override judge model | `claude-3-5-haiku-20241022` |
| `--semantics-threshold` | Minimum pass score | 0.7 |

### Temporal Options

| Option | Description | Default |
|--------|-------------|---------|
| `--eval-date` | Reference date (YYYY-MM-DD) | Today |
| `--temporal-mode` | Validation mode: `behavioral`, `structural`, `data` | `structural` |

### Distributed Execution

| Option | Description | Default |
|--------|-------------|---------|
| `--distributed`, `-d` | Use Redis queue with workers | False |
| `--workers`, `-w` | Number of worker processes | 4 |
| `--redis-url` | Redis connection URL | `redis://localhost:6379` |

### Other Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Override LLM model | Pack default |
| `--client` | Client type for tracking | `internal` |
| `--client-version` | Client version | None |
| `--description` | Run description | None |
| `--verbose`, `-v` | Detailed output | False |

### Example Workflows

```bash
# Quick RAG evaluation (50 tests)
batch run --pack rag --tag rag --label quick-test --mode generation --limit 50

# Full NL2API entity resolution baseline
batch run --pack nl2api --tag entity_resolution --label baseline-v1 --mode resolver

# RAG with semantic evaluation enabled
batch run --pack rag --tag rag --label with-semantics \
  --mode generation --semantics --semantics-threshold 0.6

# Distributed execution for large batches
batch run --pack rag --tag rag --label large-batch \
  --distributed --workers 8 --concurrency 20

# Resume interrupted batch
batch run --pack rag --tag rag --label my-run --resume abc123-def456

# Compare different models
batch run --pack nl2api --tag lookups --label haiku-test \
  --mode orchestrator --model claude-3-5-haiku-20241022
batch run --pack nl2api --tag lookups --label sonnet-test \
  --mode orchestrator --model claude-sonnet-4-20250514
```

### Viewing Results

```bash
# List recent batches
batch list

# View batch status
batch status <batch-id>

# View detailed results
batch results <batch-id>
batch results <batch-id> --failed  # Failed tests only

# Export results to JSON
batch results <batch-id> --export results.json

# Compare clients/versions
batch compare --client internal mcp_claude --start 2026-01-01

# View trends
batch trend --client internal --metric pass_rate --days 30
```

---

## Observability with OpenTelemetry

Evalkit includes full observability through OpenTelemetry, with pre-configured dashboards.

### Starting the Stack

```bash
# Start all infrastructure including OTEL stack
docker compose up -d

# Services started:
# - PostgreSQL (5432) - Primary database with pgvector
# - Redis (6379) - Caching and task queues
# - OTEL Collector (4317, 4318) - Telemetry collection
# - Prometheus (9090) - Metrics storage
# - Grafana (3000) - Dashboards
# - Jaeger (16686) - Distributed tracing
```

### Grafana Dashboards (localhost:3000)

Login with `admin/admin`, then navigate to Dashboards:

**NL2API Overview Dashboard**
- Pass rates by stage (syntax, logic, execution, semantics)
- Accuracy trends over time
- Cost tracking by model/client
- Latency distributions

**RAG Evaluation Dashboard**
- Stage-by-stage pass rates
- Retrieval metrics (recall@k, precision@k, MRR)
- Source type analysis
- Faithfulness and relevance scores

**Eval Infrastructure Dashboard**
- Worker status and throughput
- Queue depth and processing times
- Error rates by stage
- Batch completion rates

### Jaeger Tracing (localhost:16686)

Jaeger provides distributed tracing for debugging individual evaluations:

1. Open Jaeger UI at http://localhost:16686
2. Select service: `nl2api-evaluation` or `evalkit`
3. Search by:
   - `batch_id`: Find all traces for a batch
   - `test_case.id`: Find specific test evaluation
   - Time range: Narrow down to specific runs

**Trace Structure:**
```
batch_run
├── evaluate_test_case (per test)
│   ├── syntax_stage
│   ├── logic_stage
│   ├── execution_stage (if enabled)
│   └── semantics_stage (if enabled)
└── save_scorecard
```

Each span includes attributes:
- `test_case.id`, `batch_id`
- `result.passed`, `result.score`
- `stage.name`, `stage.is_gate`

### Prometheus Queries (localhost:9090)

Useful PromQL queries for monitoring:

```promql
# Pass rate by stage
sum(rate(evalkit_eval_stage_passed_total[1h])) by (stage)
  / sum(rate(evalkit_eval_stage_passed_total[1h] + evalkit_eval_stage_failed_total[1h])) by (stage)

# Total tests by pack
sum(evalkit_eval_batch_tests_total) by (pack_name)

# Cost tracking (micro-USD, divide by 1M for USD)
sum(rate(evalkit_eval_cost_usd_total[1h])) / 1000000

# Worker throughput
sum(rate(evalkit_eval_worker_tasks_processed_total[5m])) by (worker_id)

# Queue depth
evalkit_eval_queue_enqueued_total - evalkit_eval_queue_acked_total
```

### Adding Custom Instrumentation

Add tracing to your code:

```python
from src.evalkit.common.telemetry import get_tracer, get_meter

# Get tracer for your module
tracer = get_tracer(__name__)

# Create spans
async def my_operation():
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("custom.attribute", "value")
        span.set_attribute("result.count", 42)

        # Your code here
        result = await do_something()

        span.set_attribute("result.success", True)
        return result
```

Add custom metrics:

```python
from src.evalkit.common.telemetry import get_meter

meter = get_meter("my_module")

# Create counter
my_counter = meter.create_counter(
    name="my_operations_total",
    description="Total operations performed",
    unit="1",
)

# Create histogram
my_latency = meter.create_histogram(
    name="my_operation_duration_ms",
    description="Operation duration",
    unit="ms",
)

# Record metrics
my_counter.add(1, {"status": "success", "type": "query"})
my_latency.record(duration_ms, {"operation": "search"})
```

### Telemetry Configuration

Control telemetry via environment variables:

```bash
# Disable telemetry entirely
export EVALKIT_TELEMETRY_ENABLED=false

# Configure OTLP endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Service identification
export NL2API_TELEMETRY_SERVICE_NAME=my-service
export NL2API_ENV=production
```

Initialize telemetry in your application:

```python
from src.evalkit.common.telemetry import init_telemetry, shutdown_telemetry

# At startup
init_telemetry(
    service_name="my-service",
    otlp_endpoint="http://localhost:4317",
)

# At shutdown (ensures metrics/traces are flushed)
shutdown_telemetry()
```

---

## Configuration Reference

### Environment Variables

#### Storage

| Variable | Description | Default |
|----------|-------------|---------|
| `EVAL_BACKEND` | Storage backend: `postgres`, `memory`, `azure` | `postgres` |
| `EVAL_POSTGRES_URL` | PostgreSQL connection URL | `postgresql://localhost/evalkit` |
| `EVAL_AZURE_CONNECTION_STRING` | Azure Table Storage connection | None |

#### Telemetry

| Variable | Description | Default |
|----------|-------------|---------|
| `EVALKIT_TELEMETRY_ENABLED` | Enable/disable telemetry | `true` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | `http://localhost:4317` |
| `NL2API_TELEMETRY_SERVICE_NAME` | Service name for traces | `nl2api` |
| `NL2API_ENV` | Environment (development/production) | `development` |

#### LLM Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `NL2API_LLM_PROVIDER` | LLM provider: `anthropic`, `openai` | `anthropic` |
| `NL2API_ANTHROPIC_API_KEY` | Anthropic API key | None |
| `NL2API_OPENAI_API_KEY` | OpenAI API key | None |
| `EVAL_LLM_PROVIDER` | LLM provider for evaluation | `openai` |
| `EVAL_LLM_MODEL` | LLM model for generation | `gpt-5-nano` |
| `EVAL_LLM_JUDGE_MODEL` | LLM model for judge stages | `gpt-5-nano` |

#### RAG Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_UI_USE_SMALL_TO_BIG` | Enable small-to-big retrieval | `false` |
| `RAG_VECTOR_WEIGHT` | Hybrid search vector weight | `0.7` |
| `RAG_KEYWORD_WEIGHT` | Hybrid search keyword weight | `0.3` |

### RAGPackConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retrieval_enabled` | bool | `True` | Enable retrieval stage |
| `context_relevance_enabled` | bool | `True` | Enable context relevance stage |
| `faithfulness_enabled` | bool | `True` | Enable faithfulness stage |
| `answer_relevance_enabled` | bool | `True` | Enable answer relevance stage |
| `citation_enabled` | bool | `True` | Enable citation stage |
| `source_policy_enabled` | bool | `True` | Enable source policy gate |
| `policy_compliance_enabled` | bool | `True` | Enable policy compliance gate |
| `rejection_calibration_enabled` | bool | `True` | Enable rejection calibration |
| `llm_provider` | str | `None` | Provider for threshold tuning: `openai`, `anthropic` |
| `retrieval_threshold` | float | `0.5` | Pass threshold for retrieval |
| `context_relevance_threshold` | float | `0.35` | Pass threshold for context relevance |
| `faithfulness_threshold` | float | `0.4` | Pass threshold for faithfulness |
| `answer_relevance_threshold` | float | `0.5` | Pass threshold for answer relevance |
| `citation_threshold` | float | `0.6` | Pass threshold for citation |
| `custom_weights` | dict | `None` | Override stage weights |
| `parallel_stages` | bool | `False` | Run stages in parallel |

### LLMJudgeConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | `anthropic` | LLM provider |
| `model` | str | `claude-3-5-haiku-20241022` | Judge model |
| `temperature` | float | `0.0` | Sampling temperature |
| `max_tokens` | int | `512` | Max response tokens |
| `pass_threshold` | float | `0.7` | Default pass threshold |
| `timeout_ms` | int | `30000` | Request timeout |
| `max_retries` | int | `3` | Retry attempts |

### Configuration Precedence

1. **CLI flags** (highest priority)
2. **Environment variables**
3. **Config class defaults** (lowest priority)

Example:
```bash
# CLI flag takes precedence
batch run --pack rag --model claude-3-5-haiku-20241022
# ^ Uses Haiku even if EVAL_LLM_MODEL=gpt-5-nano is set
```

---

## RAG System Improvements

The RAG evaluation system has been significantly improved through systematic experimentation.

**Current Performance:**

| Metric | Baseline | Current | Improvement |
|--------|----------|---------|-------------|
| Retrieval Recall@5 | 23% | 44% | **+21% (1.9x)** |
| Context Relevance | 15% | 86% | **+71%** |
| End-to-End Pass | 3% | 30% | **+27% (10x)** |

**Key Improvements:**
- **Contextual chunking** with company/section prefixes (+27% recall)
- **Small-to-big retrieval** (search 512-char children, return 4000-char parents)
- **Entity filtering** for multi-tenant document collections
- **Provider-specific threshold tuning** (OpenAI vs Anthropic)

**Document Corpus:**
- 243K parent chunks (4000 chars each)
- 1.2M child chunks (512 chars each)
- 246 companies indexed from SEC filings

See [docs/plans/rag-00-improvements.md](docs/plans/rag-00-improvements.md) for detailed experiment results, A/B tests, and future investigation areas.

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
└── rag/                   560 cases (SEC filings evaluation)
```

### Running Tests

```bash
# Unit tests (fast, mocked dependencies)
pytest tests/unit/ -v

# Integration tests (requires docker compose up -d)
pytest tests/integration/ -v

# Accuracy tests (real LLM calls, requires API key)
pytest tests/accuracy/ -m tier1 -v   # Quick (~50 samples)
pytest tests/accuracy/ -m tier2 -v   # Standard (~200 samples)
pytest tests/accuracy/ -m tier3 -v   # Comprehensive (all)
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
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common issues and solutions |
| [docs/plans/rag-00-improvements.md](docs/plans/rag-00-improvements.md) | RAG improvement journey |

---

## License

[Add license information]
