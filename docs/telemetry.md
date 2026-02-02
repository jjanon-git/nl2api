# Telemetry & Observability Guide

This document covers OTEL integration, metrics, tracing, and the observability stack for the Evalkit project.

## Observability Stack

The observability stack runs via `docker compose up -d`:

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Primary database |
| Redis | 6379 | Caching |
| OTEL Collector | 4317 (gRPC), 4318 (HTTP) | Receives telemetry |
| Prometheus | 9090 | Metrics storage, queries |
| Grafana | 3000 | Dashboards (admin/admin) |
| Jaeger | 16686 | Distributed tracing |

**Metrics flow:**
```
Application (OTLP) → OTEL Collector (4317) → Prometheus Exporter (8889) → Prometheus (9090) → Grafana
```

## When to Add Tracing Spans

Add OTEL spans for:
- External API calls (LLM providers, LSEG APIs)
- Database operations (queries, bulk inserts)
- Multi-step orchestration flows
- Cache operations (hits/misses)
- Retry/circuit breaker events
- **Evaluation code** (evaluators, batch runners, scorers)

## Evaluation Telemetry (REQUIRED)

**All evaluation code MUST integrate with OTEL.** The evaluation pipeline is critical for measuring system quality, and observability is essential for debugging and monitoring.

| Component | Required Telemetry |
|-----------|-------------------|
| Evaluators | Spans with `test_case.id`, `result.passed`, `result.score` |
| Batch runner | Spans for batch lifecycle + `BatchMetrics` for aggregates |
| Scorers | Spans with scoring details and `duration_ms` |

Use `src.evalkit.batch.metrics.BatchMetrics` for aggregate metrics:
```python
from src.evalkit.batch.metrics import get_metrics

metrics = get_metrics()
metrics.record_test_result(scorecard, batch_id, tags)
metrics.record_batch_complete(batch_job, duration_seconds)
```

## Span Implementation Pattern

```python
from src.common.telemetry import get_tracer

tracer = get_tracer(__name__)

async def process_query(self, query: str) -> Result:
    with tracer.start_as_current_span("process_query") as span:
        span.set_attribute("query.length", len(query))

        # Child span for LLM call
        with tracer.start_as_current_span("llm_call") as llm_span:
            llm_span.set_attribute("llm.provider", self.provider)
            response = await self.llm.complete(query)
            llm_span.set_attribute("llm.tokens", response.usage.total_tokens)

        span.set_attribute("result.status", "success")
        return result
```

## Required Span Attributes

| Operation Type | Required Attributes |
|---------------|---------------------|
| LLM calls | `llm.provider`, `llm.model`, `llm.tokens` |
| Database | `db.operation`, `db.table`, `db.rows_affected` |
| External API | `http.method`, `http.url`, `http.status_code` |
| Cache | `cache.hit`, `cache.key_prefix` |
| Agent | `agent.name`, `agent.confidence`, `query.category` |
| Evaluator | `test_case.id`, `test_case.category`, `result.passed`, `result.score` |
| Batch | `batch.id`, `batch.total_tests`, `batch.passed_count`, `batch.duration_ms` |

## Metrics to Track

Record metrics for:
- Request latency (histograms)
- Error rates (counters by error type)
- Cache hit rates
- LLM token usage
- Query throughput

```python
from src.common.telemetry import get_meter

meter = get_meter(__name__)
request_counter = meter.create_counter("evalkit.requests")
latency_histogram = meter.create_histogram("evalkit.latency_ms")

async def handle_request(self, query: str):
    start = time.time()
    request_counter.add(1, {"agent": self.name})
    try:
        result = await self.process(query)
        latency_histogram.record((time.time() - start) * 1000)
        return result
    except Exception:
        request_counter.add(1, {"agent": self.name, "error": "true"})
        raise
```

## Metric Naming Convention

**IMPORTANT:**
- OTEL Collector adds `evalkit_` prefix to all metrics (configured in `config/otel-collector-config.yaml`)
- Dashboard queries must use prefixed names: `evalkit_eval_batch_tests_total`, not `eval_batch_tests_total`
- OTEL adds `_total` suffix to counters: `eval_batch_tests_passed` becomes `evalkit_eval_batch_tests_passed_total`
- If Grafana shows no data, check metric names match what's in Prometheus

## Grafana Datasource UID

**IMPORTANT:**
- Dashboard JSON files reference datasources by `uid` (e.g., `"uid": "prometheus"`)
- Datasource config in `config/grafana/provisioning/datasources/` MUST specify matching `uid`
- If dashboards show "No data", verify datasource UID matches between dashboard and config

## Prerequisite for Batch Evaluation Metrics

1. Load fixtures: `python scripts/load-nl2api-fixtures.py --all`
2. Run batch: `python -m src.evalkit.cli.main batch run --pack nl2api --limit 10`
3. View in Grafana: http://localhost:3000 → "NL2API Evaluation & Accuracy" dashboard

## MCP Server Telemetry

Add OTEL spans for MCP operations:

| Operation | Required Attributes |
|-----------|---------------------|
| MCP Server | `server.name`, `client.session_id` |
| Tool calls | `tool.name`, `tool.success` |
| Resource reads | `resource.uri`, `resource.success` |
