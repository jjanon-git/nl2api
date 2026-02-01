# Infra-04: Tempo Migration with Exemplars

**Priority:** P2 (Medium)
**Effort:** 3-5 days
**Status:** Planned
**Depends On:** infra-03-trace-scorecard-linking.md (recommended to complete first)

---

## Problem Statement

While trace-to-scorecard linking (infra-03) enables looking up traces from failures, it doesn't provide the reverse: clicking on a metric spike in Grafana to see which traces caused it.

**Exemplars** solve this by attaching trace IDs directly to metric data points. When you see a latency spike or error rate increase, you can click on the actual data point and jump to one of the traces that contributed to it.

---

## Goals

1. Replace Jaeger with Grafana Tempo for native Grafana integration
2. Enable exemplars on key metrics (latency, errors, scores)
3. Provide bidirectional navigation: metrics ↔ traces
4. Enable TraceQL for powerful trace queries

---

## Why Tempo Over Jaeger?

| Feature | Jaeger | Tempo |
|---------|--------|-------|
| **Exemplar support** | Requires manual setup | Native, automatic |
| **Grafana integration** | External links | Built-in panels, split view |
| **Query language** | Tag-based search | TraceQL (SQL-like) |
| **Storage backend** | Elasticsearch/Cassandra | Object storage (simple) |
| **Deployment** | Multiple components | Single binary |
| **Service graph** | Separate component | Built-in |
| **Cost** | Higher (complex storage) | Lower (object storage) |

### TraceQL Examples

```
# Find slow evaluations
{span.name="evaluator.evaluate"} | duration > 5s

# Find failed syntax stages
{span.name="evaluator.stage.syntax" && span.result.passed=false}

# Find evaluations for specific test case
{span.test_case.id="lookup-001"}

# Find all errors in semantics stage
{span.name="evaluator.stage.semantics" && status=error}
```

---

## Architecture Overview

### Current (Jaeger)

```
Application → OTEL Collector → Jaeger
                    ↓
              Prometheus ← Grafana (separate)
```

### Target (Tempo)

```
Application → OTEL Collector → Tempo ← Grafana (integrated)
                    ↓              ↑
              Prometheus ──────────┘
                    (exemplars link to traces)
```

---

## Implementation Plan

### Phase 1: Add Tempo to Infrastructure

**File:** `docker-compose.yml`

```yaml
services:
  # Remove or comment out jaeger
  # jaeger:
  #   image: jaegertracing/all-in-one:latest
  #   ...

  tempo:
    image: grafana/tempo:2.3.1
    command: ["-config.file=/etc/tempo.yaml"]
    volumes:
      - ./config/tempo/tempo.yaml:/etc/tempo.yaml
      - tempo-data:/var/tempo
    ports:
      - "3200:3200"    # Tempo API
      - "4317:4317"    # OTLP gRPC receiver
      - "4318:4318"    # OTLP HTTP receiver
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3200/ready"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  tempo-data:
```

### Phase 2: Tempo Configuration

**File:** `config/tempo/tempo.yaml`

```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
        http:
          endpoint: 0.0.0.0:4318

storage:
  trace:
    backend: local
    local:
      path: /var/tempo/traces
    wal:
      path: /var/tempo/wal

querier:
  search_default_result_limit: 20

metrics_generator:
  registry:
    external_labels:
      source: tempo
  storage:
    path: /var/tempo/generator/wal
    remote_write:
      - url: http://prometheus:9090/api/v1/write
        send_exemplars: true

overrides:
  defaults:
    metrics_generator:
      processors: [service-graphs, span-metrics]
```

### Phase 3: Update OTEL Collector

**File:** `config/otel-collector-config.yaml`

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  # Remove jaeger exporter
  # otlp/jaeger:
  #   endpoint: jaeger:4317
  #   tls:
  #     insecure: true

  # Add Tempo exporter
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true

  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: nl2api
    # Enable exemplars
    enable_open_metrics: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [otlp/tempo]  # Changed from otlp/jaeger

    metrics:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [prometheus]
```

### Phase 4: Update Grafana Datasources

**File:** `config/grafana/provisioning/datasources/datasources.yml`

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    uid: prometheus
    isDefault: true
    jsonData:
      # Enable exemplar support
      exemplarTraceIdDestinations:
        - name: trace_id
          datasourceUid: tempo

  - name: Tempo
    type: tempo
    access: proxy
    url: http://tempo:3200
    uid: tempo
    jsonData:
      tracesToLogsV2:
        datasourceUid: ''  # Add Loki UID if using logs
      tracesToMetrics:
        datasourceUid: prometheus
      serviceMap:
        datasourceUid: prometheus
      nodeGraph:
        enabled: true
      search:
        hide: false
      traceQuery:
        timeShiftEnabled: true
        spanStartTimeShift: '1h'
        spanEndTimeShift: '1h'
```

### Phase 5: Enable Exemplars in Application Code

**File:** `src/evalkit/common/telemetry/metrics.py`

OTEL SDK automatically attaches trace context as exemplars when recording metrics within an active span. Verify this is working:

```python
from opentelemetry import trace
from opentelemetry.metrics import get_meter

meter = get_meter(__name__)

# Histograms and counters automatically get exemplars
# when recorded within an active span
latency_histogram = meter.create_histogram(
    "eval_test_duration_ms",
    description="Evaluation duration in milliseconds",
)

def record_evaluation_metrics(duration_ms: float, passed: bool):
    """Record metrics - exemplars attached automatically from active span."""
    # If there's an active span, OTEL SDK attaches its trace_id as exemplar
    latency_histogram.record(duration_ms, {"stage": "total", "passed": str(passed)})
```

### Phase 6: Update Grafana Dashboards for Exemplars

**File:** `config/grafana/provisioning/dashboards/json/nl2api/nl2api-evaluation.json`

Enable exemplar display on histogram panels:

```json
{
  "panels": [
    {
      "title": "Evaluation Latency",
      "type": "timeseries",
      "datasource": {"uid": "prometheus"},
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum(rate(nl2api_eval_test_duration_ms_bucket[5m])) by (le))",
          "exemplar": true
        }
      ],
      "fieldConfig": {
        "defaults": {
          "custom": {
            "showPoints": "always",
            "pointSize": 5
          }
        }
      },
      "options": {
        "tooltip": {
          "mode": "single"
        }
      }
    }
  ]
}
```

### Phase 7: Add Trace Panel to Dashboard

Add a Tempo panel for trace search:

```json
{
  "title": "Recent Traces",
  "type": "traces",
  "datasource": {"uid": "tempo"},
  "targets": [
    {
      "queryType": "traceql",
      "query": "{span.name=\"evaluator.evaluate\"}"
    }
  ]
}
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `config/tempo/tempo.yaml` | Tempo configuration |
| `tests/integration/test_tempo_traces.py` | Verify trace export |

## Files to Modify

| File | Changes |
|------|---------|
| `docker-compose.yml` | Replace Jaeger with Tempo |
| `config/otel-collector-config.yaml` | Export to Tempo instead of Jaeger |
| `config/grafana/provisioning/datasources/datasources.yml` | Add Tempo, enable exemplars |
| `config/grafana/provisioning/dashboards/json/nl2api/nl2api-evaluation.json` | Enable exemplars, add trace panels |

---

## Migration Steps

1. **Backup current traces** (optional - they'll be lost)
   ```bash
   # Export important traces from Jaeger if needed
   ```

2. **Stop current stack**
   ```bash
   docker compose down
   ```

3. **Apply configuration changes**
   - Update docker-compose.yml
   - Add tempo.yaml
   - Update OTEL collector config
   - Update Grafana datasources

4. **Start new stack**
   ```bash
   docker compose up -d
   ```

5. **Verify Tempo is receiving traces**
   ```bash
   curl http://localhost:3200/ready
   # Run an evaluation
   python -m src.evalkit.cli.main batch run --pack nl2api --limit 5
   # Check traces in Grafana Explore → Tempo
   ```

6. **Verify exemplars**
   - Open Grafana → Explore → Prometheus
   - Query: `nl2api_eval_test_duration_ms_bucket`
   - Toggle "Exemplars" on
   - Should see purple dots linking to traces

---

## Testing Plan

1. **Infrastructure Tests**
   - Verify Tempo container starts healthy
   - Verify OTEL Collector connects to Tempo
   - Verify Prometheus receives metrics

2. **Trace Tests**
   - Run evaluation with telemetry enabled
   - Query traces via Tempo API
   - Query traces via Grafana Tempo datasource

3. **Exemplar Tests**
   - Run evaluations to generate metrics
   - Verify exemplars appear on histogram panels
   - Click exemplar → verify trace opens

4. **TraceQL Tests**
   - Query failed evaluations
   - Query by test_case_id
   - Query by duration

---

## Success Criteria

- [ ] Tempo running and healthy
- [ ] Traces visible in Grafana Explore → Tempo
- [ ] TraceQL queries working
- [ ] Exemplars visible on metric panels
- [ ] Click exemplar → opens trace in split view
- [ ] Service graph showing evaluation flow
- [ ] No trace data loss during migration

---

## Rollback Plan

1. Keep Jaeger configuration commented (not deleted)
2. If Tempo has issues:
   ```bash
   # Edit docker-compose.yml to re-enable Jaeger
   # Edit OTEL config to export to Jaeger
   docker compose down && docker compose up -d
   ```
3. Tempo data is isolated - won't affect Jaeger if restored

---

## Performance Considerations

- Tempo uses ~50% less memory than Jaeger for similar workloads
- Local storage is fine for development; use S3/GCS for production
- Trace retention configurable in tempo.yaml
- Consider sampling for high-volume production use

---

## Future Enhancements

After Tempo is stable:
- Add Loki for logs correlation (traces → logs)
- Enable span metrics generation (automatic RED metrics from traces)
- Set up alerts on trace-derived metrics
- Add custom TraceQL dashboards for common queries
