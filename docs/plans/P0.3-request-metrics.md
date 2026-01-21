# P0.3: Request-Level Metrics

**Priority:** P0 (Critical)
**Effort:** 3-5 days
**Status:** 🔲 Not Started

---

## Problem Statement

Currently there is no way to measure system accuracy in production:
- No structured logging of request/response details
- No way to identify failure patterns
- No metrics for A/B testing comparisons
- No data for accuracy regression detection

---

## Goals

1. Emit structured JSON metrics for every request
2. Track key accuracy indicators (routing, entity resolution, tool generation)
3. Enable offline analysis and dashboarding
4. Provide foundation for A/B testing

---

## Relationship to P0.1 (Observability)

P0.3 focuses specifically on **accuracy metrics** while P0.1 covers **operational observability** (tracing, latency).

These are complementary:
- P0.1: "How long did each step take? Where are bottlenecks?"
- P0.3: "Was the routing correct? Did we generate the right tools?"

The `RequestMetrics` class from P0.1 will be extended here with accuracy-specific fields.

---

## Implementation Plan

### Phase 1: Define Metrics Schema (1 day)

#### 1.1 Extend RequestMetrics

**File:** `src/nl2api/observability/metrics.py` (extend from P0.1)

```python
"""Request metrics collection and emission."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """
    Comprehensive metrics collected for each request.

    Categories:
    - Identification: request_id, session_id, timestamp
    - Input: query, query_length, query_type
    - Entity Resolution: entities found, resolution method
    - Routing: domain, confidence, cache status, model used
    - Context: retrieval mode, counts, latency
    - Agent Processing: rule vs LLM, tokens, latency
    - Output: tool calls generated
    - Accuracy Signals: for offline analysis
    """

    # === Identification ===
    request_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # === Input ===
    query: str = ""
    query_length: int = 0
    query_expanded: bool = False
    query_expansion_context: str | None = None  # What context triggered expansion

    # === Entity Resolution ===
    entities_input: list[str] = field(default_factory=list)  # Entities extracted
    entities_resolved: dict[str, str] = field(default_factory=dict)  # name -> RIC
    entities_unresolved: list[str] = field(default_factory=list)  # Failed to resolve
    entity_resolution_method: str = "none"  # cache, static, fuzzy, api
    entity_resolution_latency_ms: int = 0

    # === Routing ===
    routing_domain: str = ""
    routing_confidence: float = 0.0
    routing_cached: bool = False
    routing_cache_type: str | None = None  # exact, semantic
    routing_model: str | None = None  # Model used if not cached
    routing_latency_ms: int = 0
    routing_alternatives: list[dict] = field(default_factory=list)  # Other considered domains

    # === Context Retrieval ===
    context_mode: str = "local"  # local, mcp, hybrid
    context_field_codes_count: int = 0
    context_examples_count: int = 0
    context_latency_ms: int = 0

    # === Agent Processing ===
    agent_domain: str = ""
    agent_used_llm: bool = False
    agent_rule_matched: str | None = None  # Which rule pattern matched
    agent_llm_model: str | None = None
    agent_tokens_prompt: int = 0
    agent_tokens_completion: int = 0
    agent_latency_ms: int = 0

    # === Output ===
    tool_calls_count: int = 0
    tool_calls: list[dict] = field(default_factory=list)  # Serialized tool calls
    tool_names: list[str] = field(default_factory=list)

    # === Clarification ===
    needs_clarification: bool = False
    clarification_type: str | None = None  # domain, entity, ambiguity

    # === Performance ===
    total_latency_ms: int = 0
    total_tokens: int = 0

    # === Status ===
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None

    # === Accuracy Signals (for offline analysis) ===
    # These are populated by comparing against expected outputs
    accuracy_routing_correct: bool | None = None  # Set by offline evaluation
    accuracy_tools_match: bool | None = None  # Set by offline evaluation
    accuracy_score: float | None = None  # Overall accuracy score

    def __post_init__(self):
        """Compute derived fields."""
        if self.query and not self.query_length:
            self.query_length = len(self.query)

        self.total_tokens = self.agent_tokens_prompt + self.agent_tokens_completion

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string for structured logging."""
        return json.dumps(self.to_dict(), default=str)

    def to_log_summary(self) -> str:
        """Generate concise log summary."""
        return (
            f"domain={self.routing_domain} "
            f"conf={self.routing_confidence:.2f} "
            f"cached={self.routing_cached} "
            f"llm={self.agent_used_llm} "
            f"tools={self.tool_calls_count} "
            f"latency={self.total_latency_ms}ms "
            f"tokens={self.total_tokens}"
        )
```

### Phase 2: Metrics Emission (1-2 days)

#### 2.1 Create Metrics Emitter

**File:** `src/nl2api/observability/emitter.py`

```python
"""Metrics emission to various backends."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.nl2api.observability.metrics import RequestMetrics

logger = logging.getLogger(__name__)


class MetricsEmitter(ABC):
    """Base class for metrics emission."""

    @abstractmethod
    async def emit(self, metrics: "RequestMetrics") -> None:
        """Emit metrics to backend."""
        ...


class LoggingEmitter(MetricsEmitter):
    """Emit metrics to Python logging (structured JSON)."""

    def __init__(self, logger_name: str = "nl2api.metrics"):
        self._logger = logging.getLogger(logger_name)

    async def emit(self, metrics: "RequestMetrics") -> None:
        """Emit as structured JSON log."""
        # Full metrics as JSON (for log aggregation)
        self._logger.info(
            "request_metrics",
            extra={"metrics_json": metrics.to_json()},
        )

        # Summary for quick debugging
        self._logger.info(f"Request: {metrics.to_log_summary()}")


class FileEmitter(MetricsEmitter):
    """Emit metrics to JSONL file."""

    def __init__(self, file_path: str):
        self._file_path = file_path

    async def emit(self, metrics: "RequestMetrics") -> None:
        """Append metrics as JSON line."""
        with open(self._file_path, "a") as f:
            f.write(metrics.to_json() + "\n")


class CompositeEmitter(MetricsEmitter):
    """Emit to multiple backends."""

    def __init__(self, emitters: list[MetricsEmitter]):
        self._emitters = emitters

    async def emit(self, metrics: "RequestMetrics") -> None:
        """Emit to all configured backends."""
        for emitter in self._emitters:
            try:
                await emitter.emit(metrics)
            except Exception as e:
                logger.warning(f"Metrics emission failed for {type(emitter)}: {e}")


# Global emitter instance
_emitter: MetricsEmitter | None = None


def configure_emitter(emitter: MetricsEmitter) -> None:
    """Configure the global metrics emitter."""
    global _emitter
    _emitter = emitter


def get_emitter() -> MetricsEmitter:
    """Get the configured emitter (defaults to logging)."""
    global _emitter
    if _emitter is None:
        _emitter = LoggingEmitter()
    return _emitter


async def emit_metrics(metrics: "RequestMetrics") -> None:
    """Emit metrics using the configured emitter."""
    emitter = get_emitter()
    await emitter.emit(metrics)
```

### Phase 3: Orchestrator Integration (1-2 days)

#### 3.1 Instrument Orchestrator with Metrics

**File:** `src/nl2api/orchestrator.py` (modifications)

```python
from src.nl2api.observability.metrics import RequestMetrics
from src.nl2api.observability.emitter import emit_metrics

async def process(
    self,
    query: str,
    session_id: str | None = None,
    clarification_response: str | None = None,
) -> NL2APIResponse:
    """Process with full metrics collection."""

    start_time = time.perf_counter()
    metrics = RequestMetrics(
        query=query,
        session_id=session_id,
    )

    try:
        # Step 1: Session management
        session = await self._conversation_manager.get_or_create_session(...)
        metrics.session_id = str(session.id)

        # Step 2: Query expansion
        if session.total_turns > 0:
            expansion_result = self._conversation_manager.expand_query(...)
            metrics.query_expanded = expansion_result.was_expanded
            if expansion_result.was_expanded:
                metrics.query_expansion_context = expansion_result.expansion_reason

        # Step 3: Entity resolution
        entity_start = time.perf_counter()
        if self._entity_resolver:
            resolved_entities = await self._entity_resolver.resolve(effective_query)
            metrics.entities_resolved = resolved_entities
            metrics.entities_input = list(resolved_entities.keys())
            # Determine resolution method from resolver stats
            metrics.entity_resolution_method = self._get_resolution_method()
        metrics.entity_resolution_latency_ms = int((time.perf_counter() - entity_start) * 1000)

        # Step 4: Routing
        routing_start = time.perf_counter()
        domain, confidence = await self._classify_query(effective_query)
        metrics.routing_domain = domain
        metrics.routing_confidence = confidence
        # Get additional routing info from router result
        if hasattr(self, '_last_router_result'):
            metrics.routing_cached = self._last_router_result.cached
            metrics.routing_model = self._last_router_result.model_used
        metrics.routing_latency_ms = int((time.perf_counter() - routing_start) * 1000)

        # Step 5: Context retrieval
        context_start = time.perf_counter()
        field_codes, query_examples = await self._retrieve_context(...)
        metrics.context_mode = getattr(self, '_context_mode', 'local')
        metrics.context_field_codes_count = len(field_codes)
        metrics.context_examples_count = len(query_examples)
        metrics.context_latency_ms = int((time.perf_counter() - context_start) * 1000)

        # Step 6: Agent processing
        agent_start = time.perf_counter()
        result = await agent.process(context)
        metrics.agent_domain = domain
        metrics.agent_used_llm = getattr(result, 'used_llm', True)
        metrics.agent_rule_matched = getattr(result, 'rule_matched', None)
        metrics.agent_tokens_prompt = getattr(result, 'tokens_prompt', 0)
        metrics.agent_tokens_completion = getattr(result, 'tokens_completion', 0)
        metrics.agent_latency_ms = int((time.perf_counter() - agent_start) * 1000)

        # Step 7: Output
        if result.tool_calls:
            metrics.tool_calls_count = len(result.tool_calls)
            metrics.tool_names = [tc.tool_name for tc in result.tool_calls]
            metrics.tool_calls = [
                {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                for tc in result.tool_calls
            ]

        # Clarification
        if result.needs_clarification:
            metrics.needs_clarification = True
            metrics.clarification_type = "agent"

        metrics.success = True

    except Exception as e:
        metrics.success = False
        metrics.error_type = type(e).__name__
        metrics.error_message = str(e)
        raise

    finally:
        metrics.total_latency_ms = int((time.perf_counter() - start_time) * 1000)
        # Emit metrics asynchronously (fire and forget)
        try:
            await emit_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to emit metrics: {e}")

    return response
```

### Phase 4: Analysis Tools (1 day)

#### 4.1 Metrics Analysis Script

**File:** `scripts/analyze_metrics.py`

```python
"""
Analyze request metrics from JSONL log files.

Usage:
    python scripts/analyze_metrics.py metrics.jsonl

Output:
    - Summary statistics
    - Routing accuracy breakdown
    - Latency percentiles
    - Error patterns
"""

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev


@dataclass
class MetricsAnalysis:
    """Analysis results."""

    total_requests: int = 0
    success_rate: float = 0.0

    # Routing
    routing_domains: Counter = None
    routing_cached_rate: float = 0.0
    routing_confidence_mean: float = 0.0

    # Agent
    agent_llm_rate: float = 0.0
    agent_rule_patterns: Counter = None

    # Latency
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Tokens
    tokens_mean: float = 0.0
    tokens_total: int = 0

    # Errors
    error_types: Counter = None


def analyze_metrics(file_path: str) -> MetricsAnalysis:
    """Analyze metrics from JSONL file."""

    metrics_list = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                metrics_list.append(json.loads(line))

    if not metrics_list:
        return MetricsAnalysis()

    analysis = MetricsAnalysis()
    analysis.total_requests = len(metrics_list)

    # Success rate
    successes = sum(1 for m in metrics_list if m.get("success", False))
    analysis.success_rate = successes / len(metrics_list)

    # Routing
    analysis.routing_domains = Counter(m.get("routing_domain", "unknown") for m in metrics_list)
    cached = sum(1 for m in metrics_list if m.get("routing_cached", False))
    analysis.routing_cached_rate = cached / len(metrics_list)
    confidences = [m.get("routing_confidence", 0) for m in metrics_list]
    analysis.routing_confidence_mean = mean(confidences) if confidences else 0

    # Agent
    llm_used = sum(1 for m in metrics_list if m.get("agent_used_llm", False))
    analysis.agent_llm_rate = llm_used / len(metrics_list)
    analysis.agent_rule_patterns = Counter(
        m.get("agent_rule_matched") for m in metrics_list if m.get("agent_rule_matched")
    )

    # Latency
    latencies = sorted(m.get("total_latency_ms", 0) for m in metrics_list)
    analysis.latency_p50 = latencies[len(latencies) // 2]
    analysis.latency_p95 = latencies[int(len(latencies) * 0.95)]
    analysis.latency_p99 = latencies[int(len(latencies) * 0.99)]

    # Tokens
    tokens = [m.get("total_tokens", 0) for m in metrics_list]
    analysis.tokens_mean = mean(tokens) if tokens else 0
    analysis.tokens_total = sum(tokens)

    # Errors
    analysis.error_types = Counter(
        m.get("error_type") for m in metrics_list if m.get("error_type")
    )

    return analysis


def print_analysis(analysis: MetricsAnalysis) -> None:
    """Print analysis results."""

    print("\n" + "=" * 60)
    print("NL2API Metrics Analysis")
    print("=" * 60)

    print(f"\nTotal Requests: {analysis.total_requests}")
    print(f"Success Rate: {analysis.success_rate:.1%}")

    print("\n--- Routing ---")
    print(f"Cache Hit Rate: {analysis.routing_cached_rate:.1%}")
    print(f"Mean Confidence: {analysis.routing_confidence_mean:.2f}")
    print("Domain Distribution:")
    for domain, count in analysis.routing_domains.most_common():
        pct = count / analysis.total_requests * 100
        print(f"  {domain}: {count} ({pct:.1f}%)")

    print("\n--- Agent ---")
    print(f"LLM Usage Rate: {analysis.agent_llm_rate:.1%}")
    print(f"Rule-Based Rate: {1 - analysis.agent_llm_rate:.1%}")
    if analysis.agent_rule_patterns:
        print("Top Rule Patterns:")
        for pattern, count in analysis.agent_rule_patterns.most_common(5):
            print(f"  {pattern}: {count}")

    print("\n--- Latency ---")
    print(f"P50: {analysis.latency_p50}ms")
    print(f"P95: {analysis.latency_p95}ms")
    print(f"P99: {analysis.latency_p99}ms")

    print("\n--- Tokens ---")
    print(f"Mean per Request: {analysis.tokens_mean:.0f}")
    print(f"Total: {analysis.tokens_total:,}")

    if analysis.error_types:
        print("\n--- Errors ---")
        for error_type, count in analysis.error_types.most_common():
            print(f"  {error_type}: {count}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_metrics.py <metrics.jsonl>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    analysis = analyze_metrics(file_path)
    print_analysis(analysis)
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/nl2api/observability/metrics.py` | RequestMetrics dataclass (extend from P0.1) |
| `src/nl2api/observability/emitter.py` | Metrics emission backends |
| `scripts/analyze_metrics.py` | Offline metrics analysis |
| `tests/unit/nl2api/observability/test_metrics.py` | Metrics tests |
| `tests/unit/nl2api/observability/test_emitter.py` | Emitter tests |

## Files to Modify

| File | Changes |
|------|---------|
| `src/nl2api/orchestrator.py` | Add metrics collection at each step |
| `src/nl2api/agents/base.py` | Return tokens_used, used_llm, rule_matched in AgentResult |
| `src/nl2api/agents/protocols.py` | Add metrics fields to AgentResult |
| `src/nl2api/config.py` | Add metrics configuration |

---

## Metrics Schema Summary

```json
{
  "request_id": "uuid",
  "session_id": "uuid",
  "timestamp": "iso8601",

  "query": "What is Apple's EPS?",
  "query_length": 21,
  "query_expanded": false,

  "entities_resolved": {"Apple": "AAPL.O"},
  "entity_resolution_method": "static",
  "entity_resolution_latency_ms": 2,

  "routing_domain": "estimates",
  "routing_confidence": 0.92,
  "routing_cached": true,
  "routing_cache_type": "exact",
  "routing_latency_ms": 3,

  "context_mode": "local",
  "context_field_codes_count": 5,
  "context_examples_count": 3,

  "agent_domain": "estimates",
  "agent_used_llm": false,
  "agent_rule_matched": "eps_pattern",
  "agent_latency_ms": 15,

  "tool_calls_count": 1,
  "tool_names": ["get_estimates"],

  "total_latency_ms": 45,
  "total_tokens": 0,

  "success": true
}
```

---

## Testing Plan

1. **Unit Tests**
   - Test RequestMetrics serialization
   - Test each emitter backend
   - Test metrics computation

2. **Integration Tests**
   - Verify metrics emitted for each request
   - Test file emission
   - Test metrics analysis script

3. **Manual Verification**
   - Run test queries
   - Verify JSONL output
   - Run analysis script

---

## Success Criteria

- [ ] Metrics emitted for every request
- [ ] All key fields populated accurately
- [ ] Analysis script produces valid output
- [ ] < 2ms overhead for metrics collection
- [ ] 90%+ test coverage

---

## Rollback Plan

1. Metrics emission is non-blocking (fire and forget)
2. Errors in emission are logged, not raised
3. Can disable via config flag
4. No impact on core request processing
