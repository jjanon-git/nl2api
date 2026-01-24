# Eval Matrix - Implementation Plan

**Status:** Phase 0 IMPLEMENTED
**Created:** 2026-01-21
**Completed:** 2026-01-21
**Scope:** Phase 0 (Foundation)

---

## 1. Naming & Scope

**Name: "Eval Matrix"**

This is a **multi-dimensional evaluation framework** that measures:

| Dimension | What It Covers |
|-----------|----------------|
| **WHAT** (Capability) | Orchestrator, Router, Resolver, Individual Agents, Tools |
| **HOW** (Configuration) | Which LLM powers it, which prompt version, which config |
| **AGAINST** (Baseline) | Expected tool calls, expected responses, other clients |

**Why "Matrix":**
- Captures the multi-dimensional nature (capabilities × LLMs × configs)
- Not just "orchestrator" - evaluates components at multiple levels
- CLI-friendly: `eval matrix run`, `eval matrix compare`

**Validation:** The Haiku vs Sonnet routing comparison (see BACKLOG.md "Haiku Routing Spike") demonstrated this capability manually. Eval Matrix formalizes and automates this workflow.

**Example use cases:**
```bash
# How does DatastreamAgent perform with Haiku vs Sonnet?
eval matrix run --component datastream --llm haiku --tag price_queries
eval matrix run --component datastream --llm sonnet --tag price_queries
eval matrix compare --runs run1,run2

# How does the full orchestrator compare across model versions?
eval matrix run --component orchestrator --llm claude-3-5-sonnet --client internal
eval matrix run --component orchestrator --llm claude-opus-4-5 --client internal
eval matrix compare --metric accuracy,cost,latency
```

---

## 2. Current State (What Already Exists)

**Key finding:** ~80% of the infrastructure already exists. Phase 0 is primarily integration and polish.

| Component | Status | Location |
|-----------|--------|----------|
| Scorecard fields (`client_type`, `client_version`, `eval_mode`, tokens, cost) | ✅ Exists | `CONTRACTS.py:587-615` |
| Database schema (columns + indexes) | ✅ Exists | `migrations/010_multi_client_eval.sql` |
| Repository queries (`get_by_client`, `get_comparison_summary`, `get_client_trend`) | ✅ Exists | `scorecard_repo.py:459-655` |
| BatchRunner propagation of client metadata | ✅ Exists | `runner.py:349-351` |
| Response generators (resolver, routing, orchestrator, tool_only) | ✅ Exists | `response_generators.py` |
| CLI compare/trend commands | ✅ Exists | `commands/batch.py:631-818` |
| Continuous evaluation framework | ✅ Exists | `continuous/scheduler.py`, `regression.py`, `alerts.py` |
| Grafana dashboard with client filters | ✅ Exists | `evaluation-dashboard.json` |

**What's Missing (Phase 0 Work):**

| Gap | Impact | Effort |
|-----|--------|--------|
| Token extraction not wired in `OrchestratorResult` | No cost tracking | 2-3 hrs |
| `eval matrix` CLI commands | No unified interface | 2-3 hrs |
| Existing scorecards have `client_type=NULL` | Historical data not labeled | 1 hr |
| No agent factory for component mode | CLI can't instantiate agents | 1 hr |
| Cost calculation not implemented | Estimated cost always NULL | 1 hr |

---

## 3. Phase 0 Scope (1-2 Weeks)

### Goal
Demonstrate end-to-end value: Run evaluations with different components and LLMs, compare results via CLI.

### Deliverables

1. **Wire token tracking through orchestrator** (Day 1-2)
2. **Add cost calculation** (Day 2)
3. **Create `eval matrix` CLI commands** (Day 3-4)
4. **Backfill existing scorecards** (Day 4)
5. **End-to-end demo: Router Haiku vs Sonnet** (Day 5)
6. **Documentation** (Day 5)

### Out of Scope
- Real MCP passthrough (Phase 1)
- Client registry table (Phase 1)
- Statistical significance testing (Phase 2)
- Multi-turn evaluation (Phase 3)

---

## 4. Implementation Details

### 4.1 Wire Token Tracking

**Problem:** `OrchestratorResult` doesn't include token counts from LLM calls.

**Files to modify:**
- `src/nl2api/orchestrator.py` - Track tokens in `OrchestratorResult`
- `CONTRACTS.py` - Verify `OrchestratorResult` has token fields

**Changes:**
```python
# In OrchestratorResult (add if missing)
@dataclass
class OrchestratorResult:
    tool_calls: list[ToolCall]
    nl_response: str | None
    input_tokens: int = 0      # NEW
    output_tokens: int = 0     # NEW

# In orchestrator.process() - accumulate tokens from LLM calls
result.input_tokens = sum(call.usage.input_tokens for call in llm_calls)
result.output_tokens = sum(call.usage.output_tokens for call in llm_calls)
```

**Verification:**
```bash
eval matrix run --component orchestrator --limit 5
# Check scorecards have non-null input_tokens
```

### 4.2 Add Cost Calculation

**Problem:** `estimated_cost_usd` is always NULL.

**Files to modify:**
- `src/evaluation/batch/runner.py` - Add cost calculation in `_evaluate_one()`

**Implementation:**
```python
# Pricing table (per 1M tokens)
PRICING = {
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku": {"input": 0.25, "output": 1.25},
    "claude-opus-4-5": {"input": 15.0, "output": 75.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
}

def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = PRICING.get(model, PRICING["claude-3-5-sonnet"])
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
```

### 4.3 Create Eval Matrix CLI Commands

**Files to create/modify:**
- `src/evaluation/cli/commands/matrix.py` - **New:** Matrix commands
- `src/evaluation/cli/main.py` - Register matrix subcommand
- `src/nl2api/agents/__init__.py` - Add agent factory

**CLI structure:**
```python
# src/evaluation/cli/commands/matrix.py
import typer

app = typer.Typer(name="matrix", help="Multi-dimensional evaluation commands")

@app.command()
def run(
    component: str = typer.Option(..., help="Component: orchestrator, router, resolver, datastream, estimates, etc."),
    llm: str = typer.Option("claude-3-5-sonnet", help="LLM model to use"),
    client: str = typer.Option("internal", help="Client identifier for tracking"),
    tag: str = typer.Option(None, help="Filter test cases by tag"),
    limit: int = typer.Option(None, help="Limit number of test cases"),
):
    """Run evaluation matrix for a component."""
    # Component determines the response generator
    if component == "orchestrator":
        generator = create_nl2api_generator(orchestrator)
    elif component == "router":
        generator = create_routing_generator(router)
    elif component == "resolver":
        generator = create_entity_resolver_generator(resolver)
    else:
        # Individual agent (datastream, estimates, etc.)
        agent = get_agent_by_name(component, llm_provider)
        generator = create_tool_only_generator(agent)

@app.command()
def compare(
    runs: str = typer.Option(..., help="Comma-separated batch IDs to compare"),
    metric: str = typer.Option("accuracy,cost,latency", help="Metrics to compare"),
):
    """Compare multiple evaluation runs."""
```

**Agent factory:**
```python
# src/nl2api/agents/__init__.py
def get_agent_by_name(name: str, llm: LLMProvider) -> DomainAgent:
    from .datastream import DatastreamAgent
    from .estimates import EstimatesAgent
    from .fundamentals import FundamentalsAgent
    from .officers import OfficersAgent
    from .screening import ScreeningAgent

    AGENTS = {
        "datastream": DatastreamAgent,
        "estimates": EstimatesAgent,
        "fundamentals": FundamentalsAgent,
        "officers": OfficersAgent,
        "screening": ScreeningAgent,
    }
    if name not in AGENTS:
        raise ValueError(f"Unknown agent: {name}. Valid: {list(AGENTS.keys())}")
    return AGENTS[name](llm=llm)
```

### 4.4 Backfill Existing Scorecards

**Files to create:**
- `scripts/backfill_client_type.py`

**Implementation:**
```python
async def backfill():
    await pool.execute("""
        UPDATE scorecards
        SET client_type = 'internal',
            eval_mode = COALESCE(eval_mode, 'orchestrator')
        WHERE client_type IS NULL
    """)
    print(f"Updated {result} scorecards")
```

### 4.5 Verification Workflow (First Test: Router Haiku vs Sonnet)

```bash
# 1. Run matrix eval for router with Sonnet (baseline from previous work)
eval matrix run --component router --llm claude-3-5-sonnet --tag routing --limit 270
# Returns: run_id = sonnet_routing

# 2. Run matrix eval for router with Haiku
eval matrix run --component router --llm claude-3-5-haiku --tag routing --limit 270
# Returns: run_id = haiku_routing

# 3. Compare the two runs
eval matrix compare --runs sonnet_routing,haiku_routing

# Expected output (based on previous Haiku spike results in BACKLOG.md):
# ┌─────────────────────┬───────┬───────────┬───────────┬──────────┬─────────┐
# │ Run                 │ LLM   │ Pass Rate │ Avg Score │ Est Cost │ Latency │
# ├─────────────────────┼───────┼───────────┼───────────┼──────────┼─────────┤
# │ sonnet_routing      │ sonnet│ 88.9%     │ 0.89      │ ~$0.81   │ ~1.5s   │
# │ haiku_routing       │ haiku │ 94.1%     │ 0.94      │ ~$0.08   │ ~1.0s   │
# └─────────────────────┴───────┴───────────┴───────────┴──────────┴─────────┘
# Summary: Haiku is 10x cheaper AND 5% more accurate for routing
```

---

## 5. Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `src/nl2api/orchestrator.py` | Track tokens in OrchestratorResult | P0 |
| `CONTRACTS.py` | Verify OrchestratorResult has token fields | P0 |
| `src/evaluation/batch/runner.py` | Add cost calculation | P0 |
| `src/evaluation/cli/commands/matrix.py` | **New:** Eval Matrix CLI commands | P0 |
| `src/evaluation/cli/main.py` | Register matrix subcommand | P0 |
| `src/nl2api/agents/__init__.py` | Add `get_agent_by_name()` factory | P0 |
| `scripts/backfill_client_type.py` | **New:** backfill script | P0 |
| `BACKLOG.md` | Update multi-client eval status | P1 |

---

## 6. Testing Strategy

### Unit Tests
```bash
# Test cost calculation
pytest tests/unit/evaluation/test_cost_calculation.py -v

# Test agent factory
pytest tests/unit/nl2api/test_agent_factory.py -v
```

### Integration Tests
```bash
# Test full matrix flow
pytest tests/integration/evaluation/test_matrix_flow.py -v
```

### Manual Verification (First Test)
1. Run router evaluation with Sonnet
2. Run router evaluation with Haiku
3. Compare results - should match previous Haiku spike findings
4. Verify cost tracking shows ~10x difference

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Orchestrator doesn't track tokens | Medium | Check first; may need to add to LLM provider |
| Tool-only mode breaks on fixtures | Low | Use fixtures with pre-resolved entities |
| Grafana doesn't show filtered data | Low | Check datasource UID, test manually |

---

## 8. Future Phases (Not in Scope)

### Phase 1: Real MCP Passthrough
- MCP server logs tool calls from external clients
- Client registry table for structured management
- Auto-detect model version changes

### Phase 2: Continuous Pipeline
- Scheduled evaluation runner
- Regression detection with statistical significance
- Alerting on accuracy drops

### Phase 3: Multi-API Support
- Bloomberg, Quandl API providers
- Mock execution with recorded responses

---

## 9. Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Name** | Eval Matrix | Captures multi-dimensional nature (capabilities × LLMs × configs) |
| **CLI** | `eval matrix run` | New subcommand, clean separation from existing `batch` |
| **Cost tracking** | Per-query | Store on each scorecard, enables drill-down analysis |
| **Token tracking** | Real tokens | Wire through OrchestratorResult, not estimates |
| **First test** | Router Haiku vs Sonnet | Validates against known results from previous spike |

---

## 10. Summary

**Eval Matrix** is a multi-dimensional evaluation framework that answers:

1. **How does component X perform with LLM Y?**
   - `eval matrix run --component datastream --llm haiku`
   - `eval matrix run --component datastream --llm sonnet`
   - `eval matrix compare`

2. **What's the cost/accuracy tradeoff?**
   - Per-query token and cost tracking
   - Compare across runs with different configurations

3. **How do different clients compare?**
   - `--client internal` vs `--client mcp_claude`
   - Same test suite, different orchestrators

**Phase 0 delivers:** Token tracking, cost calculation, new CLI commands, component-level evaluation, comparison reports.

**First validation:** Re-run Router Haiku vs Sonnet comparison to match previous spike results.
