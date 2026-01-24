# Multi-Client Evaluation Platform Design

**Status:** In Progress - Phase 0 Complete
**Created:** 2026-01-21
**Author:** Mostly Claude, with some minor assistance from Sid

---

## Executive Summary

Expand the evaluation platform to support:
1. **Multiple APIs** beyond LSEG (extensible tool ecosystem)
2. **Dual evaluation modes**: Internal orchestrator vs MCP-exposed tools
3. **Cross-client comparison**: Claude vs ChatGPT vs custom orchestrators
4. **Continuous evaluation**: Scheduled runs, regression detection, model upgrade tracking

---

## Current State Assessment

### What Exists (Well-Architected)

| Component | Status | Notes |
|-----------|--------|-------|
| **Data models** | ✅ Ready | `TargetSystemConfig`, `Client`, `EvaluationRun` already support multi-client |
| **Storage layer** | ✅ Ready | Protocol-based, backend-swappable |
| **Batch runner** | ✅ Ready | Configurable, pluggable response generator |
| **ClientContext** | ✅ Ready | Per-request correlation in MCP server |
| **OTEL integration** | ✅ Ready | Spans include client/request IDs |

### What's Missing

| Gap | Impact | Priority |
|-----|--------|----------|
| **Execution layer** | Can't compare actual API results | P0 |
| **Tool-level evaluation** | Can only test full pipeline | P0 |
| **Client/model tracking** | No historical comparison | P0 |
| **Continuous runner** | Manual batch runs only | P1 |
| **Multi-API abstraction** | LSEG-specific tool definitions | P1 |
| **Cost/latency tracking** | No token or API cost attribution | P1 |

---

## Architecture: What You're Building

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Evaluation Platform                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   Claude     │     │   ChatGPT    │     │   Custom     │   Orchestrators │
│  │   (via MCP)  │     │   (via MCP)  │     │  NL2API Orch │                │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘                │
│         │                    │                    │                         │
│         ▼                    ▼                    ▼                         │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                    MCP Tool Layer                            │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │           │
│  │  │Datastrm │ │Estimates│ │Fundmntls│ │Officers │ │Screen  │ │           │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └───┬────┘ │           │
│  └───────┼──────────┼──────────┼──────────┼─────────────┼──────┘           │
│          │          │          │          │             │                   │
│          ▼          ▼          ▼          ▼             ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                    API Abstraction Layer                     │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │           │
│  │  │  LSEG   │ │Bloomberg│ │  Quandl │ │  Custom │  ...       │           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Evaluation Layer                                     │
│                                                                              │
│  ┌──────────────────────────┐    ┌──────────────────────────┐              │
│  │   Orchestrator Eval      │    │      Tool Eval           │              │
│  │   ─────────────────      │    │      ─────────           │              │
│  │   • Routing accuracy     │    │   • Tool call accuracy   │              │
│  │   • Multi-tool chaining  │    │   • Parameter correctness│              │
│  │   • Error recovery       │    │   • Schema compliance    │              │
│  │   • NL response quality  │    │   • Execution validity   │              │
│  └──────────────────────────┘    └──────────────────────────┘              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                    Comparison Engine                         │           │
│  │   • Cross-client accuracy (Claude vs ChatGPT vs Custom)     │           │
│  │   • Model version tracking (gpt-4o vs gpt-4.5 vs claude-4)  │           │
│  │   • Cost/latency/accuracy trade-offs                        │           │
│  │   • Regression detection                                    │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Storage & History                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │   Time-Series Evaluation Data                                │           │
│  │   ──────────────────────────                                 │           │
│  │   • Scorecards with client_id, model_version, timestamp     │           │
│  │   • Aggregated metrics per (client, model, time_bucket)     │           │
│  │   • Trend analysis and regression alerts                    │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Modes: Orchestrator vs Tool

### Mode 1: Tool Evaluation (Isolated)

Tests individual tools in isolation, regardless of orchestrator.

```
Input:  NL query + pre-resolved entities
Output: Tool call (tool_name + arguments)

Example:
  Query: "Get Apple's P/E ratio"
  Entities: {"Apple": "AAPL.O"}
  Expected: fundamentals_query(rics=["AAPL.O"], fields=["TR.PERatio"])
```

**What this tests:**
- Can the tool translate NL → correct API call?
- Are parameters correct (fields, date ranges, etc.)?
- Is the schema valid?

**What this does NOT test:**
- Routing (which tool to use)
- Entity resolution
- Multi-tool orchestration

### Mode 2: Orchestrator Evaluation (End-to-End)

Tests the full pipeline from NL query to final response.

```
Input:  NL query only
Output: Final response (tool calls + NL summary)

Example:
  Query: "Compare Apple and Microsoft's P/E ratios"
  Expected:
    - Routing: fundamentals domain
    - Entities: Apple→AAPL.O, Microsoft→MSFT.O
    - Tool: fundamentals_query(rics=[...], fields=[...])
    - NL: "Apple's P/E is X, Microsoft's is Y..."
```

**What this tests:**
- Routing accuracy
- Entity resolution accuracy
- Tool call correctness
- Response synthesis quality

### Mode 3: MCP Client Evaluation (External Orchestrator)

Tests external orchestrators (Claude, ChatGPT) using MCP tools.

```
Setup: MCP server exposes tools
Input: NL query sent to external LLM with tool access
Output: LLM's tool calls captured and evaluated

Example:
  Query: "Compare Apple and Microsoft's P/E ratios" → Claude Desktop
  Claude calls: fundamentals_query(...)
  Capture + evaluate Claude's tool usage
```

**What this tests:**
- How well does Claude/ChatGPT use the tools?
- Do they chain tools correctly?
- Do they synthesize results properly?

---

## Data Model Extensions

### 1. Enhanced Scorecard (for cross-client tracking)

```python
class Scorecard(BaseModel):
    # ... existing fields ...

    # NEW: Client identification
    client_type: ClientType  # INTERNAL_ORCHESTRATOR | MCP_CLAUDE | MCP_CHATGPT | MCP_CUSTOM
    client_version: str      # e.g., "claude-opus-4.5", "gpt-4o-2025-01"

    # NEW: Evaluation mode
    eval_mode: EvalMode  # TOOL_ONLY | ORCHESTRATOR | MCP_PASSTHROUGH

    # NEW: Cost tracking
    input_tokens: int | None
    output_tokens: int | None
    api_calls_made: int
    estimated_cost_usd: float | None

    # NEW: Tool-level breakdown
    tool_results: tuple[ToolEvalResult, ...] | None  # For multi-tool queries

class ToolEvalResult(BaseModel):
    """Result for a single tool call within a multi-tool query."""
    tool_name: str
    expected: ToolCall | None
    actual: ToolCall | None
    passed: bool
    score: float
    error_code: ErrorCode | None
```

### 2. Client Registry

```python
class ClientType(str, Enum):
    INTERNAL_ORCHESTRATOR = "internal"
    MCP_CLAUDE = "mcp_claude"
    MCP_CHATGPT = "mcp_chatgpt"
    MCP_GEMINI = "mcp_gemini"
    MCP_CUSTOM = "mcp_custom"

class RegisteredClient(BaseModel):
    """A client that can be evaluated."""
    id: str
    type: ClientType
    name: str  # Human-readable

    # Version tracking
    model_version: str  # e.g., "claude-opus-4.5-20251101"
    tool_schema_version: str  # Which version of tools they're using

    # Configuration
    mcp_endpoint: str | None  # For MCP clients
    api_key_env_var: str | None  # Environment variable name

    # Metadata
    created_at: datetime
    is_active: bool
    last_evaluated_at: datetime | None
```

### 3. Comparison Result

```python
class ClientComparison(BaseModel):
    """Comparison across clients for same test suite."""
    comparison_id: str
    test_suite_id: str
    run_at: datetime

    # Per-client results
    results: dict[str, ClientMetrics]  # client_id → metrics

    # Comparative analysis
    best_accuracy: str  # client_id
    best_latency: str
    best_cost_efficiency: str

    # Statistical significance
    accuracy_significant: bool  # Is difference statistically significant?
    p_value: float | None

class ClientMetrics(BaseModel):
    client_id: str
    client_version: str

    # Accuracy
    overall_pass_rate: float
    stage_pass_rates: dict[str, float]

    # Performance
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Cost
    total_tokens: int
    total_api_calls: int
    estimated_cost_usd: float

    # Sample size
    total_tests: int
    failed_tests: int
```

---

## What You're NOT Thinking About

### A. Tool Composition Testing

When queries require multiple tools:

```
Query: "Which S&P 500 company has the highest P/E ratio and what do analysts recommend?"

Required tools:
1. screening_query → Get S&P 500 companies ranked by P/E
2. estimates_query → Get analyst recommendations for top result

Questions:
- Did the orchestrator call them in the right order?
- Did it pass the output of tool 1 as input to tool 2?
- Did it handle partial failures (tool 1 succeeds, tool 2 fails)?
```

**Recommendation:** Add `ToolChain` evaluation that tracks:
- Expected tool sequence
- Data flow between tools
- Partial success handling

### B. Ground Truth for Orchestration

For tools: You can define expected tool calls.
For orchestration: What's "correct" routing?

```
Query: "What's Apple's stock price?"

Valid routes:
- datastream_query (real-time price)
- fundamentals_query (might have price)

Which is "correct"? Both work.
```

**Recommendation:**
- Define "acceptable" routes (set), not single "correct" route
- Use semantic equivalence: both routes would produce valid answers
- Consider user intent (real-time vs last close)

### C. Execution Layer Gap

You currently have placeholder execution. For real comparison:

```
Tool call: datastream_query(rics=["AAPL.O"], fields=["P"])

Without execution:
- Can verify tool call is syntactically correct
- Cannot verify it returns correct data

With execution:
- Can verify actual data returned
- Can compare across API providers (LSEG vs Bloomberg)
```

**Recommendation:**
- Phase 1: Mock execution with recorded responses (replay)
- Phase 2: Sandboxed real execution (rate-limited, cached)
- Phase 3: Live execution with cost tracking

### D. Semantic Equivalence

Two different tool sequences might produce the same result:

```
Query: "Apple's P/E ratio"

Option A: fundamentals_query(fields=["TR.PERatio"])
Option B: fundamentals_query(fields=["TR.PriceClose", "TR.EPSActValue"]) → calculate

Both are "correct" but A is more efficient.
```

**Recommendation:**
- Score efficiency separately from correctness
- Define canonical forms for common queries
- Allow multiple acceptable solutions with efficiency weighting

### E. Multi-Turn Evaluation

Current tests are single-turn. Real usage:

```
Turn 1: "Show me Apple's financials"
Turn 2: "Compare to Microsoft" (requires context)
Turn 3: "What about their growth rates?" (requires context from 1 & 2)
```

**Recommendation:**
- Create `ConversationTestCase` with multiple turns
- Track context accumulation accuracy
- Evaluate clarification flow (ambiguous → clarified)

### F. Data Freshness Problem

```
Test case created: 2025-01-01
Expected: Apple P/E = 28.5

Today: 2026-01-21
Actual: Apple P/E = 32.1

Test fails - but is it wrong?
```

**Recommendation:**
- Use relative comparisons ("P/E should be a number")
- Record execution data with timestamps
- Allow expected ranges, not point values
- Mark time-sensitive tests as `dynamic_expected`

### G. Cost Attribution

Different orchestrators have different costs:

| Client | Input Tokens | Output Tokens | API Calls | Total Cost |
|--------|--------------|---------------|-----------|------------|
| Claude | 1,200 | 450 | 2 | $0.018 |
| ChatGPT | 1,100 | 520 | 3 | $0.022 |
| Custom | 800 | 300 | 1 | $0.008 |

**Recommendation:**
- Track tokens per evaluation
- Track API calls per evaluation
- Calculate cost using provider pricing
- Create cost-efficiency metric: accuracy per dollar

### H. Latency Budgets

```
Interactive use: < 2 seconds
Batch processing: < 30 seconds
Report generation: < 5 minutes

Different queries have different acceptable latencies.
```

**Recommendation:**
- Define latency SLOs per query category
- Track P50/P95/P99 per client
- Alert on SLO violations
- Separate latency from accuracy scoring

### I. A/B Testing Infrastructure

When comparing Claude vs ChatGPT:

```
Problem:
- 1000 test cases
- Claude passes 850 (85%)
- ChatGPT passes 840 (84%)

Is Claude actually better, or is this noise?
```

**Recommendation:**
- Calculate statistical significance (p-value)
- Use stratified sampling by query difficulty
- Control for query category distribution
- Report confidence intervals, not point estimates

### J. Prompt/Schema Versioning

```
v1.0: tools.py defines fundamentals_query with 10 parameters
v1.1: Added 2 new parameters
v1.2: Renamed a parameter

Old test cases may fail on new schema.
```

**Recommendation:**
- Version tool schemas explicitly
- Migrate test cases when schema changes
- Support multiple schema versions simultaneously
- Track which test cases work with which versions

### K. Feedback Loop from Production

```
Production error:
  Query: "Get Tesla's earnings"
  Tool call: estimates_query(rics=["TSLA.O"], fields=["TR.Revenue"])  ← Wrong field

This should become a test case.
```

**Recommendation:**
- Log all production queries + tool calls
- Flag failed/corrected queries
- Auto-generate test cases from production errors
- Human review for edge cases

### L. Model Upgrade Detection

```
Claude updates from opus-4 to opus-4.5
ChatGPT updates from gpt-4o to gpt-4.5

How do you know without manual tracking?
```

**Recommendation:**
- Query model version at evaluation time
- Detect version changes automatically
- Trigger re-evaluation on version change
- Compare before/after metrics

---

## Continuous Evaluation Framework

### Triggers

| Trigger | Action |
|---------|--------|
| **Scheduled (cron)** | Daily/weekly full evaluation per client |
| **Model version change** | Automatic re-evaluation of affected client |
| **Tool schema change** | Re-evaluate all clients on new schema |
| **New test cases added** | Evaluate new cases against all clients |
| **Manual** | On-demand evaluation for debugging |

### Pipeline

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│  Trigger   │────▶│   Fetch    │────▶│  Execute   │────▶│   Store    │
│            │     │ Test Cases │     │ Evaluation │     │  Results   │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
                                                                │
                   ┌────────────┐     ┌────────────┐            │
                   │   Alert    │◀────│  Analyze   │◀───────────┘
                   │            │     │ Regression │
                   └────────────┘     └────────────┘
```

### Regression Detection

```python
class RegressionAlert(BaseModel):
    alert_id: str
    client_id: str
    detected_at: datetime

    # What regressed
    metric: str  # "overall_accuracy" | "routing_accuracy" | "p95_latency" | ...
    previous_value: float
    current_value: float
    change_pct: float

    # Context
    previous_run_id: str
    current_run_id: str
    affected_categories: list[str]  # Which query categories regressed

    # Severity
    severity: AlertSeverity  # LOW | MEDIUM | HIGH | CRITICAL
    is_statistically_significant: bool
```

### Alerting Rules

```python
ALERT_THRESHOLDS = {
    "overall_accuracy": {
        "warning": -0.02,   # 2% drop
        "critical": -0.05,  # 5% drop
    },
    "p95_latency_ms": {
        "warning": 1.5,     # 50% increase
        "critical": 2.0,    # 100% increase
    },
    "cost_per_query_usd": {
        "warning": 1.2,     # 20% increase
        "critical": 1.5,    # 50% increase
    },
}
```

---

## Implementation Phases

### Phase 1: Foundation (P0)

1. **Add client tracking to Scorecard**
   - `client_type`, `client_version`, `eval_mode`
   - Migrate existing scorecards (default to INTERNAL_ORCHESTRATOR)

2. **Tool-level evaluation mode**
   - Create `ToolEvaluator` that bypasses routing
   - Input: query + pre-resolved entities → expected tool call

3. **MCP passthrough evaluation**
   - Capture tool calls from MCP clients
   - Route to evaluation pipeline

4. **Basic comparison queries**
   - "Show me accuracy by client for last 7 days"
   - "Compare Claude vs ChatGPT on fundamentals queries"

### Phase 2: Continuous Pipeline (P1)

1. **Scheduled evaluation runner**
   - Cron-based trigger
   - Per-client schedules

2. **Regression detection**
   - Compare with previous run
   - Statistical significance testing
   - Alert generation

3. **Model version tracking**
   - Auto-detect version changes
   - Trigger re-evaluation

### Phase 3: Multi-API Support (P1)

1. **API abstraction layer**
   - Define `APIProvider` protocol
   - Implement LSEG provider
   - Stub Bloomberg/Quandl providers

2. **Execution layer**
   - Mock execution with recorded responses
   - Cost tracking per API call

### Phase 4: Advanced Analysis (P2)

1. **Tool composition testing**
   - Multi-tool sequence evaluation
   - Data flow validation

2. **A/B testing infrastructure**
   - Statistical significance
   - Stratified sampling

3. **Production feedback loop**
   - Log production queries
   - Auto-generate test cases

---

## Storage Schema Additions

```sql
-- Client registry
CREATE TABLE registered_clients (
    id VARCHAR(100) PRIMARY KEY,
    type VARCHAR(50) NOT NULL,  -- ClientType enum
    name VARCHAR(255) NOT NULL,
    model_version VARCHAR(100),
    tool_schema_version VARCHAR(50),
    mcp_endpoint TEXT,
    api_key_env_var VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_evaluated_at TIMESTAMPTZ
);

-- Extended scorecard columns
ALTER TABLE scorecards ADD COLUMN client_type VARCHAR(50);
ALTER TABLE scorecards ADD COLUMN client_version VARCHAR(100);
ALTER TABLE scorecards ADD COLUMN eval_mode VARCHAR(50);
ALTER TABLE scorecards ADD COLUMN input_tokens INTEGER;
ALTER TABLE scorecards ADD COLUMN output_tokens INTEGER;
ALTER TABLE scorecards ADD COLUMN api_calls_made INTEGER;
ALTER TABLE scorecards ADD COLUMN estimated_cost_usd DECIMAL(10, 6);

-- Comparison results
CREATE TABLE client_comparisons (
    id VARCHAR(100) PRIMARY KEY,
    test_suite_id VARCHAR(100) NOT NULL,
    run_at TIMESTAMPTZ NOT NULL,
    results JSONB NOT NULL,  -- ClientMetrics per client
    best_accuracy VARCHAR(100),
    best_latency VARCHAR(100),
    best_cost_efficiency VARCHAR(100),
    accuracy_significant BOOLEAN,
    p_value DECIMAL(10, 6)
);

-- Regression alerts
CREATE TABLE regression_alerts (
    id VARCHAR(100) PRIMARY KEY,
    client_id VARCHAR(100) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL,
    metric VARCHAR(100) NOT NULL,
    previous_value DECIMAL(10, 4) NOT NULL,
    current_value DECIMAL(10, 4) NOT NULL,
    change_pct DECIMAL(10, 4) NOT NULL,
    previous_run_id VARCHAR(100),
    current_run_id VARCHAR(100),
    affected_categories JSONB,
    severity VARCHAR(20) NOT NULL,
    is_statistically_significant BOOLEAN,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100)
);

-- Indexes
CREATE INDEX idx_scorecards_client ON scorecards(client_type, client_version);
CREATE INDEX idx_scorecards_eval_mode ON scorecards(eval_mode);
CREATE INDEX idx_comparisons_suite ON client_comparisons(test_suite_id);
CREATE INDEX idx_alerts_client ON regression_alerts(client_id, detected_at);
```

---

## Open Questions

1. **MCP capture mechanism**: How do we capture tool calls from Claude Desktop/ChatGPT?
   - Option A: Proxy server that logs calls
   - Option B: MCP server logs all incoming calls
   - Option C: External orchestrator reports results

2. **Cost tracking granularity**: Per-query or per-batch?
   - Per-query: More accurate, more storage
   - Per-batch: Less accurate, less overhead

3. **Multi-turn scope**: Defer or include in Phase 1?
   - Recommend: Defer to Phase 3 (complex, less common)

4. **Execution layer priority**: Mock-first or real-first?
   - Recommend: Mock-first with recorded responses (cheaper, reproducible)

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Can run tool-only evaluation | ✅ Phase 1 |
| Can compare Claude vs ChatGPT accuracy | ✅ Phase 1 |
| Can detect 5% regression automatically | ✅ Phase 2 |
| Can track cost per orchestrator | ✅ Phase 2 |
| Supports 3+ API providers | ✅ Phase 3 |
| < 5 min to add new API provider | ✅ Phase 3 |
