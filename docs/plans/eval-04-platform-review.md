# Evaluation Platform Review: Skeptical Assessment (Revised)

**Reviewer perspective:** Principal Data Scientist / ML Engineer evaluating for production use
**Date:** 2026-01-23
**Scope:** Evaluation platform only (excluding nl2api components)
**Target:** General-purpose ML evaluation framework (not just NL2API)

---

## Executive Summary

With the goal of a **general-purpose ML evaluation framework**, my assessment changes substantially:

| Aspect | Original Take | Revised Take |
|--------|---------------|--------------|
| Distributed workers | Over-engineered | ✅ Keep - needed for scale |
| Queue abstractions | Unnecessary complexity | ✅ Keep - enables different workloads |
| Protocol-based storage | Good | ✅ Excellent - core strength |
| Multi-tenant models | Remove | ⚠️ Revisit - may be needed |
| 4-stage waterfall | Rigid but fine | ❌ **Problem** - too NL2API-specific |
| Test case schema | Too specific | ❌ **Critical problem** - blocks other use cases |
| Hardcoded scoring | Bad | ❌ **Critical problem** - must be pluggable |

**Revised Verdict:** The infrastructure is actually well-suited for a general framework. The problem is that **NL2API assumptions are baked into the wrong layers**. You need to extract a generic core and make NL2API one "evaluation pack" among many.

---

## 1. WHAT'S GOOD (RE-EVALUATED)

### Infrastructure That Scales

These architectural decisions are **correct for a general framework**:

| Component | Why It's Right |
|-----------|---------------|
| **Distributed workers** | RAG evaluation at scale needs parallelism. Keep it. |
| **Queue abstraction** | Different eval workloads have different characteristics. Memory for dev, Redis for prod. |
| **Protocol-based repos** | Swap storage without changing eval logic. Essential for enterprise adoption. |
| **OTEL integration** | Eval runs are production workloads. Observability is non-negotiable. |
| **Immutable scorecards** | Audit trail for evaluation history. Required for regression tracking. |
| **Batch job tracking** | Resume interrupted runs, track progress. Essential for large evals. |

### Core Abstractions Worth Keeping

```
┌─────────────────────────────────────────────────────────────┐
│  KEEP: Generic Infrastructure                                │
├─────────────────────────────────────────────────────────────┤
│  • Storage protocols (TestCaseRepository, ScorecardRepository)
│  • Queue protocols (TaskQueue, ResultQueue)                  │
│  • Batch runner (job tracking, progress, resume)            │
│  • Worker pool (parallel execution)                          │
│  • Telemetry integration                                     │
│  • Cost tracking                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. WHAT'S WRONG (RE-EVALUATED)

### Critical: NL2API Baked Into Core

The problem isn't over-engineering—it's that **domain-specific assumptions leaked into generic layers**.

#### Problem 1: Test Case Schema Is NL2API-Specific

```python
# Current schema in CONTRACTS.py
class TestCase:
    nl_query: str                           # Assumes NL input
    expected_tool_calls: tuple[ToolCall]    # Assumes tool-calling LLM
    expected_nl_response: str               # Assumes NL output
```

**This blocks:**
- RAG evaluation (input: query, expected: relevant_chunks + answer)
- Classification (input: text, expected: label)
- Summarization (input: document, expected: summary)
- Embedding quality (input: pairs, expected: similarity scores)

#### Problem 2: 4-Stage Pipeline Is NL2API-Specific

```
Syntax → Logic → Execution → Semantics
```

This assumes:
1. Output must be parseable (Syntax) - not true for free-form generation
2. There are "tool calls" to compare (Logic) - RAG has no tool calls
3. There's an API to execute against (Execution) - not universal
4. NL response to judge (Semantics) - RAG needs different criteria

**RAG evaluation needs:**
```
Retrieval Quality → Context Relevance → Answer Faithfulness → Answer Relevance
```

#### Problem 3: Scoring Assumes Tool-Call Evaluation

```python
# Hardcoded in evaluators.py
weights = {
    "syntax": 0.1,      # JSON parsing
    "logic": 0.3,       # Tool call match
    "execution": 0.5,   # API result match
    "semantics": 0.1    # NL response quality
}
```

**RAG would need:**
```python
weights = {
    "retrieval_recall": 0.25,    # Did we find the right docs?
    "retrieval_precision": 0.15, # Did we avoid irrelevant docs?
    "faithfulness": 0.35,        # Is answer grounded in context?
    "relevance": 0.25,           # Does answer address the question?
}
```

#### Problem 4: Scorecard Assumes NL2API Results

```python
class Scorecard:
    syntax_result: StageResult      # NL2API-specific
    logic_result: StageResult       # NL2API-specific
    execution_result: StageResult   # NL2API-specific
    semantic_result: StageResult    # NL2API-specific
```

No place for RAG-specific metrics like `recall@5`, `MRR`, `NDCG`.

---

## 3. WHAT RAG EVALUATION NEEDS

For context on what "general-purpose" means, here's what RAG evaluation requires:

### RAG Metrics (Industry Standard)

| Metric | What It Measures | How to Compute |
|--------|------------------|----------------|
| **Recall@k** | Did relevant docs appear in top-k? | `len(retrieved ∩ relevant) / len(relevant)` |
| **Precision@k** | What fraction of top-k are relevant? | `len(retrieved ∩ relevant) / k` |
| **MRR** | Where does first relevant doc appear? | `1 / rank_of_first_relevant` |
| **NDCG** | Are relevant docs ranked higher? | Normalized DCG formula |
| **Context Relevance** | Is retrieved context useful for query? | LLM-as-judge or embedding similarity |
| **Faithfulness** | Is answer grounded in context? | LLM-as-judge (claim verification) |
| **Answer Relevance** | Does answer address the question? | LLM-as-judge |

### RAG Test Case Schema

```python
class RAGTestCase:
    query: str
    expected_relevant_docs: list[str]       # Doc IDs for retrieval metrics
    expected_answer: str | None             # For answer quality metrics
    context_for_generation: list[str] | None  # If testing generation only
    metadata: dict                          # Arbitrary metadata
```

### RAG Evaluation Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  RAG Evaluation Pipeline                                      │
├──────────────────────────────────────────────────────────────┤
│  1. Retrieval Stage                                           │
│     • Input: query → System: retriever → Output: doc_ids      │
│     • Metrics: recall@k, precision@k, MRR, NDCG              │
│                                                               │
│  2. Context Relevance Stage                                   │
│     • Input: query + retrieved_docs                           │
│     • Metrics: LLM-judge relevance score per doc             │
│                                                               │
│  3. Generation Stage (optional)                               │
│     • Input: query + context → System: generator → answer     │
│     • Metrics: faithfulness, relevance                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. PROPOSED ARCHITECTURE: EVALUATION PACKS

### Core Framework (Generic)

```python
# Generic test case - input/output with arbitrary schema
@dataclass(frozen=True)
class TestCase:
    id: str
    input: dict           # Arbitrary input schema
    expected: dict        # Arbitrary expected output schema
    metadata: Metadata    # Tags, category, etc.

# Generic stage result
@dataclass(frozen=True)
class StageResult:
    stage_name: str       # Arbitrary stage name
    passed: bool
    score: float
    metrics: dict         # Stage-specific metrics
    duration_ms: int
    error: str | None

# Generic scorecard - variable number of stages
@dataclass(frozen=True)
class Scorecard:
    id: str
    test_case_id: str
    stage_results: dict[str, StageResult]  # Keyed by stage name
    overall_score: float
    overall_passed: bool
    metadata: dict
```

### Evaluation Pack Protocol

```python
from typing import Protocol

class EvaluationPack(Protocol):
    """Domain-specific evaluation logic."""

    name: str

    def get_stages(self) -> list[Stage]:
        """Return ordered list of evaluation stages."""
        ...

    def get_default_weights(self) -> dict[str, float]:
        """Return default scoring weights per stage."""
        ...

    def validate_test_case(self, test_case: TestCase) -> list[str]:
        """Validate test case has required fields. Return errors."""
        ...

    def compute_overall_score(
        self,
        stage_results: dict[str, StageResult],
        weights: dict[str, float] | None = None
    ) -> float:
        """Compute weighted overall score."""
        ...

class Stage(Protocol):
    """Single evaluation stage."""

    name: str
    is_gate: bool  # If True, pipeline stops on failure

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: EvalContext
    ) -> StageResult:
        ...
```

### Example: NL2API Pack

```python
class NL2APIPack(EvaluationPack):
    name = "nl2api"

    def get_stages(self) -> list[Stage]:
        return [
            SyntaxStage(is_gate=True),      # Parse JSON
            LogicStage(is_gate=False),       # Compare tool calls
            ExecutionStage(is_gate=False),   # Verify API results
            SemanticsStage(is_gate=False),   # Judge NL response
        ]

    def get_default_weights(self) -> dict[str, float]:
        return {
            "syntax": 0.1,
            "logic": 0.3,
            "execution": 0.5,
            "semantics": 0.1,
        }

    def validate_test_case(self, tc: TestCase) -> list[str]:
        errors = []
        if "nl_query" not in tc.input:
            errors.append("Missing input.nl_query")
        if "expected_tool_calls" not in tc.expected:
            errors.append("Missing expected.expected_tool_calls")
        return errors
```

### Example: RAG Pack

```python
class RAGPack(EvaluationPack):
    name = "rag"

    def get_stages(self) -> list[Stage]:
        return [
            RetrievalStage(is_gate=False),        # recall@k, precision@k, MRR
            ContextRelevanceStage(is_gate=False), # LLM judge on retrieved docs
            FaithfulnessStage(is_gate=False),     # Is answer grounded?
            AnswerRelevanceStage(is_gate=False),  # Does answer address query?
        ]

    def get_default_weights(self) -> dict[str, float]:
        return {
            "retrieval": 0.25,
            "context_relevance": 0.15,
            "faithfulness": 0.35,
            "answer_relevance": 0.25,
        }

    def validate_test_case(self, tc: TestCase) -> list[str]:
        errors = []
        if "query" not in tc.input:
            errors.append("Missing input.query")
        # expected_relevant_docs OR expected_answer required
        if not tc.expected.get("relevant_docs") and not tc.expected.get("answer"):
            errors.append("Need expected.relevant_docs or expected.answer")
        return errors
```

---

## 5. REVISED RECOMMENDATIONS

### Phase 1: Extract Generic Core (1 week)

| Task | Description |
|------|-------------|
| **Genericize TestCase** | `input: dict`, `expected: dict` instead of NL2API fields |
| **Genericize Scorecard** | `stage_results: dict[str, StageResult]` instead of fixed 4 fields |
| **Create EvaluationPack protocol** | Define interface for domain-specific evaluation |
| **Create NL2APIPack** | Move current NL2API logic into a pack |
| **Update storage** | Store generic JSON, not NL2API-specific columns |

### Phase 2: Simple API (3 days)

```python
from evalframework import Evaluator
from evalframework.packs import NL2APIPack, RAGPack

# NL2API evaluation
evaluator = Evaluator(pack=NL2APIPack())
results = await evaluator.evaluate(test_cases, my_nl2api_system)

# RAG evaluation
evaluator = Evaluator(pack=RAGPack())
results = await evaluator.evaluate(test_cases, my_rag_system)

# Custom weights
results = await evaluator.evaluate(
    test_cases,
    my_system,
    weights={"retrieval": 0.4, "faithfulness": 0.6}
)
```

### Phase 3: RAG Pack (1 week)

| Task | Description |
|------|-------------|
| **RetrievalStage** | Implement recall@k, precision@k, MRR, NDCG |
| **ContextRelevanceStage** | LLM-as-judge for context relevance |
| **FaithfulnessStage** | Claim extraction + verification |
| **AnswerRelevanceStage** | LLM-as-judge for answer quality |
| **RAG test case fixtures** | Sample test cases for RAG evaluation |

### Phase 4: Polish (3 days)

| Task | Description |
|------|-------------|
| **CLI updates** | `eval run --pack nl2api`, `eval run --pack rag` |
| **Export formats** | JSON, CSV, pandas DataFrame |
| **Comparison tool** | `eval compare run1 run2` |
| **Getting Started docs** | Per-pack documentation |

### What to Keep vs Remove

| Component | Decision | Rationale |
|-----------|----------|-----------|
| Distributed workers | ✅ Keep | Needed for scale |
| Queue abstractions | ✅ Keep | Flexible workload handling |
| Protocol-based repos | ✅ Keep | Core strength |
| OTEL integration | ✅ Keep | Essential for production |
| Batch job tracking | ✅ Keep | Large eval runs need this |
| Cost tracking | ✅ Keep | LLM-as-judge costs money |
| 4-stage waterfall | ⚠️ Refactor | Move to NL2APIPack |
| Hardcoded weights | ⚠️ Refactor | Move to pack defaults |
| Multi-tenant models | 🔄 Defer | Revisit after core is stable |
| Azure storage stubs | ❌ Remove | Incomplete, misleading |
| Continuous scheduler | ❌ Remove | Half-baked, separate concern |

---

## 6. MIGRATION PATH

### Backwards Compatibility

Existing NL2API test cases should still work:

```python
# Old format (still supported via NL2APIPack)
{
    "id": "test-1",
    "nl_query": "What is Apple's price?",
    "expected_tool_calls": [...],
    "expected_nl_response": "..."
}

# Internally converted to generic format
{
    "id": "test-1",
    "input": {"nl_query": "What is Apple's price?"},
    "expected": {
        "tool_calls": [...],
        "nl_response": "..."
    }
}
```

### Database Migration

```sql
-- Add generic columns
ALTER TABLE test_cases ADD COLUMN input_json JSONB;
ALTER TABLE test_cases ADD COLUMN expected_json JSONB;

-- Migrate existing data
UPDATE test_cases SET
    input_json = jsonb_build_object('nl_query', nl_query),
    expected_json = jsonb_build_object(
        'tool_calls', expected_tool_calls,
        'nl_response', expected_nl_response
    );

-- Eventually drop old columns (after migration verified)
```

---

## 7. SUMMARY

### Original Assessment vs Revised

| Concern | Original | Revised |
|---------|----------|---------|
| "Over-engineered" | Yes | No - infrastructure is appropriate |
| "Too many abstractions" | Yes | No - protocols enable extensibility |
| "Remove distributed workers" | Yes | No - keep for scale |
| "6 config classes is too many" | Yes | Still yes - consolidate |
| "No simple API" | Critical | Still critical |

### The Real Problem

The platform has **good bones for a general framework** but **NL2API assumptions pollute the core**. The fix isn't removing complexity—it's moving domain-specific logic to the right layer.

### Effort Estimate

| Phase | Work | Duration |
|-------|------|----------|
| Phase 1: Extract generic core | Refactor TestCase, Scorecard, create pack protocol | 1 week |
| Phase 2: Simple API | Facade class, sensible defaults | 3 days |
| Phase 3: RAG pack | 4 stages + test fixtures | 1 week |
| Phase 4: Polish | CLI, export, docs | 3 days |
| **Total** | | **~3 weeks** |

### Recommendation

**Proceed with the general-purpose framework approach.** The infrastructure investment is justified. Focus effort on:

1. Extracting NL2API-specific logic into a pack
2. Creating the generic TestCase/Scorecard schemas
3. Building the RAG pack as proof of generality
4. Adding the simple API facade

The result will be a framework that can evaluate NL2API, RAG, and future use cases with shared infrastructure.

---

## Appendix: Files to Modify

| File | Changes |
|------|---------|
| `CONTRACTS.py` | Generic TestCase, Scorecard; EvaluationPack protocol |
| `src/evaluation/core/evaluators.py` | Pack-based pipeline instead of hardcoded stages |
| `src/evaluation/packs/nl2api/` | New directory for NL2API-specific stages |
| `src/evaluation/packs/rag/` | New directory for RAG-specific stages |
| `src/evaluation/batch/runner.py` | Accept pack parameter |
| `src/common/storage/postgres/` | Generic JSON storage for test cases |
| `src/evaluation/cli/` | `--pack` option |
