# RAG Evaluation Execution Plan

**Status:** Ready to Execute
**Prerequisites:** Phase 0 (Platform Refactoring) - Complete
**Reference:** [rag-evaluation-plan.md](./rag-evaluation-plan.md)

---

## Execution Summary

| Phase | Deliverable | Duration | Dependencies |
|-------|-------------|----------|--------------|
| **1A** | RAGPack skeleton + RetrievalStage | 2 days | None |
| **1B** | RAG Triad stages (reference-free) | 3 days | 1A |
| **2A** | Batch runner + CLI updates | 2 days | 1A |
| **2B** | Domain gates (Policy, Rejection, Citation) | 3 days | 1B |
| **3** | Test fixtures + integration tests | 2 days | 2A |
| **4** | LLM judge calibration + tuning | 2 days | 3 |
| **5** | CI integration + documentation | 2 days | 4 |

**Total: ~16 days (~3 weeks)**

---

## Phase 1A: RAGPack Skeleton + RetrievalStage (2 days)

### Goal
Get a minimal RAGPack running end-to-end with one working stage.

### Deliverables

#### 1. Create RAGPack skeleton

**File:** `src/evaluation/packs/rag/__init__.py`

```python
from .pack import RAGPack
from .stages import (
    RetrievalStage,
    ContextRelevanceStage,
    FaithfulnessStage,
    AnswerRelevanceStage,
)

__all__ = [
    "RAGPack",
    "RetrievalStage",
    "ContextRelevanceStage",
    "FaithfulnessStage",
    "AnswerRelevanceStage",
]
```

**File:** `src/evaluation/packs/rag/pack.py`

```python
from src.contracts.evaluation import EvaluationPack, Stage, StageResult
from src.contracts.core import TestCase
from .stages import RetrievalStage

class RAGPack:
    """Evaluation pack for RAG systems."""

    name = "rag"

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._stages = [
            RetrievalStage(),
            # More stages added in Phase 1B
        ]

    def get_stages(self) -> list[Stage]:
        return self._stages

    def get_default_weights(self) -> dict[str, float]:
        return {
            "retrieval": 1.0,  # Only stage for now
        }

    def validate_test_case(self, test_case: TestCase) -> list[str]:
        errors = []

        # Check for query
        if not test_case.input.get("query"):
            errors.append("Missing input['query']")

        # Need either relevant_docs (retrieval eval) or behavior (rejection eval)
        has_retrieval = bool(test_case.expected.get("relevant_docs"))
        has_behavior = bool(test_case.expected.get("behavior"))

        if not has_retrieval and not has_behavior:
            errors.append("Need expected['relevant_docs'] or expected['behavior']")

        return errors

    def compute_overall_score(
        self,
        stage_results: dict[str, StageResult],
        weights: dict[str, float] | None = None,
    ) -> float:
        weights = weights or self.get_default_weights()
        total_weight = 0.0
        weighted_sum = 0.0

        for stage_name, result in stage_results.items():
            if stage_name in weights and result.score is not None:
                w = weights[stage_name]
                weighted_sum += result.score * w
                total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def compute_overall_passed(self, stage_results: dict[str, StageResult]) -> bool:
        for stage in self._stages:
            result = stage_results.get(stage.name)
            if result and stage.is_gate and not result.passed:
                return False
        return all(r.passed for r in stage_results.values() if r is not None)
```

#### 2. Implement RetrievalStage

**File:** `src/evaluation/packs/rag/stages/retrieval.py`

```python
from src.contracts.evaluation import StageResult
from src.contracts.core import TestCase
import time

class RetrievalStage:
    """Evaluate retrieval quality with IR metrics."""

    name = "retrieval"
    is_gate = False

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: "EvalContext",
    ) -> StageResult:
        start = time.perf_counter()

        expected_docs = test_case.expected.get("relevant_docs", [])
        retrieved_docs = system_output.get("retrieved_doc_ids", [])

        # Skip if no ground truth
        if not expected_docs:
            return StageResult(
                stage_name=self.name,
                passed=True,
                score=1.0,
                metrics={"skipped": True, "reason": "no_ground_truth"},
                duration_ms=int((time.perf_counter() - start) * 1000),
            )

        # Calculate metrics
        metrics = {
            "recall_at_5": self._recall_at_k(expected_docs, retrieved_docs, k=5),
            "recall_at_10": self._recall_at_k(expected_docs, retrieved_docs, k=10),
            "precision_at_5": self._precision_at_k(expected_docs, retrieved_docs, k=5),
            "mrr": self._mrr(expected_docs, retrieved_docs),
            "hit_rate": self._hit_rate(expected_docs, retrieved_docs),
        }

        # Composite score (weighted)
        score = (
            metrics["recall_at_5"] * 0.3 +
            metrics["precision_at_5"] * 0.2 +
            metrics["mrr"] * 0.3 +
            metrics["hit_rate"] * 0.2
        )

        return StageResult(
            stage_name=self.name,
            passed=score >= 0.5,
            score=score,
            metrics=metrics,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    def _recall_at_k(self, expected: list, retrieved: list, k: int) -> float:
        if not expected:
            return 1.0
        retrieved_set = set(retrieved[:k])
        hits = len(set(expected) & retrieved_set)
        return hits / len(expected)

    def _precision_at_k(self, expected: list, retrieved: list, k: int) -> float:
        if not retrieved[:k]:
            return 0.0
        retrieved_set = set(retrieved[:k])
        hits = len(set(expected) & retrieved_set)
        return hits / len(retrieved_set)

    def _mrr(self, expected: list, retrieved: list) -> float:
        expected_set = set(expected)
        for i, doc_id in enumerate(retrieved):
            if doc_id in expected_set:
                return 1.0 / (i + 1)
        return 0.0

    def _hit_rate(self, expected: list, retrieved: list) -> float:
        expected_set = set(expected)
        return 1.0 if any(doc in expected_set for doc in retrieved) else 0.0
```

#### 3. Register RAGPack

**File:** `src/evaluation/packs/__init__.py` (update)

```python
from .nl2api import NL2APIPack
from .rag import RAGPack

__all__ = ["NL2APIPack", "RAGPack"]

PACKS = {
    "nl2api": NL2APIPack,
    "rag": RAGPack,
}

def get_pack(name: str):
    if name not in PACKS:
        raise ValueError(f"Unknown pack: {name}. Available: {list(PACKS.keys())}")
    return PACKS[name]()
```

### Tests

**File:** `tests/unit/evaluation/packs/rag/test_retrieval_stage.py`

```python
import pytest
from src.evaluation.packs.rag.stages import RetrievalStage
from src.contracts.core import TestCase

@pytest.fixture
def retrieval_stage():
    return RetrievalStage()

@pytest.mark.asyncio
async def test_perfect_retrieval(retrieval_stage):
    test_case = TestCase(
        id="test-1",
        input={"query": "test query"},
        expected={"relevant_docs": ["doc1", "doc2", "doc3"]},
    )
    system_output = {"retrieved_doc_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"]}

    result = await retrieval_stage.evaluate(test_case, system_output, None)

    assert result.passed
    assert result.metrics["recall_at_5"] == 1.0
    assert result.metrics["mrr"] == 1.0

@pytest.mark.asyncio
async def test_no_ground_truth_skips(retrieval_stage):
    test_case = TestCase(
        id="test-2",
        input={"query": "test query"},
        expected={},
    )
    system_output = {"retrieved_doc_ids": ["doc1", "doc2"]}

    result = await retrieval_stage.evaluate(test_case, system_output, None)

    assert result.passed
    assert result.metrics.get("skipped") is True
```

### Success Criteria

- [ ] `RAGPack` can be instantiated and returns stages
- [ ] `RetrievalStage` calculates correct IR metrics
- [ ] Unit tests pass
- [ ] Can run: `Evaluator(pack=RAGPack()).evaluate(test_case, system_output)`

---

## Phase 1B: RAG Triad Stages (3 days)

### Goal
Implement the reference-free LLM-as-judge stages.

### Deliverables

#### 1. ContextRelevanceStage

**File:** `src/evaluation/packs/rag/stages/context_relevance.py`

Evaluates: Is the retrieved context relevant to the query?

- Uses LLM-as-judge
- Scores each chunk, aggregates
- Reference-free (no ground truth needed)

#### 2. FaithfulnessStage

**File:** `src/evaluation/packs/rag/stages/faithfulness.py`

Evaluates: Is the response grounded in the retrieved context?

- Extract claims from response (LLM call)
- Verify each claim against context (LLM call per claim)
- Score = supported claims / total claims
- Reference-free

#### 3. AnswerRelevanceStage

**File:** `src/evaluation/packs/rag/stages/answer_relevance.py`

Evaluates: Does the response answer the question?

- LLM-as-judge
- Checks: addresses question, completeness, on-topic
- Reference-free

#### 4. LLM Judge Abstraction

**File:** `src/evaluation/packs/rag/llm_judge.py`

```python
class LLMJudge:
    """Abstraction for LLM-as-judge calls."""

    def __init__(self, config: LLMJudgeConfig):
        self.config = config
        self.llm = self._create_llm()

    async def evaluate(self, prompt: str) -> dict:
        """Call LLM and parse JSON response."""
        ...

    async def extract_claims(self, text: str) -> list[str]:
        """Extract atomic claims from text."""
        ...

    async def verify_claim(self, claim: str, context: str) -> dict:
        """Check if claim is supported by context."""
        ...
```

### Tests

- Unit tests for each stage with mocked LLM
- Test claim extraction logic
- Test score aggregation

### Success Criteria

- [ ] All 4 RAG Triad stages implemented
- [ ] LLM judge abstraction working
- [ ] Unit tests pass with mocked LLM
- [ ] Can evaluate a test case through all stages

---

## Phase 2A: Batch Runner + CLI (2 days)

### Goal
Enable running RAG evaluation via CLI.

### Deliverables

#### 1. Update Batch Runner

**File:** `src/evaluation/batch/runner.py`

```python
from src.evaluation.core.evaluator import Evaluator
from src.evaluation.packs import get_pack

class BatchRunner:
    def __init__(self, pack_name: str = "nl2api", config: dict | None = None):
        self.pack = get_pack(pack_name)
        self.evaluator = Evaluator(pack=self.pack, config=config)

    async def run(
        self,
        test_cases: list[TestCase],
        target: TargetSystem,
        **kwargs
    ) -> BatchResult:
        # Use generic evaluator instead of WaterfallEvaluator
        ...
```

#### 2. Add CLI --pack Option

**File:** `src/evaluation/cli/commands/batch.py`

```python
@click.option("--pack", default="nl2api", help="Evaluation pack: nl2api, rag")
def batch_run(pack: str, ...):
    runner = BatchRunner(pack_name=pack)
    ...
```

#### 3. Add Weights Override

```bash
eval run --pack rag \
    --weights '{"faithfulness": 0.4, "retrieval": 0.2}'
```

### Success Criteria

- [ ] `eval run --pack rag` works
- [ ] Weights can be overridden via CLI
- [ ] Results stored in database with `pack_name="rag"`

---

## Phase 2B: Domain Gates (3 days)

### Goal
Implement citation, source policy, and rejection calibration stages.

### Deliverables

#### 1. CitationStage

**File:** `src/evaluation/packs/rag/stages/citation.py`

- Citation presence check
- Citation validity (points to real chunk)
- Citation accuracy (LLM judge)
- Citation coverage (LLM judge)

#### 2. SourcePolicyStage (GATE)

**File:** `src/evaluation/packs/rag/stages/source_policy.py`

- Detect quote-only source usage
- Verify direct quotes vs paraphrasing (LLM judge)
- GATE: fails pipeline if violated

#### 3. PolicyComplianceStage (GATE)

**File:** `src/evaluation/packs/rag/stages/policy_compliance.py`

- Pattern-based detection (fast)
- LLM judge for nuanced cases
- GATE: fails pipeline if violated

#### 4. RejectionCalibrationStage

**File:** `src/evaluation/packs/rag/stages/rejection_calibration.py`

- Detect rejection patterns
- Training cutoff excuse detection
- Compare to expected behavior

### Tests

- Unit tests for each stage
- Integration test with all stages

### Success Criteria

- [ ] All domain gates implemented
- [ ] GATE stages stop pipeline on failure
- [ ] Citation metrics calculated correctly
- [ ] Source policy violations detected

---

## Phase 3: Test Fixtures + Integration (2 days)

### Goal
Create test data and verify end-to-end flow.

### Deliverables

#### 1. RAG Test Fixtures

**File:** `tests/fixtures/rag/`

```
rag/
├── should_answer_complete.json     # 20 cases
├── should_answer_partial.json      # 20 cases
├── should_reject_policy.json       # 30 cases (from gold set)
├── should_reject_no_context.json   # 10 cases
├── citation_required.json          # 10 cases
└── quote_only_sources.json         # 10 cases
```

#### 2. Fixture Loader Update

**File:** `tests/unit/nl2api/fixture_loader.py`

Add RAG fixture loading capability.

#### 3. Integration Tests

**File:** `tests/integration/evaluation/test_rag_evaluation.py`

```python
@pytest.mark.integration
async def test_rag_evaluation_e2e():
    """Full RAG evaluation pipeline test."""
    pack = RAGPack()
    evaluator = Evaluator(pack=pack)

    test_case = load_fixture("rag/should_answer_complete.json")[0]
    system_output = simulate_rag_response(test_case)

    scorecard = await evaluator.evaluate(test_case, system_output, context)

    assert scorecard.pack_name == "rag"
    assert "retrieval" in scorecard.stage_results
    assert "faithfulness" in scorecard.stage_results
```

### Success Criteria

- [ ] 100 RAG test cases created
- [ ] Fixtures load correctly
- [ ] E2E integration test passes
- [ ] Scorecards persist to database

---

## Phase 4: LLM Judge Calibration (2 days)

### Goal
Validate and tune LLM judge accuracy.

### Deliverables

#### 1. Human Labeling Set

Create 50-100 examples with human scores for:
- Faithfulness (0-1)
- Context relevance (0-1)
- Answer relevance (0-1)

#### 2. Calibration Script

**File:** `scripts/calibrate_llm_judge.py`

```python
async def calibrate():
    judge = LLMJudge(config)
    human_labels = load_human_labels()

    results = []
    for example in human_labels:
        judge_score = await judge.evaluate(example)
        results.append({
            "human": example.human_score,
            "judge": judge_score,
        })

    # Calculate correlation, kappa
    report = CalibrationReport.from_results(results)
    print(report)
```

#### 3. Anchor Examples

Create pre-graded examples for each stage to include in prompts.

### Success Criteria

- [ ] Cohen's kappa > 0.8 for faithfulness
- [ ] Cohen's kappa > 0.75 for relevance metrics
- [ ] Anchor examples created for each stage

---

## Phase 5: CI + Documentation (2 days)

### Goal
Production-ready with CI integration.

### Deliverables

#### 1. GitHub Actions Workflow

**File:** `.github/workflows/rag-evaluation.yml`

```yaml
name: RAG Evaluation
on:
  pull_request:
  schedule:
    - cron: '0 6 * * 1'  # Weekly

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run RAG evaluation
        run: |
          eval run --pack rag \
            --input tests/fixtures/rag/ \
            --threshold 0.8
```

#### 2. Grafana Dashboard

Add RAG-specific panels to evaluation dashboard:
- Stage-level metrics
- Failure mode breakdown
- Trend over time

#### 3. Documentation

**File:** `docs/rag-evaluation.md`

- How to run RAG evaluation
- Test case format
- Stage descriptions
- Weight tuning guide

### Success Criteria

- [ ] CI runs on PRs
- [ ] Grafana shows RAG metrics
- [ ] Documentation complete

---

## Execution Checklist

### Week 1

- [ ] **Day 1-2 (Phase 1A):** RAGPack skeleton + RetrievalStage
- [ ] **Day 3-5 (Phase 1B):** RAG Triad stages (Context, Faithfulness, Answer)

### Week 2

- [ ] **Day 6-7 (Phase 2A):** Batch runner + CLI updates
- [ ] **Day 8-10 (Phase 2B):** Domain gates (Citation, Source Policy, Policy, Rejection)

### Week 3

- [ ] **Day 11-12 (Phase 3):** Test fixtures + integration tests
- [ ] **Day 13-14 (Phase 4):** LLM judge calibration
- [ ] **Day 15-16 (Phase 5):** CI + documentation

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM judge inconsistency | Calibrate early (Phase 4), use anchor examples |
| Batch runner integration issues | Phase 2A delivers working CLI before domain gates |
| Test fixture quality | Start with production samples, augment with synthetic |
| Cost overruns from LLM calls | Use Haiku, batch API where possible |

---

## Definition of Done

RAG Evaluation is complete when:

1. [ ] `eval run --pack rag` executes all 8 stages
2. [ ] Scorecards persist with `pack_name="rag"` and all stage metrics
3. [ ] CI runs RAG evaluation on PRs
4. [ ] Grafana displays RAG metrics
5. [ ] Documentation explains usage
6. [ ] LLM judge calibrated (kappa > 0.75)
7. [ ] 100+ test fixtures covering all categories
