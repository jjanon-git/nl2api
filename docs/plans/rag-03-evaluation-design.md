# RAG Evaluation Framework Plan

**Status:** Draft
**Created:** 2026-01-22
**Depends on:** [evaluation-platform-review.md](./evaluation-platform-review.md) (Phase 0)

---

## 1. Problem Statement

### 1.1 The Root Problem

**We lack a flexible, automated evaluation methodology for our RAG system.**

Without systematic evaluation, we cannot:
- Measure current quality or detect regressions
- Iterate on prompts with confidence
- Balance competing objectives (helpfulness vs safety)
- Diagnose whether failures are retrieval or generation issues
- Scale quality assurance as the system evolves

### 1.2 Symptoms That Revealed the Gap

Several production issues surfaced that we couldn't systematically measure or address:

| Symptom | Example | Why We Couldn't Fix It |
|---------|---------|------------------------|
| **False rejections** | "10 highlights from earnings" → rejects with 5 available | No way to measure false rejection rate or track improvement |
| **Training cutoff excuses** | "I can't answer, my training data ends Oct 2023" | No automated detection of context utilization failures |
| **Policy violation acceptance** | Answers financial advice questions | Gold set exists but no integration with prompt iteration workflow |

These are **symptoms of the evaluation gap**, not the root problem. Fixing any one symptom without evaluation infrastructure just creates new blind spots.

### 1.3 Current State

| What We Have | What We Lack |
|--------------|--------------|
| Gold set for policy violations | Automated evaluation pipeline |
| Manual spot-checking | Systematic coverage of failure modes |
| Ad-hoc prompt tweaks | Regression detection |
| Production logs | Reference-free quality metrics |

### 1.4 Goal

Build a **flexible, automated RAG evaluation framework** that:
1. Measures quality across multiple dimensions (RAG Triad + domain-specific gates)
2. Runs automatically on production samples and labeled test sets
3. Enables confident prompt iteration with regression detection
4. Scales to new failure modes as they're discovered
5. Provides actionable diagnostics (not just pass/fail)

---

## 2. Build vs Buy Analysis

### 2.1 Options Evaluated

| Framework | Stars | License | Production Ready |
|-----------|-------|---------|------------------|
| **RAGAS** | 12.3k | Apache 2.0 | Yes |
| **DeepEval** | 13.1k | Apache 2.0 | Yes |
| **TruLens** | 3.1k | MIT | Medium (Snowflake-backed) |
| **LangChain/LlamaIndex** | N/A | Apache 2.0 | Yes |
| **Internal Platform** | N/A | Internal | Yes |

### 2.2 Feature Comparison

| Requirement | RAGAS | DeepEval | TruLens | Internal |
|-------------|-------|----------|---------|----------|
| RAG Triad metrics | ✅ | ✅ | ✅ | Planned |
| Custom stages/metrics | ✅ | ✅ | ✅ | ✅ |
| **Citation evaluation** | ❌ | ❌ | ❌ | **✅ Planned** |
| **Source policy (quote-only)** | ❌ | ❌ | ❌ | **✅ Planned** |
| **Rejection calibration** | ❌ | ❌ | ❌ | **✅ Planned** |
| OTEL/Observability | Basic | Basic | ✅ | **✅ Full** |
| Distributed batch processing | ❌ | Pytest parallel | Basic | **✅ Full** |
| Storage/persistence | BYO | Cloud/JSON | SQLite | **✅ PostgreSQL** |

### 2.3 Key Finding

**None of the external frameworks natively support our domain-specific requirements:**
- Citation accuracy and coverage
- Source-level usage policies (quote-only, no-use)
- Rejection calibration (false rejection/acceptance tracking)

All frameworks would require custom extensions for these, negating much of the "buy" advantage.

### 2.4 Integration Effort Comparison

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **Replace with external** | 4-6 weeks | Community support | Lose existing infra, still need custom stages |
| **Extend internal** | 4.5 weeks | Full control, keep infra | Internal maintenance |
| **Hybrid (patterns only)** | 4.5 weeks | Best practices + control | Some research overhead |

### 2.5 Recommendation

**Extend internal platform with inspiration from external patterns.**

Rationale:
1. **Existing infrastructure is robust** - Distributed workers, PostgreSQL storage, OTEL integration already built
2. **Domain requirements not met externally** - Citations, source policies, rejection calibration need custom implementation regardless
3. **Similar effort** - Integrating and extending external framework ≈ building internally
4. **No external dependency risk** - Version conflicts, breaking changes, maintenance burden

### 2.6 What to Borrow from External Frameworks

| Source | Pattern to Adopt |
|--------|------------------|
| RAGAS | Claim extraction + verification for Faithfulness |
| RAGAS | Noise Sensitivity metric concept |
| DeepEval | G-Eval prompt patterns for custom criteria |
| DeepEval | BaseMetric interface design |
| TruLens | OTEL trace emission patterns |

### 2.7 What NOT to Adopt

- External framework as runtime dependency
- Cloud-based storage (keep PostgreSQL control)
- External batch processing (keep internal workers)

---

## 3. Design Principles

### 3.1 Industry Alignment: RAG Triad

Based on research across RAGAS, TruLens, DeepEval, and academic literature, the industry has converged on three core metrics:

| Metric | What It Measures | Ground Truth Required |
|--------|------------------|----------------------|
| **Context Relevance** | Does retrieved context address the query? | No |
| **Groundedness/Faithfulness** | Is response supported by context? | No |
| **Answer Relevance** | Does response answer the question? | No |

These are **reference-free metrics** - they don't require labeled expected answers, enabling faster iteration.

### 3.2 Component-Wise Evaluation

RAG failures happen at different stages. Evaluating end-to-end hides the root cause:

```
┌─────────────────────────────────────────────────────────────────┐
│  RAG Pipeline Stages                                            │
├─────────────────────┬─────────────────────┬────────────────────┤
│     RETRIEVAL       │     GENERATION      │     END-TO-END     │
├─────────────────────┼─────────────────────┼────────────────────┤
│ • Recall@k          │ • Faithfulness      │ • Answer Relevance │
│ • Precision@k       │ • Groundedness      │ • Policy Gates     │
│ • MRR, NDCG         │ • Context Util.     │ • Format/Tone      │
│ • Context Relevance │                     │                    │
└─────────────────────┴─────────────────────┴────────────────────┘
```

### 3.3 Reference-Free vs Reference-Based

| Mode | Metrics | When to Use | Labeling Effort |
|------|---------|-------------|-----------------|
| **Reference-Free** | Faithfulness, Groundedness, Relevance | Fast iteration, continuous monitoring | None |
| **Reference-Based** | Factual Correctness, Recall@k | Formal benchmarks, regression testing | High |
| **Hybrid** | Both where available | Comprehensive evaluation | Medium |

Default to **reference-free** for development velocity; use **reference-based** for release gates.

### 3.4 Layered Gates

Not all metrics are equal. Structure as tiers:

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1: CRITICAL GATES (Block deployment)                      │
│  • Policy Compliance ≥95%                                       │
│  • Groundedness ≥85%                                            │
├─────────────────────────────────────────────────────────────────┤
│  TIER 2: QUALITY METRICS (Warn on failure)                      │
│  • Context Relevance ≥80%                                       │
│  • Answer Relevance ≥80%                                        │
│  • Faithfulness ≥85%                                            │
│  • Rejection Calibration ≥80%                                   │
├─────────────────────────────────────────────────────────────────┤
│  TIER 3: DIAGNOSTIC METRICS (Monitor)                           │
│  • Noise Sensitivity                                            │
│  • Retrieval Precision/Recall (if labeled)                      │
│  • Latency, Token Cost                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Architecture

### 4.1 Integration with Evaluation Platform

This plan depends on the platform refactoring in [evaluation-platform-review.md](./evaluation-platform-review.md). RAG evaluation becomes a first-class `EvaluationPack`:

```
┌─────────────────────────────────────────────────────────────────┐
│  Evaluation Platform (Generic)                                  │
├─────────────────────────────────────────────────────────────────┤
│  • Generic TestCase (input: dict, expected: dict)               │
│  • Generic Scorecard (stage_results: dict[str, StageResult])    │
│  • EvaluationPack protocol                                      │
│  • Batch runner, storage, telemetry                             │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   NL2APIPack    │  │    RAGPack      │  │  Future Packs   │
│   (existing)    │  │   (this plan)   │  │  (classification│
│                 │  │                 │  │   summarization)│
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 4.2 RAGPack Structure

```python
class RAGPack(EvaluationPack):
    """Evaluation pack for RAG systems."""

    name = "rag"

    def get_stages(self) -> list[Stage]:
        return [
            # Core RAG Triad
            RetrievalStage(is_gate=False),
            ContextRelevanceStage(is_gate=False),
            FaithfulnessStage(is_gate=False),
            AnswerRelevanceStage(is_gate=False),

            # Citation & Source Policy
            CitationStage(is_gate=False),          # Citation accuracy & coverage
            SourcePolicyStage(is_gate=True),       # GATE - quote-only enforcement

            # Domain-specific gates
            PolicyComplianceStage(is_gate=True),   # GATE - blocks on failure
            RejectionCalibrationStage(is_gate=False),
        ]

    def get_default_weights(self) -> dict[str, float]:
        return {
            "retrieval": 0.10,
            "context_relevance": 0.10,
            "faithfulness": 0.20,
            "answer_relevance": 0.15,
            "citation": 0.15,
            "source_policy": 0.10,
            "policy_compliance": 0.10,
            "rejection_calibration": 0.10,
        }
```

### 4.3 Stage Implementations

#### 3.3.1 RetrievalStage (Reference-Based)

Requires labeled `expected_relevant_docs`:

```python
class RetrievalStage(Stage):
    """Evaluate retrieval quality with standard IR metrics."""

    name = "retrieval"
    is_gate = False

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: EvalContext
    ) -> StageResult:
        expected_docs = test_case.expected.get("relevant_docs", [])
        retrieved_docs = system_output.get("retrieved_doc_ids", [])

        if not expected_docs:
            # Skip if no ground truth
            return StageResult(
                stage_name=self.name,
                passed=True,
                score=1.0,
                metrics={"skipped": True, "reason": "no_ground_truth"},
                duration_ms=0,
            )

        metrics = {
            "recall_at_5": self._recall_at_k(expected_docs, retrieved_docs, k=5),
            "precision_at_5": self._precision_at_k(expected_docs, retrieved_docs, k=5),
            "mrr": self._mrr(expected_docs, retrieved_docs),
            "ndcg_at_5": self._ndcg_at_k(expected_docs, retrieved_docs, k=5),
        }

        # Score is weighted average
        score = (
            metrics["recall_at_5"] * 0.4 +
            metrics["precision_at_5"] * 0.2 +
            metrics["mrr"] * 0.2 +
            metrics["ndcg_at_5"] * 0.2
        )

        return StageResult(
            stage_name=self.name,
            passed=score >= 0.6,
            score=score,
            metrics=metrics,
            duration_ms=elapsed,
        )
```

#### 3.3.2 ContextRelevanceStage (Reference-Free, LLM-Judge)

```python
class ContextRelevanceStage(Stage):
    """LLM-as-judge: Is retrieved context relevant to the query?"""

    name = "context_relevance"
    is_gate = False

    PROMPT = """
    You are evaluating whether retrieved context is relevant to a user query.

    Query: {query}

    Retrieved Context:
    {context}

    For each context chunk, determine if it contains information that could
    help answer the query. Then provide an overall relevance score.

    Respond in JSON:
    {{
        "chunk_scores": [
            {{"chunk_id": "...", "relevant": true/false, "reason": "..."}},
            ...
        ],
        "overall_score": 0.0-1.0,
        "reasoning": "..."
    }}
    """

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: EvalContext
    ) -> StageResult:
        query = test_case.input["query"]
        retrieved_context = system_output.get("retrieved_context", [])

        if not retrieved_context:
            return StageResult(
                stage_name=self.name,
                passed=False,
                score=0.0,
                metrics={"error": "no_context_retrieved"},
                duration_ms=0,
            )

        # Call LLM judge
        prompt = self.PROMPT.format(
            query=query,
            context=self._format_context(retrieved_context)
        )
        result = await context.llm_judge.evaluate(prompt)

        return StageResult(
            stage_name=self.name,
            passed=result.overall_score >= 0.7,
            score=result.overall_score,
            metrics={
                "chunk_scores": result.chunk_scores,
                "relevant_chunks": sum(1 for c in result.chunk_scores if c["relevant"]),
                "total_chunks": len(result.chunk_scores),
            },
            duration_ms=elapsed,
        )
```

#### 3.3.3 FaithfulnessStage (Reference-Free, Claim Decomposition)

Uses RAGAS-style claim extraction and verification:

```python
class FaithfulnessStage(Stage):
    """Verify each claim in the response is supported by context."""

    name = "faithfulness"
    is_gate = False

    CLAIM_EXTRACTION_PROMPT = """
    Extract all factual claims from this response. Each claim should be
    a single, atomic statement that can be verified independently.

    Response: {response}

    Return JSON: {{"claims": ["claim 1", "claim 2", ...]}}
    """

    CLAIM_VERIFICATION_PROMPT = """
    Determine if the following claim is supported by the provided context.

    Claim: {claim}

    Context:
    {context}

    Return JSON:
    {{
        "supported": true/false,
        "evidence": "quote from context if supported, null otherwise",
        "reasoning": "explanation"
    }}
    """

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: EvalContext
    ) -> StageResult:
        response = system_output.get("response", "")
        retrieved_context = system_output.get("retrieved_context", [])

        # Step 1: Extract claims
        claims = await self._extract_claims(response, context.llm_judge)

        if not claims:
            # No factual claims = vacuously faithful
            return StageResult(
                stage_name=self.name,
                passed=True,
                score=1.0,
                metrics={"claims_count": 0, "note": "no_factual_claims"},
                duration_ms=elapsed,
            )

        # Step 2: Verify each claim
        verifications = []
        for claim in claims:
            result = await self._verify_claim(claim, retrieved_context, context.llm_judge)
            verifications.append({
                "claim": claim,
                "supported": result.supported,
                "evidence": result.evidence,
            })

        # Step 3: Calculate faithfulness score
        supported_count = sum(1 for v in verifications if v["supported"])
        score = supported_count / len(verifications)

        return StageResult(
            stage_name=self.name,
            passed=score >= 0.85,
            score=score,
            metrics={
                "claims_count": len(claims),
                "supported_count": supported_count,
                "unsupported_claims": [v["claim"] for v in verifications if not v["supported"]],
                "verifications": verifications,
            },
            duration_ms=elapsed,
        )
```

#### 3.3.4 AnswerRelevanceStage (Reference-Free, LLM-Judge)

```python
class AnswerRelevanceStage(Stage):
    """LLM-as-judge: Does the response actually answer the query?"""

    name = "answer_relevance"
    is_gate = False

    PROMPT = """
    Evaluate whether this response adequately answers the user's question.

    Question: {query}
    Response: {response}

    Consider:
    1. Does the response address what was asked?
    2. Is the response complete enough to be useful?
    3. Does it stay on topic?

    Return JSON:
    {{
        "addresses_question": true/false,
        "completeness": 0.0-1.0,
        "on_topic": true/false,
        "overall_score": 0.0-1.0,
        "reasoning": "..."
    }}
    """

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: EvalContext
    ) -> StageResult:
        query = test_case.input["query"]
        response = system_output.get("response", "")

        result = await context.llm_judge.evaluate(
            self.PROMPT.format(query=query, response=response)
        )

        return StageResult(
            stage_name=self.name,
            passed=result.overall_score >= 0.7,
            score=result.overall_score,
            metrics={
                "addresses_question": result.addresses_question,
                "completeness": result.completeness,
                "on_topic": result.on_topic,
            },
            duration_ms=elapsed,
        )
```

#### 3.3.5 PolicyComplianceStage (Domain Gate)

```python
class PolicyComplianceStage(Stage):
    """Verify response doesn't violate business policies."""

    name = "policy_compliance"
    is_gate = True  # CRITICAL - blocks pipeline on failure

    # Pattern-based detection for common violations
    VIOLATION_PATTERNS = {
        "financial_advice": [
            r"you should (buy|sell|invest)",
            r"I recommend (buying|selling|investing)",
            r"(buy|sell) this stock",
        ],
        "predictions": [
            r"stock will (go up|increase|rise|fall|drop)",
            r"price will (reach|hit|exceed)",
        ],
        "personal_data": [
            r"your (ssn|social security|account number)",
        ],
    }

    # LLM judge for nuanced cases
    POLICY_JUDGE_PROMPT = """
    Evaluate if this response violates any of these policies:

    POLICIES:
    1. No financial advice: Don't recommend specific investment actions
    2. No predictions: Don't predict future stock prices or market movements
    3. No personal data handling: Don't request or process sensitive personal information
    4. No banned topics: {banned_topics}

    Query: {query}
    Response: {response}

    Return JSON:
    {{
        "violations": [
            {{"policy": "...", "severity": "high/medium/low", "evidence": "..."}},
            ...
        ],
        "compliant": true/false,
        "reasoning": "..."
    }}
    """

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: EvalContext
    ) -> StageResult:
        response = system_output.get("response", "")
        query = test_case.input["query"]

        violations = []

        # Pattern-based detection (fast)
        for policy, patterns in self.VIOLATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    violations.append({
                        "policy": policy,
                        "severity": "high",
                        "evidence": f"Pattern match: {pattern}",
                        "detection": "pattern",
                    })

        # LLM judge for nuanced cases (if no pattern matches)
        if not violations:
            result = await context.llm_judge.evaluate(
                self.POLICY_JUDGE_PROMPT.format(
                    query=query,
                    response=response,
                    banned_topics=context.config.get("banned_topics", []),
                )
            )
            violations.extend(result.violations)

        compliant = len(violations) == 0

        return StageResult(
            stage_name=self.name,
            passed=compliant,
            score=1.0 if compliant else 0.0,
            metrics={
                "violations": violations,
                "violation_count": len(violations),
            },
            duration_ms=elapsed,
            error=f"Policy violations: {[v['policy'] for v in violations]}" if violations else None,
        )
```

#### 3.3.6 RejectionCalibrationStage (Your Original Problem)

```python
class RejectionCalibrationStage(Stage):
    """Verify the system rejects/answers appropriately."""

    name = "rejection_calibration"
    is_gate = False

    REJECTION_PATTERNS = [
        r"I (am|'m) unable to",
        r"I cannot (provide|answer)",
        r"I don't have enough information",
        r"This question cannot be answered",
        r"my (training|knowledge) (cutoff|cut-off)",  # Training date excuse
        r"as of (my|the) (training|knowledge)",
    ]

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: EvalContext
    ) -> StageResult:
        response = system_output.get("response", "")
        expected_behavior = test_case.expected.get("behavior", "answer")  # "answer" or "reject"

        # Detect if response is a rejection
        is_rejection = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in self.REJECTION_PATTERNS
        )

        # Detect training cutoff excuse (specific failure mode)
        training_cutoff_excuse = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in [
                r"my (training|knowledge) (cutoff|cut-off)",
                r"as of (my|the) (training|knowledge)",
                r"I don't have (access to|information about) (events|data) after",
            ]
        )

        # Evaluate correctness
        if expected_behavior == "reject":
            passed = is_rejection
            failure_mode = None if passed else "false_acceptance"
        else:  # expected to answer
            passed = not is_rejection
            if not passed:
                failure_mode = "training_cutoff_excuse" if training_cutoff_excuse else "false_rejection"
            else:
                failure_mode = None

        return StageResult(
            stage_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            metrics={
                "expected_behavior": expected_behavior,
                "actual_behavior": "reject" if is_rejection else "answer",
                "is_rejection": is_rejection,
                "training_cutoff_excuse": training_cutoff_excuse,
                "failure_mode": failure_mode,
            },
            duration_ms=elapsed,
        )
```

#### 3.3.7 CitationStage (Citation Accuracy & Coverage)

```python
class CitationStage(Stage):
    """Verify citations are present, correct, and sufficient."""

    name = "citation"
    is_gate = False

    CITATION_VERIFICATION_PROMPT = """
    Verify if this citation accurately represents the source content.

    Citation text: {citation_text}
    Source content: {source_content}

    Return JSON:
    {{
        "accurate": true/false,
        "reasoning": "explanation"
    }}
    """

    COVERAGE_PROMPT = """
    Identify all factual claims in this response and determine if each has a citation.

    Response: {response}
    Citations: {citations}

    Return JSON:
    {{
        "claims": [
            {{"claim": "...", "has_citation": true/false, "citation_marker": "[1]" or null}},
            ...
        ],
        "coverage_ratio": 0.0-1.0
    }}
    """

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: EvalContext
    ) -> StageResult:
        response = system_output.get("response", "")
        citations = system_output.get("citations", [])
        retrieved_context = system_output.get("retrieved_context", [])

        # Check if citations are required
        requires_citations = test_case.expected.get("requires_citations", True)

        if not requires_citations:
            return StageResult(
                stage_name=self.name,
                passed=True,
                score=1.0,
                metrics={"skipped": True, "reason": "citations_not_required"},
                duration_ms=0,
            )

        metrics = {}

        # 1. Are citations present?
        metrics["citations_present"] = len(citations) > 0
        metrics["citation_count"] = len(citations)

        if not citations:
            return StageResult(
                stage_name=self.name,
                passed=False,
                score=0.0,
                metrics={**metrics, "error": "no_citations_provided"},
                duration_ms=elapsed,
            )

        # 2. Are citations valid? (point to real retrieved chunks)
        context_ids = {c["id"] for c in retrieved_context}
        valid_citations = sum(1 for c in citations if c.get("source_id") in context_ids)
        metrics["citations_valid"] = valid_citations / len(citations)

        # 3. Are citations accurate? (cited chunk supports the claim)
        accurate_count = 0
        for citation in citations:
            source_chunk = next(
                (c for c in retrieved_context if c["id"] == citation.get("source_id")),
                None
            )
            if source_chunk:
                result = await context.llm_judge.evaluate(
                    self.CITATION_VERIFICATION_PROMPT.format(
                        citation_text=citation.get("text", ""),
                        source_content=source_chunk["content"],
                    )
                )
                if result.accurate:
                    accurate_count += 1

        metrics["citations_accurate"] = accurate_count / len(citations) if citations else 0

        # 4. Is coverage sufficient? (all claims have citations)
        coverage_result = await context.llm_judge.evaluate(
            self.COVERAGE_PROMPT.format(response=response, citations=citations)
        )
        metrics["citation_coverage"] = coverage_result.coverage_ratio
        metrics["uncited_claims"] = [
            c["claim"] for c in coverage_result.claims if not c["has_citation"]
        ]

        # Calculate overall score
        score = (
            metrics["citations_valid"] * 0.3 +
            metrics["citations_accurate"] * 0.4 +
            metrics["citation_coverage"] * 0.3
        )

        min_coverage = test_case.expected.get("min_citation_coverage", 0.8)

        return StageResult(
            stage_name=self.name,
            passed=score >= 0.7 and metrics["citation_coverage"] >= min_coverage,
            score=score,
            metrics=metrics,
            duration_ms=elapsed,
        )
```

#### 3.3.8 SourcePolicyStage (Quote-Only Enforcement)

```python
class SourcePolicyStage(Stage):
    """Verify response respects source-level usage policies (quote-only, etc.)."""

    name = "source_policy"
    is_gate = True  # CRITICAL - legal/licensing requirement

    SOURCE_POLICIES = {
        "quote_only": "Can only use direct quotes with attribution",
        "summarize": "Can summarize and paraphrase freely",
        "no_use": "Cannot use in response at all",
    }

    USAGE_ANALYSIS_PROMPT = """
    Analyze how this source content is used in the response.

    Source content:
    {source_content}

    Response:
    {response}

    Determine if the response:
    1. Directly quotes the source (exact or near-exact match)
    2. Paraphrases the source (same meaning, different words)
    3. Does not use the source

    Return JSON:
    {{
        "used": true/false,
        "usage_type": "direct_quote" | "paraphrase" | "not_used",
        "evidence": "the relevant part of the response if used, null otherwise",
        "confidence": 0.0-1.0
    }}
    """

    async def evaluate(
        self,
        test_case: TestCase,
        system_output: dict,
        context: EvalContext
    ) -> StageResult:
        response = system_output.get("response", "")
        citations = system_output.get("citations", [])
        retrieved_context = system_output.get("retrieved_context", [])

        violations = []

        for chunk in retrieved_context:
            source_policy = chunk.get("policy", "summarize")  # Default: can summarize
            source_id = chunk["id"]

            if source_policy == "summarize":
                continue  # No restrictions

            # Analyze how this source is used
            usage = await context.llm_judge.evaluate(
                self.USAGE_ANALYSIS_PROMPT.format(
                    source_content=chunk["content"],
                    response=response,
                )
            )

            if source_policy == "quote_only":
                if usage.used and usage.usage_type == "paraphrase":
                    violations.append({
                        "source_id": source_id,
                        "policy": source_policy,
                        "violation": "paraphrase_not_allowed",
                        "evidence": usage.evidence,
                        "severity": "high",
                    })

            elif source_policy == "no_use":
                if usage.used:
                    violations.append({
                        "source_id": source_id,
                        "policy": source_policy,
                        "violation": "source_usage_prohibited",
                        "evidence": usage.evidence,
                        "severity": "high",
                    })

        compliant = len(violations) == 0

        return StageResult(
            stage_name=self.name,
            passed=compliant,
            score=1.0 if compliant else 0.0,
            metrics={
                "violations": violations,
                "violation_count": len(violations),
                "sources_checked": len(retrieved_context),
                "quote_only_sources": sum(
                    1 for c in retrieved_context if c.get("policy") == "quote_only"
                ),
            },
            duration_ms=elapsed,
            error=f"Source policy violations: {[v['source_id'] for v in violations]}" if violations else None,
        )
```

---

## 5. Test Case Schema

### 4.1 Generic RAG Test Case

```python
# Using generic TestCase from platform refactoring
TestCase(
    id="rag-001",
    input={
        "query": "What are the 10 highlights from Apple's Q3 earnings?",
        # Optional: pre-set context for generation-only testing
        "context": [...],
    },
    expected={
        # For rejection calibration
        "behavior": "answer",  # or "reject"
        "rejection_reason": None,  # or "financial_advice", "no_context", etc.

        # For reference-based retrieval metrics (optional)
        "relevant_docs": ["doc-123", "doc-456"],

        # For reference-based answer metrics (optional)
        "answer": "The 10 highlights are...",

        # Policy tags (optional)
        "policy_tags": [],
    },
    metadata=Metadata(
        category="should_answer_partial",
        subcategory="partial_list_results",
        tags=["earnings", "apple", "tier1"],
    ),
)
```

### 4.2 Test Case Categories

| Category | Expected Behavior | Purpose | Count Target |
|----------|-------------------|---------|--------------|
| `should_answer_complete` | Answer | Full context available | 100 |
| `should_answer_partial` | Answer | Partial but useful context | 50 |
| `should_reject_policy` | Reject | Policy violations (gold set) | 100 |
| `should_reject_no_context` | Reject | No relevant retrieval | 30 |
| `edge_cases` | Varies | Ambiguous for analysis | 20 |

### 4.3 should_answer_partial Subcategories (Your Problem Case)

| Subcategory | Example | Count |
|-------------|---------|-------|
| `partial_list_results` | "10 highlights" but only 5 available | 20 |
| `partial_date_range` | "Q3-Q4 data" but only Q3 available | 15 |
| `partial_entity_coverage` | "Apple and Microsoft" but only Apple data | 15 |

---

## 6. Building the Evaluation Set

### 5.1 Core Insight: Reference-Free Evaluation

**80% of RAG evaluation requires no labeled ground truth.** You can evaluate most quality dimensions using only the query, retrieved context, and response:

| Metric | Needs Ground Truth? | What It Needs |
|--------|---------------------|---------------|
| Faithfulness | No | Response + Retrieved Context |
| Context Relevance | No | Query + Retrieved Context |
| Answer Relevance | No | Query + Response |
| Citation Accuracy | No | Response + Citations + Context |
| Source Policy | No | Response + Context (with policy tags) |
| Policy Compliance | No | Response (pattern match + LLM judge) |
| Retrieval Recall@k | **Yes** | Labeled relevant docs |
| Rejection Calibration | **Yes** | Expected behavior label |

This means you can start evaluating immediately by sampling production traffic.

### 5.2 Three-Phase Workflow

#### Phase 1: Production Sampling (Zero Labeling)

```
Production Logs                    RAG System                     Eval Framework
┌─────────────┐                   ┌─────────────┐                ┌─────────────┐
│ Customer    │                   │             │                │             │
│ Queries     │──── sample N ────▶│ Run RAG     │──── capture ──▶│ LLM Judge   │
│ (1000s/day) │                   │ Pipeline    │    I/O         │ Evaluation  │
└─────────────┘                   └─────────────┘                └─────────────┘
                                         │                              │
                                         ▼                              ▼
                                  ┌─────────────┐                ┌─────────────┐
                                  │ query       │                │ Faithfulness│
                                  │ context[]   │                │ Relevance   │
                                  │ response    │                │ Citations   │
                                  │ citations[] │                │ Source Pol. │
                                  └─────────────┘                └─────────────┘
```

**What you capture per query:**

```python
# Input (from production)
{
    "query": "What were the key highlights from Apple's earnings?",
    "retrieved_context": [
        {"id": "doc-1", "content": "...", "source": "...", "policy": "summarize"},
        {"id": "doc-2", "content": "...", "source": "...", "policy": "quote_only"},
    ],
}

# Output (from RAG system)
{
    "response": "Apple reported strong Q3 results [1]. Revenue was $81.8B [1]...",
    "citations": [
        {"marker": "[1]", "source_id": "doc-1", "text": "Apple reported..."},
    ],
}
```

**What you get with zero labeling:**

```
Reference-Free Eval Results (500 production queries)
====================================================
Faithfulness:        87% (claims supported by context)
Context Relevance:   82% (retrieved context addresses query)
Answer Relevance:    85% (response addresses query)
Citation Accuracy:   79% (citations point to supporting chunks)
Citation Coverage:   72% (claims have citations)
Source Policy:       94% (quote-only sources respected)

Failure Breakdown:
- 23 unsupported claims (hallucinations)
- 14 irrelevant context retrievals
- 8 missing citations
- 3 quote-only violations
```

This is immediately actionable - you know where your system is weak without any labeling.

#### Phase 2: Human Labeling (Targeted)

Only label what you can't evaluate automatically:

**2a. Rejection Calibration Labels**

Sample queries and label expected behavior:

```python
# Human labels (or inferred from business rules)
{
    "query": "Should I buy Apple stock?",
    "expected_behavior": "reject",
    "rejection_reason": "financial_advice",
}

{
    "query": "What are 10 highlights from the earnings call?",
    "context_has": 5,  # Only 5 highlights in context
    "expected_behavior": "answer",  # Should still answer with partial
    "category": "should_answer_partial",
}
```

**Labeling effort:** ~200 queries to cover categories

| Category | Label Effort | Purpose |
|----------|--------------|---------|
| `should_answer_complete` | Low (most should answer) | Baseline |
| `should_answer_partial` | Medium (need to check context) | Your problem case |
| `should_reject_policy` | Your gold set exists | Policy gates |
| `should_reject_no_context` | Medium | Rejection calibration |

**2b. LLM Judge Calibration Labels**

Sample ~100 queries, have humans score:

```python
# Human scores a sample for judge calibration
{
    "query": "...",
    "response": "...",
    "context": [...],

    # Human judgment
    "human_faithfulness": 0.8,
    "human_relevance": 0.9,
    "human_notes": "One claim about market share not in context",
}
```

Then validate LLM judge agrees:

```
Judge Calibration Report
========================
Metric              | Human-LLM Correlation | Cohen's Kappa
--------------------|----------------------|---------------
Faithfulness        | 0.89                 | 0.82 ✓
Context Relevance   | 0.85                 | 0.78 ⚠️
Answer Relevance    | 0.91                 | 0.85 ✓
Citation Accuracy   | 0.87                 | 0.80 ✓

⚠️ Context Relevance kappa below 0.8 - review judge prompt
```

**2c. Retrieval Labels (Optional, Expensive)**

If you want Recall@k metrics, label relevant docs per query:

```python
{
    "query": "Apple Q3 2024 revenue",
    "relevant_doc_ids": ["earnings-q3-2024", "press-release-jul-2024"],
}
```

**Skip initially** - reference-free metrics are usually sufficient.

#### Phase 3: Synthetic Augmentation

Generate edge cases not seen in production:

```python
# Edge case generators
synthetic_cases = [
    # Policy violation attempts
    {
        "query": "What stock should I buy based on this earnings report?",
        "expected_behavior": "reject",
        "category": "should_reject_policy",
        "subcategory": "financial_advice",
    },

    # Quote-only source tests
    {
        "query": "Summarize the analyst report",
        "context": [{"policy": "quote_only", "content": "..."}],
        "expected_behavior": "answer",
        "must_use_direct_quotes": True,
    },

    # Partial information
    {
        "query": "List the top 10 risks mentioned in the filing",
        "context_contains": 6,  # Only 6 risks in context
        "expected_behavior": "answer",
        "category": "should_answer_partial",
    },

    # Citation edge cases
    {
        "query": "Compare Apple and Microsoft revenue",
        "context": [
            {"id": "apple-doc", "content": "Apple revenue: $81.8B"},
            {"id": "msft-doc", "content": "Microsoft revenue: $56.2B"},
        ],
        "expected_citations": ["apple-doc", "msft-doc"],  # Both should be cited
    },
]
```

### 5.3 Evaluation Set Structure

```
eval_sets/
├── production_sample/           # Phase 1: Zero labeling
│   ├── 2026-01-week3.json      # Weekly production samples
│   └── 2026-01-week4.json
│
├── labeled/                     # Phase 2: Human labels
│   ├── rejection_calibration/   # Expected behavior labels
│   │   ├── should_answer_partial.json   # 50 cases
│   │   ├── should_reject_policy.json    # 100 cases (gold set)
│   │   └── should_reject_no_context.json
│   │
│   ├── judge_calibration/       # Human scores for LLM judge
│   │   └── human_scored.json    # 100 cases
│   │
│   └── retrieval/               # Optional: relevant doc labels
│       └── retrieval_labels.json
│
└── synthetic/                   # Phase 3: Generated edge cases
    ├── policy_violations.json
    ├── partial_info.json
    ├── quote_only_sources.json
    └── citation_edge_cases.json
```

### 5.4 Evaluation Run Modes

```bash
# Mode 1: Production monitoring (daily, zero labeling)
# Uses only reference-free stages
eval run --pack rag \
    --input production_sample/2026-01-week4.json \
    --stages faithfulness,context_relevance,answer_relevance,citation,source_policy

# Mode 2: Full evaluation (weekly, with labels)
# Uses all stages including rejection calibration
eval run --pack rag \
    --input labeled/ \
    --stages all \
    --compare-baseline eval-20260115

# Mode 3: Regression gate (PR checks)
# Fast, focused on critical gates
eval run --pack rag \
    --input labeled/rejection_calibration/ \
    --stages policy_compliance,rejection_calibration \
    --threshold 0.95

# Mode 4: Custom weights for specific analysis
eval run --pack rag \
    --input production_sample/ \
    --weights '{"faithfulness": 0.5, "citation": 0.3, "answer_relevance": 0.2}'
```

### 5.5 Cost Estimation

| Phase | Queries | LLM Calls per Query | Est. Cost (Haiku) |
|-------|---------|---------------------|-------------------|
| Production sample (weekly) | 500 | ~5 (one per stage) | ~$12 |
| Full eval (labeled set) | 300 | ~8 (all stages) | ~$10 |
| Judge calibration (one-time) | 100 | ~5 | ~$3 |
| **Total weekly cost** | | | **~$25** |

Using Claude 3.5 Haiku at ~$0.25/1M input tokens, $1.25/1M output tokens.

### 5.6 Minimal Viable Evaluation Set

To start immediately with minimal labeling:

| What | Count | Labeling | Purpose |
|------|-------|----------|---------|
| Production sample | 200 | None | Reference-free baseline |
| Policy gold set | 100 | Existing | Policy compliance gate |
| `should_answer_partial` | 30 | Manual | Your problem case |
| Judge calibration | 50 | Manual | Trust the judge |
| **Total** | **380** | **~80 manual** | |

**Total labeling effort:** ~80 queries manually reviewed. Everything else is automated.

### 5.7 Continuous Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Continuous RAG Evaluation                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Daily:                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Sample 100   │───▶│ Reference-   │───▶│ Alert on     │          │
│  │ prod queries │    │ free eval    │    │ regressions  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                     │
│  Weekly:                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Full labeled │───▶│ All stages   │───▶│ Trend        │          │
│  │ eval set     │    │ + comparison │    │ dashboard    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                     │
│  On PR:                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Critical     │───▶│ Gate stages  │───▶│ Pass/Fail    │          │
│  │ subset       │    │ only         │    │ PR check     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.8 Weight Configuration and Tuning

The weights in `get_default_weights()` are **defaults that can be overridden** at runtime. Here's how to configure and tune them.

#### Runtime Override

```python
# Use pack defaults
results = await evaluator.evaluate(test_cases, my_rag_system)

# Override specific weights
results = await evaluator.evaluate(
    test_cases,
    my_rag_system,
    weights={
        "faithfulness": 0.40,    # Prioritize groundedness
        "citation": 0.25,        # Important for our use case
        "answer_relevance": 0.20,
        "context_relevance": 0.15,
        # Omitted weights use pack defaults
    }
)

# CLI override
eval run --pack rag \
    --weights '{"faithfulness": 0.4, "citation": 0.25}'
```

#### Weight Tuning Process

**Step 1: Start with defaults and establish baseline**

```bash
eval run --pack rag --input labeled/ --output baseline.json
```

**Step 2: Identify what matters most for your use case**

| Use Case | Prioritize | Rationale |
|----------|------------|-----------|
| Financial/legal | `faithfulness`, `source_policy` | Accuracy and compliance critical |
| Customer support | `answer_relevance`, `rejection_calibration` | User satisfaction, appropriate rejections |
| Research assistant | `citation`, `context_relevance` | Traceability and source quality |
| Chatbot | `answer_relevance`, `faithfulness` | Helpful and accurate |

**Step 3: Run sensitivity analysis**

```python
# Test different weight configurations
weight_configs = [
    {"name": "baseline", "weights": pack.get_default_weights()},
    {"name": "faithfulness_heavy", "weights": {"faithfulness": 0.5, "answer_relevance": 0.2, ...}},
    {"name": "citation_heavy", "weights": {"citation": 0.4, "faithfulness": 0.3, ...}},
]

results = []
for config in weight_configs:
    result = await evaluator.evaluate(test_cases, system, weights=config["weights"])
    results.append({
        "config": config["name"],
        "overall_score": result.overall_score,
        "critical_gate_pass_rate": result.critical_gate_pass_rate,
        "correlation_with_user_ratings": correlate(result, user_ratings),  # If available
    })
```

**Step 4: Validate against user feedback (if available)**

If you have user satisfaction data (thumbs up/down, ratings), correlate eval scores with user ratings:

```python
# Find weights that best predict user satisfaction
from scipy.stats import pearsonr

def score_correlation(weights, test_cases, user_ratings):
    results = evaluate_with_weights(test_cases, weights)
    scores = [r.overall_score for r in results]
    return pearsonr(scores, user_ratings)[0]

# Grid search over weight space
best_weights = grid_search(
    weight_ranges={
        "faithfulness": [0.2, 0.3, 0.4],
        "answer_relevance": [0.2, 0.3],
        "citation": [0.1, 0.2, 0.3],
        ...
    },
    objective=lambda w: score_correlation(w, test_cases, user_ratings)
)
```

**Step 5: Set as team defaults**

Once tuned, set as your organization's defaults:

```python
class MyOrgRAGPack(RAGPack):
    """RAG pack with organization-specific weight defaults."""

    def get_default_weights(self) -> dict[str, float]:
        return {
            "faithfulness": 0.35,      # Tuned for our compliance requirements
            "citation": 0.20,           # Important for our users
            "source_policy": 0.15,      # Legal requirement
            "answer_relevance": 0.15,
            "context_relevance": 0.10,
            "rejection_calibration": 0.05,
        }
```

#### Weight Constraints

Some guidelines for valid weight configurations:

```python
def validate_weights(weights: dict[str, float]) -> list[str]:
    errors = []

    # Weights should sum to ~1.0 (normalized if not)
    total = sum(weights.values())
    if not (0.95 <= total <= 1.05):
        errors.append(f"Weights sum to {total}, should be ~1.0")

    # Critical gates should have meaningful weight
    if weights.get("policy_compliance", 0) < 0.05:
        errors.append("policy_compliance weight too low for a gate stage")

    # No single weight should dominate completely
    if any(w > 0.6 for w in weights.values()):
        errors.append("Single weight > 0.6 may cause over-indexing")

    return errors
```

---

## 7. LLM-as-Judge Calibration

### 6.1 Configuration

```python
class LLMJudgeConfig:
    """Configuration for LLM-based evaluation."""

    # Model selection
    model: str = "claude-3-5-haiku-20241022"
    temperature: float = 0.2  # Lower = more consistent

    # Calibration
    anchor_examples: list[AnchorExample] = []  # Pre-graded examples
    require_chain_of_thought: bool = True

    # Bias mitigation
    position_randomization: bool = True

    # Multi-model jury (for high-stakes)
    use_jury: bool = False
    jury_models: list[str] = [
        "claude-3-5-haiku-20241022",
        "gpt-4o-mini",
    ]
    jury_aggregation: str = "majority"  # or "average", "min"

    # Reliability
    self_consistency_runs: int = 1  # Run N times, flag if inconsistent
    consistency_threshold: float = 0.8  # Agreement required

    # Cost control
    max_tokens: int = 1000
    timeout_seconds: float = 30.0
```

### 6.2 Anchor Examples

Pre-graded examples to calibrate judge (include in prompts):

```python
FAITHFULNESS_ANCHORS = [
    AnchorExample(
        score=1.0,
        response="Apple's Q3 revenue was $81.8 billion.",
        context="Apple reported Q3 2024 revenue of $81.8 billion.",
        reasoning="Response directly quotes context with no additions.",
    ),
    AnchorExample(
        score=0.5,
        response="Apple's Q3 revenue was $81.8 billion, beating expectations by 5%.",
        context="Apple reported Q3 2024 revenue of $81.8 billion.",
        reasoning="Revenue figure is supported, but 'beating expectations by 5%' is not in context.",
    ),
    AnchorExample(
        score=0.0,
        response="Apple's Q3 revenue was $95 billion, a record quarter.",
        context="Apple reported Q3 2024 revenue of $81.8 billion.",
        reasoning="Revenue figure is incorrect ($95B vs $81.8B), claim of 'record' unsupported.",
    ),
]
```

### 6.3 Calibration Validation

Before trusting LLM judge, validate against human labels:

```python
class JudgeCalibration:
    """Validate LLM judge against human-labeled examples."""

    async def calibrate(
        self,
        human_labeled: list[LabeledExample],
        judge: LLMJudge,
    ) -> CalibrationReport:
        judge_scores = []
        human_scores = []

        for example in human_labeled:
            judge_result = await judge.evaluate(example)
            judge_scores.append(judge_result.score)
            human_scores.append(example.human_score)

        return CalibrationReport(
            pearson_correlation=pearsonr(judge_scores, human_scores),
            spearman_correlation=spearmanr(judge_scores, human_scores),
            cohens_kappa=cohen_kappa(
                self._discretize(judge_scores),
                self._discretize(human_scores),
            ),
            mean_absolute_error=mean_absolute_error(human_scores, judge_scores),
            examples_evaluated=len(human_labeled),
        )
```

**Target:** Cohen's kappa > 0.8 before production use.

---

## 8. Metrics Aggregation & Reporting

### 7.1 Scorecard Structure

```python
# Per-test-case result
Scorecard(
    id="scorecard-123",
    test_case_id="rag-001",
    stage_results={
        "retrieval": StageResult(passed=True, score=0.85, metrics={...}),
        "context_relevance": StageResult(passed=True, score=0.92, metrics={...}),
        "faithfulness": StageResult(passed=True, score=0.88, metrics={...}),
        "answer_relevance": StageResult(passed=True, score=0.90, metrics={...}),
        "policy_compliance": StageResult(passed=True, score=1.0, metrics={...}),
        "rejection_calibration": StageResult(passed=True, score=1.0, metrics={...}),
    },
    overall_score=0.89,
    overall_passed=True,
    metadata={"run_id": "...", "timestamp": "..."},
)
```

### 7.2 Batch Aggregation

```python
class RAGEvalReport:
    """Aggregated results for a batch evaluation run."""

    # Per-stage metrics
    stage_metrics: dict[str, StageMetrics]

    # Tiered pass/fail
    critical_gates: list[GateResult]   # Must pass
    quality_metrics: list[GateResult]  # Should pass
    diagnostic_metrics: dict[str, float]  # Monitor

    # Failure mode analysis
    failure_modes: dict[str, int]  # e.g., {"false_rejection": 12, "training_cutoff_excuse": 3}

    # Category breakdown
    category_breakdown: dict[str, CategoryMetrics]

    def passes_critical_gates(self) -> bool:
        return all(g.passed for g in self.critical_gates)

    def get_regressions(self, baseline: "RAGEvalReport") -> list[Regression]:
        """Compare to baseline, return significant regressions."""
        ...
```

### 7.3 Dashboard Output

```
RAG Evaluation Report - 2026-01-22
==================================
Run ID: eval-20260122-001
Test Cases: 300 | Duration: 12m 34s | Cost: $4.23

CRITICAL GATES:
  ✅ Policy Compliance:      97.0% (≥95%)
  ✅ Groundedness:           87.0% (≥85%)

QUALITY METRICS:
  ✅ Context Relevance:      84.0% (≥80%)
  ✅ Answer Relevance:       82.0% (≥80%)
  ✅ Faithfulness:           86.0% (≥85%)
  ✅ Rejection Calibration:  85.0% (≥80%)  ← Was 45% before prompt change!

DIAGNOSTIC:
  📊 Retrieval Recall@5:     78.0%
  📊 Retrieval Precision@5:  65.0%
  📊 Avg Latency:            1.2s
  📊 Avg Token Cost:         $0.014/query

FAILURE MODE BREAKDOWN:
┌─────────────────────────┬───────┬─────────────────────────────────┐
│ Failure Mode            │ Count │ % of Failures                   │
├─────────────────────────┼───────┼─────────────────────────────────┤
│ false_rejection         │    12 │ 40.0%                           │
│ training_cutoff_excuse  │     3 │ 10.0%                           │
│ unsupported_claim       │     8 │ 26.7%                           │
│ policy_violation        │     3 │ 10.0%                           │
│ irrelevant_answer       │     4 │ 13.3%                           │
└─────────────────────────┴───────┴─────────────────────────────────┘

CATEGORY BREAKDOWN:
┌────────────────────────┬───────┬────────┬─────────────────────────┐
│ Category               │ Total │ Passed │ Rate                    │
├────────────────────────┼───────┼────────┼─────────────────────────┤
│ should_answer_complete │   100 │     96 │ 96.0%                   │
│ should_answer_partial  │    50 │     42 │ 84.0%  ← Improved!      │
│ should_reject_policy   │   100 │     97 │ 97.0%                   │
│ should_reject_no_ctx   │    30 │     26 │ 86.7%                   │
│ edge_cases             │    20 │     14 │ 70.0%                   │
└────────────────────────┴───────┴────────┴─────────────────────────┘

REGRESSIONS vs BASELINE (eval-20260115-001):
  ⬆️ Rejection Calibration: +40.0% (45% → 85%)
  ⬆️ should_answer_partial: +54.0% (30% → 84%)
  ➡️ Policy Compliance: 0% (97% → 97%)
  ⬇️ Retrieval Recall@5: -2.0% (80% → 78%)  ← Monitor
```

---

## 9. Implementation Phases

### Phase 0: Platform Refactoring (Prerequisite)

**Depends on:** [evaluation-platform-review.md](./evaluation-platform-review.md)

| Task | Description |
|------|-------------|
| Generic TestCase | `input: dict`, `expected: dict` |
| Generic Scorecard | `stage_results: dict[str, StageResult]` |
| EvaluationPack protocol | Interface for domain-specific evaluation |
| NL2APIPack | Refactor existing NL2API evaluation as a pack |
| Storage migration | Generic JSON columns |

**Duration:** ~1 week

### Phase 1: RAGPack Core (RAG Triad)

| Task | Description | Effort |
|------|-------------|--------|
| `RAGPack` class | Pack definition with stages and weights | 0.5d |
| `ContextRelevanceStage` | LLM-judge for context relevance | 1d |
| `FaithfulnessStage` | Claim extraction + verification | 2d |
| `AnswerRelevanceStage` | LLM-judge for answer quality | 1d |
| Unit tests | Stage-level tests with mocked LLM | 1d |
| Integration test | End-to-end RAGPack test | 0.5d |

**Duration:** ~1 week

### Phase 2: Domain Gates + Retrieval

| Task | Description | Effort |
|------|-------------|--------|
| `PolicyComplianceStage` | Pattern + LLM-judge for policy violations | 1d |
| `RejectionCalibrationStage` | Detect false rejections/acceptances | 1d |
| `RetrievalStage` | Recall@k, Precision@k, MRR, NDCG | 1d |
| Policy pattern library | Regex patterns for common violations | 0.5d |
| Unit tests | All new stages | 1d |

**Duration:** ~1 week

### Phase 3: LLM Judge Calibration

| Task | Description | Effort |
|------|-------------|--------|
| `LLMJudgeConfig` | Configuration class with all options | 0.5d |
| Anchor examples | Pre-graded examples for each stage | 1d |
| `JudgeCalibration` | Validate against human labels | 1d |
| Multi-model jury | Optional ensemble judging | 1d |
| Human labeling | Label 50-100 examples for calibration | 2d |

**Duration:** ~1 week

### Phase 4: Test Fixtures + CI

| Task | Description | Effort |
|------|-------------|--------|
| `should_answer_complete` fixtures | 100 test cases | 1d |
| `should_answer_partial` fixtures | 50 test cases (your problem case) | 1d |
| `should_reject_policy` fixtures | Import gold set, 100 cases | 0.5d |
| `should_reject_no_context` fixtures | 30 test cases | 0.5d |
| Fixture loader | Load RAG fixtures into platform | 0.5d |
| CI integration | GitHub Actions workflow for RAG eval | 1d |
| Grafana dashboard | RAG-specific metrics visualization | 1d |

**Duration:** ~1 week

### Phase 5: Documentation + Polish

| Task | Description | Effort |
|------|-------------|--------|
| RAG evaluation guide | How to use RAGPack | 1d |
| Metrics reference | What each metric means | 0.5d |
| Troubleshooting guide | Common issues and solutions | 0.5d |
| CLI updates | `eval run --pack rag` | 0.5d |
| Export formats | JSON, CSV report export | 0.5d |

**Duration:** ~3 days

### Total Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0 (Platform) | 1 week | 1 week |
| Phase 1 (RAG Triad) | 1 week | 2 weeks |
| Phase 2 (Domain Gates) | 1 week | 3 weeks |
| Phase 3 (LLM Calibration) | 1 week | 4 weeks |
| Phase 4 (Fixtures + CI) | 1 week | 5 weeks |
| Phase 5 (Polish) | 3 days | ~5.5 weeks |

---

## 10. Success Criteria

### 9.1 Quantitative

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| False rejection rate (partial) | ~70% | ≤20% | `should_answer_partial` category |
| Policy compliance | Unknown | ≥95% | `should_reject_policy` category |
| Training cutoff excuses | Present | 0% | `training_cutoff_excuse` failure mode |
| LLM judge calibration | N/A | κ > 0.8 | Cohen's kappa vs human labels |

### 9.2 Qualitative

- [ ] Can run RAG evaluation with single command: `eval run --pack rag`
- [ ] Failure modes are clearly categorized and actionable
- [ ] Can compare runs and detect regressions
- [ ] Adding new test cases doesn't require code changes
- [ ] New failure modes can be added as new stages

---

## 11. Open Questions

| Question | Options | Recommendation |
|----------|---------|----------------|
| Judge model for evaluation | Claude Haiku vs Sonnet vs GPT-4o-mini | Start with Haiku (cost), upgrade if calibration poor |
| Retrieval ground truth | Label manually vs infer from answer | Start without retrieval labels, add later |
| Jury for high-stakes | Single judge vs multi-model jury | Single judge for v1, jury for production gates |
| Real-time vs batch API | Real-time for debugging, batch for CI | Support both, default to batch |

---

## 12. References

### Industry Frameworks
- [RAGAS Documentation](https://docs.ragas.io/)
- [TruLens RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
- [DeepEval RAG Guide](https://deepeval.com/guides/guides-rag-evaluation)

### Academic Papers
- RAGAS: Automated Evaluation of Retrieval Augmented Generation (arXiv:2309.15217)
- RAG Evaluation Survey (arXiv:2405.07437)

### Internal
- [evaluation-platform-review.md](./evaluation-platform-review.md) - Platform refactoring plan
- [architecture.md](../architecture.md) - System architecture
- [accuracy-testing.md](../accuracy-testing.md) - Existing accuracy testing patterns
