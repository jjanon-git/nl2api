# RAG Evaluation Guide

Quick reference for running RAG evaluations end-to-end.

## 1. Fixture Format

Test cases live in `tests/fixtures/rag/`. Each has this structure:

```json
{
  "id": "simple_001",
  "query": "What was the company's total revenue?",
  "category": "simple_factual",
  "company": "AAPL",
  "relevant_chunk_ids": ["uuid-1", "uuid-2"],
  "answer_keywords": ["$394.3 billion", "revenue"],
  "difficulty": "easy",
  "metadata": {
    "company_name": "Apple Inc.",
    "filing_date": "2024-10-25"
  }
}
```

| Field | Purpose |
|-------|---------|
| `query` | Natural language question |
| `relevant_chunk_ids` | Ground truth: chunks containing the answer |
| `answer_keywords` | Required terms in a correct answer |
| `category` | `simple_factual`, `complex_analytical`, or `temporal_comparative` |
| `difficulty` | `easy`, `medium`, or `hard` |

## 2. Generating Test Cases

Test cases should be **retrieval-verified**: questions are generated from chunks, then verified via actual retrieval to ensure ground truth reflects real system behavior.

```bash
# Generate new evaluation dataset from SEC filings
python scripts/generate_rag_eval_dataset.py \
    --output tests/fixtures/rag/my_eval_set.json \
    --count 100 \
    --company AAPL
```

The generator:
1. Samples chunks with interesting content from SEC filings
2. Uses Claude to generate natural questions from each chunk
3. **Runs actual retrieval** for each question
4. Uses **retrieved chunks as ground truth** (not the source chunk)
   - If source chunk is in top-K: use it (ideal case)
   - If not: use top-K retrieved chunks (what the system actually returns)

### Why This Matters

Old methodology: Generate question from chunk A, use chunk A as ground truth.
Problem: Retrieval might return chunks B, C, D for that question - not chunk A.
Result: Binary 0%/100% hit rates, not measuring actual retrieval quality.

New methodology: Use what retrieval **actually returns** as ground truth.
Result: Measures real system performance, not theoretical matching.

**Note**: Existing fixtures in `sec_evaluation_set.json` may use the old methodology. Regenerate with the script above for accurate evaluation.

## 3. Loading Fixtures to Database

Before running evaluations, load fixtures into PostgreSQL:

```bash
# Start infrastructure
docker compose up -d

# Load default fixture (sec_evaluation_set.json, 466 cases)
python scripts/load_rag_fixtures.py

# Load specific fixture with options
python scripts/load_rag_fixtures.py \
    --fixture tests/fixtures/rag/my_eval_set.json \
    --clear \
    --limit 100
```

## 4. Running Evaluations

```bash
# Basic run (all RAG test cases)
python -m src.evalkit.cli.main batch run \
    --pack rag \
    --tag rag \
    --label my-experiment

# Filtered run (specific category, limited count)
python -m src.evalkit.cli.main batch run \
    --pack rag \
    --tag rag \
    --tag simple_factual \
    --label baseline-simple \
    --limit 50

# Resume interrupted batch
python -m src.evalkit.cli.main batch run \
    --pack rag \
    --tag rag \
    --resume <batch-id>
```

### Evaluation Modes

| Mode | Flag | What It Tests |
|------|------|---------------|
| Retrieval only | `--mode retrieval` | Just IR metrics (no LLM) |
| Full generation | `--mode generation` | Retrieval + LLM response (default) |
| Simulated | `--mode simulated` | Pipeline testing only |

## 5. Evaluation Stages

The RAG pack runs 8 stages:

| Stage | Weight | What It Measures |
|-------|--------|------------------|
| **Retrieval** | 15% | recall@k, precision@k, MRR, NDCG |
| **Context Relevance** | 15% | Is retrieved context relevant to query? |
| **Faithfulness** | 25% | Is response grounded in context? |
| **Answer Relevance** | 15% | Does response answer the question? |
| **Citation** | 10% | Are sources properly cited? |
| **Source Policy** | 5% | Quote-only vs summarize (GATE) |
| **Policy Compliance** | 5% | No PII/harmful content (GATE) |
| **Rejection Calibration** | 10% | Appropriate refusals |

GATE stages must pass; others contribute to weighted score.

## 6. Viewing Results

### Grafana Dashboards

Open http://localhost:3000 (admin/admin):
- **RAG Overview**: Pass rate, alerts, quick health
- **RAG Evaluation**: Stage breakdowns, trends, costs

### CLI

```bash
# List recent batches
python -m src.evalkit.cli.main batch list

# View batch details
python -m src.evalkit.cli.main batch results <batch-id>
```

### SQL

```sql
-- Compare two runs
SELECT
    sc.batch_id,
    COUNT(*) as total,
    ROUND(100.0 * SUM(CASE WHEN sc.overall_passed THEN 1 ELSE 0 END) / COUNT(*), 1) as pass_rate,
    ROUND(AVG(sc.overall_score), 3) as avg_score
FROM scorecards sc
WHERE sc.batch_id IN ('batch-1-id', 'batch-2-id')
GROUP BY sc.batch_id;

-- Breakdown by category
SELECT
    (sc.generated_output->>'category') as category,
    COUNT(*) as total,
    ROUND(100.0 * SUM(CASE WHEN sc.overall_passed THEN 1 ELSE 0 END) / COUNT(*), 1) as pass_rate
FROM scorecards sc
WHERE sc.batch_id = '<batch-id>'
GROUP BY category;
```

## 7. Typical Workflow

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Load fixtures (one-time)
python scripts/load_rag_fixtures.py

# 3. Run baseline evaluation
python -m src.evalkit.cli.main batch run \
    --pack rag --tag rag --label baseline-v1

# 4. Make changes to retriever/generator

# 5. Run comparison evaluation
python -m src.evalkit.cli.main batch run \
    --pack rag --tag rag --label improved-v2

# 6. Compare in Grafana or via SQL
```

## Key Files

| File | Purpose |
|------|---------|
| `tests/fixtures/rag/*.json` | Test case definitions |
| `scripts/generate_rag_eval_dataset.py` | Generate new test cases |
| `scripts/load_rag_fixtures.py` | Load fixtures to database |
| `src/rag/evaluation/pack.py` | RAG evaluation pack (8 stages) |
| `src/evalkit/batch/runner.py` | Batch execution engine |
| `config/grafana/.../rag/` | Dashboard definitions |
