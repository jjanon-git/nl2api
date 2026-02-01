# RAG Evaluation Guide

Measure whether your RAG system produces correct answers.

## Quick Start

```bash
# 1. Put your questions in a file (one per line)
cat > my_questions.txt << 'EOF'
What was Apple's total revenue in fiscal 2024?
How did Microsoft's cloud revenue change year-over-year?
What are Tesla's main risk factors?
EOF

# 2. Enrich with metadata (uses Claude, ~$0.001/question)
python scripts/gen-enrich-rag-questions.py my_questions.txt -o tests/fixtures/rag/my_eval.json

# 3. Load to database
python scripts/load-rag-fixtures.py --fixture tests/fixtures/rag/my_eval.json --clear

# 4. Run evaluation
python -m src.evalkit.cli.main batch run --pack rag --tag rag --mode generation --label my-test

# 5. View results
open http://localhost:3000  # Grafana → RAG Evaluation
```

## What Gets Measured

| Metric | Question Answered |
|--------|-------------------|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucinations) |
| **Answer Relevance** | Does it answer the question? Contains expected keywords? |
| **Context Relevance** | Did retrieval find relevant content? |

These are the metrics that matter. They don't require you to know which chunks are "correct" - they measure end-to-end quality.

## Input Format

The enrichment script generates this from your questions:

```json
{
  "query": "What was Apple's total revenue in fiscal 2024?",
  "category": "simple_factual",
  "company": "AAPL",
  "answer_keywords": ["revenue", "fiscal 2024", "$391 billion"],
  "difficulty": "easy"
}
```

- `answer_keywords`: Terms a correct answer should contain (auto-generated, review if needed)
- `category`: Used for filtering and breakdown in dashboards
- `difficulty`: Used for complexity analysis

## Viewing Results

**Grafana** (http://localhost:3000):
- Overall pass rate
- Breakdown by stage (faithfulness, relevance, etc.)
- Trends over time

**CLI**:
```bash
python -m src.evalkit.cli.main batch list
python -m src.evalkit.cli.main batch results <batch-id>
```

## Comparing Runs

```bash
# Run baseline
python -m src.evalkit.cli.main batch run --pack rag --tag rag --label baseline-v1

# Make changes to retriever/generator

# Run again
python -m src.evalkit.cli.main batch run --pack rag --tag rag --label improved-v2

# Compare in Grafana - select both labels
```

## Note on Retrieval Metrics

The Retrieval stage (recall@k, precision@k) requires human-verified ground truth chunks. If you don't provide `relevant_chunk_ids`, it's skipped automatically.

This is intentional: auto-generated chunk IDs (from what retrieval returns) make the metric circular. The other stages measure what actually matters - does the system produce correct, grounded answers?

## All Evaluation Stages

| Stage | Weight | Gate? | What It Checks |
|-------|--------|-------|----------------|
| Retrieval | 15% | No | Did we find relevant chunks? |
| Context Relevance | 15% | No | Is retrieved context relevant to query? |
| Faithfulness | 25% | No | Is answer grounded in context? |
| Answer Relevance | 15% | No | Does answer address the question? |
| Citation | 10% | No | Are sources properly cited? |
| Source Policy | 5% | **Yes** | Quote-only vs summarize rules |
| Policy Compliance | 5% | **Yes** | No PII/harmful content |
| Rejection Calibration | 10% | No | Appropriate refusals |

**Gate stages** must pass or the entire test fails. Other stages contribute to weighted score.

## Custom Thresholds

Override defaults in your evaluation:

```python
from src.rag.evaluation import RAGPack, RAGPackConfig

config = RAGPackConfig(
    faithfulness_threshold=0.5,    # Default: 0.4
    answer_relevance_threshold=0.8, # Default: 0.7
    retrieval_enabled=False,        # Skip retrieval metrics
)
pack = RAGPack(config=config)
```

Or via CLI (coming soon):
```bash
python -m src.evalkit.cli.main batch run \
    --pack rag \
    --pack-config '{"faithfulness_threshold": 0.5}'
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/gen-enrich-rag-questions.py` | Convert raw questions → evaluation fixtures |
| `scripts/load-rag-fixtures.py` | Load fixtures to database |
| `src/rag/evaluation/pack.py` | Evaluation logic and stage configuration |
