# SEC RAG Evaluation Test Set - Implementation Plan

## Overview

Create a comprehensive evaluation test set for RAG systems operating on SEC 10-K and 10-Q filings from S&P 500 companies.

## Current State

- **SEC Ingestion**: Running (~8 hours total), currently has 53,109 chunks from 100 companies
- **Target**: 500 S&P 500 companies, 2 years of 10-K and 10-Q filings (~4,000 filings, ~400k chunks expected)
- **RAG Pipeline**: 8-stage evaluation pack ready (retrieval, faithfulness, answer relevance, etc.)

## Two-Phase Approach

### Phase 1: Generate Questions (Execute Now)
Generate and persist ~1,500 financial analyst questions - independent of ingestion completion.

### Phase 2: Generate Answers (After Ingestion Completes)
Run questions against full database, retrieve context, generate reference answers, persist as test set.

---

## Phase 1: Question Generation

### Question Taxonomy (~1,500 total)

| Category | Count | Description |
|----------|-------|-------------|
| **Financial Metrics** | 350 | Revenue, earnings, margins, cash flow, ratios |
| **Business Operations** | 300 | Segments, products, customers, geography, strategy |
| **Risk Factors** | 250 | Competitive, regulatory, operational, macro, legal |
| **Management Discussion** | 250 | Trends, outlook, guidance, YoY changes, drivers |
| **Corporate Governance** | 150 | Executives, compensation, board, ownership |
| **Complex Queries** | 150 | Multi-hop, aggregation, cross-filing comparison |
| **Rejection Cases** | 50 | Out-of-scope, policy violations, investment advice |

### Generation Strategy: Template + LLM Variation

1. Create ~500 question templates covering all SEC sections
2. Use Claude 3.5 Haiku to generate 3 natural variations per template
3. Parameterize with company names, tickers, fiscal years
4. Use Claude 3.5 Sonnet for complex_queries category (better reasoning)

### LLM Prompts (Approved)

#### Question Variation Prompt
```
SYSTEM: You are a financial analyst generating questions about SEC filings (10-K and 10-Q reports).
Given a template question, generate 3 natural variations that ask the same thing but with different phrasing.

Rules:
- Keep the same semantic meaning
- Vary formality and structure
- Use natural financial terminology
- Do NOT change company name, ticker, or fiscal year

Output: JSON array of 3 strings
```

#### Answer Generation Prompt (Phase 2)
```
SYSTEM: You are a financial analyst answering questions based on SEC filing excerpts.

Rules:
- Answer ONLY based on provided context
- Include specific numbers, dates, and facts
- If insufficient context: {"answer": null, "reason": "...", "citations": []}

Output: JSON with "answer" (plain text) and "citations" array:
{
  "answer": "Plain text answer",
  "citations": [
    {"filing": "AAPL 10-K FY2023", "section": "Item 8", "chunk_id": "..."}
  ]
}
```

#### Rejection Case Generation Prompt
```
SYSTEM: Generate test questions that a SEC filing Q&A system should REJECT.

Categories:
- investment_advice: Buy/sell/hold, price predictions
- out_of_scope: Topics not in SEC filings
- speculation: Predictions beyond filed facts
- confidential: Non-public/insider information
- [EXPANDABLE: abuse, self-harm, etc. - add categories as needed]

Output: JSON with "question" and "rejection_reason"
```

### Output Format

```json
{
  "_meta": {
    "name": "sec_filings_rag_questions",
    "capability": "rag_evaluation",
    "schema_version": "1.0",
    "generated_at": "2026-01-23T...",
    "phase": "questions_only",
    "awaiting_answers": true
  },
  "test_cases": [
    {
      "id": "sec-rag-financial-001",
      "input": {"query": "What was Apple's total revenue in fiscal year 2023?"},
      "expected": {
        "behavior": "answer",
        "requires_citations": true
      },
      "tags": ["revenue", "10-K", "financial_metrics"],
      "category": "rag",
      "subcategory": "financial_metrics",
      "complexity": 1,
      "metadata": {
        "template_id": "revenue_lookup",
        "company_param": "Apple",
        "year_param": "2023"
      }
    }
  ]
}
```

### Files Created

1. `scripts/data/sec/sp500_companies.json` - Company metadata (tickers, CIKs, names)
2. `scripts/data/sec/question_templates.json` - Template definitions
3. `scripts/generators/sec_rag_question_generator.py` - Question generator
4. `scripts/generate_sec_rag_answers.py` - Phase 2 answer generator (stub)
5. `tests/fixtures/rag/sec_filings/questions.json` - Generated questions

---

## Phase 2: Answer Generation (Post-Ingestion)

### Trigger
Run after SEC ingestion completes.

### Process

1. **Load questions** from `tests/fixtures/rag/sec_filings/questions.json`
2. **For each question**:
   - Run hybrid retrieval against pgvector
   - Capture retrieved chunk IDs as `relevant_docs` ground truth
   - Generate reference answer using Claude with retrieved context
   - Store answer and citations metadata in test case
3. **Persist** complete test cases with ground truth

### Output Format (Complete Test Case)

```json
{
  "id": "sec-rag-financial-001",
  "input": {"query": "What was Apple's total revenue in fiscal year 2023?"},
  "expected": {
    "relevant_docs": ["chunk-uuid-1", "chunk-uuid-2"],
    "behavior": "answer",
    "answer": "Apple's total net sales were $383.3 billion for fiscal year 2023, a decrease of 3% from the prior year.",
    "citations": [
      {"filing": "AAPL 10-K FY2023", "section": "Item 8", "chunk_id": "chunk-uuid-1"}
    ],
    "requires_citations": true
  },
  "tags": ["revenue", "10-K", "financial_metrics"],
  "category": "rag",
  "subcategory": "financial_metrics"
}
```

---

## Cost Estimate

| Phase | LLM Calls | Model | Est. Cost |
|-------|-----------|-------|-----------|
| Question variations (~500 templates) | 500 | Haiku | ~$0.32 |
| Rejection questions (~50) | 50 | Haiku | ~$0.02 |
| Answer generation - standard (~1,350) | 1,350 | Haiku | ~$3.24 |
| Answer generation - complex (~150) | 150 | Sonnet | ~$1.80 |
| **Total** | | | **~$5.38** |

---

## Verification

### After Phase 1:
```bash
# Check questions generated
cat tests/fixtures/rag/sec_filings/questions.json | jq '.test_cases | length'
# Expected: ~1,500

# Check category distribution
cat tests/fixtures/rag/sec_filings/questions.json | jq '[.test_cases[].subcategory] | group_by(.) | map({key: .[0], count: length})'
```

### After Phase 2:
```bash
# Load fixtures and run evaluation
python scripts/load_fixtures_to_db.py --fixture tests/fixtures/rag/sec_filings/

# Run batch evaluation
python -m src.evaluation.cli.main batch run --pack rag --tag sec_filings --limit 50

# Check results
python -m src.evaluation.cli.main batch list
```

---

## Rejection Categories (Expandable)

Current categories:
- `investment_advice` - Buy/sell/hold recommendations
- `out_of_scope` - Topics not covered in SEC filings
- `speculation` - Predictions beyond filed facts
- `confidential` - Requests for non-public information

**Placeholder for expansion:**
- `abuse` - Harmful content requests
- `self_harm` - Self-harm related queries
- `illegal_activity` - Requests related to illegal activities
- `pii_extraction` - Attempts to extract personal information

To add new rejection categories:
1. Update `scripts/data/sec/question_templates.json` with new category templates
2. Add category to `REJECTION_CATEGORIES` in generator
3. Regenerate rejection test cases
