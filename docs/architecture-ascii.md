# Architecture Diagrams (ASCII)

> Plain-text versions of diagrams from [ARCHITECTURE.md](ARCHITECTURE.md) for terminals and editors without Mermaid support.

---

## 1.1 Evalkit Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                       Evalkit Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  src/evalkit/                                                    │
│  ├─ contracts/       Data models (TestCase, Scorecard, etc.)    │
│  ├─ batch/           Batch runner, checkpointing, metrics       │
│  ├─ core/            Evaluators (AST, temporal, semantics)      │
│  ├─ common/          Storage, telemetry, cache, resilience      │
│  ├─ distributed/     Redis queues, worker coordination          │
│  ├─ packs/           Pack registry (NL2API, RAG)                │
│  └─ cli/             CLI commands (batch, continuous, matrix)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.2 NL2API Application

```
┌─────────────────────────────────────────────────────────────────┐
│                         NL2API System                            │
├─────────────────────────────────────────────────────────────────┤
│  NL2APIOrchestrator                                              │
│  ├─ Query classification (route to domain agent)                │
│  ├─ Entity resolution (Company → RIC, 2.9M entities)            │
│  └─ Ambiguity detection → Clarification flow                    │
├─────────────────────────────────────────────────────────────────┤
│  Domain Agents (5 implemented)                                   │
│  ├─ DatastreamAgent     (price, time series, calculated fields) │
│  ├─ EstimatesAgent      (I/B/E/S forecasts, recommendations)    │
│  ├─ FundamentalsAgent   (WC codes, TR codes, financials)        │
│  ├─ OfficersAgent       (executives, compensation, governance)  │
│  └─ ScreeningAgent      (SCREEN expressions, rankings)          │
├─────────────────────────────────────────────────────────────────┤
│  Support Components                                              │
│  ├─ LLM Abstraction (Claude + OpenAI providers)                 │
│  ├─ RAG Retriever (hybrid vector + keyword, pgvector)           │
│  ├─ Conversation Manager (multi-turn, query expansion)          │
│  └─ Entity Resolver (database-backed, 99.5% accuracy)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MANAGEMENT PLANE (REST API)                            │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐      ┌─────────────────────────────────────────────────────────┐
  │  Clients    │─────▶│  Management API (ACA / Azure Functions)                 │
  │  - Web UI   │      │  ┌─────────────┬──────────────┬────────────────────┐    │
  │  - CI/CD    │      │  │ /test-cases │ /test-suites │ /target-systems    │    │
  │  - Scripts  │      │  │ /clients    │ /runs        │                    │    │
  └─────────────┘      │  └─────────────┴──────────────┴────────────────────┘    │
                       └──────────┬──────────────────────────────┬───────────────┘
                                  │                              │
                                  ▼                              ▼
                       ┌──────────────────┐           ┌─────────────────┐
                       │  Azure AI Search │           │  Service Bus    │
                       │  (Gold Store)    │           │  Queue          │
                       │  - Test Cases    │           └────────┬────────┘
                       │  - Test Suites   │                    │
                       │  - Configs       │                    │
                       └──────────────────┘                    │
                                                               │
┌──────────────────────────────────────────────────────────────┼──────────────────┐
│                           EXECUTION PLANE (Workers)          │                   │
└──────────────────────────────────────────────────────────────┼──────────────────┘
                                                               │
                                                               ▼
                       ┌──────────────────┐           ┌──────────────────┐
                       │  Azure AI Search │◀── fetch ─│  Worker (ACA)    │
                       │  (Gold Store)    │           │  - Stateless     │
                       └──────────────────┘           │  - Idempotent    │
                                                      └────────┬─────────┘
                                                               │
                                                               ▼
                                                      ┌─────────────────────────┐
                                                      │  Target System (LLM)    │
                                                      │  - GPT-4o / Claude / etc│
                                                      │  - Tool Calling         │
                                                      └────────┬────────────────┘
                                                               │
                                                               ▼
                                                      ┌─────────────────────────┐
                                                      │  Evaluation Pipeline    │
                                                      │  ┌───────────────────┐  │
                                                      │  │ Stage 1: Syntax   │  │
                                                      │  │ Stage 2: Logic    │  │
                                                      │  │ Stage 3: Exec     │  │
                                                      │  │ Stage 4: Semantic │  │
                                                      │  └───────────────────┘  │
                                                      └───────────┬─────────────┘
                                                                  │
                                                                  ▼
                                                      ┌─────────────────────────┐
                                                      │  Azure Table Storage    │
                                                      │  - Scorecards           │
                                                      │  - Run Results          │
                                                      └─────────────────────────┘
```

---

## 4.1 Evaluation Pipeline (Waterfall)

```
┌───────────┐     ┌──────────────────┐     ┌─────────────┐     ┌─────────────┐
│ TestCase  │────▶│ Generate Response│────▶│  Stage 1    │────▶│  Stage 2    │
│           │     │ (LLM call)       │     │  Syntax     │     │  Logic      │
└───────────┘     └──────────────────┘     └──────┬──────┘     └──────┬──────┘
                                                  │                   │
                                             FAIL │              PASS │
                                                  ▼                   ▼
                                           ┌──────────┐        ┌─────────────┐
                                           │  HALT    │        │  Stage 3    │
                                           │  ❌      │        │  Execution  │
                                           └──────────┘        └──────┬──────┘
                                                                      │
                                                                      ▼
                                                               ┌─────────────┐
                                                               │  Stage 4    │
                                                               │  Semantics  │
                                                               └──────┬──────┘
                                                                      │
                                                                      ▼
                                                               ┌─────────────┐
                                                               │  Scorecard  │
                                                               └─────────────┘
```

---

## 5. Batch Evaluation Flow

```
CLI/API                BatchRunner           TestCaseRepo          Pack              LLM            ScorecardRepo
   │                       │                      │                  │                │                  │
   │  batch run --limit N  │                      │                  │                │                  │
   │──────────────────────▶│                      │                  │                │                  │
   │                       │  fetch_test_cases()  │                  │                │                  │
   │                       │─────────────────────▶│                  │                │                  │
   │                       │                      │                  │                │                  │
   │                       │   List[TestCase]     │                  │                │                  │
   │                       │◀─────────────────────│                  │                │                  │
   │                       │                      │                  │                │                  │
   │                       │        ┌─────────────────────────────────────────────────────────────┐     │
   │                       │        │ loop [For each TestCase - concurrent]                       │     │
   │                       │        │                                                             │     │
   │                       │        │  evaluate(test_case)           │                │           │     │
   │                       │────────┼───────────────────────────────▶│                │           │     │
   │                       │        │                                │ generate()     │           │     │
   │                       │        │                                │───────────────▶│           │     │
   │                       │        │                                │   response     │           │     │
   │                       │        │                                │◀───────────────│           │     │
   │                       │        │                                │                │           │     │
   │                       │        │                                │ Stage 1-4      │           │     │
   │                       │        │                                │────────┐       │           │     │
   │                       │        │                                │        │       │           │     │
   │                       │        │                                │◀───────┘       │           │     │
   │                       │        │          Scorecard             │                │           │     │
   │                       │◀───────┼────────────────────────────────│                │           │     │
   │                       │        │                                │                │           │     │
   │                       │        │  save(scorecard)               │                │           │     │
   │                       │────────┼─────────────────────────────────────────────────────────────┼────▶│
   │                       │        │                                │                │           │     │
   │                       │        └─────────────────────────────────────────────────────────────┘     │
   │                       │                      │                  │                │                  │
   │   BatchResult         │                      │                  │                │                  │
   │◀──────────────────────│                      │                  │                │                  │
   │                       │                      │                  │                │                  │
```

---

## 13.2 Test Case States

```
┌──────────┐     deprecate      ┌────────────┐     archive      ┌──────────┐
│  ACTIVE  │ ─────────────────▶ │ DEPRECATED │ ───────────────▶ │ ARCHIVED │
└──────────┘                    └────────────┘                  └──────────┘
     │                                │
     │ validate                       │ revalidate
     ▼                                ▼
┌──────────┐                    ┌────────────┐
│  STALE   │ ◀──── detect ───── │  (alerts)  │
└──────────┘                    └────────────┘
     │
     │ fix & revalidate
     │
     └─────────────────────────▶ ACTIVE
```

| State | Description | Action |
|-------|-------------|--------|
| `ACTIVE` | Current, valid, running in suites | Normal evaluation |
| `STALE` | Detected drift, needs review | Exclude from pass/fail metrics |
| `DEPRECATED` | Old API version, kept for regression | Run but don't alert on failures |
| `ARCHIVED` | No longer relevant | Excluded from all runs |
