# Proposal: Distributed Evaluation Infrastructure

**Status:** Proposed
**Target Architecture:** Producer-Consumer with Shared State
**Primary Goal:** Enable horizontal scaling of evaluation workloads (19k+ test cases) across multiple worker nodes.

---

## 1. Architecture: The "Chunked Dispatch" Pattern

We will move from a monolithic "Run & Wait" model to a "Dispatch & Forget" model using **Redis** as the broker.

### Current (Monolithic)
```mermaid
[CLI/BatchRunner]
    │
    ├── Fetch 19k Tests (DB)
    ├── Loop with Semaphore (max 10)
    │     ├── Evaluate
    │     └── Update DB
    └── Update Batch Status
```

### Proposed (Distributed)
```mermaid
[CLI / Scheduler]                   [Redis Queue]                     [Worker Nodes xN]
       │                                  │                                   │
       ├── Create BatchJob (DB)           │◄──── Poll Task ───────────────    │
       ├── Chunk Tests (e.g. 50/chunk)    │                                   │
       └── Push Chunks ────────────────► [ Task: EvalChunk ] ───────────► [ Executor ]
                                          │                                   │
                                          │                                   ├── Fetch Test Data (DB)
                                          │                                   ├── Run Evaluator
                                          │                                   ├── Save Scorecards (DB)
                                          │                                   └── Atomic Increment Batch Progress (DB)
```

## 2. Technology Stack

*   **Broker:** **Redis** (Already in stack).
*   **Queue Library:** **`arq`** (Async Task Queue).
    *   *Why?* Native `asyncio` support (fits our async codebase), lightweight, integrates easily with Pydantic, no heavy "Celery" overhead.
*   **Worker Runtime:** Existing Docker container reused with a new entry point.

## 3. Detailed Data Flow

### Step 1: Scheduling (The Producer)
The `BatchRunner` is refactored into a `BatchScheduler`.
1.  Creates `BatchJob` record in Postgres (Status: `IN_PROGRESS`).
2.  Selects Test Cases based on filters.
3.  Groups Test Case IDs into **Chunks** (Configurable, default: 50).
    *   *Reasoning:* Reducing 19,000 tasks to ~380 chunks reduces queue overhead and DB contention.
4.  Enqueues `evaluate_chunk(batch_id, test_case_ids)` jobs to Redis.

### Step 2: Execution (The Worker)
New service `src/evaluation/worker/main.py` runs an `arq` worker pool.
1.  **Startup:** Initializes `NL2APIOrchestrator` (heavy init) and DB pools *once* per process.
2.  **Job Processing:**
    *   Receives list of `test_case_ids`.
    *   Fetches full `TestCase` objects from Postgres.
    *   Runs existing `WaterfallEvaluator` logic concurrently (via `asyncio.gather` within the chunk).
    *   Bulk inserts `Scorecard` results to Postgres.
3.  **Progress Update:**
    *   Updates `BatchJob` counters in Postgres using atomic increment.

### Step 3: Completion (The Watcher)
How do we know when the batch is finished?
*   **Mechanism:** The Worker checks if `completed + failed == total_tests` after every chunk update.
*   **Action:** If true, the Worker marks the `BatchJob` as `COMPLETED` and calculates final duration.

## 4. Implementation Plan

### Phase 1: Infrastructure & Dependencies (P0)
1.  Add `arq` to `pyproject.toml`.
2.  Create `src/evaluation/worker` directory.
3.  Create `src/common/queue` module to manage Redis connection settings for Arq.
4.  Add `worker` service to `docker-compose.yml`.

### Phase 2: Refactoring Codebase (P1)
1.  **Extract Evaluator:** Ensure `WaterfallEvaluator` can be initialized independently of `BatchRunner`.
2.  **Create Worker Entry Point:** Implement the `arq` worker definition and `on_startup` hooks.
3.  **Implement Job Function:** Create `run_evaluation_chunk` function.

### Phase 3: The Scheduler (P1)
1.  Create `src/evaluation/batch/scheduler.py`.
2.  Update CLI (`src/evaluation/cli/commands/batch.py`) to support a `--distributed` flag that uses the Scheduler instead of Runner.

### Phase 4: Observability (P2)
1.  Add OTEL spans to the Worker (trace context propagation).
2.  Add "Queue Depth" and "Worker Busy" metrics to Prometheus.

## 5. Migration Strategy

Side-by-side implementation:
*   `batch run --local` (Default): Uses existing `BatchRunner`.
*   `batch run --distributed`: Uses new `BatchScheduler`.
