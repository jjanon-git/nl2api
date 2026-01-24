# Distributed Evaluation Infrastructure Plan

**Status:** Ready for Review
**Created:** 2026-01-22
**Author:** Mostly Claude, with some minor assistance from Sid

---

## TL;DR - Phases 1-3 Scope (First Review Checkpoint)

**What we're building first:**

```
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Queue Abstraction                                 │
│ - TaskQueue protocol (enqueue, consume, ack, nack, dlq)   │
│ - InMemoryQueue for unit tests                            │
│ - Configuration models                                     │
├────────────────────────────────────────────────────────────┤
│ Phase 2: Redis Streams                                     │
│ - RedisStreamQueue implementation                          │
│ - Consumer groups, ACK/NACK, dead letter queue            │
│ - Stalled task detection (XPENDING/XCLAIM)                │
├────────────────────────────────────────────────────────────┤
│ Phase 3: Worker                                            │
│ - EvalWorker class with main loop                         │
│ - Graceful shutdown, retry logic                          │
│ - Integration with existing evaluator pipeline            │
│ - OTEL telemetry                                          │
└────────────────────────────────────────────────────────────┘
```

**What's NOT in phases 1-3:**
- Coordinator / CLI integration (Phase 4)
- Local worker manager / docker-compose (Phase 5)
- Rate limiting (Phase 6)
- Batch API integration (Phase 7)

**After Phase 3, you'll have:**
- Working queue infrastructure (Redis + in-memory)
- Workers that can process tasks with fault tolerance
- Full test coverage (unit + integration)
- Foundation for building Coordinator in Phase 4

---

## Executive Summary

Transform the evaluation system from single-process concurrent (`asyncio.Semaphore`) to distributed worker architecture using Redis Streams, with a clean abstraction layer enabling future migration to Azure Service Bus.

**Goals:**
1. **Fault tolerance first** - Worker crashes don't lose work; automatic retry
2. **Horizontal scale** - Add workers to increase throughput
3. **Local-first development** - Full distributed testing on single machine via Docker Compose
4. **Cloud-ready** - Same code deploys to Azure with minimal changes

**Target:** 80-100k tests/hour, scaling to tens of thousands of test cases.

---

## Rate Limit Analysis

| Mode | Anthropic Tier | RPM | Max Tests/Hour | Notes |
|------|----------------|-----|----------------|-------|
| Sync API | Tier 3 | 4,000 | ~80k | Practical limit without enterprise |
| Sync API | Tier 4+ | 8,000+ | 160k+ | Requires enterprise agreement |
| **Batch API** | Any | **Unlimited** | **100k+** | 50% cheaper, async (up to 24hr SLA) |

**Recommendation:**
- **Sync API** for small/urgent batches (< 1k tests, need results now)
- **Batch API** for large evaluations (1k+ tests, can wait hours)

The architecture supports both modes.

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI                                   │
│              eval batch run --limit 1000                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    BatchRunner                               │
│         (Single Process, asyncio.Semaphore)                 │
│                                                              │
│  1. Fetch all TestCases into memory                         │
│  2. asyncio.gather() with Semaphore(10)                     │
│  3. Evaluate all in-process                                 │
│  4. Save Scorecards to PostgreSQL                           │
└─────────────────────────────────────────────────────────────┘
```

**Limitations:**
- Single process = single CPU core
- All test cases in memory
- No fault tolerance (process crash = lost batch)
- No horizontal scaling

---

## Proposed Architecture

### Dual-Mode Design

The system supports two LLM interaction modes:

| Mode | Use Case | Throughput | Latency | Cost |
|------|----------|------------|---------|------|
| **Sync API** | Small batches, debugging, urgent | Up to 80k/hr (Tier 3) | Seconds | Standard |
| **Batch API** | Large evaluations, CI/CD | **100k+/hr** | Hours | **50% cheaper** |

### Sync API Mode

```
┌─────────────────────────────────────────────────────────────┐
│  eval batch run --distributed --workers 8 --limit 1000     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Coordinator                               │
│                                                              │
│  1. Create BatchJob                                         │
│  2. Enqueue WorkerTasks (test_case_id only)                │
│  3. Monitor progress                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Redis Streams                             │
│              eval:tasks:{batch_id}                          │
└─────┬───────────────┬───────────────┬───────────────┬───────┘
      │               │               │               │
      ▼               ▼               ▼               ▼
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
│  Worker 1 │   │  Worker 2 │   │  Worker 3 │   │  Worker N │
│           │   │           │   │           │   │           │
│ Fetch TC  │   │ Fetch TC  │   │ Fetch TC  │   │ Fetch TC  │
│ LLM Call  │◄──┼───────────┼───┼───────────┼───┤ Rate      │
│ Evaluate  │   │           │   │           │   │ Limiter   │
│ Save      │   │           │   │           │   │ (shared)  │
└─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
      │               │               │               │
      └───────────────┴───────┬───────┴───────────────┘
                              ▼
                         PostgreSQL
```

**Workers do:** Fetch test case → Call LLM → Evaluate → Save scorecard

### Batch API Mode

```
┌─────────────────────────────────────────────────────────────┐
│  eval batch run --distributed --batch-api --limit 10000    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Coordinator                               │
│                                                              │
│  1. Create BatchJob                                         │
│  2. Fetch all TestCases                                     │
│  3. Generate prompts for each                               │
│  4. Submit to Anthropic Batch API                          │
│  5. Poll until complete (background)                        │
│  6. Fetch responses                                         │
│  7. Enqueue (test_case_id, response) pairs                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Redis Streams                             │
│    eval:tasks:{batch_id} (includes pre-generated response) │
└─────┬───────────────┬───────────────┬───────────────┬───────┘
      │               │               │               │
      ▼               ▼               ▼               ▼
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
│  Worker 1 │   │  Worker 2 │   │  Worker 3 │   │  Worker N │
│           │   │           │   │           │   │           │
│ Fetch TC  │   │ Fetch TC  │   │ Fetch TC  │   │ Fetch TC  │
│ (no LLM)  │   │ (no LLM)  │   │ (no LLM)  │   │ (no LLM)  │
│ Evaluate  │   │ Evaluate  │   │ Evaluate  │   │ Evaluate  │
│ Save      │   │ Save      │   │ Save      │   │ Save      │
└─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
      │               │               │               │
      └───────────────┴───────┬───────┴───────────────┘
                              ▼
                         PostgreSQL
```

**Workers do:** Fetch test case → Evaluate pre-generated response → Save scorecard

**Key difference:** In Batch API mode, workers are **evaluation-only** (no LLM calls). The Coordinator handles all LLM interaction upfront, which allows for:
- No rate limit concerns
- 50% cost savings
- Massive parallelism during evaluation phase

---

## Component Design

### 1. Queue Abstraction Layer

**Purpose:** Enable Redis Streams now, Azure Service Bus later.

```python
# src/evaluation/distributed/queue/protocol.py

from typing import Protocol, AsyncIterator
from dataclasses import dataclass

@dataclass(frozen=True)
class QueueMessage:
    id: str                    # Message ID for acknowledgment
    payload: dict              # Task data
    attempt: int               # Retry count
    enqueued_at: datetime

@runtime_checkable
class TaskQueue(Protocol):
    """Abstract task queue interface."""

    async def enqueue(self, task: WorkerTask) -> str:
        """Add task to queue. Returns message ID."""
        ...

    async def enqueue_batch(self, tasks: list[WorkerTask]) -> list[str]:
        """Bulk enqueue for efficiency."""
        ...

    async def consume(self, consumer_id: str) -> AsyncIterator[QueueMessage]:
        """Yield messages for this consumer. Blocks when empty."""
        ...

    async def ack(self, message_id: str) -> None:
        """Acknowledge successful processing."""
        ...

    async def nack(self, message_id: str, requeue: bool = True) -> None:
        """Negative ack - requeue or send to DLQ."""
        ...

    async def get_pending_count(self) -> int:
        """Number of unprocessed messages."""
        ...

    async def get_failed_messages(self) -> list[QueueMessage]:
        """Messages in dead letter queue."""
        ...
```

**Implementations:**

| Implementation | Use Case | File |
|----------------|----------|------|
| `RedisStreamQueue` | Local dev, production | `queue/redis_stream.py` |
| `InMemoryQueue` | Unit tests | `queue/memory.py` |
| `AzureServiceBusQueue` | Future Azure migration | `queue/azure_sb.py` (stub) |

### 2. Worker

**Purpose:** Pull tasks, evaluate, report results.

```python
# src/evaluation/distributed/worker.py

class EvalWorker:
    """Distributed evaluation worker."""

    def __init__(
        self,
        worker_id: str,
        queue: TaskQueue,
        evaluator: Evaluator,
        scorecard_repo: ScorecardRepository,
        test_case_repo: TestCaseRepository,
        rate_limiter: RateLimiter | None = None,
        config: WorkerConfig = WorkerConfig(),
    ):
        ...

    async def run(self) -> None:
        """Main worker loop. Runs until shutdown signal."""
        async for message in self.queue.consume(self.worker_id):
            try:
                await self._process_task(message)
                await self.queue.ack(message.id)
            except Exception as e:
                await self._handle_failure(message, e)

    async def _process_task(self, message: QueueMessage) -> None:
        """Execute single evaluation task."""
        task = WorkerTask.model_validate(message.payload)

        # Fetch full test case (not stored in queue)
        test_case = await self.test_case_repo.get(task.test_case_id)

        # Rate limit if configured
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # Generate response (via configured generator)
        response = await self.response_generator(test_case)

        # Evaluate
        scorecard = await self.evaluator.evaluate(test_case, response)

        # Persist
        await self.scorecard_repo.save(scorecard)

        # Record metrics
        self.metrics.record_task_complete(task, scorecard)

    async def _handle_failure(self, message: QueueMessage, error: Exception) -> None:
        """Handle task failure with retry logic."""
        if message.attempt < self.config.max_retries:
            await self.queue.nack(message.id, requeue=True)
        else:
            await self.queue.nack(message.id, requeue=False)  # → DLQ
            logger.error(f"Task {message.id} failed permanently: {error}")
```

**Worker Configuration:**

```python
@dataclass
class WorkerConfig:
    max_retries: int = 3
    task_timeout_seconds: int = 300  # 5 min per task
    heartbeat_interval_seconds: int = 30
    graceful_shutdown_timeout: int = 60

    # Rate limiting (per worker)
    requests_per_minute: int | None = None  # None = no limit

    # Response generation
    eval_mode: str = "orchestrator"  # resolver, routing, tool_only, simulated
```

### 3. Coordinator

**Purpose:** Orchestrate batch execution, monitor progress, handle retries.

```python
# src/evaluation/distributed/coordinator.py

class BatchCoordinator:
    """Coordinates distributed batch evaluation."""

    def __init__(
        self,
        queue: TaskQueue,
        batch_repo: BatchJobRepository,
        test_case_repo: TestCaseRepository,
        scorecard_repo: ScorecardRepository,
        config: CoordinatorConfig = CoordinatorConfig(),
    ):
        ...

    async def start_batch(
        self,
        tags: list[str] | None = None,
        limit: int | None = None,
        eval_mode: str = "orchestrator",
    ) -> str:
        """Start a new distributed batch. Returns batch_id."""

        # 1. Fetch test case IDs (lightweight)
        test_cases = await self.test_case_repo.list(tags=tags, limit=limit)

        # 2. Create BatchJob
        batch_job = BatchJob(
            batch_id=str(uuid4()),
            total_tests=len(test_cases),
            status=BatchStatus.PENDING,
            tags=tags or [],
        )
        await self.batch_repo.create(batch_job)

        # 3. Enqueue tasks (bulk)
        tasks = [
            WorkerTask(
                task_id=str(uuid4()),
                batch_id=batch_job.batch_id,
                test_case_id=tc.id,
                eval_mode=eval_mode,
            )
            for tc in test_cases
        ]
        await self.queue.enqueue_batch(tasks)

        # 4. Update status
        await self.batch_repo.update(
            batch_job.batch_id,
            status=BatchStatus.IN_PROGRESS
        )

        return batch_job.batch_id

    async def monitor_batch(self, batch_id: str) -> BatchProgress:
        """Get current batch progress."""
        batch_job = await self.batch_repo.get(batch_id)
        scorecards = await self.scorecard_repo.get_by_batch(batch_id)
        pending = await self.queue.get_pending_count()

        return BatchProgress(
            batch_id=batch_id,
            total=batch_job.total_tests,
            completed=len(scorecards),
            passed=sum(1 for s in scorecards if s.passed),
            failed=sum(1 for s in scorecards if not s.passed),
            pending=pending,
            status=batch_job.status,
        )

    async def retry_failed(self, batch_id: str) -> int:
        """Re-enqueue failed tasks from DLQ. Returns count."""
        failed = await self.queue.get_failed_messages()
        batch_failed = [m for m in failed if m.payload.get("batch_id") == batch_id]

        for msg in batch_failed:
            task = WorkerTask.model_validate(msg.payload)
            task = task.model_copy(update={"attempt": 0})  # Reset attempt
            await self.queue.enqueue(task)

        return len(batch_failed)
```

### 4. Distributed Rate Limiter

**Purpose:** Coordinate rate limits across workers.

```python
# src/evaluation/distributed/rate_limiter.py

class DistributedRateLimiter:
    """Redis-backed token bucket rate limiter."""

    def __init__(
        self,
        redis: Redis,
        key_prefix: str = "ratelimit",
        requests_per_minute: int = 1000,
    ):
        self.redis = redis
        self.key = f"{key_prefix}:tokens"
        self.rpm = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0  # tokens per second

    async def acquire(self, tokens: int = 1) -> None:
        """Block until tokens available."""
        while True:
            available = await self._get_tokens()
            if available >= tokens:
                await self._consume(tokens)
                return

            # Wait for refill
            wait_time = (tokens - available) / self.refill_rate
            await asyncio.sleep(min(wait_time, 1.0))

    async def _get_tokens(self) -> float:
        """Get current token count, refilling as needed."""
        # Lua script for atomic get-and-refill
        ...
```

**Alternative: Per-Worker Limits**

For simpler deployments, divide total limit by worker count:
```python
# If 4 workers and 1000 RPM total → 250 RPM per worker
worker_rpm = total_rpm // worker_count
```

### 5. Worker Manager (Local Development)

**Purpose:** Run multiple workers locally for testing.

```python
# src/evaluation/distributed/manager.py

class LocalWorkerManager:
    """Manages multiple worker processes locally."""

    def __init__(self, worker_count: int = 4):
        self.worker_count = worker_count
        self.processes: list[Process] = []

    def start(self) -> None:
        """Spawn worker processes."""
        for i in range(self.worker_count):
            p = Process(
                target=run_worker,
                args=(f"worker-{i}",),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

    def stop(self) -> None:
        """Gracefully stop all workers."""
        for p in self.processes:
            p.terminate()
            p.join(timeout=30)
```

---

## Redis Streams Implementation Details

### Stream Structure

```
# Task stream (one per batch for isolation)
Stream: eval:tasks:{batch_id}
Fields:
  - task_id: str
  - test_case_id: str
  - eval_mode: str
  - attempt: int
  - enqueued_at: timestamp

# Consumer group
Group: eval-workers
Consumers: worker-0, worker-1, worker-2, ...

# Dead letter queue
Stream: eval:dlq:{batch_id}
```

### Key Operations

```python
# Enqueue task
XADD eval:tasks:{batch_id} * task_id {id} test_case_id {tc_id} ...

# Create consumer group (once per batch)
XGROUP CREATE eval:tasks:{batch_id} eval-workers 0 MKSTREAM

# Worker reads tasks (blocking)
XREADGROUP GROUP eval-workers worker-0 BLOCK 5000 STREAMS eval:tasks:{batch_id} >

# Acknowledge completion
XACK eval:tasks:{batch_id} eval-workers {message_id}

# Check pending (stalled tasks)
XPENDING eval:tasks:{batch_id} eval-workers

# Claim stalled task (after timeout)
XCLAIM eval:tasks:{batch_id} eval-workers worker-1 {min_idle_ms} {message_id}

# Move to DLQ after max retries
XADD eval:dlq:{batch_id} * ... (copy message)
XACK eval:tasks:{batch_id} eval-workers {message_id}
```

### Fault Tolerance Mechanisms

| Scenario | Detection | Recovery |
|----------|-----------|----------|
| Worker crash mid-task | XPENDING shows old entries | Coordinator claims and re-assigns |
| Task timeout | Message idle time > threshold | XCLAIM to another worker |
| Repeated failures | Attempt count ≥ max_retries | Move to DLQ |
| Coordinator crash | Batch status = IN_PROGRESS | Resume via `batch resume` CLI |

---

## CLI Changes

### New Commands

```bash
# ════════════════════════════════════════════════════════════
# DISTRIBUTED BATCH EXECUTION
# ════════════════════════════════════════════════════════════

# Sync API mode (rate-limited, immediate results)
eval batch run --distributed --workers 4 --limit 1000
eval batch run --distributed --workers 8 --limit 10000 --rpm-limit 4000

# Batch API mode (no rate limits, 50% cheaper, async)
eval batch run --distributed --batch-api --limit 100000
eval batch run --distributed --batch-api --workers 16 --limit 100000

# ════════════════════════════════════════════════════════════
# WORKER MANAGEMENT
# ════════════════════════════════════════════════════════════

# Start worker separately (for production/docker)
eval worker start --id worker-0
eval worker start --id worker-0 --queue redis://localhost:6379

# ════════════════════════════════════════════════════════════
# BATCH MANAGEMENT
# ════════════════════════════════════════════════════════════

# Monitor batch progress (live updates)
eval batch status <batch-id> --watch

# Retry failed tasks from DLQ
eval batch retry <batch-id>

# View dead letter queue
eval batch dlq <batch-id>

# Resume interrupted batch (e.g., after coordinator crash)
eval batch resume <batch-id>

# ════════════════════════════════════════════════════════════
# DEBUGGING / DEVELOPMENT
# ════════════════════════════════════════════════════════════

# Test infrastructure without LLM calls
eval batch run --distributed --mode simulated --workers 4 --limit 1000

# Single worker for debugging
eval batch run --distributed --workers 1 --limit 10 --verbose
```

### Backward Compatibility

```bash
# Original command still works (single-process mode)
eval batch run --limit 1000

# Explicitly use local mode
eval batch run --local --limit 1000
```

### Flag Summary

| Flag | Description | Default |
|------|-------------|---------|
| `--distributed` | Use distributed worker architecture | Off (local mode) |
| `--workers N` | Number of local worker processes | 4 |
| `--batch-api` | Use Anthropic Batch API (async) | Off (sync API) |
| `--rpm-limit N` | Rate limit for sync API | 4000 (Tier 3) |
| `--queue URL` | Redis URL for queue | `redis://localhost:6379` |
| `--watch` | Live progress updates | Off |

---

## Configuration

### Environment Variables

```bash
# Queue configuration
EVAL_QUEUE_BACKEND=redis              # redis | memory | azure_sb
EVAL_REDIS_URL=redis://localhost:6379
EVAL_AZURE_SB_CONNECTION_STRING=...   # Future

# Worker configuration
EVAL_WORKER_ID=worker-0
EVAL_WORKER_MAX_RETRIES=3
EVAL_WORKER_TASK_TIMEOUT=300
EVAL_WORKER_HEARTBEAT_INTERVAL=30

# Rate limiting
EVAL_RATE_LIMIT_RPM=1000              # Total across all workers
EVAL_RATE_LIMIT_MODE=distributed      # distributed | per_worker

# LLM configuration (existing)
NL2API_ANTHROPIC_API_KEY=...
NL2API_LLM_PROVIDER=anthropic
```

### Docker Compose Addition

```yaml
# docker-compose.yml additions

services:
  # ... existing services ...

  eval-worker:
    build: .
    command: python -m src.evaluation.distributed.worker
    environment:
      - EVAL_QUEUE_BACKEND=redis
      - EVAL_REDIS_URL=redis://redis:6379
      - NL2API_ANTHROPIC_API_KEY=${NL2API_ANTHROPIC_API_KEY}
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 4  # Scale workers
    restart: unless-stopped
```

---

## Directory Structure

```
src/evaluation/
├── distributed/
│   ├── __init__.py
│   ├── queue/
│   │   ├── __init__.py
│   │   ├── protocol.py          # TaskQueue protocol
│   │   ├── redis_stream.py      # Redis Streams implementation
│   │   ├── memory.py            # In-memory for tests
│   │   └── azure_sb.py          # Future: Azure Service Bus stub
│   ├── worker.py                # EvalWorker
│   ├── coordinator.py           # BatchCoordinator
│   ├── manager.py               # LocalWorkerManager
│   ├── rate_limiter.py          # DistributedRateLimiter
│   ├── config.py                # WorkerConfig, CoordinatorConfig
│   └── __main__.py              # Worker entry point
├── batch/
│   ├── runner.py                # Existing (local mode)
│   └── ...
└── cli/
    └── commands/
        ├── batch.py             # Add --distributed flag
        └── worker.py            # New: worker commands
```

---

## Implementation Phases

### ═══════════════════════════════════════════════════════════════
### PHASE GROUP A: Core Infrastructure (Phases 1-3)
### Review checkpoint after Phase 3
### ═══════════════════════════════════════════════════════════════

### Phase 1: Queue Abstraction & In-Memory Testing

**Goal:** Build queue protocol and test infrastructure.

**Deliverables:**
```
src/evaluation/distributed/
├── __init__.py
├── queue/
│   ├── __init__.py
│   ├── protocol.py          # TaskQueue protocol
│   └── memory.py            # InMemoryQueue for tests
├── config.py                # WorkerConfig, CoordinatorConfig
└── models.py                # Extended WorkerTask if needed

tests/unit/evaluation/distributed/
├── test_queue_protocol.py
└── test_memory_queue.py
```

**Tasks:**
1. [ ] Create `TaskQueue` protocol with full interface (enqueue, consume, ack, nack, etc.)
2. [ ] Create `InMemoryQueue` implementation for unit tests
3. [ ] Create configuration models (WorkerConfig, CoordinatorConfig)
4. [ ] Verify/extend `WorkerTask` model in `src/contracts/worker.py`
5. [ ] Write comprehensive unit tests for queue behavior

**Verification:**
- `pytest tests/unit/evaluation/distributed/ -v` passes
- Queue protocol covers all operations needed for fault tolerance

---

### Phase 2: Redis Streams Implementation

**Goal:** Production-ready queue with Redis Streams.

**Deliverables:**
```
src/evaluation/distributed/queue/
└── redis_stream.py          # RedisStreamQueue

tests/integration/evaluation/
└── test_redis_queue.py      # Integration tests
```

**Tasks:**
1. [ ] Implement `RedisStreamQueue` with consumer groups
2. [ ] Implement ACK/NACK with retry counting
3. [ ] Implement dead letter queue (DLQ) for permanent failures
4. [ ] Implement XPENDING monitoring for stalled task detection
5. [ ] Implement XCLAIM for task recovery
6. [ ] Add queue factory: `create_queue(config) -> TaskQueue`
7. [ ] Write integration tests (requires `docker compose up -d`)

**Key Redis Commands:**
```
XADD      - Enqueue task
XGROUP    - Create consumer group
XREADGROUP - Worker reads task (blocking)
XACK      - Acknowledge completion
XPENDING  - List unacknowledged tasks
XCLAIM    - Reassign stalled task
```

**Verification:**
- `pytest tests/integration/evaluation/test_redis_queue.py -v` passes
- Can enqueue 1000 tasks, consume with 4 workers, verify all ACKed
- Stalled task detection works (kill worker, verify task reassigned)

---

### Phase 3: Worker Implementation

**Goal:** Workers that process tasks reliably with fault tolerance.

**Deliverables:**
```
src/evaluation/distributed/
├── worker.py                # EvalWorker class
└── __main__.py              # Entry point: python -m src.evaluation.distributed

tests/unit/evaluation/distributed/
└── test_worker.py           # Worker unit tests (mocked queue)

tests/integration/evaluation/
└── test_worker_integration.py  # Worker with real Redis
```

**Tasks:**
1. [ ] Create `EvalWorker` class with main loop
2. [ ] Integrate with existing evaluator pipeline (`src/evaluation/core/evaluators.py`)
3. [ ] Integrate with existing response generators (`src/evaluation/batch/response_generators.py`)
4. [ ] Add graceful shutdown (SIGTERM/SIGINT handling)
5. [ ] Add OTEL spans for worker operations
6. [ ] Create worker entry point (`__main__.py`)
7. [ ] Write unit tests with mocked queue
8. [ ] Write integration test: spawn worker, enqueue tasks, verify scorecards saved

**Worker Loop:**
```python
async def run(self):
    async for message in self.queue.consume(self.worker_id):
        try:
            await self._process_task(message)
            await self.queue.ack(message.id)
        except Exception as e:
            await self._handle_failure(message, e)
```

**Verification:**
- Worker processes tasks from Redis queue
- Worker handles shutdown gracefully (finishes current task)
- Failed tasks are retried (up to max_retries)
- Permanently failed tasks go to DLQ
- Metrics appear in Grafana (if OTEL enabled)

---

### ══════════════════════════════════════════════════════════════
### 🔍 REVIEW CHECKPOINT: End of Phase 3
###
### At this point we have:
### - Queue abstraction (protocol + Redis + in-memory)
### - Working workers with fault tolerance
### - Unit + integration tests
###
### Before proceeding, review:
### 1. Does the queue protocol cover all needs?
### 2. Is worker fault tolerance working as expected?
### 3. Any design changes needed before building Coordinator?
### ══════════════════════════════════════════════════════════════

---

### ═══════════════════════════════════════════════════════════════
### PHASE GROUP B: Orchestration & CLI (Phases 4-5)
### ═══════════════════════════════════════════════════════════════

### Phase 4: Coordinator & CLI

**Goal:** Orchestrate distributed batches via CLI.

**Deliverables:**
```
src/evaluation/distributed/
├── coordinator.py           # BatchCoordinator

src/evaluation/cli/commands/
├── batch.py                 # Updated: --distributed flag
└── worker.py                # New: worker commands
```

**Tasks:**
1. [ ] Create `BatchCoordinator` (start_batch, monitor, retry_failed, resume)
2. [ ] Add `--distributed` flag to `eval batch run`
3. [ ] Add `eval batch retry <batch-id>` command
4. [ ] Add `eval batch dlq <batch-id>` command
5. [ ] Add `eval batch status <batch-id> --watch` for live progress
6. [ ] Add `eval worker start` command
7. [ ] Write integration tests for full distributed flow

**Verification:**
- `eval batch run --distributed --limit 100` completes successfully
- `eval batch retry` re-enqueues failed tasks
- Progress tracking works

---

### Phase 5: Local Worker Manager & Docker Compose

**Goal:** Easy local development with multiple workers.

**Deliverables:**
```
src/evaluation/distributed/
└── manager.py               # LocalWorkerManager

docker-compose.yml           # Updated: eval-worker service
```

**Tasks:**
1. [ ] Create `LocalWorkerManager` to spawn worker processes
2. [ ] Add `--workers N` flag to spawn local processes
3. [ ] Add worker health monitoring (detect crashed workers)
4. [ ] Update docker-compose.yml with eval-worker service
5. [ ] Write documentation for local development

**Verification:**
- `eval batch run --distributed --workers 4 --limit 1000` works locally
- Killing a worker doesn't lose tasks (reassigned to others)

---

### ═══════════════════════════════════════════════════════════════
### PHASE GROUP C: Scale & Production (Phases 6-8)
### ═══════════════════════════════════════════════════════════════

### Phase 6: Rate Limiting (Sync API Mode)

**Goal:** Coordinate rate limits across workers for sync API mode.

**Deliverables:**
```
src/evaluation/distributed/
└── rate_limiter.py          # DistributedRateLimiter
```

**Tasks:**
1. [ ] Implement Redis-backed token bucket rate limiter
2. [ ] Add per-model rate limits (Claude, OpenAI)
3. [ ] Add `--rpm-limit` CLI flag
4. [ ] Add rate limit metrics to Grafana
5. [ ] Write tests for rate limiter

**Verification:**
- Workers respect rate limits collectively
- No 429 errors from LLM providers during batch run

---

### Phase 7: Batch API Integration

**Goal:** Support Anthropic Batch API for high-volume, cost-efficient evaluation.

**Deliverables:**
```
src/evaluation/distributed/
├── batch_api/
│   ├── __init__.py
│   ├── submitter.py         # Submit batch to Anthropic
│   ├── poller.py            # Poll for completion
│   └── processor.py         # Process results → queue
└── coordinator.py           # Updated: batch_api mode
```

**Tasks:**
1. [ ] Create batch submitter (generate prompts, submit to Anthropic Batch API)
2. [ ] Create poller (check batch status, handle completion)
3. [ ] Create processor (fetch results, enqueue for evaluation)
4. [ ] Add `--batch-api` flag to CLI
5. [ ] Handle partial failures (some requests in batch fail)
6. [ ] Write integration tests (may need mock Batch API)

**Batch API Flow:**
```
1. Coordinator generates prompts for all test cases
2. Submit to Anthropic Batch API (returns batch_id)
3. Poll until status = "completed" (may take hours)
4. Fetch results
5. Enqueue (test_case_id, response) pairs to Redis
6. Workers evaluate (no LLM calls)
```

**Verification:**
- `eval batch run --distributed --batch-api --limit 10000` works
- Cost is ~50% lower than sync API
- Can handle 100k+ tests without rate limit issues

---

### Phase 8: Hardening & Documentation

**Goal:** Production readiness.

**Tasks:**
1. [ ] Add comprehensive error handling throughout
2. [ ] Add retry with exponential backoff for transient failures
3. [ ] Add batch timeout handling (stuck batches)
4. [ ] Add cleanup for old Redis streams
5. [ ] Create troubleshooting guide (`docs/distributed-eval-troubleshooting.md`)
6. [ ] Create operations runbook
7. [ ] Performance testing at target scale (80k tests/hour)
8. [ ] Add autoscaling to backlog (defer implementation)

**Verification:**
- Run 10k test batch successfully with sync API
- Run 100k test batch successfully with Batch API
- Recover from simulated failures (worker crash, Redis restart)
- Documentation complete

---

## Testing Strategy

### Unit Tests (No External Dependencies)

| Test | Purpose |
|------|---------|
| `test_memory_queue.py` | Queue protocol with in-memory impl |
| `test_worker_logic.py` | Worker task processing (mocked queue) |
| `test_coordinator_logic.py` | Coordinator batch management |
| `test_rate_limiter.py` | Token bucket algorithm |

### Integration Tests (Requires Docker)

| Test | Purpose |
|------|---------|
| `test_redis_queue.py` | Redis Streams operations |
| `test_distributed_batch.py` | Full batch with real Redis |
| `test_worker_recovery.py` | Task recovery after worker "crash" |
| `test_dlq_handling.py` | Dead letter queue flow |

### Chaos Testing (Manual/CI)

| Scenario | How to Test |
|----------|-------------|
| Worker crash | `docker kill eval-worker-1` mid-batch |
| Redis restart | `docker restart redis` during batch |
| Coordinator crash | Kill coordinator, run `batch resume` |
| Network partition | `docker network disconnect` |

---

## Telemetry & Observability

### Existing Infrastructure (Reuse)

The codebase already has comprehensive telemetry via `src/common/telemetry/`:

| Class | Purpose | Location |
|-------|---------|----------|
| `EvalMetrics` | Batch test counts, latency, scores | `metrics.py:159` |
| `NL2APIMetrics` | Request processing metrics | `metrics.py:23` |
| `AccuracyMetrics` | Accuracy test metrics | `metrics.py:364` |

**Existing metrics we'll reuse:**
- `eval_batch_tests_total` / `_passed` / `_failed`
- `eval_batch_duration_seconds`
- `eval_test_duration_ms`
- `eval_test_score`
- `eval_stage_passed` / `_failed`

### New Metrics (DistributedMetrics class)

```python
# src/evaluation/distributed/metrics.py

class DistributedMetrics:
    """Metrics for distributed evaluation infrastructure."""

    def __init__(self, meter_name: str = "nl2api"):
        self._meter = get_meter(meter_name)

        # ═══════════════════════════════════════════════════════════
        # WORKER METRICS
        # ═══════════════════════════════════════════════════════════

        # Worker lifecycle
        self._worker_started = self._meter.create_counter(
            name="eval_worker_started_total",
            description="Total worker starts",
        )
        self._worker_stopped = self._meter.create_counter(
            name="eval_worker_stopped_total",
            description="Total worker stops (graceful or crash)",
        )

        # Task processing
        self._worker_tasks_processed = self._meter.create_counter(
            name="eval_worker_tasks_processed_total",
            description="Tasks processed by workers",
            # Attributes: worker_id, status (success/failure/retry)
        )
        self._worker_task_duration = self._meter.create_histogram(
            name="eval_worker_task_duration_ms",
            description="Task processing duration per worker",
            # Attributes: worker_id, eval_mode
        )

        # Worker health
        self._worker_heartbeat = self._meter.create_counter(
            name="eval_worker_heartbeat_total",
            description="Worker heartbeats (for liveness detection)",
            # Attributes: worker_id
        )

        # ═══════════════════════════════════════════════════════════
        # QUEUE METRICS
        # ═══════════════════════════════════════════════════════════

        # Queue depth (gauge - use callback)
        self._queue_pending = self._meter.create_observable_gauge(
            name="eval_queue_pending_tasks",
            description="Current pending tasks in queue",
            callbacks=[self._observe_queue_pending],
            # Attributes: batch_id
        )
        self._queue_dlq_size = self._meter.create_observable_gauge(
            name="eval_queue_dlq_size",
            description="Tasks in dead letter queue",
            callbacks=[self._observe_dlq_size],
            # Attributes: batch_id
        )

        # Queue operations
        self._queue_enqueue = self._meter.create_counter(
            name="eval_queue_enqueue_total",
            description="Tasks enqueued",
            # Attributes: batch_id
        )
        self._queue_ack = self._meter.create_counter(
            name="eval_queue_ack_total",
            description="Tasks acknowledged (completed)",
            # Attributes: batch_id, worker_id
        )
        self._queue_nack = self._meter.create_counter(
            name="eval_queue_nack_total",
            description="Tasks negatively acknowledged (retry or DLQ)",
            # Attributes: batch_id, worker_id, reason (retry/dlq)
        )
        self._queue_claim = self._meter.create_counter(
            name="eval_queue_claim_total",
            description="Stalled tasks claimed by another worker",
            # Attributes: batch_id, from_worker, to_worker
        )

        # ═══════════════════════════════════════════════════════════
        # COORDINATOR METRICS
        # ═══════════════════════════════════════════════════════════

        self._batch_started = self._meter.create_counter(
            name="eval_distributed_batch_started_total",
            description="Distributed batches started",
            # Attributes: eval_mode, batch_api (true/false)
        )
        self._batch_completed = self._meter.create_counter(
            name="eval_distributed_batch_completed_total",
            description="Distributed batches completed",
            # Attributes: eval_mode, status (success/partial/failed)
        )

        # ═══════════════════════════════════════════════════════════
        # RATE LIMITER METRICS (Phase 6)
        # ═══════════════════════════════════════════════════════════

        self._rate_limit_wait = self._meter.create_histogram(
            name="eval_rate_limit_wait_ms",
            description="Time spent waiting for rate limit tokens",
            # Attributes: model
        )
        self._rate_limit_acquired = self._meter.create_counter(
            name="eval_rate_limit_acquired_total",
            description="Rate limit tokens acquired",
            # Attributes: model
        )

        # ═══════════════════════════════════════════════════════════
        # BATCH API METRICS (Phase 7)
        # ═══════════════════════════════════════════════════════════

        self._batch_api_submitted = self._meter.create_counter(
            name="eval_batch_api_submitted_total",
            description="Anthropic Batch API submissions",
        )
        self._batch_api_poll_duration = self._meter.create_histogram(
            name="eval_batch_api_poll_duration_seconds",
            description="Time spent polling Batch API for completion",
        )
```

### New Spans (Tracing)

| Component | Span Name | Attributes |
|-----------|-----------|------------|
| **Worker** | `eval.worker.process_task` | `worker_id`, `task_id`, `test_case_id`, `eval_mode` |
| **Worker** | `eval.worker.evaluate` | `test_case_id`, `result.passed`, `result.score` |
| **Worker** | `eval.worker.save_scorecard` | `test_case_id`, `batch_id` |
| **Queue** | `eval.queue.enqueue` | `batch_id`, `task_count` |
| **Queue** | `eval.queue.consume` | `worker_id`, `batch_id`, `task_id` |
| **Queue** | `eval.queue.ack` | `worker_id`, `task_id` |
| **Queue** | `eval.queue.nack` | `worker_id`, `task_id`, `reason` |
| **Queue** | `eval.queue.claim_stalled` | `task_id`, `from_worker`, `to_worker` |
| **Coordinator** | `eval.coordinator.start_batch` | `batch_id`, `test_count`, `eval_mode` |
| **Coordinator** | `eval.coordinator.monitor` | `batch_id`, `pending`, `completed` |
| **Coordinator** | `eval.coordinator.retry_failed` | `batch_id`, `retry_count` |
| **Rate Limiter** | `eval.rate_limiter.acquire` | `model`, `tokens_requested`, `wait_ms` |
| **Batch API** | `eval.batch_api.submit` | `batch_id`, `request_count` |
| **Batch API** | `eval.batch_api.poll` | `anthropic_batch_id`, `status` |
| **Batch API** | `eval.batch_api.process_results` | `batch_id`, `result_count` |

### Dashboard Updates

**New Dashboard: "Distributed Evaluation" (`evaluation-distributed.json`)**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ROW 1: OVERVIEW                                                              │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ Active Workers  │ Queue Depth     │ DLQ Size        │ Throughput (tasks/s)  │
│ (gauge)         │ (gauge)         │ (gauge)         │ (rate)                │
├─────────────────┴─────────────────┴─────────────────┴───────────────────────┤
│ ROW 2: BATCH PROGRESS                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Current Batch Progress (stacked bar: pending / processing / completed)      │
│ Pass Rate by Batch (time series)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ ROW 3: WORKER HEALTH                                                         │
├─────────────────┬─────────────────┬─────────────────────────────────────────┤
│ Tasks/Worker    │ Latency/Worker  │ Worker Status (table: id, tasks, last   │
│ (bar chart)     │ (heatmap)       │ heartbeat, status)                      │
├─────────────────┴─────────────────┴─────────────────────────────────────────┤
│ ROW 4: QUEUE OPERATIONS                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Enqueue/ACK/NACK rates (time series)                                        │
│ Stalled task claims (time series)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ROW 5: RATE LIMITING (Phase 6)                                               │
├─────────────────┬───────────────────────────────────────────────────────────┤
│ Rate Limit Wait │ Tokens Acquired vs Limit (time series)                    │
│ (histogram)     │                                                           │
├─────────────────┴───────────────────────────────────────────────────────────┤
│ ROW 6: BATCH API (Phase 7)                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Batch API Submissions | Poll Duration | Cost Savings vs Sync                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Grafana Panel Queries (PromQL)

| Panel | Query |
|-------|-------|
| Active Workers | `count(rate(nl2api_eval_worker_heartbeat_total[1m]) > 0)` |
| Queue Depth | `nl2api_eval_queue_pending_tasks` |
| DLQ Size | `nl2api_eval_queue_dlq_size` |
| Throughput | `sum(rate(nl2api_eval_worker_tasks_processed_total[1m]))` |
| Tasks/Worker | `sum by (worker_id) (rate(nl2api_eval_worker_tasks_processed_total[5m]))` |
| P95 Latency | `histogram_quantile(0.95, sum by (le) (rate(nl2api_eval_worker_task_duration_ms_bucket[5m])))` |
| Pass Rate | `sum(rate(nl2api_eval_batch_tests_passed_total[5m])) / sum(rate(nl2api_eval_batch_tests_total[5m]))` |
| Stalled Claims | `rate(nl2api_eval_queue_claim_total[5m])` |
| Rate Limit Wait P95 | `histogram_quantile(0.95, sum by (le) (rate(nl2api_eval_rate_limit_wait_ms_bucket[5m])))` |

### Alerting & Failure Detection

#### Failure Modes & Detection

| Failure Mode | How We Detect | Alert | Severity |
|--------------|---------------|-------|----------|
| **Worker crash** | Heartbeat stops | WorkerDown | Critical |
| **Worker hung** | Tasks not ACKed, heartbeat continues | WorkerStalled | Warning |
| **All workers down** | No heartbeats | AllWorkersDown | Critical |
| **Queue backing up** | Pending count growing | QueueBacklog | Warning |
| **Tasks failing repeatedly** | DLQ growing | DLQGrowing | Warning |
| **Tasks timing out** | XPENDING shows old entries | StalledTasks | Warning |
| **Batch stuck** | IN_PROGRESS for too long | BatchStuck | Warning |
| **High failure rate** | >10% tasks failing | HighFailureRate | Warning |
| **Rate limit exhausted** | P95 wait time > 5s | RateLimitSaturation | Info |
| **LLM API errors** | 429/5xx from Anthropic | LLMAPIErrors | Warning |
| **Redis down** | Connection failures | RedisDown | Critical |
| **PostgreSQL down** | Connection failures | PostgresDown | Critical |
| **Coordinator crash** | Batch IN_PROGRESS, no coordinator | CoordinatorDown | Critical |

#### Alert Definitions (Prometheus)

```yaml
# config/prometheus/alerts/distributed-eval.yml

groups:
  - name: distributed-eval-critical
    rules:
      # ═══════════════════════════════════════════════════════════
      # CRITICAL - Immediate attention required
      # ═══════════════════════════════════════════════════════════

      - alert: AllWorkersDown
        expr: |
          count(rate(nl2api_eval_worker_heartbeat_total[2m]) > 0) == 0
          and on() nl2api_eval_queue_pending_tasks > 0
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "No eval workers are running"
          description: "All workers have stopped but {{ $value }} tasks are pending"
          runbook: "1. Check worker containers: docker ps | grep eval-worker"
          action: "Restart workers: docker-compose up -d eval-worker"

      - alert: WorkerDown
        expr: |
          count(rate(nl2api_eval_worker_heartbeat_total[2m]) > 0) < 2
          and on() count(rate(nl2api_eval_worker_heartbeat_total[10m]) > 0) >= 2
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Eval worker count dropped"
          description: "Only {{ $value }} workers running (was >= 2)"
          runbook: "Check worker logs: docker logs eval-worker-N"

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Queue backend unavailable - all workers will fail"
          action: "Check Redis: docker-compose logs redis"

      - alert: PostgresDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
          description: "Cannot save scorecards or read test cases"

  - name: distributed-eval-warning
    rules:
      # ═══════════════════════════════════════════════════════════
      # WARNING - Investigate soon
      # ═══════════════════════════════════════════════════════════

      - alert: QueueBacklog
        expr: nl2api_eval_queue_pending_tasks > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Eval queue backing up"
          description: "{{ $value }} tasks pending for >10 minutes"
          runbook: "Scale workers or check for processing issues"
          action: "docker-compose up -d --scale eval-worker=8"

      - alert: DLQGrowing
        expr: increase(nl2api_eval_queue_dlq_size[1h]) > 10
        labels:
          severity: warning
        annotations:
          summary: "Dead letter queue growing"
          description: "{{ $value }} tasks moved to DLQ in past hour"
          runbook: "Investigate failed tasks: eval batch dlq <batch-id>"
          action: "Review errors, fix root cause, then: eval batch retry <batch-id>"

      - alert: StalledTasks
        expr: rate(nl2api_eval_queue_claim_total[5m]) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Tasks being reclaimed frequently"
          description: "{{ $value }} tasks/sec being reclaimed from stalled workers"
          runbook: "Workers may be crashing mid-task or timing out"

      - alert: HighFailureRate
        expr: |
          rate(nl2api_eval_worker_tasks_processed_total{status="failure"}[5m])
          / rate(nl2api_eval_worker_tasks_processed_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High task failure rate"
          description: "{{ $value | humanizePercentage }} of tasks failing"
          runbook: "Check worker logs for error patterns"

      - alert: BatchStuck
        expr: |
          (time() - nl2api_eval_batch_started_timestamp) > 3600
          and nl2api_eval_batch_status == 1  # IN_PROGRESS
        labels:
          severity: warning
        annotations:
          summary: "Batch stuck in progress"
          description: "Batch {{ $labels.batch_id }} running for >1 hour"
          action: "eval batch status <batch-id> --verbose"

      - alert: WorkerStalled
        expr: |
          rate(nl2api_eval_worker_heartbeat_total[1m]) > 0
          and rate(nl2api_eval_worker_tasks_processed_total[5m]) == 0
          and on() nl2api_eval_queue_pending_tasks > 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Worker sending heartbeats but not processing"
          description: "Worker {{ $labels.worker_id }} may be hung"

      - alert: LLMAPIErrors
        expr: rate(nl2api_llm_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM API returning errors"
          description: "{{ $value }} errors/sec from {{ $labels.provider }}"
          runbook: "Check Anthropic status page, verify API key"

  - name: distributed-eval-info
    rules:
      # ═══════════════════════════════════════════════════════════
      # INFO - Awareness, no immediate action needed
      # ═══════════════════════════════════════════════════════════

      - alert: RateLimitSaturation
        expr: |
          histogram_quantile(0.95,
            rate(nl2api_eval_rate_limit_wait_ms_bucket[5m])
          ) > 5000
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Workers waiting on rate limits"
          description: "P95 wait time is {{ $value | humanizeDuration }}"
          runbook: "Consider using Batch API or upgrading Anthropic tier"

      - alert: BatchCompleted
        expr: increase(nl2api_eval_distributed_batch_completed_total[1m]) > 0
        labels:
          severity: info
        annotations:
          summary: "Batch evaluation completed"
          description: "Batch {{ $labels.batch_id }} finished with status {{ $labels.status }}"
```

#### Notification Channels

**Grafana Alert Contact Points:**

```yaml
# config/grafana/provisioning/alerting/contactpoints.yml

apiVersion: 1
contactPoints:
  - orgId: 1
    name: default-alerts
    receivers:
      # Console logging (always enabled for local dev)
      - uid: console
        type: log
        settings: {}

      # Slack (configure SLACK_WEBHOOK_URL)
      - uid: slack
        type: slack
        settings:
          url: ${SLACK_WEBHOOK_URL}
          channel: "#eval-alerts"
          title: |
            {{ if eq .Status "firing" }}🚨{{ else }}✅{{ end }} {{ .CommonAnnotations.summary }}
          text: |
            {{ .CommonAnnotations.description }}
            {{ if .CommonAnnotations.runbook }}
            *Runbook:* {{ .CommonAnnotations.runbook }}
            {{ end }}
            {{ if .CommonAnnotations.action }}
            *Action:* `{{ .CommonAnnotations.action }}`
            {{ end }}

      # PagerDuty (for critical alerts)
      - uid: pagerduty
        type: pagerduty
        settings:
          integrationKey: ${PAGERDUTY_INTEGRATION_KEY}
          severity: '{{ if eq .CommonLabels.severity "critical" }}critical{{ else }}warning{{ end }}'
```

**Notification Policy:**

```yaml
# config/grafana/provisioning/alerting/policies.yml

apiVersion: 1
policies:
  - orgId: 1
    receiver: default-alerts
    routes:
      # Critical → PagerDuty + Slack
      - receiver: pagerduty
        matchers:
          - severity = critical
        continue: true
      - receiver: slack
        matchers:
          - severity = critical

      # Warning → Slack only
      - receiver: slack
        matchers:
          - severity = warning

      # Info → Log only (no notification)
      - receiver: console
        matchers:
          - severity = info
```

#### CLI-Based Alerting (No External Dependencies)

For local development without Grafana alerting, the CLI provides built-in monitoring:

```bash
# Watch mode with alerts (prints to console)
eval batch status <batch-id> --watch --alert

# Output:
# [14:32:01] Progress: 450/1000 (45%) | Pass: 89% | Workers: 4
# [14:32:06] Progress: 455/1000 (45.5%) | Pass: 89% | Workers: 4
# [14:32:11] ⚠️  ALERT: Worker worker-2 not responding (last heartbeat 65s ago)
# [14:32:16] Progress: 458/1000 (45.8%) | Pass: 89% | Workers: 3
# [14:32:21] ⚠️  ALERT: DLQ has 5 failed tasks
```

**CLI Alert Thresholds:**

```python
# src/evaluation/distributed/config.py

@dataclass
class AlertConfig:
    """CLI-based alert thresholds."""

    # Worker health
    worker_heartbeat_timeout_seconds: int = 60
    min_workers: int = 1

    # Queue health
    max_pending_tasks: int = 1000
    max_dlq_size: int = 10

    # Batch health
    max_batch_duration_seconds: int = 3600  # 1 hour
    max_failure_rate: float = 0.1  # 10%

    # Rate limiting
    max_rate_limit_wait_seconds: int = 30
```

#### Failure Recovery Actions

| Alert | Automatic Recovery | Manual Recovery |
|-------|-------------------|-----------------|
| WorkerDown | Docker restart policy | `docker-compose up -d eval-worker` |
| AllWorkersDown | None (needs investigation) | Check logs, restart workers |
| QueueBacklog | None | Scale workers or pause ingestion |
| DLQGrowing | None | `eval batch retry <batch-id>` after fixing root cause |
| StalledTasks | XCLAIM (automatic) | Tasks auto-reassigned to healthy workers |
| BatchStuck | None | `eval batch resume <batch-id>` or cancel |
| RedisDown | Docker restart policy | `docker-compose restart redis` |
| RateLimitSaturation | None | Switch to Batch API or wait |

#### Grafana Dashboard: Alert Overview Panel

Add to `evaluation-distributed.json`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ROW 0: ALERT STATUS (top of dashboard)                                       │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ 🔴 Critical: 0  │ 🟡 Warning: 2   │ 🟢 Info: 1      │ Last incident: 2h ago │
├─────────────────┴─────────────────┴─────────────────┴───────────────────────┤
│ Active Alerts:                                                               │
│ • [Warning] QueueBacklog - 1,234 tasks pending (started 5m ago)             │
│ • [Warning] DLQGrowing - 15 tasks in DLQ (started 1h ago)                   │
│ • [Info] RateLimitSaturation - P95 wait 6.2s                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Integration with Existing Dashboard

**Update `evaluation-dashboard.json`:**

Add new panels to existing dashboard:
1. **Distributed Mode Indicator** - Show if batch is running in distributed mode
2. **Worker Count** - Number of active workers (0 = local mode)
3. **Link to Distributed Dashboard** - Deep link when distributed mode active

### Phase-by-Phase OTEL Additions

| Phase | New Metrics | New Spans | Dashboard |
|-------|-------------|-----------|-----------|
| **1** | None | None | None |
| **2** | `queue_*` gauges | `eval.queue.*` | Queue panels (mock data) |
| **3** | `worker_*` counters/histograms | `eval.worker.*` | Worker health panels |
| **4** | `batch_started/completed` | `eval.coordinator.*` | Batch progress panels |
| **5** | None (uses Phase 3 metrics) | None | Docker Compose labels |
| **6** | `rate_limit_*` | `eval.rate_limiter.*` | Rate limit panels |
| **7** | `batch_api_*` | `eval.batch_api.*` | Batch API panels |
| **8** | Alert rules | None | Alert configuration |

### Files to Create/Modify

```
# New files
src/evaluation/distributed/metrics.py              # DistributedMetrics class
config/grafana/provisioning/dashboards/json/evaluation-distributed.json

# Modified files
config/grafana/provisioning/dashboards/json/evaluation-dashboard.json  # Add distributed mode indicator
config/prometheus/alerts/evaluation.yml            # New alert rules (Phase 8)
```

---

## Migration Path to Azure

### Phase 1: Local (Current Plan)
- Redis Streams + Docker Compose
- PostgreSQL for state

### Phase 2: Azure Container Instances
- Same code, deploy to ACI
- Azure Redis Cache instead of self-hosted
- Azure Database for PostgreSQL

### Phase 3: Azure Service Bus (Optional)
- Implement `AzureServiceBusQueue`
- Swap via `EVAL_QUEUE_BACKEND=azure_sb`
- Benefits: Better enterprise features, SLA guarantees

```python
# src/evaluation/distributed/queue/azure_sb.py (stub for now)

class AzureServiceBusQueue:
    """Azure Service Bus implementation of TaskQueue."""

    def __init__(self, connection_string: str, queue_name: str):
        # TODO: Implement when migrating to Azure
        raise NotImplementedError("Azure Service Bus not yet implemented")
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Redis single point of failure | Medium | High | Redis Sentinel/Cluster in production |
| Rate limit exceeded | High | Medium | Conservative defaults, monitoring |
| Task poison pill (always fails) | Low | Medium | Max retries → DLQ |
| Memory leak in workers | Low | Medium | Worker restart policy, monitoring |
| Queue message loss | Low | High | Redis AOF persistence, monitoring |

---

## Decisions Made

| Question | Decision | Rationale |
|----------|----------|-----------|
| Queue per batch vs shared? | **Per batch** | Better isolation, easier cleanup, simpler debugging |
| Worker autoscaling? | **Fixed initially** | Add to backlog; defer until needed |
| Batch API support? | **Yes, dual mode** | Sync for small/urgent, Batch API for large/cost-sensitive |

---

## Success Criteria

| Metric | Sync API Target | Batch API Target |
|--------|-----------------|------------------|
| Throughput | 80k tests/hour (Tier 3 limit) | 100k+ tests/hour |
| Fault tolerance | 100% completion despite 1 worker crash | Same |
| Recovery time | < 60s to detect and reassign stalled tasks | N/A (coordinator handles) |
| DLQ rate | < 1% of tasks | < 0.1% (no rate limit issues) |
| P95 task latency | < 30s (excluding rate limit waits) | Hours (async by design) |
| Cost efficiency | Baseline | 50% cheaper |

---

## Backlog Items (Deferred)

| Item | Trigger to Implement |
|------|---------------------|
| **Worker autoscaling** | When manual scaling becomes painful |
| **Azure Service Bus migration** | When enterprise compliance requires it |
| **Multi-region deployment** | When latency to LLM providers matters |
| **Kubernetes Helm chart** | When deploying to AKS |

---

## Appendix: Existing Code References

| Component | Location | Notes |
|-----------|----------|-------|
| Current BatchRunner | `src/evaluation/batch/runner.py:80` | Replace with Coordinator |
| WorkerTask model | `src/contracts/worker.py:20` | Already exists |
| BatchJob model | `src/contracts/worker.py:60` | Already exists |
| Response generators | `src/evaluation/batch/response_generators.py` | Reuse in workers |
| Evaluator pipeline | `src/evaluation/core/evaluators.py` | Reuse in workers |
| Scorecard repo | `src/common/storage/postgres/scorecard_repo.py` | Workers write here |
| Batch repo | `src/common/storage/postgres/batch_repo.py` | Coordinator uses |
| Redis cache | `src/common/cache/redis_cache.py` | Extend for queue |
