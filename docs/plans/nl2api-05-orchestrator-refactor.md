# Orchestrator Refactoring Plan

**Status:** Not Started
**Priority:** Medium (but High Impact)
**Author:** Mostly Claude, with some minor assistance from Sid
**Created:** 2026-01-24

---

## Problem Statement

`NL2APIOrchestrator` is an 803-line god class with several architectural problems that make it difficult to test, extend, and maintain.

### Current Problems

| Problem | Impact | Location |
|---------|--------|----------|
| Hidden lazy initialization | `_router_initialized` flag creates implicit state machine | Lines 150-180 |
| 11-parameter `__init__` | Complex conditionals, hard to construct correctly | Lines 80-140 |
| Error/clarification conflation | Failures and clarification requests use same code paths | Lines 400-450 |
| `_RAGContextAdapter` anti-pattern | Adapter wrapping adapter, unnecessary indirection | Lines 60-75 |
| Mixed responsibilities | Routing, resolution, context, agent dispatch all in one | Throughout |

### Current Constructor Signature

```python
def __init__(
    self,
    llm: LLMProvider,
    router: QueryRouter | None = None,
    entity_resolver: EntityResolver | None = None,
    rag_retriever: RAGRetriever | None = None,
    context_retriever: ContextRetriever | None = None,
    agents: list[DomainAgent] | None = None,
    config: NL2APIConfig | None = None,
    routing_llm: LLMProvider | None = None,
    cache_client: RedisClient | None = None,
    postgres_pool: PostgresPool | None = None,
    request_metrics: RequestMetrics | None = None,
)
```

---

## Goals

1. **Simpler construction** - Builder pattern with sensible defaults
2. **Explicit initialization** - No hidden state machines
3. **Separated concerns** - Distinct classes for routing, resolution, context
4. **Testable** - Easy to mock individual components
5. **Clear error handling** - Failures vs clarifications are distinct

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  OrchestratorBuilder                     │
│  .with_llm(llm)                                         │
│  .with_router(router)                                   │
│  .with_entity_resolver(resolver)                        │
│  .with_agents([...])                                    │
│  .build() → NL2APIOrchestrator                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  NL2APIOrchestrator                      │
│  - Thin coordinator, delegates to components            │
│  - process(query) → NL2APIResponse                      │
└─────────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ QueryRouter │ │ EntityFlow  │ │ ContextFlow │
│             │ │             │ │             │
│ classify()  │ │ resolve()   │ │ retrieve()  │
└─────────────┘ └─────────────┘ └─────────────┘
```

---

## Phases

### Phase 1: Extract Builder Pattern (2 days)

Create `OrchestratorBuilder` to handle construction complexity.

**Tasks:**
- [ ] Create `src/nl2api/orchestrator_builder.py`
- [ ] Move default creation logic to builder
- [ ] Builder validates configuration before building
- [ ] Orchestrator constructor becomes simple assignment

**New Usage:**
```python
# Before
orchestrator = NL2APIOrchestrator(
    llm=llm,
    router=None,  # Will be created lazily
    entity_resolver=resolver,
    rag_retriever=rag,
    context_retriever=None,  # Will wrap rag_retriever
    agents=agents,
    config=config,
    routing_llm=None,  # Will be created from config
    cache_client=redis,
    postgres_pool=pool,
    request_metrics=metrics,
)

# After
orchestrator = (
    OrchestratorBuilder()
    .with_llm(llm)
    .with_entity_resolver(resolver)
    .with_rag(rag)
    .with_agents(agents)
    .with_config(config)
    .with_cache(redis)
    .with_metrics(metrics)
    .build()
)
```

**Builder Implementation:**
```python
class OrchestratorBuilder:
    def __init__(self):
        self._llm: LLMProvider | None = None
        self._router: QueryRouter | None = None
        # ... other fields

    def with_llm(self, llm: LLMProvider) -> Self:
        self._llm = llm
        return self

    def build(self) -> NL2APIOrchestrator:
        # Validate required components
        if self._llm is None:
            raise ValueError("LLM provider is required")

        # Create defaults for optional components
        router = self._router or self._create_default_router()

        # Construct with validated, complete config
        return NL2APIOrchestrator(
            llm=self._llm,
            router=router,
            # ... fully initialized components
        )
```

### Phase 2: Remove Lazy Initialization (1 day)

Eliminate `_router_initialized` and `_ensure_router_initialized()`.

**Tasks:**
- [ ] Move router creation to builder
- [ ] Remove `_router_initialized` flag
- [ ] Remove `_ensure_router_initialized()` method
- [ ] Router is always ready when orchestrator is constructed

**Current Anti-Pattern:**
```python
async def _classify_query(self, query: str) -> str:
    self._ensure_router_initialized()  # Hidden state change
    return await self._router.classify(query)
```

**After:**
```python
async def _classify_query(self, query: str) -> str:
    return await self._router.classify(query)  # Always initialized
```

### Phase 3: Separate Error Handling from Clarifications (2 days)

Currently, both errors and clarification requests flow through similar code paths, making it hard to distinguish between "something went wrong" and "need more info from user".

**Tasks:**
- [ ] Create `ClarificationRequest` dataclass (distinct from errors)
- [ ] Create `ProcessingError` for actual failures
- [ ] Update `process()` to return `NL2APIResponse | ClarificationRequest`
- [ ] Or use discriminated union: `ProcessResult = Success | Clarification | Failure`

**New Types:**
```python
@dataclass(frozen=True)
class ClarificationRequest:
    """User input needed to proceed."""
    question: str
    options: list[str] | None = None
    context: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ProcessingFailure:
    """Unrecoverable error during processing."""
    error_code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

ProcessResult = NL2APIResponse | ClarificationRequest | ProcessingFailure
```

### Phase 4: Remove RAGContextAdapter (1 day)

`_RAGContextAdapter` wraps `RAGRetriever` to implement `ContextRetriever`. This is unnecessary indirection.

**Tasks:**
- [ ] Have `HybridRAGRetriever` implement `ContextRetriever` directly
- [ ] Remove `_RAGContextAdapter` class
- [ ] Update builder to use RAG retriever directly

**Current:**
```python
class _RAGContextAdapter:
    def __init__(self, rag: RAGRetriever):
        self._rag = rag

    async def retrieve(self, query: str) -> list[str]:
        results = await self._rag.retrieve(query)
        return [r.content for r in results]
```

**After:**
```python
class HybridRAGRetriever(RAGRetriever, ContextRetriever):
    async def retrieve_context(self, query: str) -> list[str]:
        results = await self.retrieve(query)
        return [r.content for r in results]
```

### Phase 5: Extract Domain Flows (3 days)

Split orchestrator into focused components for each domain concern.

**Tasks:**
- [ ] Create `EntityResolutionFlow` - handles entity extraction and resolution
- [ ] Create `ContextRetrievalFlow` - handles RAG/MCP context
- [ ] Create `AgentDispatchFlow` - handles routing and agent execution
- [ ] Orchestrator becomes thin coordinator of flows

**Flow Pattern:**
```python
class EntityResolutionFlow:
    def __init__(self, resolver: EntityResolver):
        self._resolver = resolver

    async def resolve(self, query: str) -> ResolvedEntities:
        # Extract entities from query
        # Resolve to RICs
        # Return structured result

class NL2APIOrchestrator:
    def __init__(
        self,
        entity_flow: EntityResolutionFlow,
        context_flow: ContextRetrievalFlow,
        dispatch_flow: AgentDispatchFlow,
    ):
        self._entity = entity_flow
        self._context = context_flow
        self._dispatch = dispatch_flow

    async def process(self, query: str) -> ProcessResult:
        entities = await self._entity.resolve(query)
        context = await self._context.retrieve(query, entities)
        return await self._dispatch.execute(query, entities, context)
```

---

## Migration Strategy

1. **Phase 1-2**: Non-breaking - Builder is additive, old constructor still works
2. **Phase 3**: May require caller updates for new return types
3. **Phase 4-5**: Internal refactoring, API unchanged

**Deprecation Path:**
```python
# Old constructor still works but emits warning
def __init__(self, llm, router=None, ...):
    warnings.warn(
        "Direct construction deprecated. Use OrchestratorBuilder.",
        DeprecationWarning
    )
    # Delegate to builder internally
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Lines in orchestrator.py | 803 | <300 |
| Constructor parameters | 11 | 3-4 (flows) |
| Lazy initialization flags | 1 | 0 |
| Test setup complexity | High (many mocks) | Low (mock flows) |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Deprecation warnings, backward compat shim |
| Increased file count | Clear naming, documented structure |
| Over-abstraction | Keep flows focused, no premature generalization |

---

## Dependencies

- None - can start immediately
- Should complete before MCP dual-mode implementation (cleaner base)

---

## Effort Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: Builder pattern | 2 days |
| Phase 2: Remove lazy init | 1 day |
| Phase 3: Error/clarification split | 2 days |
| Phase 4: Remove adapter | 1 day |
| Phase 5: Extract flows | 3 days |
| **Total** | **9 days** |
