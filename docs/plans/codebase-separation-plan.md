# Codebase Separation Plan

## Executive Summary

Separate the monolithic `nl2api` codebase into three distinct components:

1. **`evalkit`** (working name) - The core evaluation framework (publishable)
2. **`nl2api`** - Natural language to API translation (example application)
3. **`rag`** - RAG implementation for SEC filings (example application)

The evaluation framework becomes the top-level project; NL2API and RAG become example implementations that demonstrate how to build evaluation packs.

**Target: External publication** - The evalkit framework will be externalized as publishable packages.

---

## Migration Strategy: Two-Stage Approach

### Stage 1: Namespace Packages (This Plan)
- Consolidate scattered code into clear namespaces
- Validate all imports, tests pass
- Establish clean boundaries between framework and applications
- **Outcome:** Working codebase with logical separation

### Stage 2: Monorepo with Publishable Packages (Future)
- Convert namespaces into separate packages with their own `pyproject.toml`
- Enable independent versioning and publishing
- Structure for PyPI distribution
- **Prerequisite:** Stage 1 complete and stable

This staged approach reduces risk by validating the logical separation before restructuring for publication.

---

## Current State Analysis

### Directory Structure (as-is)
```
src/
├── common/              # Shared infrastructure (storage, telemetry, cache, resilience)
├── contracts/           # Shared data models (TestCase, Scorecard, ToolCall, etc.)
├── evaluation/          # Evaluation framework
│   ├── batch/          # Batch runner, metrics, pricing
│   ├── cli/            # CLI commands
│   ├── continuous/     # Continuous monitoring
│   ├── core/           # AST comparator, temporal, evaluators
│   ├── distributed/    # Distributed workers
│   └── packs/          # Domain-specific evaluation logic
│       ├── nl2api.py   # NL2API 4-stage pack
│       └── rag/        # RAG 8-stage pack
├── nl2api/              # NL2API system
│   ├── agents/         # Domain agents
│   ├── conversation/   # Multi-turn
│   ├── ingestion/      # SEC filing ingestion (MISPLACED - should be with RAG)
│   ├── llm/            # LLM providers
│   ├── observability/  # Metrics
│   ├── rag/            # RAG retriever (MISPLACED - should be with RAG)
│   ├── resolution/     # Entity resolution
│   └── routing/        # Query routing
├── rag_ui/              # Streamlit RAG UI
└── mcp_servers/         # MCP servers
```

### Key Dependencies (Clean Layering)
```
CONTRACTS (foundation - data models)
    ↓
src/common (infrastructure - storage, telemetry, cache)
    ↓
src/evaluation/core (generic evaluators - AST comparator, temporal)
    ↓
src/evaluation/packs/* (domain-specific evaluation logic)
    ↓
src/evaluation/batch + cli (orchestration layer)
```

### Problems to Solve
1. **RAG code scattered**: `nl2api/rag/`, `nl2api/ingestion/`, `rag_ui/`, `evaluation/packs/rag/`
2. **Misleading project name**: "nl2api" doesn't reflect evaluation framework focus
3. **No clear boundary**: Framework vs application code intermingled
4. **SEC ingestion misplaced**: Lives in nl2api but only used by RAG

---

## Recommended Approach: Namespace Packages

Keep a single repo but reorganize into clear namespaces:

### Target Structure
```
src/
├── evalkit/                 # Core evaluation framework (publishable)
│   ├── __init__.py
│   ├── contracts/          # Data models (from src/contracts/)
│   ├── common/             # Infrastructure (from src/common/)
│   ├── core/               # Evaluators (from src/evaluation/core/)
│   ├── batch/              # Batch runner (from src/evaluation/batch/)
│   ├── cli/                # CLI commands (from src/evaluation/cli/)
│   ├── distributed/        # Workers (from src/evaluation/distributed/)
│   ├── continuous/         # Monitoring (from src/evaluation/continuous/)
│   └── packs/              # Pack interface/protocol ONLY
│       ├── __init__.py
│       └── base.py         # Pack protocol, Stage protocol
│
├── nl2api/                  # NL2API application (example)
│   ├── agents/
│   ├── conversation/
│   ├── llm/
│   ├── observability/
│   ├── resolution/         # Entity resolution (shared - extraction deferred to Stage 2)
│   ├── routing/
│   └── evaluation/         # NL2API pack implementation
│       └── pack.py         # NL2APIPack (from evaluation/packs/nl2api.py)
│
├── rag/                     # RAG application (example, consolidated)
│   ├── retriever/          # From nl2api/rag/
│   ├── ingestion/          # From nl2api/ingestion/sec_filings/
│   ├── ui/                 # From rag_ui/
│   ├── embedders/          # From nl2api/rag/embedders.py
│   └── evaluation/         # RAG pack implementation
│       └── pack.py         # RAGPack (from evaluation/packs/rag/)
│
└── mcp_servers/             # Keep in place for Stage 1 (entity resolution shared)
    └── entity_resolution/
```

---

## Migration Phases

### Phase 1: Consolidate RAG Code (Low Risk)

Move all RAG-related code under `src/rag/`:

```bash
# Create structure
mkdir -p src/rag/{retriever,ingestion,ui,evaluation}

# Move files
git mv src/nl2api/rag/*.py src/rag/retriever/
git mv src/nl2api/ingestion/sec_filings/ src/rag/ingestion/
git mv src/rag_ui/* src/rag/ui/
```

**Files to update imports:** ~15-20 files
**Run tests, commit checkpoint**

### Phase 2: Create Evalkit Namespace (Medium Risk)

Move evaluation framework under `evalkit`:

```bash
# Create structure
mkdir -p src/evalkit/{contracts,common,core,batch,cli,distributed,continuous,packs}

# Move core modules
git mv src/contracts/* src/evalkit/contracts/
git mv src/common/* src/evalkit/common/
git mv src/evaluation/core/* src/evalkit/core/
git mv src/evaluation/batch/* src/evalkit/batch/
git mv src/evaluation/cli/* src/evalkit/cli/
git mv src/evaluation/distributed/* src/evalkit/distributed/
git mv src/evaluation/continuous/* src/evalkit/continuous/
```

**Compatibility shims (KEEP - job running):**
```python
# src/contracts/__init__.py
# TODO: Remove after validation period
from evalkit.contracts import *  # Re-export for backwards compat

# src/common/__init__.py
# TODO: Remove after validation period
from evalkit.common import *
```

**Run tests, commit checkpoint**

### Phase 3: Reorganize Packs (Interface + Implementations)

```bash
# Create pack interface in evalkit
mkdir -p src/evalkit/packs
# Move protocol/base classes to evalkit/packs/base.py

# Move NL2API pack to nl2api application
mkdir -p src/nl2api/evaluation
git mv src/evaluation/packs/nl2api.py src/nl2api/evaluation/pack.py

# Move RAG pack to rag application
mkdir -p src/rag/evaluation
git mv src/evaluation/packs/rag/* src/rag/evaluation/
```

**Run tests, commit checkpoint**

### Phase 4: Update CLI Entry Points

```toml
# pyproject.toml
[project.scripts]
evalkit = "evalkit.cli.main:app"
# entity-mcp stays unchanged for now (entity resolution extraction deferred to Stage 2)
```

**Note:** MCP servers stay in `src/mcp_servers/` for Stage 1. Entity resolution is used by both nl2api and rag, so extracting it properly is deferred to Stage 2.

**Run tests, commit checkpoint**

### Phase 5: Documentation & Cleanup

- Update CLAUDE.md
- Update docs/architecture.md
- Remove empty directories
- Add TODO comments for compatibility shim removal

---

## Files Affected (Estimated)

| Category | Count | Risk |
|----------|-------|------|
| RAG-related imports | ~20 files | Low |
| Evaluation framework imports | ~80 files | Medium |
| Test files | ~50 files | Low |
| Config/scripts | ~10 files | Low |
| **Total** | ~160 files | - |

---

## Verification Plan

After each phase:
1. Run unit tests: `pytest tests/unit/ -x -v --tb=short`
2. Run integration tests: `pytest tests/integration/ -x -v --tb=short`
3. Run linting: `ruff check .`
4. Verify OTEL stack: Ensure telemetry imports resolve, run `docker compose up -d` and verify metrics flow to Prometheus/Grafana
5. Fix any failures before proceeding

End-to-end verification (after all phases):
1. Test CLI: `evalkit batch list`
2. Test RAG UI: `python -m src.rag.ui.app`
3. Verify OTEL end-to-end: Run batch eval, check traces in Jaeger, metrics in Grafana

---

## Risk Mitigation

1. **Git mv for moves**: Preserve file history with `git mv`
2. **Scripted import updates**: Predictable patterns (`from src.X` → `from evalkit.X`)
3. **Incremental commits**: Commit after each successful phase
4. **Test after each move**: Run full test suite before proceeding
5. **Compatibility shims**: Temporary re-exports at old paths

## Execution Approach

Claude will execute each phase:
1. Move files with `git mv`
2. Update imports programmatically (Edit tool)
3. Run tests to validate
4. Verify OTEL stack working
5. Commit checkpoint
6. **Checkpoint with user**: Summarize changes, tests run, reflect on what else needed
7. Proceed to next phase only after user approval

---

## Decisions (Resolved)

1. **Naming**: `evalkit` ✓
2. **Pack location**: Interface in `evalkit/packs/base.py`, implementations with applications:
   - `src/nl2api/evaluation/pack.py` - NL2API pack
   - `src/rag/evaluation/pack.py` - RAG pack
3. **Compatibility shims**: Keep temporarily (job running). TODO: Remove after validation period.
4. **MCP servers & entity resolution**: Keep in place for Stage 1 (entity resolution is shared by nl2api and rag - proper extraction deferred to Stage 2)

## Additional Requirements

- Include observability (o11y) plumbing in migrations
- Run full test suite after each phase
- Fix any test failures before proceeding

## Deferred to Stage 2

- **Entity resolution extraction**: Currently in `nl2api/resolution/` but used by both nl2api and rag. Need to decide:
  - Move to `evalkit/common/resolution/` (shared infrastructure)?
  - Create standalone `resolution/` module?
  - Keep as-is with cross-app imports?
- **MCP server placement**: Depends on entity resolution decision
- Final PyPI package names and structure
- Version pinning strategy between packages
- CI/CD for multi-package publishing
- Compatibility shim removal (after validation period)
