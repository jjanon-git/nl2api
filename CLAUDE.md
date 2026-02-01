# CLAUDE.md - Project Context for Claude Code

## Project Overview

**Evalkit** is an evaluation framework for LLM-powered systems, with applications for NL2API (natural language to API translation) and RAG (retrieval-augmented generation). Includes batch evaluation, accuracy testing, and observability infrastructure (~19k test cases).

## Quick Commands

```bash
# Start infrastructure
docker compose up -d

# Run tests
pytest tests/unit/ -v --tb=short -x              # All unit tests
./scripts/ci-test-changed.sh                      # Only changed modules
pytest tests/integration/ -v --tb=short -x        # Integration tests

# Lint
ruff check .

# Batch evaluation
python scripts/load-nl2api-fixtures.py --all      # Load fixtures (required first)
python -m src.evalkit.cli.main batch run --pack nl2api --limit 10
python -m src.evalkit.cli.main batch list

# Accuracy tests (costs API credits)
pytest tests/accuracy/ -m tier1 -v                # Quick (~50 samples)
```

---

## Critical Rules

These are non-negotiable. Follow them always.

### 1. Run Tests After Code Changes

```bash
./scripts/ci-test-changed.sh    # Fast feedback
pytest tests/unit/ -v -x        # Before committing
```

See [docs/testing.md](docs/testing.md) for module-to-test mapping and coverage requirements.

### 2. Verify Batch Evaluation Results

Never declare batch evaluation "working" without checking actual results:
- Wait for at least 5 scorecards to complete
- Verify `generated_nl_response` is not empty
- Check you're looking at the current batch_id

See [docs/evaluation.md](docs/evaluation.md) for verification queries.

### 3. Review LLM Prompts Before Running

Before running any script that calls LLM APIs:
1. Show the full system prompt and example user prompt
2. Calculate and show estimated cost
3. Get explicit user confirmation

### 4. Track Work in BACKLOG.md

All planned work must be tracked in [BACKLOG.md](BACKLOG.md):
- Check if work exists before starting
- Update status as work progresses
- Move completed items with date

### 5. Every Capability Needs Evaluation

No capability is complete without:
- Definition of "correct" output
- Evaluation fixtures with expected results
- Baseline metrics recorded
- Tracking over time

---

## Self-Improvement Loop (MANDATORY)

**Before marking any significant task complete, you MUST complete this reflection checklist:**

### Post-Task Reflection Checklist

After fixing a bug, adding a feature, or debugging an issue, ask yourself:

- [ ] **What was the root cause?** (Not the symptom - the actual cause)
- [ ] **Why wasn't this caught earlier?** (Missing test? Missing docs? Missing validation?)
- [ ] **What instruction would have prevented this?** (Be specific)
- [ ] **Have I added that instruction to CLAUDE.md?** (If no, do it now)

### When to Trigger This Checklist

- After any debugging session longer than 10 minutes
- After discovering a config mismatch or integration issue
- After finding undocumented requirements
- After manual testing reveals something unit tests missed
- Before committing a fix

### Examples

| Situation | Root Cause | Action |
|-----------|------------|--------|
| Grafana shows no data | Datasource UID mismatch | Add to `docs/troubleshooting.md` |
| Metrics not appearing | Missing `_total` suffix | Add to `docs/troubleshooting.md` |
| Batch eval fails silently | No fixtures in DB | Add prerequisite to CLAUDE.md (process) |
| FastAPI returns 422 for valid JSON | `from __future__ import annotations` breaks introspection | Add comment in affected file (code gotcha) |
| Batch runs but results empty | Field not set in Scorecard creation | Trace data flow end-to-end before declaring done |
| Fix applied but batch still broken | Running process started before fix | Kill old process, restart, verify NEW batch_id |

### Where to Document

| Issue Type | Where to Document | Example |
|------------|-------------------|---------|
| Process/standards gap | CLAUDE.md | "Always run manual tests before committing servers" |
| Code pattern/gotcha | Comment in affected file | "Don't use X with Y because..." |
| Config/integration issue | [docs/troubleshooting.md](docs/troubleshooting.md) | "If Grafana shows no data, check UID" |
| Regression risk | Add a test | Unit test that would have caught it |

**CLAUDE.md is for process, not implementation trivia.** If the instruction is specific to one file or framework, it belongs in code comments or docs, not here.

### The Standard

**Every debugging session should result in one of:**
1. A process instruction in CLAUDE.md, OR
2. A code comment/docstring in the affected file, OR
3. An entry in `docs/troubleshooting.md`, OR
4. A test that catches the regression

**If you finish a task without updating one of these - you haven't finished.**

---

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for comprehensive documentation.

### Codebase Structure

```
src/
├── evalkit/        # Evaluation framework (publishable)
│   ├── contracts/  # Data models
│   ├── common/     # Infrastructure (storage, telemetry, cache)
│   ├── core/       # Evaluators (AST, temporal, semantics)
│   ├── batch/      # Batch runner and metrics
│   ├── cli/        # CLI commands
│   └── packs/      # Pack registry
├── nl2api/         # NL2API application
│   ├── agents/     # Domain agents
│   ├── resolution/ # Entity resolution
│   └── evaluation/ # NL2API pack
├── rag/            # RAG application
│   ├── retriever/  # Hybrid retriever
│   ├── ingestion/  # SEC EDGAR ingestion
│   └── evaluation/ # RAG pack
└── mcp_servers/    # MCP servers
```

### Key Patterns

1. **Protocol-Based Design** - `DomainAgent`, `Stage` protocols for extensibility
2. **Frozen Pydantic Models** - All data models are immutable
3. **Async-First** - All repository operations, agents, evaluators are async
4. **Factory Pattern** - Use `create_entity_resolver(config)` not direct instantiation

---

## Detailed Documentation

| Topic | Document |
|-------|----------|
| Testing standards | [docs/testing.md](docs/testing.md) |
| Evaluation pipeline | [docs/evaluation.md](docs/evaluation.md) |
| Telemetry/OTEL | [docs/telemetry.md](docs/telemetry.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Accuracy testing | [docs/accuracy-testing.md](docs/accuracy-testing.md) |
| Evaluation data | [docs/evaluation-data.md](docs/evaluation-data.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `EVAL_BACKEND` | "postgres" \| "memory" \| "azure" |
| `NL2API_LLM_PROVIDER` | "anthropic" \| "openai" |
| `NL2API_ANTHROPIC_API_KEY` | Claude API key |
| `EVALKIT_TELEMETRY_ENABLED` | "true" to enable OTEL |

---

## CI/CD

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR to main | Lint + unit tests |
| `accuracy.yml` | PR (tier1), daily (tier2), weekly (tier3) | Accuracy tests |

---

## Security Standards

See global `~/.claude/CLAUDE.md` for security checklist. Key points:
- Use parameterized queries only (`$1, $2` syntax)
- No secrets in code or logs
- Validate all external input

---

## Error Handling

See global `~/.claude/rules/testing.md` for patterns. Key points:
- Catch specific exceptions, never bare `except:`
- Preserve exception chains with `from`
- Don't log and raise at the same level

---

## MCP Server Development

Use the official MCP SDK:

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server(name="my-server", version="1.0.0", lifespan=my_lifespan)

@server.list_tools()
async def list_tools() -> list[Tool]: ...
```

Never roll custom protocol handling.

---

## Known Gaps

| Gap | Workaround |
|-----|------------|
| Migration guidance | Create migrations in `migrations/` with naming `NNN_description.sql` |
| Backwards compatibility | Avoid breaking changes; discuss with team first |
| Performance testing | Manual testing with `ab` or `wrk` |

---

## Important Files

### Evalkit Framework
- `src/evalkit/contracts/` - Data models
- `src/evalkit/batch/runner.py` - Batch evaluation
- `src/evalkit/packs/` - Pack registry

### Applications
- `src/nl2api/orchestrator.py` - NL2API entry point
- `src/nl2api/evaluation/pack.py` - NL2API 4-stage pack
- `src/rag/evaluation/pack.py` - RAG 8-stage pack

### Testing
- `tests/unit/nl2api/fixture_loader.py` - Fixture loading
- `tests/accuracy/core/evaluator.py` - Accuracy testing

**Total Tests: 1964 (unit) + accuracy tests**
