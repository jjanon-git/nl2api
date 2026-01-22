# NL2API - Claude Code Guidelines

## ALWAYS (check every time)

```
BEFORE writing code:
□ Read existing code first - never modify code you haven't read

AFTER writing code:
□ Run tests: .venv/bin/python -m pytest tests/unit/ -v --tb=short -x
□ New code = new tests (no exceptions)

BEFORE committing:
□ Tests pass (unit + integration)
□ No secrets in diff
□ Lint passes: ruff check .

AFTER debugging (>10 min OR config/integration issue):
□ What was the ROOT CAUSE? (not the symptom)
□ Why wasn't it caught earlier? (missing test? missing docs? design flaw?)
□ What SPECIFIC fix prevents recurrence? (add test, add docs, fix design)
□ Make that fix NOW, before moving on
```

**That's it.** Everything below is reference material - consult when relevant.

---

## Where to Document Issues

| Issue Type | Location |
|------------|----------|
| Process/standards gap | This file (ALWAYS section or relevant section below) |
| Config/integration gotcha | `docs/troubleshooting.md` |
| Code-specific gotcha | Comment in the affected file |
| New test case needed | `tests/` with regression test |

---

## Quick Commands

```bash
# Tests
.venv/bin/python -m pytest tests/unit/ -v --tb=short -x      # Unit (required)
.venv/bin/python -m pytest tests/integration/ -v --tb=short  # Integration
.venv/bin/python -m pytest tests/accuracy/ -m tier1 -v       # Accuracy (costs $)

# Lint
ruff check .

# Start services
docker compose up -d

# Batch evaluation
.venv/bin/python -m src.evaluation.cli.main batch run --limit 10
```

---

## Project Overview

**NL2API** translates natural language queries into LSEG financial API calls.

**Architecture:**
```
User Query → Orchestrator → [Entity Resolution] → [Domain Agent] → Tool Calls
                               ↓                        ↓
                          Company→RIC              Datastream/Estimates/
                                                   Fundamentals/Officers/
                                                   Screening
```

**Key files:**
- `CONTRACTS.py` - Data models (read first)
- `src/nl2api/orchestrator.py` - Main entry point
- `src/nl2api/agents/*.py` - Domain agents
- `tests/fixtures/lseg/generated/` - ~19k test fixtures

---

## Testing Standards

### Test Requirements by Change Type

| You Create | You Also Create |
|------------|-----------------|
| New class | Test class with tests for public methods |
| New function | Tests for happy path + error cases |
| Bug fix | Regression test that reproduces the bug |
| Config change | Tests for default value + behavior |

### Test Types

| Scenario | Unit Test | Integration Test | Manual Verify |
|----------|-----------|------------------|---------------|
| Pure function logic | ✅ | | |
| Class with dependencies | ✅ (mock) | | |
| Database operations | | ✅ | |
| Multi-component flows | | ✅ | |
| External APIs (GLEIF, etc.) | ✅ (mock) | | ✅ |
| Servers/CLIs | | | ✅ |

### Accuracy Tests (cost money)

| Tier | Samples | When |
|------|---------|------|
| tier1 | ~50 | PR checks |
| tier2 | ~200 | Daily CI |
| tier3 | All | Weekly |

---

## Evaluation Pipeline

**Capabilities need evaluation data.** No capability is complete without:
1. Test cases with expected outputs
2. Baseline metrics recorded
3. Regression tracking over time

### Fixture Requirements

```python
TestCase(
    id="unique-id",
    nl_query="What is Apple's PE ratio?",
    expected_tool_calls=(ToolCall(...),),
    category="lookups",
    subcategory="single_field",
)
```

### Batch Evaluation Modes

| Mode | When to Use |
|------|-------------|
| `resolver` (default) | Real accuracy tracking |
| `orchestrator` | Full pipeline (costs API credits) |
| `simulated` | Infrastructure testing only |

---

## Security Checklist

- [ ] No secrets in code (use env vars)
- [ ] No secrets in logs
- [ ] Parameterized SQL queries only (`$1, $2` syntax)
- [ ] User input validated before use

```python
# CORRECT
await conn.fetch("SELECT * FROM users WHERE id = $1", user_id)

# WRONG - SQL injection
await conn.fetch(f"SELECT * FROM users WHERE id = {user_id}")
```

---

## Error Handling

1. **Catch specific exceptions** - never bare `except:`
2. **Preserve chains** - use `raise X from e`
3. **Don't log and raise** - do one or the other
4. **Fail fast at boundaries** - validate early

```python
class NL2APIError(Exception): pass
class EntityResolutionError(NL2APIError): pass
class AgentProcessingError(NL2APIError): pass
class StorageError(NL2APIError): pass
```

---

## Telemetry (OTEL)

Add spans for: LLM calls, DB operations, external APIs, cache operations.

**Required attributes:**

| Operation | Attributes |
|-----------|------------|
| LLM calls | `llm.provider`, `llm.model`, `llm.tokens` |
| Database | `db.operation`, `db.table` |
| Agent | `agent.name`, `agent.confidence` |
| MCP Server | `server.name`, `client.session_id` |

---

## MCP Server Development

**Always use the official MCP SDK** (`mcp` package):

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server(name="my-server", version="1.0.0", lifespan=my_lifespan)

@server.list_tools()
async def list_tools() -> list[Tool]: ...

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]: ...
```

**Never** roll custom protocol handling.

---

## LLM Prompt Review

**Before running batch LLM calls, show user:**
1. Full system prompt
2. Example user prompt
3. Expected output format
4. Cost estimate (tokens × rate)

---

## Directory Structure

```
nl2api/
├── CONTRACTS.py              # Data models
├── src/
│   ├── nl2api/               # Core system
│   │   ├── orchestrator.py   # Entry point
│   │   ├── agents/           # Domain agents
│   │   ├── resolution/       # Entity resolution
│   │   └── llm/              # LLM providers
│   ├── common/               # Shared utilities
│   └── evaluation/           # Eval pipeline
├── tests/
│   ├── unit/                 # Mocked dependencies
│   ├── integration/          # Real DB
│   ├── accuracy/             # Real LLM
│   └── fixtures/             # ~19k test cases
└── docker-compose.yml        # PostgreSQL, Redis, OTEL
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NL2API_ANTHROPIC_API_KEY` | Claude API key |
| `NL2API_LLM_PROVIDER` | "anthropic" or "openai" |
| `NL2API_TELEMETRY_ENABLED` | Enable OTEL |
| `EVAL_BACKEND` | "postgres", "memory", or "azure" |

---

## Known Gotchas

| Issue | Solution |
|-------|----------|
| FastAPI returns 422 for valid JSON | Don't use `from __future__ import annotations` in FastAPI files |
| Grafana shows no data | Check metric names have `nl2api_` prefix and `_total` suffix for counters |
| Batch eval fails silently | Run `python scripts/load_fixtures_to_db.py --all` first |
| Orchestrator fails with "API key not set" | Pass router explicitly to avoid hidden NL2APIConfig dependency |

---

## Backlog

All planned work tracked in [BACKLOG.md](BACKLOG.md). Update it:
- Before starting significant work
- While working (status updates)
- When completing work (move to Completed)
