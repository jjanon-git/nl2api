# EvalPlatform

Distributed evaluation framework for testing LLM tool-calling at scale (~400k test cases).

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Start PostgreSQL (required for storage)
docker compose up -d

# Run tests
pytest tests/unit/ -v

# Run single test case evaluation
python -m src.cli.main run tests/fixtures/search_products.json

# Run batch evaluation
python -m src.cli.main batch run --limit 10

# View batch results
python -m src.cli.main batch list
```

## Features

### Evaluation Pipeline
- **Stage 1 (Syntax)**: JSON structure validation - hard stop on failure
- **Stage 2 (Logic)**: AST-based tool call comparison with order-independence and type coercion
- **Stage 3 (Execution)**: Live API verification (not yet implemented)
- **Stage 4 (Semantics)**: LLM-as-Judge NL comparison (not yet implemented)

### Batch Processing
- Concurrent evaluation with configurable parallelism
- Progress tracking with Rich
- Batch status and results commands
- Optional OpenTelemetry metrics

### Storage
- PostgreSQL backend with pgvector
- In-memory repositories for unit tests
- Protocol-based abstraction for future Azure integration

## Documentation

- **ARCHITECTURE.md** - Full system design and API specifications
- **CLAUDE.md** - Quick reference for AI assistants
- **\*_REFERENCE.md** - LSEG API reference docs for test case authoring
