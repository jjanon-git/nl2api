# Contributing to NL2API

Thank you for your interest in contributing to NL2API! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (Python version, OS, etc.)
- Relevant logs or error messages

### Suggesting Features

Feature requests are welcome. Please provide:

- A clear description of the feature
- The problem it solves or use case it enables
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests and linting (see below)
5. Commit with a clear message
6. Push to your fork
7. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for PostgreSQL)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nl2api.git
cd nl2api

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Start PostgreSQL
docker compose up -d

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all unit tests
.venv/bin/python -m pytest tests/unit/ -v

# Run with coverage
.venv/bin/python -m pytest tests/unit/ -v --cov=src --cov-report=term-missing

# Run specific test file
.venv/bin/python -m pytest tests/unit/nl2api/test_orchestrator.py -v
```

### Code Quality

We use the following tools to maintain code quality:

```bash
# Format code
ruff format .

# Lint code
ruff check . --fix

# Type checking
mypy src/

# All checks (via pre-commit)
pre-commit run --all-files
```

## Coding Standards

### Style Guide

- Follow PEP 8 (enforced by ruff)
- Use type hints for all public functions
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Documentation

- Add docstrings to all public modules, classes, and functions
- Use Google-style docstrings
- Update README.md if adding new features

### Testing

- Write tests for all new functionality
- Maintain or improve test coverage
- Tests should be independent and repeatable
- Use descriptive test names

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Reference issues when applicable (`Fixes #123`)

Example:
```
Add fuzzy matching to entity resolver

- Implement rapidfuzz for company name matching
- Add threshold configuration
- Update tests for new functionality

Fixes #42
```

## Project Structure

```
nl2api/
├── src/
│   ├── nl2api/           # Core NL2API system
│   │   ├── agents/       # Domain agents
│   │   ├── llm/          # LLM providers
│   │   ├── rag/          # RAG retrieval
│   │   └── ...
│   ├── common/           # Shared utilities
│   └── evaluation/       # Evaluation pipeline
├── tests/
│   ├── unit/             # Unit tests
│   └── fixtures/         # Test fixtures
└── docs/                 # Documentation
```

## Getting Help

- Check existing [issues](https://github.com/YOUR_USERNAME/nl2api/issues)
- Read the [documentation](docs/)
- Open a new issue for questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
