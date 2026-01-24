# Public Repository Release Audit

**Date:** 2026-01-21
**Status:** In Progress - P0 Items Pending

---

## Executive Summary

The nl2api codebase is well-structured with good test coverage (907 tests). However, several critical issues must be addressed before public release, particularly around **security** (exposed API key) and **missing open-source standard files** (LICENSE, CONTRIBUTING, etc.).

---

## P0 - BLOCKERS (Must Fix Before Public Release)

### 1. CRITICAL: Exposed API Key in .env

**Location:** `.env` (local file)
**Risk:** HIGH - Live Anthropic API key could be accidentally committed

**Proposed Fix:**
1. Immediately revoke the API key via Anthropic dashboard
2. Ensure `.env` is not tracked (it's in `.gitignore` but verify)
3. Add `.env` pattern to `.git/info/exclude` as extra protection
4. Update `.env.example` with clear placeholder patterns

### 2. Missing LICENSE File

**Risk:** Without a license, the code is "all rights reserved" and cannot be legally used

**Proposed Fix:** Add MIT License (most permissive, widely adopted)

```
MIT License

Copyright (c) 2026 [Organization Name]

Permission is hereby granted, free of charge...
```

### 3. Missing CONTRIBUTING.md

**Proposed Content:**
- Development environment setup
- Coding standards (ruff, mypy)
- Pull request process
- Testing requirements
- Issue reporting guidelines

### 4. Missing CODE_OF_CONDUCT.md

**Proposed Fix:** Add Contributor Covenant v2.1 (industry standard)

### 5. Missing SECURITY.md

**Proposed Content:**
- Supported versions
- How to report vulnerabilities
- Disclosure policy
- Security contact

### 6. Missing GitHub Templates

**Proposed Files:**
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/PULL_REQUEST_TEMPLATE.md`

---

## P1 - IMPORTANT (Should Fix Before Public Release)

### 7. Incomplete pyproject.toml Metadata

**Current:** Missing authors, license, repository, keywords, classifiers

**Proposed Addition:**
```toml
[project]
authors = [{ name = "NL2API Team" }]
license = { text = "MIT" }
keywords = ["nlp", "api", "lseg", "financial-data", "natural-language"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
]

[project.urls]
Homepage = "https://github.com/[org]/nl2api"
Documentation = "https://github.com/[org]/nl2api#readme"
Repository = "https://github.com/[org]/nl2api.git"
Issues = "https://github.com/[org]/nl2api/issues"
```

### 8. MCP Client Stub Documentation

**Location:** `src/nl2api/mcp/client.py`
**Issue:** Placeholder implementation that raises "not yet implemented"

**Proposed Fix:** Add module docstring clearly marking as experimental:
```python
"""
MCP (Model Context Protocol) Client - EXPERIMENTAL

This module provides MCP client functionality for future integration
with MCP-compatible servers. Currently in development.

Status: Not yet implemented - API subject to change
"""
```

### 9. Evaluation Stages 3 & 4 Documentation

**Location:** `src/evaluation/core/evaluators.py`
**Issue:** Stage 3 (Execution) and Stage 4 (Semantics) are stubs

**Proposed Fix:** Add clear docstring:
```python
"""
Note: Stages 3 (Execution) and 4 (Semantics) are planned for future
releases. Currently these stages pass through for compatibility.
See: https://github.com/[org]/nl2api/issues/XX
"""
```

### 10. Azure Backend Documentation

**Location:** `src/common/storage/factory.py`
**Issue:** Azure backend option exists but raises NotImplementedError

**Proposed Fix:** Update config.py docstring and README to indicate postgres-only for now

### 11. Deprecation Warnings

**Location:** `src/nl2api/agents/protocols.py`
**Issue:** `can_handle()` marked deprecated but no runtime warning

**Proposed Fix:**
```python
import warnings

async def can_handle(self, query: str) -> float:
    warnings.warn(
        "can_handle() is deprecated and will be removed in v2.0. "
        "Use LLMToolRouter for query routing instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ...
```

### 12. Module Exports Completeness

**Location:** `src/nl2api/agents/__init__.py`
**Issue:** Only EstimatesAgent exported, not other agents

**Proposed Fix:** Export all public agents or document why selective

---

## P2 - NICE TO HAVE (Post-Release)

### 13. README Enhancements

**Proposed Additions:**
- CI/CD status badges
- Coverage badge
- PyPI version badge (if publishing)
- Quick start for end-users (not just developers)
- Link to API documentation

### 14. Silent Exception Handlers

**Locations:** 19 occurrences of `except: pass`
**Proposed Fix:** Add `logger.debug()` for observability

### 15. Type Hint Completeness

**Current:** mypy runs with `continue-on-error: true`
**Proposed:** Fix mypy errors incrementally, remove continue-on-error

### 16. CHANGELOG.md

**Proposed:** Add changelog following Keep a Changelog format

---

## Implementation Plan

### Phase 1: Critical Security & Legal (P0)
1. Verify API key not in git history
2. Add LICENSE (MIT)
3. Add CONTRIBUTING.md
4. Add CODE_OF_CONDUCT.md
5. Add SECURITY.md
6. Add GitHub issue/PR templates

### Phase 2: Documentation & Metadata (P1)
1. Update pyproject.toml with full metadata
2. Add experimental notices to MCP client
3. Document evaluation stage limitations
4. Add deprecation warnings
5. Fix module exports

### Phase 3: Polish (P2)
1. Add README badges
2. Create CHANGELOG.md
3. Address silent exception handlers
4. Type hint improvements

---

## Files to Create

| File | Priority | Description |
|------|----------|-------------|
| `LICENSE` | P0 | MIT License |
| `CONTRIBUTING.md` | P0 | Contributor guidelines |
| `CODE_OF_CONDUCT.md` | P0 | Contributor Covenant |
| `SECURITY.md` | P0 | Security policy |
| `.github/ISSUE_TEMPLATE/bug_report.md` | P0 | Bug report template |
| `.github/ISSUE_TEMPLATE/feature_request.md` | P0 | Feature request template |
| `.github/PULL_REQUEST_TEMPLATE.md` | P0 | PR template |
| `CHANGELOG.md` | P2 | Version history |

## Files to Modify

| File | Priority | Changes |
|------|----------|---------|
| `pyproject.toml` | P1 | Add metadata (authors, license, urls, classifiers) |
| `README.md` | P1/P2 | Add badges, improve quick start |
| `src/nl2api/mcp/client.py` | P1 | Add experimental notice |
| `src/nl2api/agents/__init__.py` | P1 | Export all agents |
| `src/nl2api/agents/protocols.py` | P1 | Add deprecation warning |
| `src/evaluation/core/evaluators.py` | P1 | Document deferred stages |
| `.env.example` | P0 | Verify safe placeholder values |

---

## Estimated Effort

| Phase | Items | Effort |
|-------|-------|--------|
| Phase 1 (P0) | 7 files to create | ~30 min |
| Phase 2 (P1) | 6 files to modify | ~20 min |
| Phase 3 (P2) | Optional polish | ~30 min |

**Total:** ~1-2 hours for P0+P1

---

## Questions for Approval

1. **License:** Is MIT acceptable, or prefer Apache 2.0?
2. **Organization name:** What name for LICENSE copyright?
3. **Repository URL:** What will the public repo URL be?
4. **MCP client:** Keep as experimental, or remove entirely?
5. **Changelog:** Start from v0.1.0 or document all commits?
