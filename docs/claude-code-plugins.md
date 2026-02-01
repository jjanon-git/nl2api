# Claude Code Plugins

Adapted from [everything-claude-code](https://github.com/affaan-m/everything-claude-code) for Python/pytest eval framework.

## Installed Plugins

Location: `~/.claude/`

### Commands (`~/.claude/commands/`)

| Command | File | Usage | Description |
|---------|------|-------|-------------|
| `/acc` | `acc.md` | `/acc tier1` | Run accuracy tests (tier1/2/3 pytest markers) |
| `/refactor-clean` | `refactor-clean.md` | `/refactor-clean` | Find and remove dead Python code |

### Agents (`~/.claude/agents/`)

| Agent | File | Invoke With | Description |
|-------|------|-------------|-------------|
| code-reviewer | `code-reviewer.md` | "Use code-reviewer agent" | Python code review with CRITICAL/HIGH/MEDIUM/LOW severity |
| tdd-guide | `tdd-guide.md` | "Use tdd-guide agent" | Test-driven development workflow for pytest |
| refactor-cleaner | `refactor-cleaner.md` | "Use refactor-cleaner agent" | Dead code detection and safe removal |

### Hooks (`~/.claude/hooks/`)

Auto-triggered on tool use:

| Trigger | Action |
|---------|--------|
| Edit `.py` file | Auto-fix with `ruff check --fix`, type-check with `mypy` |
| Write `.py` file | Auto-format with `ruff format` |
| Edit `test_*.py` | Remind to run pytest |
| Run expensive eval | Warn about API costs |
| Git push | Show pre-push checklist |
| Stop (any) | Check for debug breakpoints in modified files |

### Rules (`~/.claude/rules/`)

| Rule | File | Description |
|------|------|-------------|
| testing | `testing.md` | Coverage targets, TDD workflow, eval tiers |

---

## Not Imported (and Why)

### Agents

| Agent | Why Not Imported |
|-------|------------------|
| **security-reviewer** | JS/npm focused (npm audit, eslint-plugin-security). Python equivalent would need bandit, safety, pip-audit. Your CLAUDE.md already has security checklist. |
| **architect** | Generic enough to use as-is, but read-only (no Edit). Lower priority - use when needed. |
| **build-error-resolver** | TypeScript/Next.js specific (tsc, webpack). Python equivalent would be for mypy/ruff errors - already covered by hooks. |
| **e2e-runner** | Playwright/JS specific. Would need adaptation for pytest + selenium/playwright-python. |
| **doc-updater** | Auto-updates README/docs. Risk of unwanted doc changes. Use manually when needed. |
| **planner** | Generic planning agent. Claude Code has built-in `/plan` mode that's better integrated. |

### Commands

| Command | Why Not Imported |
|---------|------------------|
| **build-fix** | TypeScript specific (tsc errors). Python has hooks for ruff/mypy. |
| **checkpoint** | Useful but git stash/branch workflow is simpler. Consider adding if needed. |
| **code-review** | Invokes code-reviewer agent. Can just ask directly. |
| **e2e** | Playwright/JS specific. |
| **learn** | Extracts patterns to memory. Interesting but experimental. |
| **orchestrate** | Runs multiple agents in sequence. Complex, rarely needed. |
| **plan** | Built into Claude Code already (`/plan`). |
| **setup-pm** | Package manager detection (npm/yarn/pnpm). Not needed for Python. |
| **tdd** | Invokes tdd-guide agent. Can just ask directly. |
| **test-coverage** | JS-specific coverage tools. Python uses `pytest --cov`. |
| **update-codemaps** | Generates codebase maps. Interesting but experimental. |
| **update-docs** | Auto-updates docs. Risk of unwanted changes. |
| **verify** | Runs lint+typecheck+tests. Python: just run `pytest && ruff check && mypy`. |

### Rules

| Rule | Why Not Imported |
|------|------------------|
| **agents.md** | Meta-rules about using agents. Not needed - agents are self-documenting. |
| **coding-style.md** | JS/React specific (TypeScript, ESLint, Prettier). |
| **git-workflow.md** | Generic git rules. Your CLAUDE.md already covers this. |
| **hooks.md** | Documents hook system. Not needed as rule. |
| **patterns.md** | React/Next.js patterns. Not applicable. |
| **performance.md** | JS bundle optimization. Not applicable. |
| **security.md** | Mostly JS-focused. Your CLAUDE.md has Python security checklist. |

### Skills

| Skill | Why Not Imported |
|-------|------------------|
| **backend-patterns** | Express/Node patterns. Not applicable. |
| **clickhouse-io** | ClickHouse specific. Not using ClickHouse. |
| **coding-standards** | JS/TS standards. |
| **continuous-learning** | Experimental pattern extraction. |
| **eval-harness** | JS testing harness. You have your own eval framework. |
| **frontend-patterns** | React/Next.js patterns. |
| **project-guidelines-example** | Example template. |
| **security-review** | JS security patterns. |
| **strategic-compact** | Context management. Interesting but experimental. |
| **tdd-workflow** | JS TDD. Already have Python version. |
| **verification-loop** | JS verification. |

---

## How to Use

### Run a Command

```
/eval tier1
/refactor-clean
```

### Invoke an Agent

```
Use the code-reviewer agent to review my changes
Use the tdd-guide agent to help me write tests
Use the refactor-cleaner agent to find dead code
```

### Hooks Run Automatically

Hooks trigger on specific tool uses - no action needed.

### Dependencies

Required for refactor-cleaner:
```bash
pip install vulture autoflake
```

Already in your venv: `ruff`, `mypy`, `pytest`

---

## Adding More Plugins

### To Import Another Agent

1. Copy from `/tmp/ecc/agents/` to `~/.claude/agents/`
2. Adapt for Python:
   - Replace npm/yarn with pip
   - Replace tsc with mypy
   - Replace eslint with ruff
   - Replace jest/vitest with pytest
3. Update tool permissions in frontmatter

### To Import Another Command

1. Copy from `/tmp/ecc/commands/` to `~/.claude/commands/`
2. Adapt bash commands for Python tooling

### To Add Custom Hooks

Edit `~/.claude/hooks/hooks.json`:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "tool == \"Edit\" && tool_input.file_path matches \"\\.py$\"",
        "hooks": [
          {
            "type": "command",
            "command": "your-command-here"
          }
        ],
        "description": "Description of what this hook does"
      }
    ]
  }
}
```

---

## Source Repository

Full collection: https://github.com/affaan-m/everything-claude-code

Cloned to: `/tmp/ecc/` (temporary, will be deleted on reboot)

To re-clone:
```bash
git clone https://github.com/affaan-m/everything-claude-code /tmp/ecc
```
