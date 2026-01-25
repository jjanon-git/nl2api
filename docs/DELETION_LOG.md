# Deletion Log

Record of dead code cleanup actions.

## 2026-01-25

### Changes Made

| File | Change | Reason |
|------|--------|--------|
| `src/evalkit/common/telemetry/setup.py` | Prefixed unused params with `_` in no-op classes | Silence vulture warnings for intentionally-unused params in `_NoOpCounter.add()`, `_NoOpHistogram.record()`, `_NoOpGauge.set()` |
| `src/nl2api/resolution/resolver.py:298` | Removed redundant `if x or True:` condition | Always-true condition was dead code. Simplified to unconditional API call with same behavior. |

### False Positives (No Action)

| File | Warning | Reason for No Action |
|------|---------|---------------------|
| `src/evalkit/contracts/evaluation.py:707,724,725` | Unused variables `actual_data`, `expected_text`, `actual_text` | These are abstract method parameters - they define the interface but don't use the values. Implementations will use them. |

### Verification

- **Tests before:** 2851 passed, 11 skipped
- **Tests after:** 2851 passed, 11 skipped
- **vulture findings reduced:** 5 → 3 (remaining 3 are false positives)
