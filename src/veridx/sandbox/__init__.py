"""表达式沙箱：AST 白名单 → 名字白名单 → 静态上限 → 常驻子进程 + 超时。

对外三件事：

```python
from veridx.sandbox import evaluate, compare, record_verdict

evaluate("1/6")        # ok, value=0.1666…
evaluate("2<x<3")      # unsupported（合法数学，不是攻击）
evaluate("9**9**9")    # rejected, reason=nested_pow（能力型）
evaluate("1j")         # unsupported: complex_result（归一化为 1*I）
compare("e-1", "1.718")  # Comparison(equal=True)
```

判定五态见 `verdict.py`；`rejected` 的唯一判据是**具备能力**（ADR-015），
集合为 `CAPABILITY_REASONS`（M17 的唯一口径）。
"""

from __future__ import annotations

from veridx.sandbox.core import clear_cache, compare, evaluate, expr_digest
from veridx.sandbox.guard import CONSTANT_NAMES, FUNCTION_NAMES, SYMBOL_NAMES, static_check
from veridx.sandbox.ledger import (
    capability_reasons_seen,
    record_verdict,
    rejection_rate,
)
from veridx.sandbox.normalize import normalize_complex_literal
from veridx.sandbox.pool import get_pool, shutdown_pool
from veridx.sandbox.verdict import (
    CAPABILITY_REASONS,
    ERROR_REASONS,
    UNSUPPORTED_REASONS,
    Comparison,
    Reason,
    SandboxResult,
    Verdict,
    is_capability,
    reason_prefix,
)

__all__ = [
    "CAPABILITY_REASONS",
    "CONSTANT_NAMES",
    "Comparison",
    "ERROR_REASONS",
    "FUNCTION_NAMES",
    "Reason",
    "SYMBOL_NAMES",
    "SandboxResult",
    "UNSUPPORTED_REASONS",
    "Verdict",
    "capability_reasons_seen",
    "clear_cache",
    "compare",
    "evaluate",
    "expr_digest",
    "get_pool",
    "is_capability",
    "normalize_complex_literal",
    "reason_prefix",
    "record_verdict",
    "rejection_rate",
    "shutdown_pool",
    "static_check",
]
