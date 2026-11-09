"""表达式沙箱：AST 白名单 → 名字白名单 → 静态上限 → 子进程超时。

对外只有三件事：

```python
from veridx.sandbox import evaluate, compare, record_verdict

evaluate("1/6")            # SandboxResult(verdict=ok, value=0.1666…)
evaluate("2<x<3")           # verdict=unsupported（合法数学，不是攻击）
evaluate("9**9**9")         # verdict=rejected, reason=nested_pow
compare("e-1", "1.718")     # Comparison(equal=True)
```

判定只有五态（`ok / unsupported / rejected / timeout / error`），见 `verdict.py`。
"""

from __future__ import annotations

from veridx.sandbox.core import clear_cache, compare, evaluate
from veridx.sandbox.guard import FUNCTION_NAMES, SYMBOL_NAMES, static_check
from veridx.sandbox.ledger import record_verdict, rejection_rate
from veridx.sandbox.verdict import Comparison, Reason, SandboxResult, Verdict

__all__ = [
    "Comparison",
    "FUNCTION_NAMES",
    "Reason",
    "SYMBOL_NAMES",
    "SandboxResult",
    "Verdict",
    "clear_cache",
    "compare",
    "evaluate",
    "record_verdict",
    "rejection_rate",
    "static_check",
]
