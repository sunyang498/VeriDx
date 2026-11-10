"""沙箱判定结果与**唯一的**能力原因集合。

## 五态

| verdict | 含义 | 计入 M17 |
|---|---|---|
| `ok` | 可安全求值，结果是**有限实数** | — |
| `unsupported` | 合法输入，但超出能力边界（符号/复数/非有限/过大/未知符号…） | — |
| `rejected` | **具备能力**的结构：能逃逸，或能烧资源 | ✅ |
| `timeout` | 通过静态检查但计算超预算 | — |
| `error` | 工程故障（spawn 失败 / 子进程崩溃 / 内部错误） | — |

## `rejected` 的唯一判据（ADR-015）

> **具备能力**：能逃逸出求值层，或能消耗不可控资源。

因此以下三类**不是** rejected（它们既不能逃逸也不能烧资源，只是"写错了"或"太大了"）：

* `unknown_symbol`（拼写错误）
* `function_as_value`（把函数当数用）
* `oversize`（超容量但未超硬上限）

## 为什么要有 `CAPABILITY_REASONS`

M17（沙箱拒绝率）的口径必须**一处定义、多处引用**。历史上有三种算法并存（按 verdict 数、
按 reason 数、按样本分组数），必然漂移。现在只有这一个常量，测试会断言
`CAPABILITY_REASONS == I4 的 rejected 原因集`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"


class Reason(StrEnum):
    """原因码 = **前缀**；细节以 `prefix:detail` 形式追加（如 `forbidden_syntax:Attribute`）。"""

    # --- 能力型（≤> rejected）---
    FORBIDDEN_SYNTAX = "forbidden_syntax"  # Attribute / Subscript / 推导式 / lambda / import / f-string
    FORBIDDEN_CALL = "forbidden_call"  # 非白名单函数调用
    FORBIDDEN_NAME = "forbidden_name"  # dunder 名（内省主通道）
    EXPR_TOO_LONG = "expr_too_long"  # 超过硬上限 4096
    TOO_MANY_NODES = "too_many_nodes"
    NESTED_POW = "nested_pow"
    EXPONENT_TOO_LARGE = "exponent_too_large"
    INT_LITERAL_TOO_LARGE = "int_literal_too_large"

    # --- 能力边界（→ unsupported）---
    OVERSIZE = "oversize"  # 257–4096：合法长答案，不是攻击
    UNKNOWN_SYMBOL = "unknown_symbol"  # 拼写错误
    FUNCTION_AS_VALUE = "function_as_value"  # sqrt + 1
    UNSUPPORTED_NODE = "unsupported_node"  # Compare / BoolOp / List / Tuple / Set / Dict
    SYMBOLIC = "symbolic"  # 含自由符号
    COMPLEX_RESULT = "complex_result"  # 值域是复数（含复数字面量）
    NON_FINITE = "non_finite"  # inf / nan（`non_finite:inf` / `non_finite:nan`）
    UNDEFINED = "undefined"  # zoo（1/0、log(0)）
    NON_NUMERIC = "non_numeric"  # 无值
    PARSE_ERROR = "parse_error"  # 语法错但无危险结构
    EMPTY_EXPR = "empty_expr"
    NOT_A_STRING = "not_a_string"  # 调用方类型 bug

    # --- 运行时（→ timeout / error）---
    EVAL_TIMEOUT = "eval_timeout"
    SPAWN_FAILED = "spawn_failed"
    EVAL_ERROR = "eval_error"  # `eval_error:user_input:*` / `eval_error:sandbox:*`
    INTERNAL = "internal"


#: **M17 的唯一口径**：只有这些原因算"具备能力"，才计入拒绝率。
CAPABILITY_REASONS: frozenset[str] = frozenset(
    {
        Reason.FORBIDDEN_SYNTAX.value,
        Reason.FORBIDDEN_CALL.value,
        Reason.FORBIDDEN_NAME.value,
        Reason.EXPR_TOO_LONG.value,
        Reason.TOO_MANY_NODES.value,
        Reason.NESTED_POW.value,
        Reason.EXPONENT_TOO_LARGE.value,
        Reason.INT_LITERAL_TOO_LARGE.value,
    }
)

#: 允许表达为 `unsupported` 的原因（用于校验"classification 是否自洽"）
UNSUPPORTED_REASONS: frozenset[str] = frozenset(
    {
        Reason.OVERSIZE.value,
        Reason.UNKNOWN_SYMBOL.value,
        Reason.FUNCTION_AS_VALUE.value,
        Reason.UNSUPPORTED_NODE.value,
        Reason.SYMBOLIC.value,
        Reason.COMPLEX_RESULT.value,
        Reason.NON_FINITE.value,
        Reason.UNDEFINED.value,
        Reason.NON_NUMERIC.value,
        Reason.PARSE_ERROR.value,
        Reason.EMPTY_EXPR.value,
        Reason.NOT_A_STRING.value,
    }
)

#: 允许表达为 `error` 的原因
ERROR_REASONS: frozenset[str] = frozenset(
    {Reason.SPAWN_FAILED.value, Reason.EVAL_ERROR.value, Reason.INTERNAL.value}
)


def reason_prefix(reason: str) -> str:
    return reason.split(":", 1)[0]


def is_capability(reason: str) -> bool:
    return reason_prefix(reason) in CAPABILITY_REASONS


@dataclass(frozen=True)
class SandboxResult:
    verdict: Verdict
    reason: str = ""
    expr_digest: str = ""
    expr_len: int = 0
    elapsed_ms: int = 0
    value: float | None = None
    exact: str | None = None
    #: R3 归一化后的式子（`1j` → `1*I`）；未改写时为 None。审计用。
    normalized: str | None = None

    @property
    def is_accepted(self) -> bool:
        """`ok` 与 `unsupported` 都表示"没有危险结构"（unsupported 依然不执行）。"""
        return self.verdict in (Verdict.OK, Verdict.UNSUPPORTED)

    @property
    def is_rejected(self) -> bool:
        return self.verdict is Verdict.REJECTED

    @property
    def is_numeric(self) -> bool:
        """可参与数值判等：**必须**是 ok 且有限（`inf`/`nan` 一律不算）。"""
        if self.verdict is not Verdict.OK or self.value is None:
            return False
        return math.isfinite(self.value)

    def to_payload(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "value": self.value,
            "exact": self.exact,
            "elapsed_ms": self.elapsed_ms,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class Comparison:
    """`compare()` 的结果：`equal is None` 表示"无法判定"（**不是**"不相等"）。"""

    equal: bool | None
    left: SandboxResult
    right: SandboxResult
    reason: str = ""
