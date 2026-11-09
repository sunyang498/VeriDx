"""沙箱判定结果。

三分类是**风险评估**（§4.6）：

* `ok` —— 可安全求值
* `unsupported` —— **合法数学但不支持**（`2<x<3`、区间、符号表达式）→ 上层按 `cannot_parse` 处理，
  **不算攻击、不计入 M17**
* `rejected` —— 危险结构或超限 → 拒绝 + 落痕 + 计入 M17

另有两种**运行时**结果（不是风险分类，但必须可见，否则会被误当成"模型错"）：

* `timeout` —— 通过前三层但运行超过预算（进程被 terminate）
* `error` —— 子进程异常（子进程崩了、序列化失败等）

> 把 timeout 折进 `rejected` 会污染 M17（攻击尝试率）；把 timeout 折进 `unsupported` 会让
> "预算不足"伪装成"数学不支持"。**宁可多一个枚举值，也不要一个说谎的分类。**
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"


class Reason(StrEnum):
    """拒绝/不支持的原因码。机器可读，用于 M17 的原因分布。"""

    EXPR_TOO_LONG = "expr_too_long"
    TOO_MANY_NODES = "too_many_nodes"
    INT_LITERAL_TOO_LARGE = "int_literal_too_large"
    EXPONENT_TOO_LARGE = "exponent_too_large"
    NESTED_POW = "nested_pow"
    FORBIDDEN_SYNTAX = "forbidden_syntax"
    FORBIDDEN_CALL = "forbidden_call"
    UNKNOWN_NAME = "unknown_name"
    PARSE_ERROR = "parse_error"
    EMPTY_EXPR = "empty_expr"
    SYMBOLIC = "symbolic"
    NON_NUMERIC = "non_numeric"
    UNSUPPORTED_NODE = "unsupported_node"
    EVAL_TIMEOUT = "eval_timeout"
    EVAL_ERROR = "eval_error"


@dataclass(frozen=True)
class SandboxResult:
    verdict: Verdict
    reason: str = ""
    expr_digest: str = ""
    expr_len: int = 0
    elapsed_ms: int = 0
    value: float | None = None
    exact: str | None = None

    @property
    def is_accepted(self) -> bool:
        """`ok` 与 `unsupported` 都表示"没有危险结构"（unsupported 依然不执行）。"""
        return self.verdict in (Verdict.OK, Verdict.UNSUPPORTED)

    @property
    def is_rejected(self) -> bool:
        return self.verdict is Verdict.REJECTED

    @property
    def is_numeric(self) -> bool:
        return self.verdict is Verdict.OK and self.value is not None

    def to_payload(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "value": self.value,
            "exact": self.exact,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class Comparison:
    """`compare()` 的结果：`equal is None` 表示"无法判定"（不是"不相等"）。"""

    equal: bool | None
    left: SandboxResult
    right: SandboxResult
    reason: str = ""
