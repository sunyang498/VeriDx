""" "归一化结果能不能真被下游吃掉"的唯一判据（审查 §3.3 / §3.5 第 2 条）。

**为什么单独立一个模块**：这是本项目已经踩过 5 次的那类错误的第 6 次 ——
「判据是**上游自报成功**，而不是**下游真的能吃**」。所以判据只能有一份实现，
由 `normalize.gate` 与 `eval.preflight` **共用**，不许各写一份。

* 不 spawn 子进程（只做 AST 静态检查）：单条 1ms 量级，可以逐条跑全量数据集；
* 判据是 `static_check(expr) is None`：它同时覆盖语法错、非白名单调用、危险语法、未知符号。
"""

from __future__ import annotations

__all__ = ["static_parse_ok"]

try:  # pragma: no cover - 仅用于隔离导入期的循环依赖
    from veridx.sandbox.guard import static_check
except ImportError:  # pragma: no cover
    static_check = None  # type: ignore[assignment]


#: `static_check` 的 reason 里，**不属于"解析不了"** 的那一类：
#: `not_representable:digits=N` 是**容量**判据（ADR-018 三分法），不是语法问题 ——
#: 表达式本身完全合法，只是数值太大。把它算成"解析失败"会**掩盖真正的终态**
#: （答案不可表示 → `answer_unrepresentable`，而不是 `normalization_unverified`）。
_CAPACITY_PREFIXES = ("not_representable",)


def static_parse_ok(expr: str) -> bool:
    """`expr` 能不能被沙箱**静态**接受（不执行、不 spawn）。

    注意它比"Python 能 parse"更严：`__import__('os')` 也能被 `ast.parse` 解析，
    但会被 `static_check` 判 `forbidden_call` —— 那正是我们要的。

    但**容量**不算失败：`not_representable:digits=5403` 的表达式语法完全合法，
    只是数值太大；那一层由 `runner` 的必6 闸负责给出正确终态。
    """
    if not isinstance(expr, str) or not expr.strip():
        return False
    if static_check is None:  # pragma: no cover
        return False
    verdict = static_check(expr)
    if verdict is None:
        return True
    _verdict, reason = verdict
    return str(reason).startswith(_CAPACITY_PREFIXES)
