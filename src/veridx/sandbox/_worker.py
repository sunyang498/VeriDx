"""常驻 worker 的子进程入口。

**三条硬约束**（不满足则在 Windows / 并发下直接不可用）：

1. **模块级函数**——Windows 用 `spawn`，子进程会重新 import 本模块；闭包无法 pickle。
2. **导入时无副作用**——`spawn` 会执行模块顶层代码，所以这里只有 import 与函数定义。
   调用方（脚本）必须有 `if __name__ == "__main__":` 守卫，否则会递归 spawn。
   即使忘了，父进程侧也会把它降级成 `error: spawn_failed:*`（不是抛异常）。
3. **一次 spawn 服务多次求值**——`import sympy` ≈1.08s，是单次调用成本的主体；
   常驻后稳态只剩计算时间（<10ms）。

协议（父 → 子 → 父）：

```
子: {"ready": true, "pid": 1234}          ← import 完成
父: {"expr": "...", "symbols": [...]}
子: {"ok": true, "exact": "1/6", "value": 0.166..., "kind": "real"|"complex"|"symbolic"|"undefined"}
子: {"ok": false, "reason": "user_input:TypeError", "detail": "..."}
父: None                                   ← 关闭
```
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import sympy

from veridx.sandbox.guard import CONSTANT_NAMES, SYMBOL_NAMES

_CONSTANTS: dict[str, Any] = {
    "e": sympy.E,
    "E": sympy.E,
    "pi": sympy.pi,
    "I": sympy.I,
    "inf": sympy.oo,
    "oo": sympy.oo,
}

_FUNCTIONS: dict[str, Any] = {
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "log": sympy.log,
    "ln": sympy.log,
    "exp": sympy.exp,
    "Abs": sympy.Abs,
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
}


def build_namespace(symbols: list[str] | None = None) -> dict[str, Any]:
    names = symbols if symbols is not None else list(SYMBOL_NAMES)
    namespace: dict[str, Any] = {name: sympy.Symbol(name) for name in names}
    for name in names:
        if name in _CONSTANTS:
            namespace[name] = _CONSTANTS[name]
    namespace.update(_FUNCTIONS)
    return namespace


def evaluate_payload(expr: str, symbols: list[str] | None) -> dict[str, Any]:
    """求值并分类值域。**不抛异常**：所有失败都转成 `{"ok": false, ...}`。"""
    try:
        value = sympy.sympify(expr, locals=build_namespace(symbols), evaluate=True)
    except (TypeError, ValueError, AttributeError, SyntaxError, OverflowError, RecursionError) as exc:
        return {"ok": False, "reason": f"user_input:{type(exc).__name__}", "detail": str(exc)[:160]}
    except BaseException as exc:  # noqa: BLE001 - 子进程不能崩在未捕获异常上
        return {"ok": False, "reason": f"sandbox:{type(exc).__name__}", "detail": str(exc)[:160]}

    exact = str(value) if value is not None else None
    if getattr(value, "free_symbols", None):
        return {"ok": True, "kind": "symbolic", "exact": exact, "value": None}
    # zoo（复无穷）= 数学未定义（1/0、log(0)）：必须在转 complex 之前判，
    # 否则 complex(zoo) 会给出 (nan+nanj)，被误分类成 nan 或复数
    if value is sympy.zoo or (hasattr(value, "has") and value.has(sympy.zoo)):
        return {"ok": True, "kind": "undefined", "exact": exact, "value": None}

    try:
        numeric = complex(sympy.N(value))
    except (TypeError, ValueError, OverflowError) as exc:
        return {"ok": True, "kind": "non_numeric", "exact": exact, "value": None, "detail": str(exc)[:80]}

    # nan 必须先判：nan 与任何数比较都为 True/False 的怪异语义会让 "nan != 0" 成立，
    # 从而把 0/0 误判成"复数"（实测踩过）
    real = numeric.real
    imag = numeric.imag
    if math.isnan(real) or math.isnan(imag):
        return {"ok": True, "kind": "nan", "exact": exact, "value": None}
    if math.isinf(real) or math.isinf(imag):
        return {"ok": True, "kind": "inf", "exact": exact, "value": None}
    if imag != 0:
        return {"ok": True, "kind": "complex", "exact": exact, "value": None}
    return {"ok": True, "kind": "real", "exact": exact, "value": float(real)}


def one_shot(expr: str, symbols: list[str] | None, conn: Any) -> None:
    """一次性求值（测试与冷启动兜底用）。"""
    with contextlib.suppress(Exception):
        conn.send(evaluate_payload(expr, symbols))
    with contextlib.suppress(Exception):
        conn.close()


def serve(conn: Any) -> None:
    """常驻循环：一次 spawn 服务多次求值。"""
    with contextlib.suppress(Exception):
        conn.send({"ready": True})
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        expr, symbols = message
        with contextlib.suppress(Exception):
            conn.send(evaluate_payload(expr, symbols))
    # pragma: no cover - 循环只在 EOF/None 时退出


__all__ = ["CONSTANT_NAMES", "SYMBOL_NAMES", "build_namespace", "evaluate_payload", "one_shot", "serve"]
