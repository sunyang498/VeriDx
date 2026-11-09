"""第 4 层防御：子进程求值入口。

**必须满足两条约束，否则在 Windows 上直接不可用**：

1. **模块级函数**——Windows 用 `spawn`，子进程会重新 import 本模块；
   闭包 / lambda 无法 pickle（P0.6 审查 §4.3①）。
2. **导入时无副作用**——`spawn` 会执行模块顶层代码，所以这里只有 import 与函数定义。

为什么必须是**进程**而不是线程/信号（§4.5）：

| 方案 | 为什么不行 |
|---|---|
| `signal.alarm` | Windows 不支持，且只能主线程 |
| `threading.join(timeout)` | 杀不掉线程，超时后继续烧 CPU |
| `asyncio.wait_for` | 只是"不再等"，计算仍在跑 |
| **子进程 + terminate()** | 唯一能真正中断的方式 |
"""

from __future__ import annotations

import contextlib
from typing import Any

import sympy

from veridx.sandbox.guard import FUNCTION_NAMES, SYMBOL_NAMES

_CONSTANTS: dict[str, Any] = {
    "e": sympy.E,
    "E": sympy.E,
    "pi": sympy.pi,
    "I": sympy.I,
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


def main(expr: str, symbols: list[str] | None, conn: Any) -> None:
    """子进程入口：求值 → 通过 Pipe 回传一个 JSON 可序列化的 dict。"""
    try:
        value = sympy.sympify(expr, locals=build_namespace(symbols), evaluate=True)
        free_symbols = bool(getattr(value, "free_symbols", False))
        payload: dict[str, Any] = {
            "ok": True,
            "exact": str(value),
            "free_symbols": free_symbols,
            "value": None,
        }
        if not free_symbols:
            try:
                payload["value"] = float(sympy.N(value))
            except (TypeError, ValueError, OverflowError):
                payload["value"] = None
        conn.send(payload)
    except BaseException as exc:  # noqa: BLE001 - 子进程不能把异常抛回父进程
        with contextlib.suppress(Exception):
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
    finally:
        with contextlib.suppress(Exception):
            conn.close()


__all__ = ["FUNCTION_NAMES", "build_namespace", "main"]
