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


#: 整数→`float` 的可表示上界：`2**1024 > sys.float_info.max`。
#: 超过它的整数**不是** `inf`，而是"超出可表示范围"——`complex(sympy.N(v))` 会静默
#: 溢出成 inf，把它误判成 `non_finite`（审查 必8 的实测：`999999999**200` 是有限整数）。
_FLOAT_MAX_BITS = 1024

#: `int → str` 的位长阈值（≈4214 位十进制，CPython 上限 4300 位）。超过它 `str()` 会抛。
_STR_MAX_BITS = 14000


def _too_large(value: Any) -> bool:
    """任何"物化它"的动作（`str()` / `complex()`）都会失败或失真 → 直接判容量超限。"""
    return _digit_estimate(value) is not None


def _digit_estimate(value: Any) -> int | None:
    """超出可表示范围的整数的**十进制位数估计**；不需要物化它。"""
    if not isinstance(value, sympy.Integer):
        return None
    try:
        bits = value.p.bit_length()
    except AttributeError:  # pragma: no cover - 防御
        return None
    if bits <= _FLOAT_MAX_BITS:
        return None
    # log10(2) ≈ 0.30103；bit_length 给的是上界，四舍五入到整数位即可
    return int(bits * 0.30103) + 1


def evaluate_payload(expr: str, symbols: list[str] | None) -> dict[str, Any]:
    """求值并分类值域。**不抛异常**：所有失败都转成 `{"ok": false, ...}`。"""
    try:
        value = sympy.sympify(expr, locals=build_namespace(symbols), evaluate=True)
    except (TypeError, ValueError, AttributeError, SyntaxError, OverflowError, RecursionError) as exc:
        return {"ok": False, "reason": f"user_input:{type(exc).__name__}", "detail": str(exc)[:160]}
    except BaseException as exc:  # noqa: BLE001 - 子进程不能崩在未捕获异常上
        return {"ok": False, "reason": f"sandbox:{type(exc).__name__}", "detail": str(exc)[:160]}

    try:
        return _classify_value(value)
    except BaseException as exc:  # noqa: BLE001 - 见 必9：**这一层必须有回包**
        return {"ok": False, "reason": f"sandbox:{type(exc).__name__}", "detail": str(exc)[:160]}


def _classify_value(value: Any) -> dict[str, Any]:
    """值域分类。**所有会产生异常的语句都在这里**（含 `str(value)`）。"""
    if getattr(value, "free_symbols", None):
        return {"ok": True, "kind": "symbolic", "exact": _safe_str(value), "value": None}
    # zoo（复无穷）= 数学未定义（1/0、log(0)）：必须在转 complex 之前判，
    # 否则 complex(zoo) 会给出 (nan+nanj)，被误分类成 nan 或复数
    if value is sympy.zoo or (hasattr(value, "has") and value.has(sympy.zoo)):
        return {"ok": True, "kind": "undefined", "exact": _safe_str(value), "value": None}

    # 必8：超大整数先按位长拦，**不要**先 str() 也不要先 complex()
    digits = _digit_estimate(value)
    if digits is not None:
        return {"ok": True, "kind": "too_large", "exact": None, "value": None, "digits": digits}

    exact = _safe_str(value)
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


def _safe_str(value: Any) -> str | None:
    """`str(value)` 会因超过 4300 位抛 `ValueError` —— 吞掉它，返回 None（审查 必9）。"""
    try:
        if isinstance(value, sympy.Integer):
            try:
                if value.p.bit_length() > _STR_MAX_BITS:
                    return None
            except AttributeError:  # pragma: no cover
                pass
        return str(value)
    except (ValueError, MemoryError):
        return None


def one_shot(expr: str, symbols: list[str] | None, conn: Any) -> None:
    """一次性求值（测试与冷启动兜底用）。"""
    with contextlib.suppress(Exception):
        conn.send(evaluate_payload(expr, symbols))
    with contextlib.suppress(Exception):
        conn.close()


def _reply(payload: Any) -> dict[str, Any]:
    """**一请求必有一回包**（审查 必9）。

    父进程拿不到回包就会判"超时"并毒丸杀掉一个健康进程 —— 那既污染
    timeout 计数与 `static_gap` 观测，又要多付一次 0.9s 冷启动。
    """
    if isinstance(payload, dict):
        return payload
    return {"ok": False, "reason": "sandbox:NonDictPayload", "detail": type(payload).__name__}


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
        try:
            payload = _reply(evaluate_payload(expr, symbols))
        except BaseException as exc:  # noqa: BLE001 - **绝不静默吞掉后什么都不回**
            payload = {"ok": False, "reason": f"sandbox:{type(exc).__name__}", "detail": "serve"}
        try:
            conn.send(payload)
        except (EOFError, OSError, ValueError):
            return
    # pragma: no cover - 循环只在 EOF/None 时退出


__all__ = ["CONSTANT_NAMES", "SYMBOL_NAMES", "build_namespace", "evaluate_payload", "one_shot", "serve"]
