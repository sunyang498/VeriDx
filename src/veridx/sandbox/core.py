"""沙箱对外入口：归一化 → 静态判定（1–3 层）→ 常驻 worker 求值（第 4 层）→ LRU。

## 值域政策（ADR-016）

| 结果 | verdict | reason |
|---|---|---|
| 有限实数 | `ok` | — |
| `inf` / `nan` | `unsupported` | `non_finite:inf` / `non_finite:nan` |
| 复数（`1j`、`sqrt(-1)`、`I`） | `unsupported` | `complex_result` |
| `zoo`（`1/0`、`log(0)`） | `unsupported` | `undefined` |
| 含自由符号 | `unsupported` | `symbolic` |

**`ok` 的定义里就含"有限实数"**：溢出成 `inf` 不算成功。原因是一旦允许 `ok, inf`，
`compare` 的容差公式 `abs(l-r) <= tol*max(1,|l|,|r|)` 会退化成 `inf <= inf` 恒真 ——
**任何溢出值都会和任何大数判等**。

## 归一化（R3）

只把 Python 复数字面量 `1j` 改写成 `1*I`，与 `I` 同等待遇。
隐式乘法（`3I`、`2x`）**延后到 P1.8**：那条规则会让 `1e-5` 静默变成 `1*e-5`（改值），
见 `normalize.py` 的说明。

## 缓存键

键 = `(归一化后的式子, 符号表)`。用归一化后的式子，是为了让 `1j` 与 `1*I` 命中同一条缓存；
原式只进 `SandboxResult.normalized` 与 evidence 审计字段。
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict

from veridx import config
from veridx.sandbox import pool as pool_module
from veridx.sandbox.guard import FUNCTION_NAMES, SYMBOL_NAMES, static_check
from veridx.sandbox.normalize import normalize_complex_literal
from veridx.sandbox.verdict import Comparison, Reason, SandboxResult, Verdict

_CACHE: OrderedDict[tuple[str, tuple[str, ...]], SandboxResult] = OrderedDict()
_CACHEABLE = frozenset({Verdict.OK, Verdict.UNSUPPORTED, Verdict.REJECTED})


def expr_digest(expr: str) -> str:
    return hashlib.sha256(expr.encode("utf-8", "replace")).hexdigest()[:16]


def _normalize_symbols(symbols: frozenset[str] | None) -> tuple[str, ...]:
    return tuple(sorted(symbols)) if symbols is not None else tuple(sorted(SYMBOL_NAMES))


def _remember(key: tuple[str, tuple[str, ...]], result: SandboxResult) -> None:
    _CACHE[key] = result
    _CACHE.move_to_end(key)
    while len(_CACHE) > config.SANDBOX_LRU_SIZE:
        _CACHE.popitem(last=False)


def clear_cache() -> None:
    _CACHE.clear()


def _classify(
    payload: dict, digest: str, length: int, elapsed_ms: int, normalized: str | None
) -> SandboxResult:
    """把 worker payload 翻译成 SandboxResult（值域判定在这里收口）。"""
    if not payload.get("ok"):
        reason = str(payload.get("reason", "sandbox:unknown"))
        return SandboxResult(
            verdict=Verdict.ERROR,
            reason=f"{Reason.EVAL_ERROR.value}:{reason}",
            expr_digest=digest,
            expr_len=length,
            elapsed_ms=elapsed_ms,
            normalized=normalized,
        )

    kind = payload.get("kind")
    exact = payload.get("exact")
    common = {
        "expr_digest": digest,
        "expr_len": length,
        "elapsed_ms": elapsed_ms,
        "exact": str(exact) if exact is not None else None,
        "normalized": normalized,
    }
    if kind == "real":
        value = float(payload["value"])
        if not math.isfinite(value):  # 双保险：worker 已判，这里不信上游
            return SandboxResult(
                verdict=Verdict.UNSUPPORTED,
                reason=f"{Reason.NON_FINITE.value}:{'nan' if value != value else 'inf'}",
                **common,
            )
        return SandboxResult(verdict=Verdict.OK, value=value, **common)
    if kind == "inf":
        return SandboxResult(verdict=Verdict.UNSUPPORTED, reason=f"{Reason.NON_FINITE.value}:inf", **common)
    if kind == "nan":
        return SandboxResult(verdict=Verdict.UNSUPPORTED, reason=f"{Reason.NON_FINITE.value}:nan", **common)
    if kind == "complex":
        return SandboxResult(verdict=Verdict.UNSUPPORTED, reason=Reason.COMPLEX_RESULT.value, **common)
    if kind == "symbolic":
        return SandboxResult(verdict=Verdict.UNSUPPORTED, reason=Reason.SYMBOLIC.value, **common)
    if kind == "undefined":
        return SandboxResult(verdict=Verdict.UNSUPPORTED, reason=Reason.UNDEFINED.value, **common)
    # non_numeric：无法转成数值
    return SandboxResult(verdict=Verdict.UNSUPPORTED, reason=Reason.NON_NUMERIC.value, **common)


def evaluate(
    expr: object,
    *,
    symbols: frozenset[str] | None = None,
    timeout_seconds: float | None = None,
    use_cache: bool = True,
) -> SandboxResult:
    """判定 + 求值。**永不抛异常**，任何漏网异常都降级成 `error:internal:*`。"""
    try:
        return _evaluate(expr, symbols=symbols, timeout_seconds=timeout_seconds, use_cache=use_cache)
    except BaseException as exc:  # noqa: BLE001 - 契约：永不抛
        return SandboxResult(
            verdict=Verdict.ERROR,
            reason=f"{Reason.INTERNAL.value}:{type(exc).__name__}",
            expr_digest=expr_digest(expr if isinstance(expr, str) else repr(expr)),
            expr_len=len(expr) if isinstance(expr, str) else 0,
        )


def _evaluate(
    expr: object,
    *,
    symbols: frozenset[str] | None,
    timeout_seconds: float | None,
    use_cache: bool,
) -> SandboxResult:
    normalized_symbols = _normalize_symbols(symbols)

    if not isinstance(expr, str):
        return SandboxResult(
            verdict=Verdict.UNSUPPORTED,
            reason=f"{Reason.NOT_A_STRING.value}:{type(expr).__name__}",
            expr_digest=expr_digest(repr(expr)),
        )

    rewritten = normalize_complex_literal(expr)
    effective = rewritten or expr
    key = (effective, normalized_symbols)

    if use_cache:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached

    digest = expr_digest(effective)
    length = max(len(expr), len(effective))

    static = static_check(effective, frozenset(normalized_symbols))
    if static is not None:
        verdict, reason = static
        result = SandboxResult(
            verdict=verdict,
            reason=reason,
            expr_digest=digest,
            expr_len=length,
            normalized=rewritten,
        )
        if use_cache and verdict in _CACHEABLE:
            _remember(key, result)
        return result

    budget = config.SANDBOX_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    started = time.perf_counter()
    try:
        payload = pool_module.get_pool().run(effective, normalized_symbols, budget)
    except TimeoutError:
        return SandboxResult(
            verdict=Verdict.TIMEOUT,
            reason=Reason.EVAL_TIMEOUT.value,
            expr_digest=digest,
            expr_len=length,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            normalized=rewritten,
        )
    except pool_module.WorkerUnavailable as exc:
        return SandboxResult(
            verdict=Verdict.ERROR,
            reason=f"{Reason.SPAWN_FAILED.value}:{type(exc).__name__}",
            expr_digest=digest,
            expr_len=length,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            normalized=rewritten,
        )
    except BaseException as exc:  # noqa: BLE001
        return SandboxResult(
            verdict=Verdict.ERROR,
            reason=f"{Reason.INTERNAL.value}:{type(exc).__name__}",
            expr_digest=digest,
            expr_len=length,
            normalized=rewritten,
        )

    result = _classify(payload, digest, length, int((time.perf_counter() - started) * 1000), rewritten)
    if use_cache and result.verdict in _CACHEABLE:
        _remember(key, result)
    return result


def compare(
    left: str,
    right: str,
    *,
    symbols: frozenset[str] | None = None,
    rel_tol: float | None = None,
) -> Comparison:
    """数值判等。

    **三层保护**（ADR-016）：① `is_numeric` 要求 `ok` 且有限；② 结果再断言 `isfinite`；
    ③ 任何一侧失败 → `equal=None`（"判不了"），**绝不**返回 `False`（"不相等"）。
    """
    left_result = evaluate(left, symbols=symbols)
    right_result = evaluate(right, symbols=symbols)
    tolerance = config.REL_TOL if rel_tol is None else rel_tol

    if not left_result.is_numeric or not right_result.is_numeric:
        blocked = left_result if not left_result.is_numeric else right_result
        return Comparison(
            equal=None,
            left=left_result,
            right=right_result,
            reason=f"not_comparable:{blocked.verdict.value}:{blocked.reason or 'none'}",
        )

    lvalue = left_result.value
    rvalue = right_result.value
    if lvalue is None or rvalue is None or not (math.isfinite(lvalue) and math.isfinite(rvalue)):
        return Comparison(equal=None, left=left_result, right=right_result, reason="non_finite_guard")

    difference = abs(lvalue - rvalue)
    scale = max(1.0, abs(lvalue), abs(rvalue))
    equal = difference <= tolerance * scale
    if not math.isfinite(difference):  # 理论上已被上面的守卫拦住
        return Comparison(equal=None, left=left_result, right=right_result, reason="non_finite_difference")
    return Comparison(equal=equal, left=left_result, right=right_result)


__all__ = [
    "FUNCTION_NAMES",
    "SYMBOL_NAMES",
    "clear_cache",
    "compare",
    "evaluate",
    "expr_digest",
]
