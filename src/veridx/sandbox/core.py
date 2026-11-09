"""沙箱对外入口：静态判定（1–3 层）+ 子进程求值（第 4 层）+ LRU。

**超时预算**（P0.6 审查 §4.3②，值都在 `config.py`）：

```
单次沙箱求值         ≤ SANDBOX_TIMEOUT_SECONDS (3.0s)
单题本地求值次数     ≤ LOCAL_TOOL_MAX_CALLS   (3)
  → 沙箱最坏总时长   ≤ 9s
单题 wall-clock      ≤ WALL_CLOCK_SECONDS     (180s)
单题 P95 目标        ≤ 25s（NFR-1）
  → 7 次 LLM 调用 + 9s 沙箱仍留有余量；真正的硬闸门是 180s 的 wall-clock
```

`evaluate()` 只做**数值**求值：表达式含自由符号时返回 `unsupported:symbolic`（而不是硬算）。
符号等价（`(x-1)e^x` vs `e^x(x-1)`）是 P1.8 步骤定位的职责，不在这一层假装能做。
"""

from __future__ import annotations

import hashlib
import multiprocessing
import time
from collections import OrderedDict

from veridx import config
from veridx.sandbox import _worker
from veridx.sandbox.guard import SYMBOL_NAMES, static_check
from veridx.sandbox.verdict import Comparison, Reason, SandboxResult, Verdict

#: 缓存键 = (表达式, 符号表)。**符号表必须进键**：同一个算式在不同符号集下合法性不同
_CACHE: OrderedDict[tuple[str, tuple[str, ...]], SandboxResult] = OrderedDict()
_CACHEABLE = frozenset({Verdict.OK, Verdict.UNSUPPORTED, Verdict.REJECTED})


def expr_digest(expr: str) -> str:
    return hashlib.sha256(expr.encode("utf-8")).hexdigest()[:16]


def _normalize_symbols(symbols: frozenset[str] | None) -> tuple[str, ...]:
    return tuple(sorted(symbols)) if symbols is not None else tuple(sorted(SYMBOL_NAMES))


def _remember(key: tuple[str, tuple[str, ...]], result: SandboxResult) -> None:
    _CACHE[key] = result
    _CACHE.move_to_end(key)
    while len(_CACHE) > config.SANDBOX_LRU_SIZE:
        _CACHE.popitem(last=False)


def clear_cache() -> None:
    _CACHE.clear()


def _run_child(expr: str, symbols: tuple[str, ...], timeout_seconds: float) -> SandboxResult:
    context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(target=_worker.main, args=(expr, list(symbols), child_conn), daemon=True)

    started = time.perf_counter()
    payload: dict[str, object] | None = None
    try:
        process.start()
        child_conn.close()
        deadline = started + timeout_seconds
        while time.perf_counter() < deadline:
            if parent_conn.poll(0.02):
                payload = parent_conn.recv()
                break
            if not process.is_alive():
                break
    finally:
        if process.is_alive():
            process.terminate()
        process.join(config.SANDBOX_TERMINATE_GRACE_SECONDS)
        if process.is_alive():  # pragma: no cover - terminate 理论上一定生效
            process.kill()
            process.join(config.SANDBOX_TERMINATE_GRACE_SECONDS)
        parent_conn.close()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    digest = expr_digest(expr)
    length = len(expr)

    if payload is None:
        if elapsed_ms >= timeout_seconds * 1000:
            return SandboxResult(
                verdict=Verdict.TIMEOUT,
                reason=Reason.EVAL_TIMEOUT.value,
                expr_digest=digest,
                expr_len=length,
                elapsed_ms=elapsed_ms,
            )
        return SandboxResult(
            verdict=Verdict.ERROR,
            reason=f"{Reason.EVAL_ERROR.value}:child_exited",
            expr_digest=digest,
            expr_len=length,
            elapsed_ms=elapsed_ms,
        )

    if not payload.get("ok"):
        return SandboxResult(
            verdict=Verdict.ERROR,
            reason=f"{Reason.EVAL_ERROR.value}:{payload.get('error', '')}",
            expr_digest=digest,
            expr_len=length,
            elapsed_ms=elapsed_ms,
        )

    exact = str(payload["exact"]) if payload.get("exact") is not None else None
    value = payload.get("value")
    if value is None:
        verdict = Verdict.UNSUPPORTED
        reason = Reason.SYMBOLIC.value if payload.get("free_symbols") else Reason.NON_NUMERIC.value
        return SandboxResult(
            verdict=verdict,
            reason=reason,
            expr_digest=digest,
            expr_len=length,
            elapsed_ms=elapsed_ms,
            exact=exact,
        )

    return SandboxResult(
        verdict=Verdict.OK,
        expr_digest=digest,
        expr_len=length,
        elapsed_ms=elapsed_ms,
        value=float(value),  # type: ignore[arg-type]
        exact=exact,
    )


def evaluate(
    expr: str,
    *,
    symbols: frozenset[str] | None = None,
    timeout_seconds: float | None = None,
    use_cache: bool = True,
) -> SandboxResult:
    """判定 + 求值。返回值永远是 5 种 verdict 之一，**不抛异常**。"""
    normalized = _normalize_symbols(symbols)
    key = (expr if isinstance(expr, str) else repr(expr), normalized)

    if use_cache:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached

    static = static_check(expr, frozenset(normalized))
    if static is not None:
        verdict, reason = static
        result = SandboxResult(
            verdict=verdict,
            reason=reason,
            expr_digest=expr_digest(key[0]),
            expr_len=len(key[0]),
        )
    else:
        result = _run_child(key[0], normalized, timeout_seconds or config.SANDBOX_TIMEOUT_SECONDS)

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
    """数值判等。`equal is None` = 无法判定（**不是**不相等）。"""
    left_result = evaluate(left, symbols=symbols)
    right_result = evaluate(right, symbols=symbols)
    tolerance = config.REL_TOL if rel_tol is None else rel_tol

    if not left_result.is_numeric or not right_result.is_numeric:
        verdict = left_result.verdict if left_result.verdict is not Verdict.OK else right_result.verdict
        return Comparison(
            equal=None, left=left_result, right=right_result, reason=f"not_numeric:{verdict.value}"
        )

    assert left_result.value is not None and right_result.value is not None
    equal = abs(left_result.value - right_result.value) <= tolerance * max(
        1.0, abs(left_result.value), abs(right_result.value)
    )
    return Comparison(equal=equal, left=left_result, right=right_result)
