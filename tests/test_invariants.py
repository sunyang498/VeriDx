"""不变量测试（I1–I8 + 并发/回收）。

样本证明"这 87 条对"；**不变量才证明"任何输入都对"**。
这是审批意见 §5 新增 7 项要求的落地处：

* +1 I4 的 rejected 原因集**不含** function_as_value / oversize / unknown_symbol
* +2 `CAPABILITY_REASONS` 是唯一常量，且有测试断言它与 I4 一致
* +3 并发安全：2 线程 × 20 次求值，结果与串行一致
* +4 退出回收：结束后 worker 进程数归零
* +6 `1e-5` 仍为 `1e-05`
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from veridx import config
from veridx.sandbox import (
    CAPABILITY_REASONS,
    ERROR_REASONS,
    UNSUPPORTED_REASONS,
    Verdict,
    clear_cache,
    compare,
    evaluate,
    get_pool,
    reason_prefix,
    shutdown_pool,
)
from veridx.sandbox.samples import ATTACKS, DOMAIN, NORMAL, UNSUPPORTED

#: 非能力原因：绝不能出现在 rejected 里
NON_CAPABILITY_REASONS = frozenset(
    {"function_as_value", "oversize", "unknown_symbol", "non_finite", "complex_result", "undefined"}
)


@pytest.fixture(autouse=True)
def _fresh() -> None:
    clear_cache()
    yield
    clear_cache()


# --- I1 ----------------------------------------------------------------------
def test_i1_ok_implies_finite_value() -> None:
    """任何 `ok` 的结果，value 必须是有限实数。"""
    for sample in NORMAL + DOMAIN:
        result = evaluate(sample.expr)
        if result.verdict is Verdict.OK:
            assert result.value is not None, sample.expr
            assert math.isfinite(result.value), f"{sample.expr!r} 得到非有限值 {result.value}"


# --- I2 ----------------------------------------------------------------------
def test_i2_equal_true_requires_both_sides_numeric() -> None:
    """`equal=True` 只能发生在两侧都是 `ok` 且有限的时候。"""
    pairs = [
        ("1/3", "0.333333"),
        ("2**1e9", "1e308"),
        ("0/0", "0"),
        ("I", "0"),
        ("x+1", "1"),
        ("1e400", "1e308"),
    ]
    for left, right in pairs:
        if compare(left, right).equal is True:
            left_result = evaluate(left)
            right_result = evaluate(right)
            assert left_result.is_numeric and right_result.is_numeric, (left, right)


# --- I3 ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [None, 123, b"1+1", {}, [], "（", "1+", "\x00", "😀", 3.14, True, object(), "", "   ", ")" * 300],
)
def test_i3_evaluate_never_raises(bad: object) -> None:
    result = evaluate(bad, use_cache=False)
    assert result.verdict in set(Verdict), f"{bad!r} 返回了非法 verdict"


# --- I4 + 审批 +1/+2 ---------------------------------------------------------
def test_i4_rejected_iff_capability() -> None:
    """`rejected` ⟺ reason ∈ `CAPABILITY_REASONS`（双向）。这是 M17 的唯一口径。

    单向断言（只查 rejected ⊆ capability）挡不住"把 oversize 塞进 rejected"，
    因为攻击样本里没有 oversize —— 必须双向，且覆盖**全部**样本组。
    """
    from veridx.sandbox.samples import ALL_SAMPLES

    for sample in ALL_SAMPLES:
        result = evaluate(sample.expr)
        prefix = reason_prefix(result.reason) if result.reason else ""
        if result.verdict is Verdict.REJECTED:
            assert prefix in CAPABILITY_REASONS, (
                f"{sample.expr!r} 被 rejected 但原因不是能力型：{result.reason}"
            )
        elif prefix in CAPABILITY_REASONS:
            assert result.verdict is Verdict.REJECTED, (
                f"{sample.expr!r} 是能力型原因却不是 rejected：{result.verdict}:{result.reason}"
            )


def test_capability_reasons_excludes_benign_causes() -> None:
    """审批 +1：function_as_value / oversize / unknown_symbol 绝不能算能力型。"""
    assert CAPABILITY_REASONS.isdisjoint(NON_CAPABILITY_REASONS)
    assert CAPABILITY_REASONS.isdisjoint(UNSUPPORTED_REASONS)
    assert CAPABILITY_REASONS.isdisjoint(ERROR_REASONS)


def test_reason_sets_partition_the_enum() -> None:
    """三组原因集合互不重叠，且覆盖全部已知原因（新增 Reason 必须归组）。"""
    from veridx.sandbox.verdict import Reason

    all_reasons = {r.value for r in Reason}
    classified = (
        CAPABILITY_REASONS
        | UNSUPPORTED_REASONS
        | ERROR_REASONS
        | {
            "eval_timeout"  # timeout 单独一类
        }
    )
    assert all_reasons == classified, f"未归类的原因：{all_reasons - classified}"


# --- I5 ----------------------------------------------------------------------
def test_i5_same_math_object_same_verdict() -> None:
    """同一数学对象的两种写法必须同 verdict 类别。"""
    pairs = [
        ("1j", "1*I"),
        ("2.5j", "2.5*I"),
        ("Abs(3+4j)", "Abs(3+4*I)"),
    ]
    for literal, explicit in pairs:
        a = evaluate(literal, use_cache=False)
        b = evaluate(explicit, use_cache=False)
        assert a.verdict is b.verdict, (literal, a, b)
    assert evaluate("1j").reason == evaluate("1*I").reason == "complex_result"


# --- I6 ----------------------------------------------------------------------
def test_i6_every_result_has_parseable_reason() -> None:
    known = CAPABILITY_REASONS | UNSUPPORTED_REASONS | ERROR_REASONS | {"eval_timeout"}
    for sample in ATTACKS + NORMAL + UNSUPPORTED + DOMAIN:
        result = evaluate(sample.expr)
        if result.verdict is Verdict.OK:
            assert result.reason == "", sample.expr
            continue
        assert reason_prefix(result.reason) in known, f"{sample.expr!r} -> {result.reason!r}"


# --- I8 ----------------------------------------------------------------------
def test_i8_cache_does_not_change_verdict() -> None:
    clear_cache()
    first = evaluate("e-1")
    second = evaluate("e-1")
    assert first == second
    clear_cache()
    third = evaluate("e-1")
    assert third.verdict is first.verdict and third.value == pytest.approx(first.value or 0)


# --- 审批 +6：R4 延后期间的回归保护 ------------------------------------------
def test_scientific_notation_is_never_rewritten() -> None:
    for expr, expected in (("1e-5", 1e-05), ("1e+5", 100000.0), (".5e3", 500.0)):
        result = evaluate(expr)
        assert result.verdict is Verdict.OK
        assert result.value == pytest.approx(expected), f"{expr} 被改写了"


# --- 审批 +3：并发安全 ------------------------------------------------------
def test_concurrent_evaluation_matches_serial() -> None:
    exprs = [sample.expr for sample in NORMAL[:10]] * 2
    serial = [evaluate(expr, use_cache=False).value for expr in exprs]
    clear_cache()

    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(lambda e: evaluate(e, use_cache=False).value, exprs))

    assert len(parallel) == len(serial)
    for got, want in zip(parallel, serial, strict=True):
        assert got == pytest.approx(want or 0)


def test_concurrent_sandbox_rejects_do_not_cross_talk() -> None:
    """两个线程同时跑不同表达式，结果不能串。"""
    jobs = [("1/6", 1 / 6), ("sqrt(2)", math.sqrt(2)), ("2**10", 1024.0), ("3.14", 3.14)] * 5
    errors: list[str] = []

    def worker(expr: str, expected: float) -> None:
        try:
            value = evaluate(expr, use_cache=False).value
            if value is None or abs(value - expected) > 1e-9:
                errors.append(f"{expr} -> {value} != {expected}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(f"{expr} raised {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=job) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


# --- 审批 +4：退出回收 ------------------------------------------------------
def test_worker_pool_releases_processes_on_shutdown() -> None:
    evaluate("1+1", use_cache=False)
    pool = get_pool()
    processes = pool.live_processes()
    assert processes, "求值后应至少有一个 worker"

    shutdown_pool()
    for process in processes:
        process.join(config.SANDBOX_TERMINATE_GRACE_SECONDS)
        assert not process.is_alive(), f"worker 未被回收：pid={process.pid}"
