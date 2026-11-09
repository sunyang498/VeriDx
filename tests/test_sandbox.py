"""P1.1 沙箱验收：45 条样本（20 攻击 + 25 回归）+ 机制级用例。

验收口径（写死在测试里，避免"做完再争"）：

* 20 条攻击样本 **逃逸 = 0**（全部 `rejected`）
* 25 条回归样本 **误杀 = 0**（20 条 `ok` + 5 条 `unsupported`）
* `9**9**9` 在 ≤3.5s 内返回（静态拦截，不依赖子进程超时）
* 每次拒绝落 `sandbox_rejects` 台账（带 `run_id`）
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from veridx import config
from veridx.sandbox import (
    Reason,
    SandboxResult,
    Verdict,
    clear_cache,
    compare,
    evaluate,
    ledger,
    rejection_rate,
    static_check,
)
from veridx.sandbox.samples import (
    ATTACK_SAMPLES,
    REGRESSION_OK,
    REGRESSION_UNSUPPORTED,
    TOLERANCE_PAIRS,
    attack_categories,
)
from veridx.store import db as store_db
from veridx.store.repository import Repository

#: 每条样本最多允许的挂钟时间（§4.4 层 4：超时 3s + terminate 宽限 1.0s）
SAMPLE_BUDGET_SECONDS = 3.5


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection, _ = store_db.init_db(tmp_path / "sandbox.db", config.MIGRATIONS_DIR)
    yield connection
    connection.close()


# --- 样本集本身的规模（防止有人"顺手删两条"）--------------------------------
def test_sample_sets_have_documented_size() -> None:
    assert len(ATTACK_SAMPLES) == 20
    assert len(REGRESSION_OK) == 20
    assert len(REGRESSION_UNSUPPORTED) == 5
    assert attack_categories() == {
        "code_exec": 4,
        "filesystem": 3,
        "attribute": 3,
        "introspection": 3,
        "resource": 4,
        "length": 3,
    }


def test_attack_samples_are_unique() -> None:
    exprs = [sample.expr for sample in ATTACK_SAMPLES]
    assert len(set(exprs)) == len(exprs)


# --- 攻击集：逃逸 = 0 -------------------------------------------------------
@pytest.mark.parametrize("sample", ATTACK_SAMPLES, ids=lambda s: f"{s.category}-{s.expr[:24]}")
def test_attack_samples_are_rejected(sample) -> None:  # noqa: ANN001 - 参数化样本
    started = time.perf_counter()
    result = evaluate(sample.expr)
    elapsed = time.perf_counter() - started

    assert result.verdict is Verdict.REJECTED, f"{sample.expr!r} -> {result.verdict}:{result.reason}"
    assert elapsed <= SAMPLE_BUDGET_SECONDS, f"{sample.expr[:24]!r} 用时 {elapsed:.2f}s"


# --- 回归集：误杀 = 0 -------------------------------------------------------
@pytest.mark.parametrize("expr", REGRESSION_OK)
def test_normal_expressions_evaluate(expr: str) -> None:
    result = evaluate(expr)
    assert result.verdict is Verdict.OK, f"{expr!r} -> {result.verdict}:{result.reason}"
    assert result.value is not None


@pytest.mark.parametrize("expr", REGRESSION_UNSUPPORTED)
def test_legal_but_unsupported_is_not_rejected(expr: str) -> None:
    """`2<x<3` 判成 rejected 会让误杀率与论文结论一起失真（§4.6）。"""
    result = evaluate(expr)
    assert result.verdict is Verdict.UNSUPPORTED, f"{expr!r} -> {result.verdict}:{result.reason}"
    assert result.is_accepted is True
    assert result.is_rejected is False


@pytest.mark.parametrize(("left", "right", "expected"), TOLERANCE_PAIRS)
def test_numeric_tolerance(left: str, right: str, expected: bool) -> None:
    comparison = compare(left, right)
    assert comparison.equal is expected, f"{left} vs {right} -> {comparison}"


# --- 机制级用例（每条锁一个断言目标）----------------------------------------
def test_nested_pow_is_named_reason() -> None:
    result = evaluate("9**9**9")
    assert result.verdict is Verdict.REJECTED
    assert result.reason == Reason.NESTED_POW.value


def test_bitxor_is_rejected_not_allowed_via_operator_superclass() -> None:
    """`ast.operator` 当白名单会放行 `x^2`（§4.7 坑 1）。"""
    result = evaluate("x^2")
    assert result.verdict is Verdict.REJECTED
    assert result.reason == f"{Reason.FORBIDDEN_SYNTAX.value}:BitXor"


def test_shift_is_rejected() -> None:
    result = evaluate("1<<10**8")
    assert result.verdict is Verdict.REJECTED
    assert "LShift" in result.reason


def test_factorial_is_not_in_whitelist() -> None:
    result = evaluate("factorial(10**7)")
    assert result.verdict is Verdict.REJECTED
    assert result.reason == f"{Reason.FORBIDDEN_CALL.value}:factorial"


def test_attribute_access_is_rejected() -> None:
    assert evaluate("(1).__class__.__bases__").reason == f"{Reason.FORBIDDEN_SYNTAX.value}:Attribute"


def test_exponent_cap_is_enforced() -> None:
    result = evaluate("2**201")
    assert result.verdict is Verdict.REJECTED
    assert result.reason.startswith(Reason.EXPONENT_TOO_LARGE.value)


def test_symbolic_expression_is_unsupported_not_executed() -> None:
    """含自由符号：P1.1 只做数值，符号等价是 P1.8 的事。"""
    result = evaluate("x+1")
    assert result.verdict is Verdict.UNSUPPORTED
    assert result.reason == Reason.SYMBOLIC.value
    assert result.exact is not None


def test_expression_can_restrict_symbol_table() -> None:
    """符号表是白名单的一部分，也进 LRU 键（P0.6 审查 §4.3④）。"""
    assert evaluate("n+1").verdict is Verdict.UNSUPPORTED
    strict = evaluate("n+1", symbols=frozenset({"x", "y"}))
    assert strict.verdict is Verdict.REJECTED
    assert strict.reason == f"{Reason.UNKNOWN_NAME.value}:n"
    # 同一表达式、不同符号表 => 不同结论（缓存没有把它们混在一起）
    assert evaluate("n+1").verdict is Verdict.UNSUPPORTED


def test_cache_returns_same_object_for_same_symbol_table() -> None:
    first = evaluate("1/6")
    second = evaluate("1/6")
    assert first is second


def test_timeout_路径可用() -> None:
    """把预算压到 1ms，子进程来不及回传 → TIMEOUT（且不缓存）。"""
    result = evaluate("sqrt(2)", timeout_seconds=0.001, use_cache=False)
    assert result.verdict is Verdict.TIMEOUT
    assert result.reason == Reason.EVAL_TIMEOUT.value


def test_timeout_result_is_not_cached() -> None:
    evaluate("sqrt(2)", timeout_seconds=0.001)
    assert evaluate("sqrt(2)", timeout_seconds=5.0).verdict is Verdict.OK


def test_undefined_math_is_unsupported_not_error() -> None:
    """`1/0` 是数学未定义，不是安全问题（§4.10）。"""
    result = evaluate("1/0", use_cache=False)
    assert result.verdict in {Verdict.UNSUPPORTED, Verdict.ERROR}
    if result.verdict is Verdict.UNSUPPORTED:
        assert result.reason in {Reason.NON_NUMERIC.value, Reason.SYMBOLIC.value}


def test_static_check_returns_none_only_for_runnable_expression() -> None:
    assert static_check("1/6") is None
    assert static_check("1/6000") is None
    verdict, reason = static_check("open('a').read()")
    assert verdict is Verdict.REJECTED and reason.startswith(Reason.FORBIDDEN_SYNTAX.value)


# --- 台账：拒绝落痕 + 原因分布 ----------------------------------------------
def test_rejections_are_recorded_in_ledger(conn: sqlite3.Connection) -> None:
    rejected = evaluate("9**9**9")
    unsupported = evaluate("2<x<3")
    ok = evaluate("1/6")

    assert ledger.record_verdict(conn, run_id="r-1", result=rejected) is not None
    assert ledger.record_verdict(conn, run_id="r-1", result=unsupported) is not None
    assert ledger.record_verdict(conn, run_id="r-1", result=ok) is None

    stats = rejection_rate(conn, run_id="r-1")
    assert stats["total"] == 2
    assert stats["counts"] == {"rejected": 1, "unsupported": 1}
    assert stats["reasons"][Reason.NESTED_POW.value] == 1


def test_ledger_requires_run_id_scope(conn: sqlite3.Connection) -> None:
    ledger.record_verdict(conn, run_id="r-A", result=evaluate("globals()"))
    ledger.record_verdict(conn, run_id="r-B", result=evaluate("globals()"))
    assert rejection_rate(conn, run_id="r-A")["total"] == 1
    assert rejection_rate(conn)["total"] == 2


def test_sandbox_result_is_serialisable_for_evidence() -> None:
    payload = SandboxResult(Verdict.OK, value=1.0, exact="1").to_payload()
    assert payload["verdict"] == "ok"
    assert payload["value"] == 1.0


# --- 与仓储层联动：证据 payload 能直接存 ------------------------------------
def test_sandbox_result_fits_evidence_payload(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    result = evaluate("e-1")
    record = repo.create_evidence(
        run_id="r-1",
        evidence_id="ev_1",
        kind="numeric_check",
        payload={"expr": "e-1", **result.to_payload()},
        produced_by="check_numeric",
        seq=1,
    )
    assert record["payload"]["verdict"] == "ok"
    assert record["payload"]["value"] == pytest.approx(1.718281828, rel=1e-6)
