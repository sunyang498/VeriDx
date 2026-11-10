"""沙箱验收：87 条样本（6 组 / K1–K21）+ 机制级用例。

分组口径（内容 v3.2）：

| 组 | 条数 | 判据 |
|---|---|---|
| A 攻击 | 20 | 全部 `rejected`，且 reason ∈ `CAPABILITY_REASONS` |
| B 正常 | 30 | 全部 `ok` |
| C 合法不支持 | 15 | 归类正确，**不计入 M17** |
| D 值域边界 | 10 | 绝不判 `ok` |
| E 输入形态 | 8 | 永不抛 |
| F 运行时故障 | 4 | `timeout`/`error`，不进 rejected |
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from veridx import config
from veridx.sandbox import (
    CAPABILITY_REASONS,
    Reason,
    SandboxResult,
    Verdict,
    clear_cache,
    compare,
    evaluate,
    ledger,
    normalize_complex_literal,
    reason_prefix,
    rejection_rate,
    shutdown_pool,
    static_check,
)
from veridx.sandbox.samples import (
    ATTACKS,
    DOMAIN,
    INPUT_FORMS,
    NON_COMPARABLE_PAIRS,
    NORMAL,
    TOLERANCE_PAIRS,
    UNSUPPORTED,
    group_counts,
)
from veridx.store import db as store_db
from veridx.store.repository import Repository


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection, _ = store_db.init_db(tmp_path / "sandbox.db", config.MIGRATIONS_DIR)
    yield connection
    connection.close()


def _label(sample) -> str:  # noqa: ANN001 - 参数化 id
    text = sample.expr if isinstance(sample.expr, str) else repr(sample.expr)
    return f"{sample.k}-{text[:22]}"


# --- 样本集规模（防止"顺手删两条"）-----------------------------------------
def test_group_counts_match_design() -> None:
    counts = group_counts()
    assert counts["A_attacks"] == 20
    assert counts["B_normal"] == 30
    assert counts["C_unsupported"] == 15
    assert counts["D_domain"] == 10
    assert counts["E_input_forms"] == 8
    assert counts["F_runtime_faults"] == 4
    assert counts["total"] == 87


# --- A 攻击：逃逸 = 0，且必须记在能力集合里 ---------------------------------
@pytest.mark.parametrize("sample", ATTACKS, ids=_label)
def test_attack_is_rejected_and_capability_classified(sample) -> None:  # noqa: ANN001
    result = evaluate(sample.expr)
    assert result.verdict is Verdict.REJECTED, f"{sample.why} -> {result.verdict}:{result.reason}"
    assert reason_prefix(result.reason) in CAPABILITY_REASONS, "rejected 必须落在能力集合内"
    assert result.reason.startswith(sample.reason_prefix), sample.why


# --- B 正常：误杀 = 0 -------------------------------------------------------
@pytest.mark.parametrize("sample", NORMAL, ids=_label)
def test_normal_expression_is_ok(sample) -> None:  # noqa: ANN001
    result = evaluate(sample.expr)
    assert result.verdict is Verdict.OK, f"{sample.why} -> {result.verdict}:{result.reason}"
    assert result.value is not None


# --- C 合法不支持：归类正确且不计入 M17 -------------------------------------
@pytest.mark.parametrize("sample", UNSUPPORTED, ids=_label)
def test_legal_but_unsupported(sample) -> None:  # noqa: ANN001
    result = evaluate(sample.expr)
    assert result.verdict is Verdict.UNSUPPORTED, f"{sample.why} -> {result.verdict}:{result.reason}"
    assert result.reason.startswith(sample.reason_prefix), sample.why
    assert reason_prefix(result.reason) not in CAPABILITY_REASONS


# --- D 值域边界：绝不 ok ----------------------------------------------------
@pytest.mark.parametrize("sample", DOMAIN, ids=_label)
def test_domain_boundary_is_never_ok(sample) -> None:  # noqa: ANN001
    result = evaluate(sample.expr)
    assert result.verdict is Verdict.UNSUPPORTED, f"{sample.why} -> {result.verdict}:{result.reason}"
    assert result.reason.startswith(sample.reason_prefix), sample.why


# --- E 输入形态：永不抛 -----------------------------------------------------
@pytest.mark.parametrize("sample", INPUT_FORMS, ids=_label)
def test_input_form_never_raises(sample) -> None:  # noqa: ANN001
    result = evaluate(sample.expr)
    assert isinstance(result, SandboxResult)
    assert result.verdict is Verdict.UNSUPPORTED, sample.why
    assert result.reason.startswith(sample.reason_prefix), sample.why


# --- F 运行时故障（注入）----------------------------------------------------
def test_compute_timeout_is_timeout_and_worker_is_replaced() -> None:
    # 预算 0 秒：deadline 立刻过期 → 确定的超时路径（不依赖机器快慢）
    slow = evaluate("sqrt(2)", timeout_seconds=0.0, use_cache=False)
    assert slow.verdict is Verdict.TIMEOUT, f"{slow.verdict}:{slow.reason}"
    assert slow.reason == Reason.EVAL_TIMEOUT.value

    healthy = evaluate("sqrt(2)", use_cache=False)
    assert healthy.verdict is Verdict.OK, "毒丸之后必须能重建 worker"


def test_child_failure_becomes_error_not_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from veridx.sandbox import core

    class Boom:
        def run(self, *_a, **_k):
            raise OSError("pipe broken")

    monkeypatch.setattr(core.pool_module, "get_pool", lambda: Boom())
    result = evaluate("1+1", use_cache=False)
    assert result.verdict is Verdict.ERROR
    assert reason_prefix(result.reason) in {Reason.SPAWN_FAILED.value, Reason.INTERNAL.value}


def test_spawn_failure_is_degraded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    from veridx.sandbox import pool as pool_module

    def boom(self):  # noqa: ANN001
        raise pool_module.WorkerUnavailable("no __main__ guard")

    monkeypatch.setattr(pool_module.WorkerPool, "_acquire", boom)
    shutdown_pool()
    result = evaluate("1+1", use_cache=False)
    assert result.verdict is Verdict.ERROR
    assert result.reason.startswith(Reason.SPAWN_FAILED.value)
    shutdown_pool()


# --- 容差与非判等 -----------------------------------------------------------
@pytest.mark.parametrize(("left", "right", "expected"), TOLERANCE_PAIRS)
def test_tolerance_pairs(left: str, right: str, expected: bool) -> None:
    assert compare(left, right).equal is expected


@pytest.mark.parametrize(("left", "right", "why"), NON_COMPARABLE_PAIRS)
def test_non_comparable_returns_none_not_false(left: str, right: str, why: str) -> None:
    """`None`（判不了）与 `False`（不相等）是两件事——混用会直接制造误判。"""
    assert compare(left, right).equal is None, why


def test_inf_never_compares_equal() -> None:
    """历史 bug：容差公式在 inf 上退化成 `inf <= inf` 恒真。"""
    result = compare("2**1e9", "1e308")
    assert result.equal is None
    assert "non_finite" in result.reason or "not_comparable" in result.reason


# --- 机制级 ----------------------------------------------------------------
def test_bitxor_and_shift_rejected() -> None:
    assert evaluate("x^2").reason == "forbidden_syntax:BitXor"
    assert evaluate("1<<2").reason == "forbidden_syntax:LShift"


def test_mod_and_floordiv_allowed() -> None:
    assert evaluate("37%5").value == pytest.approx(2.0)
    assert evaluate("7//2").value == pytest.approx(3.0)


def test_nested_pow_rejected_by_structure_not_value() -> None:
    assert evaluate("1**1**1**1").reason == Reason.NESTED_POW.value


def test_function_as_value_is_unsupported_not_rejected() -> None:
    result = evaluate("sqrt")
    assert result.verdict is Verdict.UNSUPPORTED
    assert result.reason == "function_as_value:sqrt"
    assert result.exact is None, "函数对象不能进 exact（会泄漏地址且不可复现）"


def test_dunder_name_is_rejected() -> None:
    assert evaluate("__builtins__").reason == "forbidden_name:__builtins__"


def test_unknown_symbol_is_unsupported_not_rejected() -> None:
    assert evaluate("foo+1").verdict is Verdict.UNSUPPORTED
    assert evaluate("foo+1").reason == "unknown_symbol:foo"


def test_oversize_is_unsupported_and_hard_cap_is_rejected() -> None:
    long_but_legal = "+".join(["x"] * 179)  # 357 字符
    assert evaluate(long_but_legal).reason.startswith(Reason.OVERSIZE.value)
    assert evaluate("1" + "+1" * 3000).verdict is Verdict.REJECTED


def test_symbol_table_restriction() -> None:
    assert evaluate("n+1").verdict is Verdict.UNSUPPORTED
    strict = evaluate("n+1", symbols=frozenset({"x", "y"}))
    assert strict.reason == "unknown_symbol:n"


def test_static_check_contract() -> None:
    assert static_check("1/6") is None
    verdict, reason = static_check("open('a')")
    assert verdict is Verdict.REJECTED and reason.startswith("forbidden_call")


# --- R3 归一化 --------------------------------------------------------------
@pytest.mark.parametrize(
    ("literal", "normalized"),
    [("1j", "1*I"), ("2.5j", "2.5*I"), ("1+2j", "1+2*I"), ("1e3j", "1e3*I")],
)
def test_complex_literal_normalized(literal: str, normalized: str) -> None:
    assert normalize_complex_literal(literal) == normalized


def test_complex_literal_and_explicit_form_agree() -> None:
    """I5：同一数学对象的两种写法必须同 verdict、同值。"""
    from_literal = evaluate("1j", use_cache=False)
    from_explicit = evaluate("1*I", use_cache=False)
    assert from_literal.verdict is from_explicit.verdict is Verdict.UNSUPPORTED
    assert from_literal.reason == from_explicit.reason == Reason.COMPLEX_RESULT.value
    assert from_literal.normalized == "1*I"


def test_normalization_does_not_touch_scientific_notation() -> None:
    """R4 延后期间的回归保护：科学计数法绝不能被改写。"""
    for expr, expected in (("1e-5", 1e-05), ("1e+5", 100000.0), (".5e3", 500.0), ("1e-3", 0.001)):
        assert normalize_complex_literal(expr) is None
        result = evaluate(expr)
        assert result.verdict is Verdict.OK
        assert result.value == pytest.approx(expected)


def test_cache_key_uses_normalized_expression() -> None:
    """`1j` 与 `1*I` 命中同一条缓存；原式只进审计字段。"""
    first = evaluate("1j")
    second = evaluate("1*I")
    assert first == second
    assert first.normalized == "1*I"


# --- 台账 ------------------------------------------------------------------
def test_ledger_records_non_ok_only(conn: sqlite3.Connection) -> None:
    assert ledger.record_verdict(conn, run_id="r-1", result=evaluate("9**9**9")) is not None
    assert ledger.record_verdict(conn, run_id="r-1", result=evaluate("2<x<3")) is not None
    assert ledger.record_verdict(conn, run_id="r-1", result=evaluate("1/6")) is None


def test_rejection_rate_keys_and_m17_semantics(conn: sqlite3.Connection) -> None:
    ledger.record_verdict(conn, run_id="r-1", result=evaluate("globals()"))
    ledger.record_verdict(conn, run_id="r-1", result=evaluate("2<x<3"))
    stats = rejection_rate(conn, run_id="r-1")

    assert set(stats) >= {
        "total_non_ok",
        "counts",
        "capability_count",
        "capability_share",
        "reasons_by_verdict",
        "capability_reasons",
    }
    assert "shared_rejected" not in stats, "键名曾写错成 shared_rejected"
    assert stats["capability_count"] == 1, "unsupported 不算能力型"
    assert stats["reasons_by_verdict"]["rejected"] == {"forbidden_call:globals": 1}


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
