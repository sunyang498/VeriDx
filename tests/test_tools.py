"""P1.2 工具层验收。

验收口径（P1.2 卡 + 审批 §5 +8/+9）：

* `dispatch` 只认白名单，编一个工具名无效；
* 4 个工具 × 3 种坏参数 → 全部返回结构化 `tool_error`，**无一抛出**；
* `check_numeric` 产出的 evidence 能被 `Repository.create_evidence` 接收，`seq` 由 run 上下文发号；
* 沙箱非 `ok` 结果**同时**写台账与回灌 evidence；
* 证据查询/写入一律带 `run_id`。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from veridx import config
from veridx.sandbox import clear_cache, shutdown_pool
from veridx.sandbox.verdict import Verdict
from veridx.store import db as store_db
from veridx.tools import ToolContext, ToolName, dispatch
from veridx.tools.spec import TOOL_SCHEMAS

BAD_PARAMS: dict[str, list[dict]] = {
    ToolName.CHECK_NUMERIC: [
        {},
        {"expr": "1", "value": None},
        {"expr": "   ", "value": "1"},
        {"expr": 123, "value": "1"},
    ],
    ToolName.STEP_DIFF: [
        {},
        {"model_steps": [], "student_steps": ["a"]},
        {"model_steps": [1], "student_steps": ["a"]},
        {"model_steps": ["a"], "student_steps": None},
    ],
    ToolName.KB_LOOKUP: [
        {},
        {"kp_ids": []},
        {"kp_ids": [None]},
        {"kp_ids": "kp_1"},
    ],
    ToolName.ESCALATE: [
        {},
        {"reason_code": None},
        {"reason_code": ""},
        {"reason_code": "user_cancel"},  # 少个 ed：手写漂移
    ],
}


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()
    shutdown_pool()


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection, _ = store_db.init_db(tmp_path / "tools.db", config.MIGRATIONS_DIR)
    yield connection
    connection.close()


def _ctx(conn: sqlite3.Connection | None = None) -> ToolContext:
    return ToolContext(run_id="r-test", conn=conn)


# --- 白名单 ----------------------------------------------------------------
def test_only_whitelisted_tools_exist() -> None:
    assert {schema["function"]["name"] for schema in TOOL_SCHEMAS} == {
        ToolName.CHECK_NUMERIC,
        ToolName.STEP_DIFF,
        ToolName.KB_LOOKUP,
        ToolName.ESCALATE,
    }


@pytest.mark.parametrize("name", ["__import__", "open", "eval", "check_numeric ", "", None, 123, ["x"]])
def test_unknown_tool_is_refused(name: object) -> None:
    result = dispatch(name, {}, _ctx())
    assert result["status"] == "tool_error"
    assert result["error"] == "unknown_tool"
    assert ToolName.CHECK_NUMERIC in result["allowed"]


def test_non_dict_arguments_are_refused() -> None:
    result = dispatch(ToolName.CHECK_NUMERIC, "expr=1", _ctx())
    assert result["error"] == "invalid_arguments"


# --- 4 工具 × 坏参数：无一抛出 ----------------------------------------------
@pytest.mark.parametrize("tool", list(BAD_PARAMS))
def test_bad_parameters_never_raise(tool: str) -> None:
    for args in BAD_PARAMS[tool]:
        result = dispatch(tool, args, _ctx())
        assert isinstance(result, dict), f"{tool} {args} 返回了非 dict"
        assert result["status"] == "tool_error", f"{tool} {args} -> {result}"


# --- check_numeric ---------------------------------------------------------
def test_check_numeric_equal(conn: sqlite3.Connection) -> None:
    ctx = _ctx(conn)
    result = dispatch(ToolName.CHECK_NUMERIC, {"expr": "1", "value": "1.0"}, ctx)

    assert result["status"] == "ok"
    assert result["equal"] is True
    assert result["evidence_id"].startswith("ev_")

    evidence = conn.execute(
        "SELECT * FROM evidence WHERE run_id = 'r-test' AND id = ?", (result["evidence_id"],)
    ).fetchone()
    assert evidence is not None
    assert evidence["kind"] == "numeric_check"
    assert evidence["status"] == "ok"
    assert evidence["produced_by"] == ToolName.CHECK_NUMERIC


def test_check_numeric_not_equal(conn: sqlite3.Connection) -> None:
    result = dispatch(ToolName.CHECK_NUMERIC, {"expr": "1", "value": "e-1"}, _ctx(conn))
    assert result["status"] == "ok"
    assert result["equal"] is False


def test_check_numeric_cannot_parse_marks_evidence_skipped(conn: sqlite3.Connection) -> None:
    """无法判定时 evidence 必须 status=skipped —— E2 只接受 ok，所以引用它的 claim 会被拒。"""
    result = dispatch(ToolName.CHECK_NUMERIC, {"expr": "2<x<3", "value": "2"}, _ctx(conn))

    assert result["status"] == "cannot_parse"
    assert result["equal"] is None
    assert "reason" in result
    row = conn.execute(
        "SELECT status FROM evidence WHERE run_id = 'r-test' AND id = ?", (result["evidence_id"],)
    ).fetchone()
    assert row["status"] == "skipped"


def test_check_numeric_records_sandbox_ledger(conn: sqlite3.Connection) -> None:
    """审批 +9：沙箱非 ok 结果要同时写台账。"""
    dispatch(ToolName.CHECK_NUMERIC, {"expr": "9**9**9", "value": "1"}, _ctx(conn))
    rows = conn.execute("SELECT verdict, reason FROM sandbox_rejects WHERE run_id = 'r-test'").fetchall()
    assert len(rows) == 1
    assert rows[0]["verdict"] == Verdict.REJECTED.value
    assert rows[0]["reason"] == "nested_pow"


def test_check_numeric_without_db_still_works() -> None:
    """无连接时返回草稿（P1.6 可自己决定落库时机）。"""
    result = dispatch(ToolName.CHECK_NUMERIC, {"expr": "1/3", "value": "0.333333"}, _ctx())
    assert result["status"] == "ok"
    assert result["equal"] is True


def test_check_numeric_infinite_result_is_not_ok(conn: sqlite3.Connection) -> None:
    result = dispatch(ToolName.CHECK_NUMERIC, {"expr": "2**1e9", "value": "1e308"}, _ctx(conn))
    assert result["status"] == "cannot_parse"
    assert result["equal"] is None


# --- seq 单一发号器 --------------------------------------------------------
def test_seq_comes_from_run_context(conn: sqlite3.Connection) -> None:
    ctx = _ctx(conn)
    first = dispatch(ToolName.CHECK_NUMERIC, {"expr": "1", "value": "1"}, ctx)
    second = dispatch(ToolName.CHECK_NUMERIC, {"expr": "2", "value": "2"}, ctx)

    assert first["evidence_id"] == "ev_1"
    assert second["evidence_id"] == "ev_2"
    assert ctx.next_seq() == 3
    seqs = [
        row["seq"]
        for row in conn.execute("SELECT seq FROM evidence WHERE run_id='r-test' ORDER BY seq").fetchall()
    ]
    assert seqs == [1, 2]


def test_evidence_is_isolated_by_run(conn: sqlite3.Connection) -> None:
    ctx_a = ToolContext(run_id="r-A", conn=conn)
    ctx_b = ToolContext(run_id="r-B", conn=conn)
    a = dispatch(ToolName.CHECK_NUMERIC, {"expr": "1", "value": "1"}, ctx_a)
    b = dispatch(ToolName.CHECK_NUMERIC, {"expr": "2", "value": "2"}, ctx_b)

    assert a["evidence_id"] == b["evidence_id"] == "ev_1"  # 同号不同 run
    current = conn.execute("SELECT DISTINCT run_id FROM evidence WHERE id='ev_1'").fetchall()
    assert {row["run_id"] for row in current} == {"r-A", "r-B"}


# --- step_diff（占位）------------------------------------------------------
def test_step_diff_is_honest_about_not_implemented(conn: sqlite3.Connection) -> None:
    result = dispatch(
        ToolName.STEP_DIFF,
        {"model_steps": ["a", "b"], "student_steps": ["a", "c"]},
        _ctx(conn),
    )
    assert result["status"] == "unsupported"
    assert result["locator"] == "model"
    row = conn.execute(
        "SELECT status FROM evidence WHERE run_id='r-test' AND id=?", (result["evidence_id"],)
    ).fetchone()
    assert row["status"] == "skipped"


# --- kb_lookup -------------------------------------------------------------
def test_kb_lookup_returns_found_and_missing(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO kp (id, name) VALUES ('kp_int_by_parts', '分部积分')")
    result = dispatch(ToolName.KB_LOOKUP, {"kp_ids": ["kp_int_by_parts", "kp_missing"]}, _ctx(conn))
    assert result["status"] == "ok"
    assert result["found"] == [{"id": "kp_int_by_parts", "name": "分部积分", "parent_id": None}]
    assert result["missing"] == ["kp_missing"]


def test_kb_lookup_without_db_is_unavailable() -> None:
    result = dispatch(ToolName.KB_LOOKUP, {"kp_ids": ["kp_1"]}, _ctx())
    assert result["status"] == "unavailable"


# --- escalate --------------------------------------------------------------
def test_escalate_accepts_known_reason_code() -> None:
    result = dispatch(ToolName.ESCALATE, {"reason_code": "insufficient_evidence"}, _ctx())
    assert result["status"] == "escalated"
    assert result["terminal"] == "escalated"


def test_escalate_rejects_unknown_reason_code() -> None:
    result = dispatch(ToolName.ESCALATE, {"reason_code": "because_i_said_so"}, _ctx())
    assert result["status"] == "tool_error"
    assert "reason_code" in result["message"]
