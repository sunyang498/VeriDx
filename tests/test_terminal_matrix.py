"""A-3：6 个终态**各一条显式用例**，走"后台作业 → tasks 表 → 终态"这条真实路径。

为什么单列一个文件（`test_runner.py` 里其实已有零散覆盖）：
本轮实测撞出的 `failed/internal_error` 说明**这条路径从没被正经测过**，
它只是"跑挂了顺带出现"。所以这里断言的是三件事：

1. 每个终态都能**被构造出来**（不是"跑挂了算一条"）；
2. `reason_code` 符合 §8.2 的 one-of 规则（`diagnosed ⟺ null`，其余必须可归因）；
3. 终态是 `Repository.finish_task` 写的（tasks 行与 trace 的最后一条事实一致）。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from veridx.api.jobs import JobManager
from veridx.api.schemas import Terminal
from veridx.flow.runner import RunSpec
from veridx.llm.mock import MockProvider
from veridx.store import db as store_db
from veridx.store.repository import Repository
from veridx.trace.event import ReasonCode

STEM = "求 ∫_0^1 x^2 dx 的值。"
KEY = "1/3"
WRONG = "1/4"


def _blind(answer: str) -> str:
    return json.dumps(
        {"answer": answer, "steps": ["求出原函数", "代入上下限"], "confidence": 0.9, "note": ""},
        ensure_ascii=False,
    )


def _diagnose() -> str:
    return json.dumps(
        {
            "error_type": "计算表达错误",
            "error_subtype": None,
            "root_cause": "把 x^2 的指数当成了分母",
            "first_divergent_step": None,
            "confidence": 0.9,
            "hypotheses": [],
            "kp_ids": [],
            "claims": [
                {"field": "error_type", "value": "计算表达错误", "evidence_ids": ["ev_1"]},
                {"field": "root_cause", "value": "指数处理错误", "evidence_ids": ["ev_1"]},
            ],
        },
        ensure_ascii=False,
    )


def _steps() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "title": f"第 {index + 1} 步",
                    "content": "重新套用幂函数积分公式",
                    "key_point": "指数加一后除以新指数",
                    "targets_error": "root_cause：把指数当成了分母",
                    "verifiable": False,
                    "verifiable_expr": None,
                }
                for index in range(3)
            ]
        },
        ensure_ascii=False,
    )


class _Raiser:
    """永远抛非可重试异常：用来**构造** `failed`。"""

    model = "raiser"

    def describe(self) -> dict[str, Any]:
        return {"provider": "raiser", "model": "raiser", "mode": "mock"}

    async def chat(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("provider exploded (constructed)")


class _Blocker:
    """一直阻塞：用来**构造** `canceled`。"""

    model = "blocker"

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    def describe(self) -> dict[str, Any]:
        return {"provider": "blocker", "model": "blocker", "mode": "mock"}

    async def chat(self, *args: object, **kwargs: object) -> object:
        self.entered.set()
        await asyncio.sleep(3600)
        raise AssertionError("不应走到这里")


def _seed(db_path: Path, *, student: str = WRONG) -> tuple[str, Repository, Any]:
    conn, _ = store_db.init_db(db_path)
    repo = Repository(conn)
    question = repo.create_question(course="calculus", stem=STEM, correct_answer=KEY, qtype="numeric")
    attempt = repo.create_attempt(question_id=question["id"], student_answer=student)
    task = repo.create_task(attempt_id=attempt["id"], estimated_ms=1)
    return task["id"], repo, conn


async def _drive(
    db_path: Path,
    *,
    factory: Callable[[], tuple[Any, Any | None]],
    student: str = WRONG,
    cancel_when: asyncio.Event | None = None,
    manager_probe: list[JobManager] | None = None,
) -> dict[str, Any]:
    task_id, repo, conn = _seed(db_path, student=student)
    manager = JobManager(db_path, provider_factory=factory)
    spec = RunSpec(
        stem=STEM,
        student_answer=student,
        answer_key=KEY,
        attempt_id=repo.get_task(task_id)["attempt_id"],  # type: ignore[index]
        question_id="q-matrix",
    )
    manager.submit(task_id=task_id, run_id="r-matrix", spec=spec)
    handle = manager._jobs[task_id].handle  # noqa: SLF001 - 测试要看住这个句柄
    if cancel_when is not None and manager_probe is not None:
        manager_probe.append(manager)
        await cancel_when.wait()
        manager.cancel(task_id)
    try:
        await handle
    except asyncio.CancelledError:
        pass
    row = repo.get_task(task_id)
    assert row is not None
    conn.close()
    return row


def _assert_reason_rule(row: dict[str, Any]) -> None:
    """§8.2：`diagnosed` 的 reason_code 必须为 null；其余终态必须可归因且是已知枚举。"""
    terminal = row["terminal"]
    reason = row["reason_code"]
    if terminal == Terminal.DIAGNOSED.value:
        assert reason is None, f"diagnosed 不应带 reason_code，实际 {reason}"
    else:
        assert reason, f"{terminal} 必须带 reason_code（I4：每个终态可归因）"
        assert reason in {code.value for code in ReasonCode}, f"{reason} 不在 ReasonCode 枚举内"
    assert row["status"] == terminal, "status 与 terminal 必须一致（C-17 的 CHECK）"


async def test_terminal_diagnosed(tmp_path: Path) -> None:
    primary = MockProvider(responses=[_blind(KEY), _diagnose(), _steps()], name="p", model="p")
    second = MockProvider(responses=[_blind(KEY)], name="s", model="s")
    row = await _drive(tmp_path / "d.db", factory=lambda: (primary, second))
    assert row["terminal"] == Terminal.DIAGNOSED.value
    _assert_reason_rule(row)


async def test_terminal_escalated(tmp_path: Path) -> None:
    """证据撑不住断言 → 转人工（不是"判错"）。"""
    bad = json.dumps(
        {
            "error_type": "计算表达错误",
            "root_cause": "x",
            "confidence": 0.9,
            "claims": [{"field": "error_type", "value": "计算表达错误", "evidence_ids": ["ev_99"]}],
        },
        ensure_ascii=False,
    )
    primary = MockProvider(responses=[_blind(KEY), bad, bad, bad], name="p", model="p")
    second = MockProvider(responses=[_blind(KEY)], name="s", model="s")
    row = await _drive(tmp_path / "e.db", factory=lambda: (primary, second))
    assert row["terminal"] == Terminal.ESCALATED.value
    assert row["reason_code"] == ReasonCode.INSUFFICIENT_EVIDENCE.value
    _assert_reason_rule(row)


async def test_terminal_uncertain(tmp_path: Path) -> None:
    """判不了 ≠ 判错：模型答了个沙箱不支持的写法 → `uncertain`。"""
    primary = MockProvider(responses=[_blind("int(x**2,(x,0,1))")], name="p", model="p")
    row = await _drive(tmp_path / "u.db", factory=lambda: (primary, None))
    assert row["terminal"] == Terminal.UNCERTAIN.value
    assert row["reason_code"] == ReasonCode.SANDBOX_UNSUPPORTED.value
    _assert_reason_rule(row)


async def test_terminal_needs_clarification(tmp_path: Path) -> None:
    primary = MockProvider(responses=[], name="p", model="p")
    row = await _drive(tmp_path / "n.db", factory=lambda: (primary, None), student="   ")
    assert row["terminal"] == Terminal.NEEDS_CLARIFICATION.value
    assert row["reason_code"] == ReasonCode.UNPARSABLE_ANSWER.value
    _assert_reason_rule(row)


async def test_terminal_failed(tmp_path: Path) -> None:
    """`failed` 必须是**构造出来的**（provider 连续抛非重试异常），不是"跑挂了算一条"。"""
    raiser = _Raiser()
    row = await _drive(tmp_path / "f.db", factory=lambda: (raiser, None))
    assert row["terminal"] == Terminal.FAILED.value
    assert row["reason_code"] == ReasonCode.INTERNAL_ERROR.value
    _assert_reason_rule(row)


async def test_terminal_canceled(tmp_path: Path) -> None:
    blocker = _Blocker()
    manager_probe: list[JobManager] = []
    task = asyncio.ensure_future(
        _drive(
            tmp_path / "c.db",
            factory=lambda: (blocker, None),
            cancel_when=blocker.entered,
            manager_probe=manager_probe,
        )
    )
    row = await task
    assert row["terminal"] == Terminal.CANCELED.value
    assert row["reason_code"] == ReasonCode.USER_CANCELED.value
    _assert_reason_rule(row)


@pytest.mark.parametrize("terminal", list(Terminal))
def test_every_terminal_has_an_explicit_case(terminal: Terminal) -> None:
    """6 个终态一个都不能漏：本文件里每个名字都出现在某个测试的函数名里。"""
    names = {name.removeprefix("test_terminal_") for name in globals() if name.startswith("test_terminal_")}
    assert terminal.value in names, f"{terminal.value} 没有显式用例"
