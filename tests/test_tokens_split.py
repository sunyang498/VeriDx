"""实施单 §2：token **输入/输出分开记**。

原来的两个字段都填"总量"（`input_tokens=self._tokens` / `output_tokens=self._tokens`，
`jobs.py` 干脆写 `tokens_out=0`），于是"每道题花多少 token / 多少钱"的分析全是假的，
`PER_CALL_CTX_TOKENS=8000` 的预算也无法校准。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veridx.api.jobs import JobManager
from veridx.flow.runner import RunLimits, Runner, RunSpec
from veridx.llm.mock import MockProvider
from veridx.store import db as store_db
from veridx.store.repository import Repository

STEM = "求 ∫_0^1 x^2 dx 的值。"
KEY = "1/3"
WRONG = "1/4"


def _blind() -> str:
    return json.dumps({"answer": KEY, "steps": ["求原函数"], "confidence": 0.9, "note": ""})


def _diagnose() -> str:
    return json.dumps(
        {
            "error_type": "计算表达错误",
            "root_cause": "指数处理错误",
            "confidence": 0.9,
            "claims": [{"field": "error_type", "value": "计算表达错误", "evidence_ids": ["ev_1"]}],
        },
        ensure_ascii=False,
    )


def _steps() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "title": f"第 {i + 1} 步",
                    "content": "重新套用公式",
                    "key_point": "指数加一",
                    "targets_error": "root_cause：指数处理错误",
                    "verifiable": False,
                    "verifiable_expr": None,
                }
                for i in range(3)
            ]
        },
        ensure_ascii=False,
    )


def _providers() -> tuple[MockProvider, MockProvider]:
    return (
        MockProvider(responses=[_blind(), _diagnose(), _steps()], name="p", model="p"),
        MockProvider(responses=[_blind()], name="s", model="s"),
    )


async def test_runner_splits_input_and_output_tokens() -> None:
    primary, second = _providers()
    runner = Runner(primary=primary, second=second, run_id="r-tok")
    result = await runner.run(
        RunSpec(stem=STEM, student_answer=WRONG, answer_key=KEY, attempt_id="a-1", question_id="q-1")
    )

    expected_in = sum(call_usage.input_tokens for call_usage in [primary.total_usage, second.total_usage])
    assert result.tokens_in == expected_in
    assert result.tokens_out > 0, "输出 token 不许再是 0"
    assert result.tokens_in > result.tokens_out, "输入（含 prompt）必然大于输出"
    assert result.tokens == result.tokens_in + result.tokens_out


async def test_job_persists_both_directions(tmp_path: Path) -> None:
    """`tasks.tokens_out` 不许再恒为 0；且与 trace 的 tokens_in 自洽。"""
    db_path = tmp_path / "tok.db"
    conn, _ = store_db.init_db(db_path)
    repo = Repository(conn)
    question = repo.create_question(course="calculus", stem=STEM, correct_answer=KEY, qtype="numeric")
    attempt = repo.create_attempt(question_id=question["id"], student_answer=WRONG)
    task = repo.create_task(attempt_id=attempt["id"], estimated_ms=1)
    conn.close()

    manager = JobManager(db_path, provider_factory=_providers)
    spec = RunSpec(
        stem=STEM, student_answer=WRONG, answer_key=KEY, attempt_id=attempt["id"], question_id=question["id"]
    )
    manager.submit(task_id=task["id"], run_id="r-tok-job", spec=spec)
    await manager._jobs[task["id"]].handle  # noqa: SLF001 - 等作业跑完

    conn = store_db.connect(db_path)
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
    trace_in = conn.execute(
        "SELECT COALESCE(SUM(tokens_in), 0) AS s FROM trace_events WHERE run_id = ?", ("r-tok-job",)
    ).fetchone()["s"]
    trace_out = conn.execute(
        "SELECT COALESCE(SUM(tokens_out), 0) AS s FROM trace_events WHERE run_id = ?", ("r-tok-job",)
    ).fetchone()["s"]
    conn.close()

    assert row["tokens_in"] > 0
    assert row["tokens_out"] > 0, "任务表必须记下输出 token"
    # 同一个 run 内自洽：tasks 的两个数 == trace 里各次调用的和
    assert row["tokens_in"] == trace_in
    assert row["tokens_out"] == trace_out


async def test_llm_calls_and_tokens_stay_in_sync_under_budget_stop() -> None:
    """预算停下时也不能把没发生的调用记到 token 上。"""
    primary, second = _providers()
    runner = Runner(primary=primary, second=second, run_id="r-tok-b", limits=RunLimits(max_llm_calls=0))
    result = await runner.run(
        RunSpec(stem=STEM, student_answer=WRONG, answer_key=KEY, attempt_id="a-1", question_id="q-1")
    )
    assert result.llm_calls == 0
    assert result.tokens_in == 0 and result.tokens_out == 0


@pytest.mark.parametrize("field", ["tokens_in", "tokens_out"])
def test_run_result_exposes_both_fields(field: str) -> None:
    from veridx.flow.runner import RunResult

    assert field in RunResult.__dataclass_fields__


def test_stream_done_reports_both_directions(tmp_path: Path) -> None:
    """SSE 的 `done.tokens` 也必须 in/out 分开（以前两个都填总量）。"""
    from fastapi.testclient import TestClient

    from veridx import config
    from veridx.api.app import create_app

    original = config.LLM_MODE
    config.LLM_MODE = "mock"
    try:
        app = create_app(db_path=tmp_path / "sse.db")
        with TestClient(app) as client:
            attempt = client.post(
                "/api/v1/attempts",
                json={
                    "question": {"course": "calculus", "stem": STEM, "correct_answer": KEY},
                    "student_answer": WRONG,
                },
            ).json()
            task_id = client.post(f"/api/v1/attempts/{attempt['attempt_id']}/diagnose").json()["task_id"]
            body = client.get(f"/api/v1/tasks/{task_id}/stream").text
    finally:
        config.LLM_MODE = original

    done = [line for line in body.splitlines() if line.startswith("data: ") and '"tokens"' in line]
    assert done, body
    payload = json.loads(done[-1][len("data: ") :])
    tokens = payload["tokens"]
    assert tokens["in"] > 0
    assert tokens["out"] > 0, "SSE 的 out 不许再是 0"
    assert tokens["in"] != tokens["out"]
