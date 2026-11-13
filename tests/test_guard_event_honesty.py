"""B-1 / B-2 的后端验收：guard 事件与 done 耗时**不许再编数**。

背景（审查 B-1 / B-2 的实测）：`api/routes.py` 曾经用一个布尔 `escalate` 同时冒充
`schema` / `evidence_binding` 两个判断，`retries` 恒为 0（detail 里根本没这个字段）；
`done.latency_ms` 报的是"SSE 流活了多久"，所以 failed 永远显示 `0.0s`。

这里盯三件事：
1. 三档口径 `pass / partial / fail_after_retry` 由**真实分子分母**算出来；
2. `retries` 取真实 schema 重试次数，老数据没有就是 `null`（不是 0）；
3. `done.latency_ms` 拿不到就是 `null`（前端据此不显示 `· 0.0s`）。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from veridx.api import schemas as S
from veridx.api.routes import _event_frames, _task_latency_ms
from veridx.flow.runner import Runner, RunSpec
from veridx.guard.evidence import binding_label
from veridx.llm.mock import MockProvider

STEM = "求 ∫_0^1 x^2 dx 的值。"
KEY = "1/3"
WRONG = "1/4"


def _blind(answer: str) -> str:
    return json.dumps({"answer": answer, "steps": ["求原函数"], "confidence": 0.9, "note": ""})


def _diagnose(*, good_field: str, bad_field: str | None = None) -> str:
    claims = [{"field": good_field, "value": "计算表达错误", "evidence_ids": ["ev_1"]}]
    if bad_field:
        claims.append({"field": bad_field, "value": "编一个根因", "evidence_ids": ["ev_99"]})
    return json.dumps(
        {
            "error_type": "计算表达错误",
            "root_cause": "指数处理错误",
            "confidence": 0.9,
            "claims": claims,
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


async def _run(*responses: str) -> tuple[Any, list[Any]]:
    primary = MockProvider(responses=list(responses), name="p", model="p")
    second = MockProvider(responses=[_blind(KEY)], name="s", model="s")
    from veridx.trace.recorder import TraceRecorder

    tracer = TraceRecorder(conn=None, run_id="r-honest")
    runner = Runner(primary=primary, second=second, tracer=tracer, run_id="r-honest")
    result = await runner.run(
        RunSpec(stem=STEM, student_answer=WRONG, answer_key=KEY, attempt_id="a-1", question_id="q-1")
    )
    guard_events = [event for event in tracer.events() if event.node == "guard"]
    return result, guard_events


def test_three_tier_labels() -> None:
    """三档口径的唯一实现：部分通过必须与"通过"区分。"""
    assert binding_label(2, 0, escalate=False) == "pass"
    assert binding_label(1, 1, escalate=False) == "partial"
    assert binding_label(0, 1, escalate=False) == "fail_after_retry"
    assert binding_label(1, 1, escalate=True) == "fail_after_retry"


async def test_partial_binding_is_produced_by_the_pipeline() -> None:
    """error_type 立住、root_cause 立不住 → **部分通过**（以前会显示成"通过"）。"""
    result, guard_events = await _run(_blind(KEY), _diagnose(good_field="error_type", bad_field="root_cause"))
    assert result.decision is not None and result.decision.escalate is False
    detail = guard_events[0].detail
    assert detail["accepted"] == 1 and detail["rejected"] == 1
    assert detail["evidence_binding"] == "partial"
    # 另外两个判据必须各自独立可见
    assert detail["schema"] == "pass"
    assert detail["schema_retries"] == 0


async def test_guard_frame_carries_real_numbers() -> None:
    _result, guard_events = await _run(
        _blind(KEY), _diagnose(good_field="error_type", bad_field="root_cause")
    )
    row = {
        "node": "guard",
        "status": "ok",
        "end_ms": 0,
        "detail": json.dumps(guard_events[0].detail, ensure_ascii=False),
    }
    frames = list(_event_frames(row, lambda: 1))
    payload = json.loads(frames[0].split("data: ", 1)[1])
    assert payload["evidence_binding"] == "partial"
    assert payload["accepted"] == 1 and payload["total"] == 2
    assert payload["schema"] == "pass"
    assert payload["retries"] == 0


async def test_schema_retry_is_reported_not_invented() -> None:
    """模型第一次给了**不合法的错因枚举**、第二次给对 → `retry` + 真实重试次数 1。

    （用"截断"制造重试是不行的：必4 已规定截断是确定性失败，不重试。）
    """
    bad = json.dumps(
        {"error_type": "不存在的错因大类", "root_cause": "x", "confidence": 0.9, "claims": []},
        ensure_ascii=False,
    )
    _result, guard_events = await _run(_blind(KEY), bad, _diagnose(good_field="error_type"), _steps())
    assert guard_events, "guard 事件必须存在"
    detail = guard_events[0].detail
    assert detail["schema"] == "retry"
    assert detail["schema_retries"] == 1


def test_old_guard_row_reports_null_instead_of_fake_zero() -> None:
    """老 trace 行没有 `schema_retries`/`schema` → 发 `null`，**不许**顶替成 0 / pass。"""
    row = {
        "node": "guard",
        "status": "ok",
        "end_ms": 0,
        "detail": json.dumps({"escalate": False, "accepted": 2, "rejected": 0}, ensure_ascii=False),
    }
    payload = json.loads(list(_event_frames(row, lambda: 1))[0].split("data: ", 1)[1])
    assert payload["retries"] is None
    assert payload["schema"] is None
    # 分子分母是真实存在的 → 仍然要能算出"全部通过"
    assert payload["evidence_binding"] == "pass"
    assert payload["accepted"] == 2 and payload["total"] == 2


def test_done_latency_is_null_when_it_cannot_be_known() -> None:
    """秒级时间戳相同 = 亚秒级 run：返回 None，让界面不显示 `· 0.0s`。"""
    same = {"created_at": "2026-11-13T12:00:00Z", "updated_at": "2026-11-13T12:00:00Z"}
    assert _task_latency_ms(same) is None
    longer = {"created_at": "2026-11-13T12:00:00Z", "updated_at": "2026-11-13T12:00:03Z"}
    assert _task_latency_ms(longer) == 3000
    assert _task_latency_ms({"created_at": None, "updated_at": None}) is None


def test_done_event_accepts_null_latency() -> None:
    data = S.DoneEventData(
        terminal=S.Terminal.FAILED,
        reason_code="internal_error",
        run_id="r-1",
        latency_ms=None,
        tokens=S.TokenUsage(input_tokens=0, output_tokens=0),
    )
    assert data.latency_ms is None


@pytest.mark.parametrize("binding", ["pass", "partial", "fail_after_retry", "skipped"])
def test_guard_event_schema_accepts_all_four_bindings(binding: str) -> None:
    data = S.GuardEventData.model_validate(
        {"schema": "pass", "evidence_binding": binding, "retries": None, "accepted": 1, "total": 2}
    )
    assert data.evidence_binding.value == binding
