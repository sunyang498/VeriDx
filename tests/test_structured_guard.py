"""P1.3 结构化守卫验收：回喂重试、独立 trace 事件、预算闸门、降级留痕、密钥不外泄。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from veridx import config
from veridx.guard import Budget, StructuredOutputError, call_structured
from veridx.llm.mock import MockProvider
from veridx.store import db as store_db
from veridx.trace.recorder import TraceRecorder
from veridx.trace.secrets import contains_secret, scrub


class Diagnosis(BaseModel):
    error_type: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


GOOD = '{"error_type": "程序性错误", "confidence": 0.82}'


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection, _ = store_db.init_db(tmp_path / "guard.db", config.MIGRATIONS_DIR)
    yield connection
    connection.close()


# --- 基本路径 --------------------------------------------------------------
async def test_good_first_response(conn: sqlite3.Connection) -> None:
    tracer = TraceRecorder(conn, "r-1")
    provider = MockProvider(responses=[GOOD])

    result = await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis, tracer=tracer)

    assert result.model.error_type == "程序性错误"
    assert len(result.attempts) == 1
    events = tracer.events()
    assert [e.node for e in events].count("diagnose") == 1
    assert events[-1].status == "ok"


async def test_second_attempt_succeeds_after_feedback(conn: sqlite3.Connection) -> None:
    """错误回喂的核心断言：第 2 次带上了具体错误说明。"""
    tracer = TraceRecorder(conn, "r-1")
    provider = MockProvider(responses=['{"error_type": "程序性错误"}', GOOD])  # 第一次缺 confidence

    result = await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis, tracer=tracer)

    assert result.model.confidence == pytest.approx(0.82)
    assert len(provider.calls) == 2
    feedback = provider.calls[1]["messages"][-1]["content"]
    assert "confidence" in feedback, "回喂必须包含具体字段错误"
    assert "上一次输出不合法" in feedback


async def test_retry_is_independent_trace_event(conn: sqlite3.Connection) -> None:
    """I4：重试**不覆盖**原事件，且带 attempt_index 与原始坏文本留档。"""
    tracer = TraceRecorder(conn, "r-1")
    provider = MockProvider(responses=["not json at all", GOOD])

    await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis, tracer=tracer)

    diagnose_events = [e for e in tracer.events() if e.node == "diagnose"]
    assert len(diagnose_events) == 2, "两次尝试必须是两条独立事件"
    first, second = diagnose_events
    assert first.status == "retry"
    assert first.reason_code == "schema_invalid"
    assert first.detail["attempt_index"] == 0
    assert first.detail["raw_head"] == "not json at all"
    assert len(first.detail["raw_sha"]) == 16
    assert second.status == "ok"
    assert second.detail["attempt_index"] == 1


async def test_retries_exhausted_raises_structured_error(conn: sqlite3.Connection) -> None:
    tracer = TraceRecorder(conn, "r-1")
    provider = MockProvider(responses=["{}", "{}", "{}"])

    with pytest.raises(StructuredOutputError) as excinfo:
        await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis, tracer=tracer)

    assert len(excinfo.value.attempts) == config.GUARD_MAX_RETRIES + 1
    final = [e for e in tracer.events() if e.node == "diagnose"][-1]
    assert final.status == "fail"
    assert final.detail["degraded"] is True, "降级必须留痕，不许静默"


async def test_retry_count_is_bounded(conn: sqlite3.Connection) -> None:
    provider = MockProvider(responses=["x"] * 10)
    with pytest.raises(StructuredOutputError):
        await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis)
    assert len(provider.calls) == config.GUARD_MAX_RETRIES + 1


# --- 预算闸门 --------------------------------------------------------------
async def test_budget_gate_stops_retries(conn: sqlite3.Connection) -> None:
    """重试是"放大结构"：预算不够就降级，不许烧完再报错。"""
    tracer = TraceRecorder(conn, "r-1")
    provider = MockProvider(responses=["bad", "bad", "bad"])
    budget = Budget(remaining_tokens=0, estimated_call_tokens=1500)

    with pytest.raises(StructuredOutputError):
        await call_structured(
            provider, [{"role": "user", "content": "q"}], Diagnosis, tracer=tracer, budget=budget
        )

    assert len(provider.calls) == 1, "预算不足时不允许重试"
    skipped = [e for e in tracer.events() if e.status == "skipped"]
    assert skipped and skipped[0].reason_code == "skipped_budget"


async def test_budget_is_charged_on_success(conn: sqlite3.Connection) -> None:
    budget = Budget(remaining_tokens=10_000)
    provider = MockProvider(responses=[GOOD])
    result = await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis, budget=budget)
    assert result.usage_total > 0
    assert budget.remaining_tokens < 10_000


# --- provider 选择可观察（假能力预警）--------------------------------------
async def test_provider_selection_is_observable(conn: sqlite3.Connection) -> None:
    tracer = TraceRecorder(conn, "r-1")
    provider = MockProvider(responses=[GOOD])
    await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis, tracer=tracer)

    select = tracer.events()[0]
    assert select.node == "parse"
    assert select.detail["provider"]["provider"] == "mock"
    assert select.detail["provider"]["model"] == "mock-model"


# --- 容错解析与守卫的联动 ---------------------------------------------------
async def test_fenced_json_is_repaired_transparently(conn: sqlite3.Connection) -> None:
    tracer = TraceRecorder(conn, "r-1")
    provider = MockProvider(responses=[f"```json\n{GOOD}\n```"])

    result = await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis, tracer=tracer)

    assert result.model.confidence == pytest.approx(0.82)
    assert "strip_fence" in tracer.events()[-1].detail["transformations"]


# --- provider 故障 ---------------------------------------------------------
async def test_retryable_provider_error_is_retried(conn: sqlite3.Connection) -> None:
    provider = MockProvider(responses=[GOOD], fail_times=1)
    result = await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis)
    assert result.model.error_type == "程序性错误"


# --- 密钥三不落 -------------------------------------------------------------
async def test_trace_never_contains_secrets(conn: sqlite3.Connection) -> None:
    """把 trace 全量序列化后 grep：`sk-` / `Bearer` / `api_key` / `?key=` 必须 0 命中。"""
    tracer = TraceRecorder(conn, "r-1")
    leaky = (
        '{"error_type": "程序性错误", "confidence": 0.8, '
        '"debug": "Bearer sk-live-ABCDEFGH12345678 api_key=supersecret ?key=alsosupersecret"}'
    )
    provider = MockProvider(responses=[leaky])

    await call_structured(provider, [{"role": "user", "content": "q"}], Diagnosis, tracer=tracer)

    blob = json.dumps([event.detail for event in tracer.events()], ensure_ascii=False)
    assert not contains_secret(blob), f"trace 中残留疑似密钥：{blob[:400]}"


def test_scrub_removes_common_secret_shapes() -> None:
    cases = [
        "Authorization: Bearer sk-abcdef123456",
        "https://api.example.com/v1?key=SECRET123&model=x",
        '{"api_key": "SECRET123"}',
        "api_key=SECRET123",
    ]
    for case in cases:
        cleaned = scrub(case)
        assert not contains_secret(cleaned), f"{case!r} -> {cleaned!r}"
        assert "SECRET123" not in cleaned
