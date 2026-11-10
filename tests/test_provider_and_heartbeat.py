"""P1.3 验收：Provider 抽象 + 事件循环不被阻塞 + SSE 心跳不断。

关键一条（审查 §8.6）：**并发 3 个慢请求时 SSE 心跳不断** ——
而且用**保留的 P1 缺陷**（常量表达式绕过指数上限 → 超时 3–4s）去制造真实慢请求，
不是造一个 mock 慢接口。这是那条缺陷作为基线素材的第一个正面用途。
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from veridx import config
from veridx.api import schemas as S
from veridx.api.stream import heartbeat_stream, sse_frame
from veridx.llm.base import ProviderError
from veridx.llm.factory import ProviderConfigError, build_provider
from veridx.llm.mock import MockProvider
from veridx.llm.openai_compat import OpenAICompatProvider
from veridx.sandbox import clear_cache, evaluate
from veridx.sched.aio import run_blocking, run_blocking_many
from veridx.trace.secrets import contains_secret

SECRET = "sk-live-ABCDEFGH1234567890"


@pytest.fixture(autouse=True)
def _fresh() -> None:
    clear_cache()
    yield
    clear_cache()


# --- Provider 契约 ---------------------------------------------------------
async def test_mock_provider_is_structurally_async() -> None:
    provider = MockProvider(responses=['{"a":1}'])
    result = await provider.chat([{"role": "user", "content": "x"}])
    assert result.text == '{"a":1}'
    assert result.usage.total > 0
    assert result.meta["provider"] == "mock"


def test_describe_never_contains_secret() -> None:
    provider = OpenAICompatProvider(model="m", base_url="https://api.example.com/v1", api_key=SECRET)
    described = json.dumps(provider.describe(), ensure_ascii=False)
    assert not contains_secret(described)
    assert SECRET not in described
    assert described.count("api.example.com") == 1


def test_provider_refuses_empty_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatProvider(model="m", base_url="https://x/v1", api_key="")


def test_factory_auto_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_MODE", "auto")
    monkeypatch.setattr(config, "API_KEY", "")
    provider = build_provider()
    assert isinstance(provider, MockProvider)


def test_factory_real_mode_without_key_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_MODE", "real")
    monkeypatch.setattr(config, "API_KEY", "")
    with pytest.raises(ProviderConfigError):
        build_provider()


# --- HTTP 层：重试只为可重试错误 -------------------------------------------
def _transport(responses: list[httpx.Response]) -> httpx.MockTransport:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Bearer ")
        return queue.pop(0) if queue else httpx.Response(500)

    return httpx.MockTransport(handler)


async def test_http_provider_parses_usage_and_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.0)
    body = {
        "id": "cmpl-1",
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    provider = OpenAICompatProvider(
        model="m",
        base_url="https://api.example.com/v1",
        api_key=SECRET,
        transport=_transport([httpx.Response(200, json=body)]),
    )
    result = await provider.chat([{"role": "user", "content": "hi"}])
    assert result.text == "hello"
    assert result.usage.total == 15


async def test_http_provider_retries_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.0)
    body = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    provider = OpenAICompatProvider(
        model="m",
        base_url="https://api.example.com/v1",
        api_key=SECRET,
        transport=_transport([httpx.Response(503), httpx.Response(200, json=body)]),
    )
    assert (await provider.chat([{"role": "user", "content": "hi"}])).text == "ok"
    assert provider.last_attempts == 2


async def test_http_provider_does_not_retry_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.0)
    provider = OpenAICompatProvider(
        model="m",
        base_url="https://api.example.com/v1",
        api_key=SECRET,
        transport=_transport([httpx.Response(400, text="bad request")]),
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat([{"role": "user", "content": "hi"}])
    assert excinfo.value.retryable is False
    assert provider.last_attempts == 1


async def test_http_error_message_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.0)
    provider = OpenAICompatProvider(
        model="m",
        base_url="https://api.example.com/v1?key=LEAKED123456",
        api_key=SECRET,
        transport=_transport([httpx.Response(500, text=f"upstream error key={SECRET}")]),
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat([{"role": "user", "content": "hi"}])
    assert not contains_secret(str(excinfo.value)), str(excinfo.value)


# --- 事件循环不被阻塞 -------------------------------------------------------
async def test_run_blocking_does_not_block_loop() -> None:
    ticks = 0

    async def ticker(stop: asyncio.Event) -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    stop = asyncio.Event()
    ticker_task = asyncio.create_task(ticker(stop))
    await run_blocking(time.sleep, 0.2)
    stop.set()
    await ticker_task
    assert ticks >= 5, f"阻塞调用期间事件循环停了（ticks={ticks}）"


async def test_slow_sandbox_calls_do_not_starve_the_loop() -> None:
    """用保留的 P1 缺陷制造真实慢请求（常量表达式绕过指数上限 → 超时）。"""
    slow = "10**(3*3*3*3*3*3*3*3*3*3)"
    ticks = 0

    async def ticker(stop: asyncio.Event) -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.05)

    stop = asyncio.Event()
    ticker_task = asyncio.create_task(ticker(stop))

    results = await run_blocking_many(
        (evaluate, (slow,)),
        (evaluate, ("1+1",)),
        (evaluate, ("2+2",)),
    )

    stop.set()
    await ticker_task
    assert results[1].verdict.value == "ok"
    assert ticks >= 5, f"慢请求阻塞了事件循环（ticks={ticks}）"
    assert results[0].verdict.value in {"timeout", "rejected", "error"}


# --- SSE 心跳 --------------------------------------------------------------
async def test_heartbeat_continues_during_slow_work() -> None:
    """审查 §8.6 的关键验收：慢请求期间心跳不断，`done` 恰好一次且在最后。"""
    slow = "10**(3*3*3*3*3*3*3*3*3*3)"

    async def work() -> object:
        return await run_blocking(evaluate, slow)

    frames = [frame async for frame in heartbeat_stream(work, run_id="r-slow", interval=0.2)]

    assert frames[0].startswith("event: stage"), "首帧必须是 stage（3s 内可知活着）"
    heartbeats = [f for f in frames if '"heartbeat"' in f]
    assert len(heartbeats) >= 3, f"心跳太少：{len(heartbeats)}"
    assert frames[-1].startswith("event: done"), "done 必须在最后"
    assert sum(1 for f in frames if f.startswith("event: done")) == 1


async def test_heartbeat_emits_done_on_failure() -> None:
    async def boom() -> object:
        raise RuntimeError("kaboom")

    frames = [frame async for frame in heartbeat_stream(boom, run_id="r-err", interval=0.05)]
    assert frames[-1].startswith("event: done")
    done = S.SSEFrame.parse("done", frames[-1].split("data: ", 1)[1].strip())
    assert isinstance(done.data, S.DoneEventData)
    assert done.data.terminal is S.Terminal.FAILED


async def test_three_concurrent_slow_requests_keep_heartbeat() -> None:
    """3 个并发慢请求，每个都在自己的 SSE 流里发心跳。"""
    slow = "10**(3*3*3*3*3*3*3*3*3*3)"

    async def one(run_id: str) -> list[str]:
        async def work() -> object:
            return await run_blocking(evaluate, slow)

        return [frame async for frame in heartbeat_stream(work, run_id=run_id, interval=0.2)]

    streams = await asyncio.gather(*(one(f"r-{i}") for i in range(3)))
    for run_id, frames in enumerate(streams):
        assert sum(1 for f in frames if '"heartbeat"' in f) >= 2, f"run-{run_id} 心跳不足"
        assert frames[-1].startswith("event: done")


def test_sse_frame_roundtrip() -> None:
    text = sse_frame(
        S.SSEEventName.STAGE,
        S.StageEventData(stage="heartbeat", status="ok", ms=1, seq=1),
    )
    assert text.startswith("event: stage\n")
    assert text.endswith("\n\n")


# --- 入口闸门（§8.5）：不修沙箱，只挡"最廉价的路" --------------------------
_QUESTION = {"course": "calculus", "stem": "s", "correct_answer": "1"}


def test_entry_gate_caps_student_answer_length() -> None:
    too_long = "1" * (config.SANDBOX_MAX_EXPR_LEN + 1)
    with pytest.raises(Exception):  # pydantic ValidationError
        S.AttemptCreate(question=_QUESTION, student_answer=too_long)

    ok = S.AttemptCreate(question=_QUESTION, student_answer="1" * config.SANDBOX_MAX_EXPR_LEN)
    assert len(ok.student_answer) == config.SANDBOX_MAX_EXPR_LEN


def test_entry_gate_does_not_stop_the_retained_defect() -> None:
    """重要：入口闸门**挡不住** 25 字符的绕过样本 —— 基线劣化依然真实存在。"""
    bypass = "10**(3*3*3*3*3*3*3*3*3*3)"
    assert len(bypass) <= config.SANDBOX_MAX_EXPR_LEN
    model = S.AttemptCreate(question=_QUESTION, student_answer=bypass)
    assert model.student_answer == bypass
