"""SSE 帧流的最小实现（P2.1 的底座，P1.3 只用它的心跳能力）。

为什么 P1.3 就需要它：审查 §8.6 要求验收"**并发 3 个慢请求时 SSE 心跳不断**"。
那条验收用保留的 P1 缺陷（常量表达式超时）制造真实慢请求，
所以这里提供 `heartbeat_stream()`：在业务任务跑在线程池期间，按固定间隔发心跳。

契约（与 §8.2 对齐）：
* 首帧必须是 `stage`（客户端 3s 内可知"活着"）；
* `done` **恰好一次**，且一定在最后；
* 任意异常都转成 `done(terminal=failed)`，不把异常漏给客户端。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from veridx.api import schemas as S

HEARTBEAT_INTERVAL_SECONDS = 0.2


def sse_frame(event: S.SSEEventName, data: S.EventData) -> str:
    return S.SSEFrame(event=event, data=data).to_text()


async def heartbeat_stream(
    work: Callable[[], Awaitable[Any]],
    *,
    run_id: str,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
    on_progress: Callable[[], list[str]] | None = None,
) -> AsyncIterator[str]:
    """边等 `work()` 边按 interval 发心跳；`work` 结束后发 `done`。

    心跳用 `stage` 事件（`status=ok`）承载，`stage="heartbeat"` 便于前端区分。
    """
    started = time.perf_counter()
    seq = 0

    def stage(name: str, status: str = "ok") -> str:
        nonlocal seq
        seq += 1
        return sse_frame(
            S.SSEEventName.STAGE,
            S.StageEventData(
                stage=name,
                status=status,  # type: ignore[arg-type]
                ms=int((time.perf_counter() - started) * 1000),
                seq=seq,
            ),
        )

    yield stage("accepted")

    task = asyncio.create_task(work())
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                break
            for line in on_progress() if on_progress else []:
                yield line
            yield stage("heartbeat")
        result = task.result()
        terminal = getattr(result, "terminal", "diagnosed")
        tokens = getattr(result, "tokens", None) or {"in": 0, "out": 0}
        yield sse_frame(
            S.SSEEventName.DONE,
            S.DoneEventData(
                terminal=S.Terminal(terminal),
                reason_code=getattr(result, "reason_code", None),
                run_id=run_id,
                latency_ms=int((time.perf_counter() - started) * 1000),
                tokens=S.TokenUsage(
                    input_tokens=int(tokens.get("in", 0)), output_tokens=int(tokens.get("out", 0))
                ),
            ),
        )
    except asyncio.CancelledError:
        task.cancel()
        raise
    except BaseException as exc:  # noqa: BLE001 - 必须发 done，不能漏给客户端
        yield sse_frame(
            S.SSEEventName.DONE,
            S.DoneEventData(
                terminal=S.Terminal.FAILED,
                reason_code=f"{type(exc).__name__}",
                run_id=run_id,
                latency_ms=int((time.perf_counter() - started) * 1000),
                tokens=S.TokenUsage(input_tokens=0, output_tokens=0),
            ),
        )


__all__ = ["HEARTBEAT_INTERVAL_SECONDS", "heartbeat_stream", "sse_frame"]
