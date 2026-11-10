"""把同步的阻塞调用（sqlite / 沙箱 / 工具）搬离事件循环。

## 为什么这是"必须"而不是"优化"

P1 的缺陷被**刻意保留**（常量表达式可绕过指数上限），于是 `check_numeric` 的最坏耗时
从 2ms 变成 **≈9.8s**（2 次求值各超时 4s + 冷启动 0.9s）。

把它直接放进 `async def` 端点 = **阻塞事件循环 9.8 秒**：
期间 SSE 心跳、其他请求、健康检查全部卡死。

`await asyncio.to_thread(...)` 是最小且正确的解法：不引 aiosqlite、不改仓储层。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


async def run_blocking(func: Callable[..., T], /, *args: object, **kwargs: object) -> T:
    """在默认线程池里执行同步调用。异常原样透传（`to_thread` 的语义）。"""
    return await asyncio.to_thread(func, *args, **kwargs)


async def run_blocking_many(*calls: tuple[Callable[..., object], tuple[object, ...]]) -> list[object]:
    """并发执行多个同步调用（各自在线程池里，互不阻塞事件循环）。"""
    return list(await asyncio.gather(*(run_blocking(func, *args) for func, args in calls)))


__all__ = ["run_blocking", "run_blocking_many"]
