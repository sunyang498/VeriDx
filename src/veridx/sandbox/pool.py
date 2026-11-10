"""常驻 worker 池（§4.12）。

## 为什么必须池化

实测成本拆解：

```
spawn 空进程（不含 import sympy）        204 ms
spawn + import sympy + 求值 "1+1"       1282 ms      ← 差值 1078ms 就是 import sympy
```

单次 spawn 式求值 = 1.1s，而真正的计算 <10ms。
**调大超时治不了这个，只能复用进程。**

## 池化引入的三个风险（必须同时解决）

1. **线程安全**：两个请求拿到同一 worker → 管道串话、结果错配。
   对策：`queue.Queue` 天然互斥，同一 worker 同一时刻只被一个线程持有。
2. **退出回收**：uvicorn 重启/测试结束 → 孤儿进程越积越多。
   对策：`atexit` + `shutdown()`；测试断言进程 `is_alive()` 为 False。
3. **毒丸**：worker 被 `terminate()` 后不能复用 → 一次超时会让它**永久失效**。
   对策：超时/异常 → terminate + **移出池** + 下次重新 spawn。

## 超时口径

* **冷启动**：从 `spawn` 到收到 `ready`（含 import sympy）→ `SANDBOX_COLD_TIMEOUT_SECONDS`（8s）
* **稳态计算**：从写入表达式到收到结果 → `SANDBOX_TIMEOUT_SECONDS`（3s）

两者分开，否则首次调用会把整个预算耗在 import 上，慢机器上必然误判 timeout。
"""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from veridx import config
from veridx.sandbox import _worker


class WorkerUnavailable(RuntimeError):
    """worker 无法启动（通常是没有 `if __name__ == "__main__":` 守卫）。"""


@dataclass
class _Worker:
    process: Any
    conn: Any


class WorkerPool:
    def __init__(self, size: int | None = None) -> None:
        self.size = size or config.SANDBOX_WORKER_POOL_SIZE
        self._idle: queue.Queue[_Worker] = queue.Queue(maxsize=max(self.size, 1))
        self._lock = threading.Lock()
        self._all: dict[int, Any] = {}  # pid -> Process（用于回收断言）
        self._closed = False

    # ---- 生命周期 ----
    def _spawn(self) -> _Worker:
        context = multiprocessing.get_context("spawn")
        parent_conn, child_conn = context.Pipe(duplex=True)
        process = context.Process(target=_worker.serve, args=(child_conn,), daemon=True)
        try:
            process.start()
        except BaseException as exc:  # noqa: BLE001 - spawn 失败要降级，不是崩溃
            raise WorkerUnavailable(f"{type(exc).__name__}: {exc}") from exc
        child_conn.close()
        deadline = time.monotonic() + config.SANDBOX_COLD_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if parent_conn.poll(0.02):
                message = parent_conn.recv()
                if isinstance(message, dict) and message.get("ready"):
                    with self._lock:
                        self._all[process.pid or id(process)] = process
                    return _Worker(process=process, conn=parent_conn)
                raise WorkerUnavailable(f"worker 握手异常：{message!r}")
            if not process.is_alive():
                raise WorkerUnavailable("worker 启动后立即退出（检查 __main__ 守卫）")
        self._kill(_Worker(process=process, conn=parent_conn))
        raise WorkerUnavailable("worker 冷启动超时")

    def _kill(self, worker: _Worker) -> None:
        """terminate → join → kill 兜底。被 kill 的 worker **绝不能**再归还池。"""
        process = worker.process
        with contextlib.suppress(Exception):
            process.terminate()
        with contextlib.suppress(Exception):
            process.join(config.SANDBOX_TERMINATE_GRACE_SECONDS)
        with contextlib.suppress(Exception):
            if process.is_alive():
                process.kill()
                process.join(config.SANDBOX_TERMINATE_GRACE_SECONDS)
        with contextlib.suppress(Exception):
            worker.conn.close()
        with self._lock:
            self._all.pop(process.pid or -1, None)

    def _acquire(self) -> _Worker:
        if self._closed:
            raise WorkerUnavailable("池已关闭")
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            # 首次创建 / 池被毒丸清空后重建
            return self._spawn()

    def _release(self, worker: _Worker) -> None:
        if self._closed:
            self._kill(worker)
            return
        try:
            self._idle.put_nowait(worker)
        except queue.Full:
            self._kill(worker)

    def run(self, expr: str, symbols: tuple[str, ...], timeout_seconds: float) -> dict[str, Any]:
        """在池中执行一次求值。返回 worker 的原始 payload（含 `ok` 字段）。

        `WorkerUnavailable` 由调用方翻译成 `Verdict.ERROR`，**池自身不抛**给上层业务。
        """
        worker = self._acquire()
        try:
            worker.conn.send((expr, list(symbols)))
            deadline = time.monotonic() + timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # 超时：terminate + **不归还**（毒丸）
                    self._kill(worker)
                    raise TimeoutError("compute budget exceeded")
                if worker.conn.poll(min(remaining, 0.05)):
                    payload = worker.conn.recv()
                    if isinstance(payload, dict):
                        self._release(worker)
                        return payload
                    raise WorkerUnavailable(f"worker 回包异常：{payload!r}")
                if not worker.process.is_alive():
                    raise WorkerUnavailable("worker 在计算中死亡")
        except (TimeoutError, WorkerUnavailable):
            raise
        except BaseException:
            self._kill(worker)
            raise

    # ---- 关闭 ----
    def shutdown(self) -> None:
        self._closed = True
        while True:
            try:
                worker = self._idle.get_nowait()
            except queue.Empty:
                break
            self._kill(worker)

    def live_pids(self) -> set[int]:
        with self._lock:
            return set(self._all)

    def live_processes(self) -> list[Any]:
        """供测试断言回收；生产代码不该用它。"""
        with self._lock:
            return list(self._all.values())


_POOL: WorkerPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> WorkerPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = WorkerPool()
            atexit.register(_POOL.shutdown)
        return _POOL


def shutdown_pool() -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown()
            _POOL = None
