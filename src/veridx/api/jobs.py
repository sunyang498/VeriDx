"""后台诊断作业（P2.1 的最小落地）。

## 为什么需要它，以及它不做什么

`POST /attempts/{id}/diagnose` 是 **202 Accepted**：立刻返回 `task_id`，真正的诊断在后台跑。
本模块就是那个"后台"——进程内的 `asyncio` 任务，**不是**任务队列。

* 单用户毕设规模（NFR）：一个进程 + SQLite 足够，引入 Celery/Redis 只会增加"演示跑不起来"的
  失败面；
* **有意的限制**：进程重启后正在跑的任务会丢失（`status=running` 的行会留在库里）。
  这在单用户场景可以接受，但**必须写进限制**，不能假装它有持久性。

## 两条纪律

1. **作业自己开连接**：FastAPI 的 `RepoDep` 每请求一条连接、请求结束即关闭，
   后台作业活得更久，绝不能复用那条连接。
2. **终态只由 `finish_task` 写**（P0.6 约定），且失败也要落盘（`failed` + `internal_error`）。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veridx.flow.runner import RunLimits, Runner, RunSpec
from veridx.llm import factory as llm_factory
from veridx.store import db as store_db
from veridx.store.repository import Repository
from veridx.trace.event import ReasonCode
from veridx.trace.recorder import TraceRecorder


@dataclass
class _Job:
    task_id: str
    run_id: str
    handle: asyncio.Task[None]


class JobManager:
    """进程内的后台作业表（task_id → asyncio.Task）。

    `provider_factory` 可注入：测试用确定性的 provider，避免真打网络。
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        provider_factory: Callable[[], tuple[Any, Any | None]] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._jobs: dict[str, _Job] = {}
        self._provider_factory = provider_factory or _default_providers

    # ---- 查询 ----
    def is_running(self, task_id: str) -> bool:
        job = self._jobs.get(task_id)
        return bool(job and not job.handle.done())

    @property
    def active_count(self) -> int:
        return sum(1 for job in self._jobs.values() if not job.handle.done())

    # ---- 提交 / 取消 ----
    def submit(self, *, task_id: str, run_id: str, spec: RunSpec) -> None:
        """起一个后台作业。**调用方必须先建好 task 行**（status=queued）。"""
        handle = asyncio.create_task(
            self._run(task_id=task_id, run_id=run_id, spec=spec),
            name=f"diagnose:{task_id}",
        )
        self._jobs[task_id] = _Job(task_id=task_id, run_id=run_id, handle=handle)
        handle.add_done_callback(lambda _: self._jobs.pop(task_id, None))

    def cancel(self, task_id: str) -> bool:
        job = self._jobs.get(task_id)
        if job is None or job.handle.done():
            return False
        job.handle.cancel()
        return True

    async def shutdown(self) -> None:
        for job in list(self._jobs.values()):
            job.handle.cancel()
        for job in list(self._jobs.values()):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await job.handle
        self._jobs.clear()

    # ---- 作业主体 ----
    async def _run(self, *, task_id: str, run_id: str, spec: RunSpec) -> None:
        conn = store_db.connect(self.db_path)
        try:
            repo = Repository(conn)
            repo.start_task(task_id, run_id=run_id)
            tracer = TraceRecorder(conn=conn, run_id=run_id)
            primary, second = self._provider_factory()
            runner = Runner(
                primary=primary,
                second=second,
                extract=_extract_provider(),
                conn=conn,
                limits=RunLimits(),
                tracer=tracer,
                run_id=run_id,
            )
            result = await runner.run(spec)
            with contextlib.suppress(Exception):
                repo.add_tokens(task_id, tokens_in=result.tokens_in, tokens_out=result.tokens_out)
            with contextlib.suppress(Exception):
                repo.finish_task(task_id, result.terminal.value, reason_code=result.reason_code)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                Repository(conn).finish_task(task_id, "canceled", reason_code=ReasonCode.USER_CANCELED.value)
            raise
        except Exception as exc:  # noqa: BLE001 - 后台作业的异常必须落成终态
            with contextlib.suppress(Exception):
                Repository(conn).finish_task(task_id, "failed", reason_code=ReasonCode.INTERNAL_ERROR.value)
            with contextlib.suppress(Exception):
                TraceRecorder(conn=conn, run_id=run_id).event(
                    "submit",
                    status="fail",
                    reason_code=ReasonCode.INTERNAL_ERROR.value,
                    detail={"error": f"{type(exc).__name__}: {exc}"[:200]},
                )
        finally:
            conn.close()


def _default_providers() -> tuple[Any, Any | None]:
    primary = llm_factory.build_provider(role="primary")
    second = llm_factory.build_provider(role="second") if _second_available() else None
    return primary, second


def _extract_provider() -> Any | None:
    """隔离提取器（方案 §4）：**没有就返回 None**，由 runner fail closed。

    注意它**不能**退回主模型：那会把"无特权的小模型"变成"有特权的大模型"，
    隔离就白做了（方案 §1 铁律）。
    """
    try:
        return llm_factory.build_provider(role="extract")
    except llm_factory.ProviderConfigError:
        return None


def _second_available() -> bool:
    try:
        return llm_factory.effective_mode("second") == "real"
    except llm_factory.ProviderConfigError:
        return False


def job_spec_from(attempt: dict[str, Any], question: dict[str, Any]) -> RunSpec:
    return RunSpec(
        stem=question["stem"],
        student_answer=attempt["student_answer"],
        answer_key=question["correct_answer"],
        attempt_id=attempt["id"],
        question_id=question["id"],
    )


__all__ = ["JobManager", "job_spec_from"]
