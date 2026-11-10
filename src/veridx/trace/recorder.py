"""trace 落盘器：**每条事件独立一行**，`seq` 由 run 级计数器发号。

三条硬规则（不变量 I4）：

1. **重试是独立事件，绝不覆盖原事件**（`parent_seq` 指向父节点，`detail.attempt_index` 区分第几次）；
2. **失败也要落盘**——失败样本恰是评测最需要的；
3. **每个终态必须有 `reason_code`**。

`detail` 里的敏感信息由写入方负责脱敏；`secrets.py` 提供统一入口。
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from veridx.clock import new_id, now_utc
from veridx.trace.event import NodeStatus, TraceEvent


class TraceRecorder:
    """一次 run 的 trace 写入器（一个实例只服务一个 run）。"""

    def __init__(self, conn: sqlite3.Connection | None, run_id: str, *, start_seq: int = 1) -> None:
        self.conn = conn
        self.run_id = run_id
        self._counter = itertools.count(start_seq)
        self._events: list[TraceEvent] = []

    def event(
        self,
        node: str,
        *,
        status: str | NodeStatus = NodeStatus.OK,
        parent_seq: int | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        reason_code: str | None = None,
        detail: dict[str, Any] | None = None,
        elapsed_ms: int = 0,
        item_id: str | None = None,
    ) -> int:
        """写一条事件，返回它的 `seq`（父节点可用它做 `parent_seq`）。"""
        seq = next(self._counter)
        event = TraceEvent(
            run_id=self.run_id,
            node=node,
            seq=seq,
            start_ms=0,
            end_ms=max(elapsed_ms, 0),
            status=str(status),
            parent_seq=parent_seq,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reason_code=reason_code,
            detail=dict(detail or {}),
            item_id=item_id,
        )
        self._events.append(event)
        if self.conn is not None:
            self.conn.execute(
                "INSERT INTO trace_events"
                " (id, run_id, item_id, node, seq, parent_seq, start_ms, end_ms,"
                "  tokens_in, tokens_out, status, reason_code, detail)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id("te"),
                    event.run_id,
                    event.item_id,
                    event.node,
                    event.seq,
                    event.parent_seq,
                    event.start_ms,
                    event.end_ms,
                    event.tokens_in,
                    event.tokens_out,
                    event.status,
                    event.reason_code,
                    json.dumps(event.detail, ensure_ascii=False),
                ),
            )
        return seq

    @contextmanager
    def node(
        self, node: str, *, parent_seq: int | None = None, item_id: str | None = None
    ) -> Iterator[_NodeHandle]:
        """`with tracer.node("diagnose") as n: n.detail = {...}`；异常也会落成 `status=fail`。"""
        seq = self.event(node, status=NodeStatus.OK, parent_seq=parent_seq, item_id=item_id)
        handle = _NodeHandle(seq=seq)
        try:
            yield handle
        except BaseException as exc:  # noqa: BLE001 - 失败也要落盘
            self.conn and _update(
                self.conn,
                seq=seq,
                run_id=self.run_id,
                status=NodeStatus.FAIL.value,
                reason_code=handle.reason_code,
                detail={**handle.detail, "exception": type(exc).__name__},
                tokens_in=handle.tokens_in,
                tokens_out=handle.tokens_out,
            )
            raise
        else:
            self.conn and _update(
                self.conn,
                seq=seq,
                run_id=self.run_id,
                status=handle.status,
                reason_code=handle.reason_code,
                detail=handle.detail,
                tokens_in=handle.tokens_in,
                tokens_out=handle.tokens_out,
            )

    # ---- 读取（回放/测试用）----
    def events(self) -> list[TraceEvent]:
        if self.conn is None:
            return list(self._events)
        rows = self.conn.execute(
            "SELECT * FROM trace_events WHERE run_id = ? ORDER BY seq", (self.run_id,)
        ).fetchall()
        return [
            TraceEvent(
                run_id=str(row["run_id"]),
                node=str(row["node"]),
                seq=int(row["seq"]),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]) if row["end_ms"] is not None else None,
                status=str(row["status"]),
                parent_seq=int(row["parent_seq"]) if row["parent_seq"] is not None else None,
                tokens_in=int(row["tokens_in"]),
                tokens_out=int(row["tokens_out"]),
                reason_code=row["reason_code"],
                detail=json.loads(row["detail"]),
                item_id=row["item_id"],
            )
            for row in rows
        ]


class _NodeHandle:
    def __init__(self, *, seq: int) -> None:
        self.seq = seq
        self.status = NodeStatus.OK.value
        self.reason_code: str | None = None
        self.detail: dict[str, Any] = {}
        self.tokens_in = 0
        self.tokens_out = 0


def _update(
    conn: sqlite3.Connection,
    *,
    seq: int,
    run_id: str,
    status: str,
    reason_code: str | None,
    detail: dict[str, Any],
    tokens_in: int,
    tokens_out: int,
) -> None:
    conn.execute(
        "UPDATE trace_events SET status = ?, reason_code = ?, detail = ?,"
        " tokens_in = ?, tokens_out = ? WHERE run_id = ? AND seq = ?",
        (
            status,
            reason_code,
            json.dumps(detail, ensure_ascii=False),
            tokens_in,
            tokens_out,
            run_id,
            seq,
        ),
    )


__all__ = ["TraceRecorder", "now_utc"]
