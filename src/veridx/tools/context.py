"""工具层的运行上下文：**run 级单一发号器** + 证据/台账落库口。

## 为什么 `seq` 必须集中发号

`evidence.id = ev_{seq}`，表上有 `UNIQUE(run_id, seq)`。如果让每个工具各自从 1 开始数，
第二个工具产出的证据就会撞唯一索引。所以：

* **一个 run 一个计数器**，工具只负责"要一个号"，不自己造；
* 同一个计数器也用于 claim 的 `seq`（E4「禁止倒因为果」要求 claim.seq > evidence.seq）；
* `ToolContext` 是唯一入口，`Repository` 不再暴露裸 `seq` 参数滥用口。

## 两条纪律（来自 P0.6 审查 §4.2）

1. 证据查询/写入**一律带 `run_id`**（`ev_N` 只在 run 内唯一）；
2. 沙箱的非 `ok` 结果要**同时**写 `sandbox_rejects` 台账与 evidence（否则 M17 算不出）。
"""

from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from veridx.clock import evidence_id
from veridx.sandbox import SandboxResult, record_verdict
from veridx.store.repository import Repository


@dataclass
class ToolContext:
    run_id: str
    conn: sqlite3.Connection | None = None
    symbols: frozenset[str] | None = None
    #: 可注入的计数器起点（测试用）；默认从 1 开始
    start_seq: int = 1
    _counter: Callable[[], int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._counter = itertools.count(self.start_seq).__next__

    # ---- 发号 ----
    def next_seq(self) -> int:
        return self._counter()

    # ---- 落库 ----
    def record_sandbox(self, result: SandboxResult) -> str | None:
        """非 `ok` 的判定写台账（M17 的原料）。无连接时静默跳过。"""
        if self.conn is None:
            return None
        return record_verdict(self.conn, run_id=self.run_id, result=result)

    def record_evidence(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        produced_by: str,
        status: str = "ok",
    ) -> dict[str, Any]:
        """分配序号并写入证据。返回的 dict 至少含 `id` / `seq` / `kind` / `status`。

        没有连接时返回"未持久化"的草稿（P1.6 的控制流可以自己决定落库时机），
        这样工具层可以脱离数据库单测。
        """
        seq = self.next_seq()
        record = {
            "id": evidence_id(seq),
            "run_id": self.run_id,
            "kind": kind,
            "payload": payload,
            "status": status,
            "produced_by": produced_by,
            "seq": seq,
        }
        if self.conn is None:
            return record
        return Repository(self.conn).create_evidence(
            run_id=self.run_id,
            evidence_id=record["id"],
            kind=kind,
            payload=payload,
            status=status,
            produced_by=produced_by,
            seq=seq,
        )
