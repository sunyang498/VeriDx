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

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from veridx.clock import evidence_id
from veridx.sandbox import SandboxResult, record_verdict
from veridx.store.repository import Repository


def ensure_finite_payload(payload: dict[str, Any], *, path: str = "payload") -> dict[str, Any]:
    """N5：非有限数**永不入库**，无论上游有没有挡住。

    用 `json.dumps(allow_nan=False)` 当唯一判据（而不是自己递归找 float）：
    它就是"能不能落成合法 JSON"的定义，不会与序列化行为漂移。
    """
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise ValueError(f"{path} 含非有限数，拒绝落库：{exc}") from exc
    return payload


@dataclass
class ToolContext:
    run_id: str
    conn: sqlite3.Connection | None = None
    symbols: frozenset[str] | None = None
    #: 可注入的计数器起点（测试用）；默认从 1 开始
    start_seq: int = 1
    #: 无连接时的内存台账（`record_evidence` 的返回值），E2 需要"本 run 的全部证据"
    drafts: list[dict[str, Any]] = field(default_factory=list)
    _next_seq: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._next_seq = self.start_seq

    # ---- 发号 ----
    def next_seq(self) -> int:
        seq = self._next_seq
        self._next_seq += 1
        return seq

    def peek_next_seq(self) -> int:
        """下一个将要发出的号（不消耗）。用于断言"占号时机"这类顺序性质。"""
        return self._next_seq

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
        """分配序号并写入证据。

        没有连接时返回"未持久化"的草稿（挂进 `drafts` 形成内存台账），
        这样工具层与控制流都能脱离数据库单测。

        **N5（全局数值契约的最后一道闸）**：`payload` 是自由 dict，不受
        `StrictModel.allow_inf_nan=False` 保护，落库前必须自己过一遍
        `json.dumps(..., allow_nan=False)`。前三次"非有限数穿透"都发生在
        "以为上游挡住了，所以下游不查"的地方，这里是存储层的最后关口。
        """
        seq = self.next_seq()
        record = {
            "id": evidence_id(seq),
            "run_id": self.run_id,
            "kind": kind,
            "payload": ensure_finite_payload(payload),
            "status": status,
            "produced_by": produced_by,
            "seq": seq,
        }
        if self.conn is None:
            self.drafts.append(record)
            return record
        return Repository(self.conn).create_evidence(
            run_id=self.run_id,
            evidence_id=record["id"],
            kind=kind,
            payload=record["payload"],
            status=status,
            produced_by=produced_by,
            seq=seq,
        )
