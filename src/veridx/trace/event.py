"""trace 契约（不变量 I4 的载体，设计文档 §11.1）。

三条硬规则：
1. **重试是兄弟/子事件，绝不覆盖原事件**（`parent_seq` 指向父节点）
2. **失败也要落盘**——失败样本恰是评测最需要的
3. **每个终态必须有 `reason_code`**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeStatus(StrEnum):
    OK = "ok"
    RETRY = "retry"
    FAIL = "fail"
    SKIPPED = "skipped"
    CANCELED = "canceled"


class NodeName(StrEnum):
    PARSE = "parse"
    BLIND_SOLVE = "blind_solve"
    CHECK_NUMERIC = "check_numeric"
    STEP_DIFF = "step_diff"
    DIAGNOSE = "diagnose"
    EXPLAIN = "explain"
    GUARD = "guard"
    SUBMIT = "submit"
    ESCALATE = "escalate"
    SANDBOX = "sandbox"


class ReasonCode(StrEnum):
    """§12.1 枚举 + P0 冻结时补齐的 C-06/C-07（见 doc/CHANGELOG-contract.md）。"""

    UNPARSABLE_STEM = "unparsable_stem"
    UNPARSABLE_ANSWER = "unparsable_answer"
    SANDBOX_REJECTED = "sandbox_rejected"
    SANDBOX_UNSUPPORTED = "sandbox_unsupported"
    TOOL_ERROR = "tool_error"
    SCHEMA_INVALID = "schema_invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LOW_CONFIDENCE = "low_confidence"
    SUSPECTED_BAD_ANSWER_KEY = "suspected_bad_answer_key"
    PROVIDER_5XX = "provider_5xx"
    RATE_LIMITED = "rate_limited"
    TIMED_OUT = "timed_out"
    SKIPPED_BUDGET = "skipped_budget"
    CALL_BUDGET_EXCEEDED = "call_budget_exceeded"
    USER_CANCELED = "user_canceled"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    node: str
    seq: int
    start_ms: int
    status: str
    item_id: str | None = None
    parent_seq: int | None = None
    end_ms: int | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    reason_code: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seq < 1:
            raise ValueError("seq 必须 ≥ 1（run 内全局单调递增）")
        if self.parent_seq is not None and self.parent_seq >= self.seq:
            raise ValueError("parent_seq 必须小于 seq（父子关系不能倒置）")
        if self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms 不能早于 start_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "item_id": self.item_id,
            "node": self.node,
            "seq": self.seq,
            "parent_seq": self.parent_seq,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        return cls(**data)
