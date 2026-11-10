"""守卫层：schema/枚举校验 + 证据绑定裁决 + 错误回喂 + 置信度阈值。

守卫有**否决权**：它决定能否落库/返回，而不是"记录校验结果"（设计文档 §2.1 / §3.3）。

P1.3 已实现：`tolerant_parse`（F1–F12）+ `call_structured`（回喂重试 ≤2）。
证据绑定 E1–E5 在 P1.4。
"""

from __future__ import annotations

from veridx.guard.claim import (
    STRONG_EVIDENCE_KINDS,
    Claim,
    ClaimField,
    Evidence,
    EvidenceKind,
    EvidenceStatus,
    confidence_cap,
)
from veridx.guard.schema import ParseOutcome, tolerant_parse
from veridx.guard.structured import (
    Budget,
    StructuredOutputError,
    StructuredResult,
    call_structured,
)

__all__ = [
    "STRONG_EVIDENCE_KINDS",
    "Budget",
    "Claim",
    "ClaimField",
    "Evidence",
    "EvidenceKind",
    "EvidenceStatus",
    "ParseOutcome",
    "StructuredOutputError",
    "StructuredResult",
    "call_structured",
    "confidence_cap",
    "tolerant_parse",
]
