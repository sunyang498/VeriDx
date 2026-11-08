"""证据与断言契约（设计文档 §3.3）—— 本项目第一机制。

把"不许编错因"从提示词里的一句话，变成**会拒绝写库的校验器**：
每个诊断字段（claim）都必须引用 ≥1 条程序产生的 `status=ok` 证据，
且证据的产生序号必须早于断言（禁止倒因为果）。

本模块只定义**契约与常量**；E1–E5 的裁决实现见 P1.4 的 `guard/evidence.py`。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceKind(StrEnum):
    NUMERIC_CHECK = "numeric_check"
    STEP_DIFF = "step_diff"
    CONDITION_CHECK = "condition_check"
    KB_LOOKUP = "kb_lookup"
    STUDENT_TEXT = "student_text"


class EvidenceStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


class ClaimField(StrEnum):
    ERROR_TYPE = "error_type"
    ERROR_SUBTYPE = "error_subtype"
    ROOT_CAUSE = "root_cause"
    FIRST_DIVERGENT_STEP = "first_divergent_step"
    KP_IDS = "kp_ids"


class BindingRule(StrEnum):
    """E1–E5：五条全部机器可判（§3.3.3）。"""

    E1_NON_EMPTY = "E1"
    E2_EXISTS_AND_OK = "E2"
    E3_EVIDENCE_CLASS = "E3"
    E4_CAUSAL_ORDER = "E4"
    E5_ESCALATE = "E5"


class BindingViolation(StrEnum):
    EMPTY_EVIDENCE = "empty_evidence"
    UNKNOWN_EVIDENCE = "unknown_evidence"
    EVIDENCE_NOT_OK = "evidence_not_ok"
    EVIDENCE_CLASS_MISMATCH = "evidence_class_mismatch"
    CAUSAL_INVERSION = "causal_inversion"
    UNSATISFIABLE = "unsatisfiable"


#: 强程序证据：概念性/程序性/计算表达必须至少有一条
STRONG_EVIDENCE_KINDS: frozenset[EvidenceKind] = frozenset(
    {EvidenceKind.NUMERIC_CHECK, EvidenceKind.STEP_DIFF}
)

#: 半程序证据：只有"审题性错误"允许用它单独支撑
SEMI_PROGRAM_EVIDENCE_KINDS: frozenset[EvidenceKind] = frozenset({EvidenceKind.CONDITION_CHECK})

#: 只引用学生原话不足以支撑 error_type（E3）
TEXT_EVIDENCE_KINDS: frozenset[EvidenceKind] = frozenset({EvidenceKind.STUDENT_TEXT})

#: 证据类型 → 诊断置信度上限（§3.3.6：condition_check 是半程序证据）
EVIDENCE_CONFIDENCE_CAP: dict[EvidenceKind, float] = {EvidenceKind.CONDITION_CHECK: 0.5}


class Evidence(BaseModel):
    id: str = Field(pattern=r"^ev_\d+$")
    kind: EvidenceKind
    payload: dict[str, Any] = Field(default_factory=dict)
    status: EvidenceStatus = EvidenceStatus.OK
    produced_by: str
    seq: int = Field(ge=1)


class Claim(BaseModel):
    field: ClaimField
    value: str | list[str]
    evidence_ids: list[str] = Field(min_length=1)
    seq: int = Field(ge=1)


def confidence_cap(evidence: list[Evidence]) -> float:
    """本次诊断允许的最大置信度（取所有被引用证据里最低的 cap）。"""
    caps = [EVIDENCE_CONFIDENCE_CAP[e.kind] for e in evidence if e.kind in EVIDENCE_CONFIDENCE_CAP]
    return min(caps) if caps else 1.0


def strong_evidence(kinds: set[EvidenceKind]) -> bool:
    return bool(kinds & STRONG_EVIDENCE_KINDS)
