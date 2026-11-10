"""证据绑定 E1–E5 的**唯一裁决实现**（设计文档 §3.3.3 / P1.4）。

## 五条判据

| 判据 | 内容 | 违反时 |
|---|---|---|
| E1 | 断言必须引用 ≥1 条证据 | `empty_evidence` |
| E2 | 每条被引用的证据必须**存在**且 `status=ok`，且**属于本 run** | `unknown_evidence` / `evidence_not_ok` |
| E3 | 证据**类别**要撑得起这个字段（分级见下） | `evidence_class_mismatch` |
| E4 | `evidence.seq < claim.seq`（禁止倒因为果） | `causal_inversion` |
| E5 | 撑不起就 escalate，不许硬猜 | `unsatisfiable` |

## 两条反作弊前提（审查 §6.1）

1. **`claim.seq` 由服务端发号**（`allocate_claim` → 与 evidence 同一个 `next_seq()`）。
   模型自带 `seq` 会被 pydantic `extra="forbid"` 直接拒掉。
2. **跨 run 串证据必须被拒**：`ev_N` 只在 run 内唯一（P0.6 主键修复），
   所以 E2 同时比对 `entry.run_id == run_id`。

## E3 的分级（审查 §6.2：机制性 vs 能力性）

* 概念性 / 程序性 / 计算表达 → 必须有**强程序证据**（`numeric_check` / `step_diff`）；
* 审题性 → 允许半程序证据 `condition_check` 单独支撑，但**置信度上限 0.5 + `needs_review`**；
* `first_divergent_step` → 只认 `step_diff`；`kp_ids` → 只认 `kb_lookup`（查表本身就是验证）。

E3 撑不起来时，`step_diff` 占位、`numeric_check` 遇符号/复数都会给 `skipped` ——
这**不是守卫太严，是能力边界**。所以 escalate 要分成两个分母分别统计
（`EscalationKind.CAPABILITY` / `MECHANISM`），否则会为了压指标去放宽 E3。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from veridx.guard.claim import (
    EVIDENCE_CONFIDENCE_CAP,
    SEMI_PROGRAM_EVIDENCE_KINDS,
    STRONG_EVIDENCE_KINDS,
    BindingViolation,
    Claim,
    ClaimField,
    Evidence,
    EvidenceKind,
    EvidenceStatus,
)
from veridx.sandbox.verdict import UNSUPPORTED_REASONS, Reason
from veridx.taxonomy import ErrorType

__all__ = [
    "BindingDecision",
    "ClaimVerdict",
    "ClaimWithSeq",
    "EscalationKind",
    "LedgerEntry",
    "OverEscalationReport",
    "allocate_claim",
    "bind_claims",
    "field_requirements",
    "over_escalation",
]

#: `condition_check` 单独支撑审题性错误时的置信度上限（§3.3.6）
SEMI_EVIDENCE_CAP = EVIDENCE_CONFIDENCE_CAP[EvidenceKind.CONDITION_CHECK]

#: `first_divergent_step` 只认逐步对齐
STEP_FIELD_KINDS: frozenset[EvidenceKind] = frozenset({EvidenceKind.STEP_DIFF})

#: `kp_ids` 只认查表证据
LOOKUP_KINDS: frozenset[EvidenceKind] = frozenset({EvidenceKind.KB_LOOKUP})

#: 能力性阻塞原因：合法输入、超出沙箱能力边界（含超时）
CAPABILITY_BLOCKED_REASONS: frozenset[str] = UNSUPPORTED_REASONS | {Reason.EVAL_TIMEOUT.value}

#: 概念/程序/计算表达都必须有强程序证据（E3 默认档）
STRONG_REQUIRED_FIELDS: frozenset[ClaimField] = frozenset(
    {
        ClaimField.ERROR_TYPE,
        ClaimField.ERROR_SUBTYPE,
        ClaimField.ROOT_CAUSE,
    }
)


def field_requirements(*, error_type: ErrorType | None = None) -> dict[ClaimField, frozenset[EvidenceKind]]:
    """字段 → 允许的证据类别。`error_type` 决定是否放宽到半程序证据。"""
    relaxed = error_type is ErrorType.READING
    strong = STRONG_EVIDENCE_KINDS
    if relaxed:
        strong = strong | SEMI_PROGRAM_EVIDENCE_KINDS
    requirements: dict[ClaimField, frozenset[EvidenceKind]] = {
        ClaimField.FIRST_DIVERGENT_STEP: STEP_FIELD_KINDS,
        ClaimField.KP_IDS: LOOKUP_KINDS,
    }
    for name in STRONG_REQUIRED_FIELDS:
        requirements[name] = strong
    return requirements


class EscalationKind(StrEnum):
    """两个 over-escalation 分母（审查 §6.2）。"""

    MECHANISM = "mechanism"  # 机制性：空证据 / 不存在 / 非 ok / 类别不符 / 倒序
    CAPABILITY = "capability"  # 能力性：证据拿不到数值（符号/复数/超界/占位 step_diff）


@dataclass(frozen=True)
class LedgerEntry:
    """证据台账的一条记录。`run_id` 必须参与比对，否则跨 run 串证据就绕过了 E2。"""

    run_id: str
    evidence: Evidence


@dataclass(frozen=True)
class ClaimWithSeq:
    """已经由服务端发过号的断言。**构造它只能走 `allocate_claim`。**"""

    claim: Claim
    seq: int


@dataclass
class ClaimVerdict:
    claim: Claim
    seq: int
    violations: list[BindingViolation] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return not self.violations


@dataclass
class BindingDecision:
    verdicts: list[ClaimVerdict]
    escalate: bool
    reason_code: str | None = None
    escalation_kind: EscalationKind | None = None
    confidence_cap: float = 1.0
    needs_review: bool = False

    @property
    def accepted(self) -> list[ClaimWithSeq]:
        return [ClaimWithSeq(v.claim, v.seq) for v in self.verdicts if v.accepted]

    @property
    def rejected(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if not v.accepted]

    @property
    def violations(self) -> list[tuple[ClaimField, BindingViolation]]:
        return [(v.claim.field, b) for v in self.rejected for b in v.violations]

    def field_value(self, name: ClaimField) -> str | list[str] | None:
        for verdict in self.verdicts:
            if verdict.accepted and verdict.claim.field is name:
                return verdict.claim.value
        return None


def allocate_claim(claim: Claim, next_seq: Callable[[], int]) -> ClaimWithSeq:
    """由**服务端**发号器给断言分配序号（唯一合法构造入口）。"""
    return ClaimWithSeq(claim=claim, seq=next_seq())


def bind_claims(
    *,
    run_id: str,
    claims: Sequence[ClaimWithSeq],
    ledger: Sequence[LedgerEntry],
) -> BindingDecision:
    """逐条裁决 E1–E4，再按 E5 得出是否必须 escalate。"""
    index: dict[str, LedgerEntry] = {}
    for entry in ledger:
        # 只收本 run 的证据；串进来的别 run 记录一律当"不存在"
        if entry.run_id == run_id:
            index.setdefault(entry.evidence.id, entry)

    error_type = _declared_error_type(claims)
    requirements = field_requirements(error_type=error_type)
    verdicts: list[ClaimVerdict] = []
    caps: list[float] = []
    semi_only = False

    for item in claims:
        verdict = ClaimVerdict(claim=item.claim, seq=item.seq)
        referenced = _check_references(item, index, run_id, verdict)
        if referenced:
            _check_class(item, referenced, requirements, verdict)
            caps.extend(
                EVIDENCE_CONFIDENCE_CAP[entry.evidence.kind]
                for entry in referenced
                if entry.evidence.kind in EVIDENCE_CONFIDENCE_CAP
            )
            if item.claim.field is ClaimField.ERROR_TYPE and all(
                entry.evidence.kind not in STRONG_EVIDENCE_KINDS for entry in referenced
            ):
                semi_only = True
        verdicts.append(verdict)

    escalate, reason_code, kind = _decide_escalation(verdicts, ledger, run_id)
    confidence_cap = min(caps) if caps else 1.0
    needs_review = semi_only and not escalate
    if needs_review:
        confidence_cap = min(confidence_cap, SEMI_EVIDENCE_CAP)

    return BindingDecision(
        verdicts=verdicts,
        escalate=escalate,
        reason_code=reason_code,
        escalation_kind=kind,
        confidence_cap=round(confidence_cap, 4),
        needs_review=needs_review,
    )


def _declared_error_type(claims: Sequence[ClaimWithSeq]) -> ErrorType | None:
    for item in claims:
        if item.claim.field is not ClaimField.ERROR_TYPE:
            continue
        raw = item.claim.value
        if isinstance(raw, str):
            try:
                return ErrorType(raw)
            except ValueError:
                return None
    return None


def _check_references(
    item: ClaimWithSeq,
    index: dict[str, LedgerEntry],
    run_id: str,
    verdict: ClaimVerdict,
) -> list[LedgerEntry]:
    """E1 / E2 / E4：引用非空、存在且 ok、且早于断言。"""
    ids = item.claim.evidence_ids
    if not ids:
        verdict.violations.append(BindingViolation.EMPTY_EVIDENCE)
        return []

    found: list[LedgerEntry] = []
    for evidence_id in ids:
        entry = index.get(evidence_id)
        if entry is None or entry.run_id != run_id:
            # 跨 run 的 ev_N 在这里被拒：它与本 run 的 ev_N 同名但不同物
            verdict.violations.append(BindingViolation.UNKNOWN_EVIDENCE)
            verdict.detail.append(f"{evidence_id} 不存在于 run {run_id}")
            continue
        if entry.evidence.status is not EvidenceStatus.OK:
            verdict.violations.append(BindingViolation.EVIDENCE_NOT_OK)
            verdict.detail.append(f"{evidence_id} status={entry.evidence.status}")
            continue
        if entry.evidence.seq >= item.seq:
            verdict.violations.append(BindingViolation.CAUSAL_INVERSION)
            verdict.detail.append(f"{evidence_id}.seq={entry.evidence.seq} >= claim.seq={item.seq}")
            continue
        found.append(entry)
    return found


def _check_class(
    item: ClaimWithSeq,
    referenced: list[LedgerEntry],
    requirements: dict[ClaimField, frozenset[EvidenceKind]],
    verdict: ClaimVerdict,
) -> None:
    """E3：证据类别要撑得起这个字段。"""
    allowed = requirements.get(item.claim.field)
    if allowed is None:
        return
    if any(entry.evidence.kind in allowed for entry in referenced):
        return
    verdict.violations.append(BindingViolation.EVIDENCE_CLASS_MISMATCH)
    kinds = sorted({entry.evidence.kind.value for entry in referenced})
    verdict.detail.append(f"{item.claim.field.value} 需要 {sorted(k.value for k in allowed)}，实际 {kinds}")


def _decide_escalation(
    verdicts: Sequence[ClaimVerdict],
    ledger: Sequence[LedgerEntry],
    run_id: str,
) -> tuple[bool, str | None, EscalationKind | None]:
    """E5：error_type 立不住就必须 escalate（不许硬猜）。"""
    error_verdict = next(
        (v for v in verdicts if v.claim.field is ClaimField.ERROR_TYPE),
        None,
    )
    if error_verdict is not None and error_verdict.accepted:
        return False, None, None

    reason = BindingViolation.EMPTY_EVIDENCE if not verdicts else BindingViolation.UNSATISFIABLE
    kind = EscalationKind.CAPABILITY if _has_capability_block(ledger, run_id) else EscalationKind.MECHANISM
    return True, reason.value, kind


def _has_capability_block(ledger: Iterable[LedgerEntry], run_id: str) -> bool:
    """本 run 里是否存在"合法输入但沙箱给不出值"的证据（能力边界，不是机制缺陷）。"""
    for entry in ledger:
        if entry.run_id != run_id:
            continue
        evidence = entry.evidence
        if evidence.kind not in STRONG_EVIDENCE_KINDS:
            continue
        if evidence.status is EvidenceStatus.OK:
            continue
        if _blocked_reasons(evidence.payload) & CAPABILITY_BLOCKED_REASONS:
            return True
    return False


def _blocked_reasons(payload: dict[str, object]) -> set[str]:
    reasons: set[str] = set()
    for key in ("blocked_reason", "reason"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            reasons.add(value.split(":", 1)[0])
    for side in ("left", "right"):
        side_payload = payload.get(side)
        if isinstance(side_payload, dict):
            value = side_payload.get("reason")
            if isinstance(value, str) and value:
                reasons.add(value.split(":", 1)[0])
    return reasons


@dataclass
class OverEscalationReport:
    """两个分母分开报（审查 §6.2）：机制性 / 能力性。"""

    total: int = 0
    mechanism: int = 0
    capability: int = 0

    @property
    def mechanism_rate(self) -> float:
        return self.mechanism / self.total if self.total else 0.0

    @property
    def capability_rate(self) -> float:
        return self.capability / self.total if self.total else 0.0

    @property
    def total_rate(self) -> float:
        return (self.mechanism + self.capability) / self.total if self.total else 0.0


def over_escalation(decisions: Sequence[BindingDecision]) -> OverEscalationReport:
    report = OverEscalationReport(total=len(decisions))
    for decision in decisions:
        if not decision.escalate:
            continue
        if decision.escalation_kind is EscalationKind.CAPABILITY:
            report.capability += 1
        else:
            report.mechanism += 1
    return report
