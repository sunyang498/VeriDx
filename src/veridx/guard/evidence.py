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
    STRONG_EVIDENCE_KINDS,
    BindingViolation,
    Claim,
    ClaimField,
    Evidence,
    EvidenceKind,
    EvidenceStatus,
)
from veridx.taxonomy import READING_SUBTYPES, ErrorType

__all__ = [
    "BindingDecision",
    "ClaimSeqBlock",
    "ClaimVerdict",
    "ClaimWithSeq",
    "EscalationKind",
    "LedgerEntry",
    "OverEscalationReport",
    "allocate_claim",
    "bind_claims",
    "over_escalation",
    "reserve_claim_seqs",
]

#: `condition_check` 单独支撑审题性错误时的置信度上限（§3.3.6）
SEMI_EVIDENCE_CAP = EVIDENCE_CONFIDENCE_CAP[EvidenceKind.CONDITION_CHECK]

#: `first_divergent_step` 只认逐步对齐
STEP_FIELD_KINDS: frozenset[EvidenceKind] = frozenset({EvidenceKind.STEP_DIFF})

#: `kp_ids` 只认查表证据
LOOKUP_KINDS: frozenset[EvidenceKind] = frozenset({EvidenceKind.KB_LOOKUP})

#: 只引用学生原话永远不足以支撑错因（E3 的下限）
TEXT_KINDS: frozenset[EvidenceKind] = frozenset({EvidenceKind.STUDENT_TEXT})


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
    """给**单条**断言分配序号。

    ⚠️ 时机要求（ADR-020）：序号必须在**产生 claim 的那次 LLM 调用之前**领取。
    在 `bind_claims()` 之前顺手领号会让 `claim.seq` 大于所有证据 → **E4 恒真**。
    流程里请用 `reserve_claim_seqs()` 一次性占号（见其 docstring）。
    """
    return ClaimWithSeq(claim=claim, seq=next_seq())


class ClaimSeqBlock:
    """**在 LLM 调用之前**预留的一段 claim 序号（ADR-020）。

    语义：`claim.seq` = "我开始下这个结论的时刻"。整段序号都小于"调用之后"产生的任何证据，
    因此模型若引用了调用之后才出现的证据，E4 必然拒绝。
    """

    __slots__ = ("_next", "seqs")

    def __init__(self, seqs: list[int]) -> None:
        if not seqs:
            raise ValueError("claim 序号块不能为空")
        self.seqs = tuple(seqs)
        self._next = 0

    @property
    def start(self) -> int:
        return self.seqs[0]

    @property
    def remaining(self) -> int:
        return len(self.seqs) - self._next

    def take(self) -> int:
        """领下一个号（不得新增号：号必须已经在调用前占好）。"""
        if self._next >= len(self.seqs):
            raise IndexError("claim 序号块已用尽：说明模型给出的 claim 数超过预留量")
        seq = self.seqs[self._next]
        self._next += 1
        return seq


def reserve_claim_seqs(next_seq: Callable[[], int], size: int) -> ClaimSeqBlock:
    """**一次性占号**：必须写在"产生 claim 的那次 LLM 调用"**之前**。

    编排上的等价说法：**先领一张空白结论单，再去问模型**——
    而不是"问完模型再补一张单子"（后者会让 E4 结构上恒真）。
    """
    if size < 1:
        raise ValueError("size 必须 ≥ 1")
    return ClaimSeqBlock([next_seq() for _ in range(size)])


def bind_claims(
    *,
    run_id: str,
    claims: Sequence[ClaimWithSeq],
    ledger: Sequence[LedgerEntry],
) -> BindingDecision:
    """逐条裁决 E1–E4，再按 E5 得出是否必须 escalate。

    两遍扫描是刻意的：先做**结构**判定（E1/E2/E4），再用"结构上站得住的
    `error_subtype`"去做 E3 的档位判定 —— 这样档位既不由模型自述单独决定，
    也不会被一条本身立不住的 claim 带动。
    """
    index: dict[str, LedgerEntry] = {}
    for entry in ledger:
        # 只收本 run 的证据；串进来的别 run 记录一律当"不存在"
        if entry.run_id == run_id:
            index.setdefault(entry.evidence.id, entry)

    verdicts: list[ClaimVerdict] = []
    referenced_by: dict[int, list[LedgerEntry]] = {}

    # 第一遍：E1 / E2 / E4（结构判定）
    for position, item in enumerate(claims):
        verdict = ClaimVerdict(claim=item.claim, seq=item.seq)
        referenced_by[position] = _check_references(item, index, run_id, verdict)
        verdicts.append(verdict)

    # 第二遍：E3（档位判定 —— 由**证据**触发，不由模型声明触发）
    subtype = _structurally_valid_subtype(claims, verdicts)
    for position, verdict in enumerate(verdicts):
        referenced = referenced_by[position]
        if referenced:
            _check_class(verdict, referenced, subtype=subtype)

    caps = [
        EVIDENCE_CONFIDENCE_CAP[entry.evidence.kind]
        for referenced in referenced_by.values()
        for entry in referenced
        if entry.evidence.kind in EVIDENCE_CONFIDENCE_CAP
    ]
    semi_only = _error_type_uses_semi_evidence(verdicts, referenced_by)

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


def _structurally_valid_subtype(
    claims: Sequence[ClaimWithSeq], verdicts: Sequence[ClaimVerdict]
) -> str | None:
    """只取"通过 E1/E2/E4"的 `error_subtype` 值。立不住的 subtype 不参与档位判定。"""
    for item, verdict in zip(claims, verdicts, strict=True):
        if item.claim.field is not ClaimField.ERROR_SUBTYPE:
            continue
        if not verdict.accepted:
            continue
        raw = item.claim.value
        if isinstance(raw, str):
            return raw.strip()
    return None


def _error_type_uses_semi_evidence(
    verdicts: Sequence[ClaimVerdict],
    referenced_by: dict[int, list[LedgerEntry]],
) -> bool:
    """`error_type` 是否只靠半程序证据站住（决定 cap 0.5 + needs_review）。"""
    for position, verdict in enumerate(verdicts):
        if verdict.claim.field is not ClaimField.ERROR_TYPE or not verdict.accepted:
            continue
        kinds = {entry.evidence.kind for entry in referenced_by[position]}
        if not (kinds & STRONG_EVIDENCE_KINDS) and kinds & {EvidenceKind.CONDITION_CHECK}:
            return True
    return False


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
    verdict: ClaimVerdict,
    referenced: list[LedgerEntry],
    *,
    subtype: str | None,
) -> None:
    """E3：证据类别要撑得起这个字段（档位**由证据触发**，见 `_class_ok`）。"""
    kinds = {entry.evidence.kind for entry in referenced}
    if _class_ok(verdict.claim, kinds, subtype):
        return
    verdict.violations.append(BindingViolation.EVIDENCE_CLASS_MISMATCH)
    verdict.detail.append(
        f"{verdict.claim.field.value} 需要 {_required_label(verdict.claim, subtype)}，"
        f"实际 {sorted(k.value for k in kinds)}"
    )


def _class_ok(claim: Claim, kinds: set[EvidenceKind], subtype: str | None) -> bool:
    """E3 唯一判据。**放宽只由"引用了 condition_check 证据"触发**（ADR-021）。

    | 字段 | 判据 |
    |---|---|
    | `kp_ids` | 必须引用 `kb_lookup` |
    | `first_divergent_step` | 必须引用 `step_diff` |
    | `root_cause` / `error_subtype` | **永远**要强程序证据（不随 `error_type` 放宽） |
    | `error_type=审题性错误` | 强程序证据，或（实际引用了 `condition_check` **且** 子类在预枚举审题类内） |
    | `error_type=其他` | `student_text` + 任意 1 条其他证据（§3.3.3 第三档） |
    | 其余 | 强程序证据 |
    """
    strong = bool(kinds & STRONG_EVIDENCE_KINDS)

    if claim.field is ClaimField.KP_IDS:
        return bool(kinds & LOOKUP_KINDS)
    if claim.field is ClaimField.FIRST_DIVERGENT_STEP:
        return bool(kinds & STEP_FIELD_KINDS)
    if claim.field in {ClaimField.ROOT_CAUSE, ClaimField.ERROR_SUBTYPE}:
        return strong
    if claim.field is not ClaimField.ERROR_TYPE:
        return strong

    declared = claim.value if isinstance(claim.value, str) else ""
    if declared == ErrorType.READING.value:
        relaxed = EvidenceKind.CONDITION_CHECK in kinds and subtype in READING_SUBTYPES
        return strong or relaxed
    if declared == ErrorType.OTHER.value:
        return bool(kinds & TEXT_KINDS) and len(kinds - TEXT_KINDS) >= 1
    return strong


def _required_label(claim: Claim, subtype: str | None) -> list[str]:
    if claim.field is ClaimField.KP_IDS:
        return ["kb_lookup"]
    if claim.field is ClaimField.FIRST_DIVERGENT_STEP:
        return ["step_diff"]
    declared = claim.value if isinstance(claim.value, str) else ""
    if claim.field is ClaimField.ERROR_TYPE and declared == ErrorType.READING.value:
        allowed = subtype in READING_SUBTYPES
        return (
            ["numeric_check|step_diff"] + (["condition_check"] if allowed else [])
            if allowed
            else ["numeric_check|step_diff（未声明预枚举审题子类，不放宽）"]
        )
    if claim.field is ClaimField.ERROR_TYPE and declared == ErrorType.OTHER.value:
        return ["student_text + 任意一条其他证据"]
    return ["numeric_check|step_diff"]


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
    kind = EscalationKind.CAPABILITY if _has_capability_gap(ledger, run_id) else EscalationKind.MECHANISM
    return True, reason.value, kind


def _has_capability_gap(ledger: Iterable[LedgerEntry], run_id: str) -> bool:
    """本 run 里是否存在"能力边界"造成的非 ok 证据（审查 §6-必3 的分派表）。

    判据绑**证据类型 + 状态**，不绑 payload 的字段名：
    上一版去找 `payload["reason"]`，于是 `step_diff` 占位
    （`{"status": "not_implemented"}`）被误判成"机制故障"，报告会得出与事实相反的结论。

    | kind | 状态 | 判定 |
    |---|---|---|
    | `step_diff` | `skipped` | **capability**（还没有逐步对齐能力） |
    | `numeric_check` | `skipped` 且任一侧 `verdict != ok` | **capability**（沙箱给不出值） |
    | `condition_check` | `skipped` | **capability** |
    | 任意 | `failed` | mechanism（机制故障） |
    | 任意 | 非 ok 但无任何原因痕迹 | mechanism |
    """
    for entry in ledger:
        if entry.run_id != run_id:
            continue
        evidence = entry.evidence
        if evidence.status is EvidenceStatus.OK:
            continue
        if evidence.status is EvidenceStatus.FAILED:
            continue  # 机制故障
        if _is_capability_gap(evidence):
            return True
    return False


def _is_capability_gap(evidence: Evidence) -> bool:
    payload = evidence.payload
    if evidence.kind is EvidenceKind.STEP_DIFF:
        # 占位实现（`status="not_implemented"`）与"无法对齐"都属能力边界
        return payload.get("status") != "ok"
    if evidence.kind is EvidenceKind.NUMERIC_CHECK:
        if payload.get("blocked_reason"):
            return True
        for side in ("left", "right"):
            side_payload = payload.get(side)
            if not isinstance(side_payload, dict):
                continue
            if side_payload.get("verdict") not in (None, "ok"):
                return True
            if side_payload.get("verdict") is None and side_payload.get("reason"):
                return True
        return False
    # `condition_check` 目前没有实现，任何 skipped 都是能力边界
    return evidence.kind is EvidenceKind.CONDITION_CHECK


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


def capability_gap(ledger: Iterable[LedgerEntry], run_id: str) -> bool:
    """公开口径：本 run 是否存在"能力边界"造成的非 ok 证据（报告分类用）。"""
    return _has_capability_gap(ledger, run_id)


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
