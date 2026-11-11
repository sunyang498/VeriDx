"""P1.4 验收：证据绑定 E1–E5（设计文档 §3.3.3 / 审查 §6）。

**这一层的全部价值在于"判据的输入不来自被验证方"**，所以最关键的用例不是
"合法输入能过"，而是：

* 模型自带 `seq` 会被拒（否则 E4 形同虚设，审查 §6.1）
* 跨 run 串证据会被拒（`ev_2` 在别的 run 里也是 `ev_2`，P0.6 主键修复的回归）
* 能力边界导致的 escalate 与机制缺陷导致的 escalate **分开计数**（审查 §6.2）
"""

from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from veridx.guard.claim import (
    BindingViolation,
    Claim,
    ClaimField,
    Evidence,
    EvidenceKind,
    EvidenceStatus,
)
from veridx.guard.evidence import (
    SEMI_EVIDENCE_CAP,
    ClaimWithSeq,
    EscalationKind,
    LedgerEntry,
    allocate_claim,
    bind_claims,
    over_escalation,
    over_escalation_from_outcomes,
)
from veridx.tools.context import ToolContext
from veridx.tools.impl import kb_lookup

RUN = "run-1"


# --- 构造小工具 -------------------------------------------------------------
def _evidence(
    evidence_id: str,
    seq: int,
    kind: EvidenceKind = EvidenceKind.NUMERIC_CHECK,
    status: EvidenceStatus = EvidenceStatus.OK,
    payload: dict | None = None,
    run_id: str = RUN,
) -> LedgerEntry:
    return LedgerEntry(
        run_id=run_id,
        evidence=Evidence(
            id=evidence_id,
            kind=kind,
            status=status,
            produced_by="test",
            seq=seq,
            payload=payload or {},
        ),
    )


def _claim(field: ClaimField | str, value: str, ids: list[str], seq: int) -> ClaimWithSeq:
    return ClaimWithSeq(
        claim=Claim(field=field, value=value, evidence_ids=ids),  # type: ignore[arg-type]
        seq=seq,
    )


def _counter(start: int = 1):
    value = start - 1

    def nxt() -> int:
        nonlocal value
        value += 1
        return value

    return nxt


# --- §6.1：序号由服务端发，模型不许碰 ---------------------------------------
def test_claim_rejects_model_supplied_seq() -> None:
    """模型写 `"seq": 999999` 就能让任何证据"早于"自己 —— 所以这个字段根本不许存在。"""
    with pytest.raises(ValidationError, match="seq"):
        Claim(field="root_cause", value="x", evidence_ids=["ev_1"], seq=999999)


def test_claim_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Claim(field="root_cause", value="x", evidence_ids=["ev_1"], seq_hint=1)


def test_allocate_claim_uses_the_run_counter() -> None:
    claim = Claim(field="root_cause", value="x", evidence_ids=["ev_1"])
    first = allocate_claim(claim, _counter(5))
    second = allocate_claim(claim, _counter(5))
    assert (first.seq, second.seq) == (5, 5), "计数器由调用方持有，这里是独立实例"

    shared = _counter(5)
    assert (allocate_claim(claim, shared).seq, allocate_claim(claim, shared).seq) == (5, 6)


# --- E1 非空 ---------------------------------------------------------------
def test_e1_empty_evidence_is_rejected() -> None:
    claim = Claim.model_construct(field=ClaimField.ROOT_CAUSE, value="x", evidence_ids=[])
    decision = bind_claims(run_id=RUN, claims=[ClaimWithSeq(claim, 9)], ledger=[])
    assert decision.escalate is True
    assert (ClaimField.ROOT_CAUSE, BindingViolation.EMPTY_EVIDENCE) in decision.violations


# --- E2 存在且 ok，且必须在本 run 内 ---------------------------------------
def test_e2_unknown_evidence_is_rejected() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_9"], seq=10)],
        ledger=[_evidence("ev_1", 1)],
    )
    assert (ClaimField.ERROR_TYPE, BindingViolation.UNKNOWN_EVIDENCE) in decision.violations


def test_e2_skipped_evidence_is_rejected() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 1, status=EvidenceStatus.SKIPPED)],
    )
    assert (ClaimField.ERROR_TYPE, BindingViolation.EVIDENCE_NOT_OK) in decision.violations


def test_e2_cross_run_evidence_is_rejected() -> None:
    """邻 run 的同名 `ev_1`（status=ok）不得通过 —— P0.6 主键修复的回归用例。"""
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 1, run_id="run-2")],
    )
    assert (ClaimField.ERROR_TYPE, BindingViolation.UNKNOWN_EVIDENCE) in decision.violations
    assert decision.escalate is True


# --- E4 禁止倒因为果 -------------------------------------------------------
@pytest.mark.parametrize("evidence_seq", [10, 11, 99])
def test_e4_evidence_must_precede_claim(evidence_seq: int) -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", evidence_seq)],
    )
    assert (ClaimField.ERROR_TYPE, BindingViolation.CAUSAL_INVERSION) in decision.violations


def test_e4_earlier_evidence_passes() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 9)],
    )
    assert decision.escalate is False
    assert decision.accepted[0].claim.value == "程序性错误"


# --- E3 分级 ---------------------------------------------------------------
def test_e3_procedural_error_needs_strong_evidence() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert (ClaimField.ERROR_TYPE, BindingViolation.EVIDENCE_CLASS_MISMATCH) in decision.violations


def test_e3_reading_error_accepts_condition_check_with_cap_and_review() -> None:
    """审题性要放宽，必须**同时**满足：实际引用 condition_check + 子类在预枚举内。"""
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ERROR_SUBTYPE, "misread_condition", ["ev_1"], seq=11),
        ],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert decision.escalate is False
    assert decision.confidence_cap == SEMI_EVIDENCE_CAP
    assert decision.needs_review is True


def test_e3_reading_error_without_condition_check_is_not_relaxed() -> None:
    """声明审题性但只引用判不了的 `numeric_check`（skipped）→ 必须 escalate。

    （审查 §6-必1 指定的反向用例：档位不许由模型自述换。）
    """
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ERROR_SUBTYPE, "misread_condition", ["ev_1"], seq=11),
        ],
        ledger=[
            _evidence(
                "ev_1",
                1,
                kind=EvidenceKind.NUMERIC_CHECK,
                status=EvidenceStatus.SKIPPED,
                payload={"blocked_reason": "symbolic", "left": {"verdict": "unsupported"}},
            )
        ],
    )
    assert decision.escalate is True
    assert (ClaimField.ERROR_TYPE, BindingViolation.EVIDENCE_NOT_OK) in decision.violations


def test_e3_reading_relaxation_needs_enumerated_subtype() -> None:
    """有 condition_check 但子类不在预枚举里（模型自造子类）→ 按强证据要求处理。"""
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ERROR_SUBTYPE, "我随便写一个", ["ev_1"], seq=11),
        ],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert decision.escalate is True
    assert (ClaimField.ERROR_TYPE, BindingViolation.EVIDENCE_CLASS_MISMATCH) in decision.violations


def test_e3_reading_relaxation_requires_subtype_claim_at_all() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert decision.escalate is True


def test_e3_procedural_cannot_borrow_the_reading_tier() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ERROR_SUBTYPE, "misread_condition", ["ev_1"], seq=11),
        ],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert decision.escalate is True
    assert (ClaimField.ERROR_TYPE, BindingViolation.EVIDENCE_CLASS_MISMATCH) in decision.violations


def test_e3_other_error_needs_student_text_plus_one_more() -> None:
    """§3.3.3 第三档：`其他` 允许 `student_text` + 任意一条其他证据。"""
    only_text = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "其他", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.STUDENT_TEXT)],
    )
    assert only_text.escalate is True, "只引用学生原话永远撑不起错因"

    with_more = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "其他", ["ev_1", "ev_2"], seq=10)],
        ledger=[
            _evidence("ev_1", 1, kind=EvidenceKind.STUDENT_TEXT),
            _evidence("ev_2", 2, kind=EvidenceKind.KB_LOOKUP),
        ],
    )
    assert with_more.escalate is False


def test_e3_reading_error_with_strong_evidence_is_not_capped() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ERROR_SUBTYPE, "unit_confusion", ["ev_1"], seq=11),
        ],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.NUMERIC_CHECK)],
    )
    assert decision.escalate is False
    assert decision.confidence_cap == 1.0
    assert decision.needs_review is False


def test_e3_kp_ids_needs_lookup_evidence() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.KP_IDS, ["kp_1"], ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.NUMERIC_CHECK)],
    )
    assert (ClaimField.KP_IDS, BindingViolation.EVIDENCE_CLASS_MISMATCH) in decision.violations


def test_e3_first_divergent_step_needs_step_diff() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.FIRST_DIVERGENT_STEP, "2", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.NUMERIC_CHECK)],
    )
    assert (
        ClaimField.FIRST_DIVERGENT_STEP,
        BindingViolation.EVIDENCE_CLASS_MISMATCH,
    ) in decision.violations


def test_e3_root_cause_is_never_relaxed() -> None:
    """`root_cause` 不随 `error_type` 放宽（自由文本，最容易被编）。"""
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ERROR_SUBTYPE, "misread_condition", ["ev_1"], seq=11),
            _claim(ClaimField.ROOT_CAUSE, "把条件看反了", ["ev_1"], seq=12),
        ],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert (ClaimField.ROOT_CAUSE, BindingViolation.EVIDENCE_CLASS_MISMATCH) in decision.violations
    assert decision.escalate is False, "error_type 与 error_subtype 仍可成立"
    assert decision.needs_review is True


def test_e3_reading_subtype_itself_is_accepted() -> None:
    """审题性 + 预枚举子类 + condition_check → `error_subtype` **必须** accepted。

    上一版把 `error_subtype` 归进"永不放宽"，与"放宽 `error_type` 必须读 subtype"
    自相矛盾，导致审题性错误只剩一个字段（审查 §6-必1，是上轮裁定的副作用）。
    """
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ERROR_SUBTYPE, "misread_condition", ["ev_1"], seq=11),
        ],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    fields = {item.claim.field for item in decision.accepted}
    assert fields == {ClaimField.ERROR_TYPE, ClaimField.ERROR_SUBTYPE}
    assert decision.violations == []
    assert decision.confidence_cap == SEMI_EVIDENCE_CAP


def test_e3_subtype_must_be_in_the_enumerated_set_even_when_relaxed() -> None:
    """枚举字段的"在集合内"本身就是约束：自造子类不放宽。"""
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ERROR_SUBTYPE, "我随手写一个", ["ev_1"], seq=11),
        ],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert (ClaimField.ERROR_SUBTYPE, BindingViolation.EVIDENCE_CLASS_MISMATCH) in decision.violations
    assert (ClaimField.ERROR_TYPE, BindingViolation.EVIDENCE_CLASS_MISMATCH) in decision.violations


def test_e3_subtype_never_relaxed_by_non_reading_error_type() -> None:
    """`error_type=程序性错误` 时，枚举子类也不能靠 `condition_check` 通过。"""
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ERROR_SUBTYPE, "misread_condition", ["ev_1"], seq=11),
        ],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert (ClaimField.ERROR_SUBTYPE, BindingViolation.EVIDENCE_CLASS_MISMATCH) in decision.violations


# --- E5 escalate + 两个分母 ------------------------------------------------
def test_e5_no_claims_escalates_as_mechanism() -> None:
    decision = bind_claims(run_id=RUN, claims=[], ledger=[])
    assert decision.escalate is True
    assert decision.escalation_kind is EscalationKind.MECHANISM
    assert decision.reason_code == BindingViolation.EMPTY_EVIDENCE.value


def test_e5_rejected_error_type_escalates() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "概念性错误", ["ev_9"], seq=10)],
        ledger=[_evidence("ev_1", 1)],
    )
    assert decision.escalate is True
    assert decision.reason_code == BindingViolation.UNSATISFIABLE.value


def test_capability_block_is_counted_separately_from_mechanism() -> None:
    """`step_diff` 占位、`numeric_check` 遇符号 → 证据 skipped。这是能力边界，不是守卫太严。"""
    capability = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10)],
        ledger=[
            _evidence(
                "ev_1",
                1,
                kind=EvidenceKind.NUMERIC_CHECK,
                status=EvidenceStatus.SKIPPED,
                payload={"blocked_reason": "symbolic", "left": {"verdict": "unsupported"}},
            )
        ],
    )
    assert capability.escalate is True
    assert capability.escalation_kind is EscalationKind.CAPABILITY

    mechanism = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_9"], seq=10)],
        ledger=[],
    )
    assert mechanism.escalation_kind is EscalationKind.MECHANISM

    report = over_escalation([capability, mechanism, capability])
    assert report.total == 3
    assert report.capability == 2
    assert report.mechanism == 1
    assert report.capability_rate == pytest.approx(2 / 3)
    assert report.mechanism_rate == pytest.approx(1 / 3)


def test_timeout_is_a_capability_block_not_a_mechanism_fault() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "计算表达错误", ["ev_1"], seq=10)],
        ledger=[
            _evidence(
                "ev_1",
                1,
                status=EvidenceStatus.SKIPPED,
                payload={"left": {"verdict": "timeout", "reason": "eval_timeout"}},
            )
        ],
    )
    assert decision.escalation_kind is EscalationKind.CAPABILITY


def test_step_diff_placeholder_counts_as_capability() -> None:
    """`step_diff` 占位（`status=not_implemented`）是**能力边界**，不是机制故障。

    上一版去 payload 里找 `reason` 字段 → 找不到 → 误判成 mechanism，
    报告会得出"机制性问题很严重"的相反结论（审查 §6-必3）。
    """
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_9"], seq=10)],
        ledger=[
            _evidence(
                "ev_1",
                1,
                kind=EvidenceKind.STEP_DIFF,
                status=EvidenceStatus.SKIPPED,
                payload={
                    "status": "not_implemented",
                    "locator": "model",
                    "note": "P1.8 才做",
                },
            )
        ],
    )
    assert decision.escalate is True
    assert decision.escalation_kind is EscalationKind.CAPABILITY
    report = over_escalation([decision])
    assert (report.capability, report.mechanism) == (1, 0)


def test_failed_evidence_is_a_mechanism_fault() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10)],
        ledger=[
            _evidence(
                "ev_1",
                1,
                status=EvidenceStatus.FAILED,
                payload={"error": "spawn_failed"},
            )
        ],
    )
    assert decision.escalation_kind is EscalationKind.MECHANISM


def test_evidence_without_any_block_reason_is_mechanism() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "程序性错误", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 1, status=EvidenceStatus.SKIPPED, payload={})],
    )
    assert decision.escalation_kind is EscalationKind.MECHANISM


# --- P7：kb_lookup 产证据 --------------------------------------------------
@pytest.fixture()
def conn(tmp_path) -> sqlite3.Connection:  # noqa: ANN001
    from veridx import config
    from veridx.store import db as store_db

    connection, _ = store_db.init_db(tmp_path / "kp.db", config.MIGRATIONS_DIR)
    connection.execute("INSERT INTO kp (id, name) VALUES ('kp_1', '一元二次方程')")
    connection.commit()
    yield connection
    connection.close()


def test_kb_lookup_produces_evidence(conn: sqlite3.Connection) -> None:
    ctx = ToolContext(run_id=RUN, conn=conn)
    result = kb_lookup({"kp_ids": ["kp_1"]}, ctx)
    assert result["status"] == "ok"
    assert result["evidence_id"] == "ev_1"
    assert ctx.next_seq() == 2, "查表也要占一个序号（否则 evidence 与 claim 会撞号）"


def test_kb_lookup_marks_partial_lookup_as_skipped(conn: sqlite3.Connection) -> None:
    ctx = ToolContext(run_id=RUN, conn=conn)
    result = kb_lookup({"kp_ids": ["kp_1", "kp_nope"]}, ctx)
    assert result["missing"] == ["kp_nope"]
    rows = conn.execute(
        "SELECT kind, status FROM evidence WHERE run_id = ? AND id = ?",
        (RUN, result["evidence_id"]),
    ).fetchall()
    assert (rows[0]["kind"], rows[0]["status"]) == ("kb_lookup", "skipped")


def test_kb_lookup_evidence_can_support_kp_ids(conn: sqlite3.Connection) -> None:
    ctx = ToolContext(run_id=RUN, conn=conn)
    result = kb_lookup({"kp_ids": ["kp_1"]}, ctx)
    ledger = [
        LedgerEntry(
            run_id=RUN,
            evidence=Evidence(
                id=result["evidence_id"],
                kind=EvidenceKind.KB_LOOKUP,
                produced_by="kb_lookup",
                seq=1,
            ),
        )
    ]
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.KP_IDS, ["kp_1"], [result["evidence_id"]], seq=2)],
        ledger=ledger,
    )
    assert decision.accepted, decision.violations


# --- 对外口径：分母 = 全部题数，分子排除"正确的 escalate" -------------------
class _Outcome:
    """最小替身：`RunResult` 的 (terminal, reason_code, escalation_kind) 三属性。"""

    def __init__(self, terminal: str, reason_code: str | None, kind: str | None = None) -> None:
        self.terminal = terminal
        self.reason_code = reason_code
        self.escalation_kind = EscalationKind(kind) if kind else None


def test_over_escalation_denominator_is_all_items() -> None:
    """分母必须是**全部题数**：只数"跑到 bind_claims 的题"是另一个总体。"""
    outcomes = [
        _Outcome("diagnosed", None),
        _Outcome("diagnosed", None),
        _Outcome("escalated", "insufficient_evidence", "mechanism"),
        _Outcome("uncertain", "timed_out"),
        _Outcome("failed", "internal_error"),
    ]
    report = over_escalation_from_outcomes(outcomes)
    assert report.total == 5
    assert report.diagnosed == 2
    assert report.uncertain == 1
    assert report.failed == 1
    assert report.mechanism == 1
    assert report.mechanism_rate == pytest.approx(1 / 5)


@pytest.mark.parametrize(
    "reason",
    ["student_answer_correct", "suspected_bad_answer_key", "unparsable_stem", "unparsable_answer"],
)
def test_justified_escalations_are_not_over_escalation(reason: str) -> None:
    """学生本来就对 / 题目有误 / 输入不可用 —— 这些是**正确终局**，不是过度拒绝。"""
    report = over_escalation_from_outcomes([_Outcome("escalated", reason, "mechanism")])
    assert report.justified == 1
    assert report.over_escalation_total == 0
    assert report.justified_rate == pytest.approx(1.0)


def test_report_reports_three_numbers_separately() -> None:
    outcomes = [
        _Outcome("diagnosed", None),
        _Outcome("escalated", "low_confidence", "capability"),
        _Outcome("escalated", "low_confidence", "capability"),
        _Outcome("escalated", "insufficient_evidence", "mechanism"),
        _Outcome("escalated", "student_answer_correct", None),
    ]
    data = over_escalation_from_outcomes(outcomes).as_dict()
    assert data["over_escalation_mechanism"] == 1
    assert data["over_escalation_capability"] == 2
    assert data["justified_escalation"] == 1
    assert data["diagnosed"] == 1
    assert data["total"] == 5
    assert data["over_escalation_rate"] == pytest.approx(0.6)


def test_unclassified_escalation_counts_as_mechanism() -> None:
    """没标注口径的 escalate 一律算机制性：宁可高估自己的问题，不美化指标。"""
    report = over_escalation_from_outcomes([_Outcome("escalated", "tool_error")])
    assert report.mechanism == 1


def test_outcome_aggregator_accepts_dicts() -> None:
    report = over_escalation_from_outcomes(
        [{"terminal": "diagnosed", "reason_code": None, "escalation_kind": None}]
    )
    assert report.diagnosed == 1
