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
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10)],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert decision.escalate is False
    assert decision.confidence_cap == SEMI_EVIDENCE_CAP
    assert decision.needs_review is True


def test_e3_reading_error_with_strong_evidence_is_not_capped() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[_claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10)],
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


def test_e3_root_cause_can_rely_on_semi_evidence_when_reading_error() -> None:
    decision = bind_claims(
        run_id=RUN,
        claims=[
            _claim(ClaimField.ERROR_TYPE, "审题性错误", ["ev_1"], seq=10),
            _claim(ClaimField.ROOT_CAUSE, "把条件看反了", ["ev_1"], seq=11),
        ],
        ledger=[_evidence("ev_1", 1, kind=EvidenceKind.CONDITION_CHECK)],
    )
    assert decision.escalate is False
    assert len(decision.accepted) == 2


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
                payload={"left": {"reason": "eval_timeout"}},
            )
        ],
    )
    assert decision.escalation_kind is EscalationKind.CAPABILITY


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
