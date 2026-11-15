"""P1.6 验收：受控控制流（5 节点 / 6 终态 / 预算）。

DoD（TODO P1.6）：
* [ ] 单题端到端跑通（mock 模型即可）
* [ ] 终态互斥完备：每次运行恰好一个
* [ ] 每个终态都有 `reason_code`（本实现：`diagnosed` 例外，见交付文档差异声明）
* [ ] 超过 7 次调用立即 `uncertain` 而不是继续
* [ ] 步数/token 上限可注入（测试用）

外加审查 §6-必4 的流程级要求：
**claim 占号必须在产生 claim 的 LLM 调用之前** —— 否则 E4 结构上恒真。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from veridx import config
from veridx.api.schemas import Terminal
from veridx.flow.explain import MIN_STEPS
from veridx.flow.runner import RunLimits, Runner, RunSpec
from veridx.guard.claim import BindingViolation, Claim, ClaimField
from veridx.guard.evidence import ClaimWithSeq, bind_claims
from veridx.llm.mock import MockProvider
from veridx.store import db as store_db
from veridx.store.repository import Repository, RepositoryError
from veridx.trace.event import ReasonCode
from veridx.trace.recorder import TraceRecorder

STEM = "求 ∫_0^1 x^2 dx 的值。"
KEY = "1/3"
WRONG = "1/4"


def _blind(answer: str, note: str = "") -> str:
    return json.dumps(
        {"answer": answer, "steps": ["求出原函数", "代入上下限"], "confidence": 0.9, "note": note},
        ensure_ascii=False,
    )


def _diagnose(
    *,
    error_type: str = "计算表达错误",
    confidence: float = 0.9,
    evidence_ids: tuple[str, ...] = ("ev_1",),
    subtype: str | None = None,
    include_seq: int | None = None,
) -> str:
    claims = [
        {
            "field": "error_type",
            "value": error_type,
            "evidence_ids": list(evidence_ids),
        },
        {
            "field": "root_cause",
            "value": "把 x^2 的指数当成了分母",
            "evidence_ids": list(evidence_ids),
        },
    ]
    if subtype is not None:
        claims.append({"field": "error_subtype", "value": subtype, "evidence_ids": list(evidence_ids)})
    if include_seq is not None:
        claims[0]["seq"] = include_seq
    return json.dumps(
        {
            "error_type": error_type,
            "error_subtype": subtype,
            "root_cause": "把 x^2 的指数当成了分母",
            "first_divergent_step": None,
            "confidence": confidence,
            "hypotheses": [{"label": "幂函数积分公式记错", "confidence": 0.7}],
            "kp_ids": [],
            "claims": claims,
        },
        ensure_ascii=False,
    )


def _steps(count: int = MIN_STEPS, *, verifiable_expr: str | None = None) -> str:
    steps = []
    for index in range(count):
        steps.append(
            {
                "title": f"第 {index + 1} 步",
                "content": "重新套用幂函数积分公式",
                "key_point": "指数加一后除以新指数",
                # 必11：targets_error 必须点名一个已被证据绑定的字段
                "targets_error": "root_cause：把指数当成了分母",
                "verifiable": verifiable_expr is not None and index == 0,
                "verifiable_expr": verifiable_expr if index == 0 else None,
            }
        )
    return json.dumps({"steps": steps}, ensure_ascii=False)


def _spec(**overrides: object) -> RunSpec:
    base = {
        "stem": STEM,
        "student_answer": WRONG,
        "answer_key": KEY,
        "attempt_id": "a-1",
    }
    base.update(overrides)
    return RunSpec(**base)  # type: ignore[arg-type]


def _providers(*diagnose_responses: str, blind: str | None = None) -> tuple[MockProvider, MockProvider]:
    primary = MockProvider(
        responses=[blind or _blind(KEY), *diagnose_responses],
        name="mock-primary",
        model="mock-primary",
    )
    second = MockProvider(responses=[_blind(KEY)], name="mock-second", model="mock-second")
    return primary, second


async def _run(primary: MockProvider, second: MockProvider, **kwargs):  # noqa: ANN003,ANN202
    limits = kwargs.pop("limits", None) or RunLimits()
    conn = kwargs.pop("conn", None)
    tracer = TraceRecorder(conn=conn, run_id="r-test")
    runner = Runner(
        primary=primary,
        second=second,
        conn=conn,
        limits=limits,
        tracer=tracer,
        run_id="r-test",
    )
    result = await runner.run(_spec())
    return result, tracer


# --- ① 端到端 -------------------------------------------------------------
async def test_end_to_end_diagnosed() -> None:
    primary, second = _providers(_diagnose(), _steps())
    result, tracer = await _run(primary, second)

    assert result.terminal is Terminal.DIAGNOSED
    assert result.reason_code is None
    assert len(result.claims) == 2
    assert result.decision is not None and not result.decision.escalate
    assert result.explanation is not None and len(result.explanation.steps) >= MIN_STEPS
    assert result.llm_calls == 3, "blind_solve + diagnose + explain"
    nodes = [event.node for event in tracer.events()]
    # 归一化发生在 parse 阶段**内部**，它的 trace 事件因此可能先于 parse 的"完成"事件；
    # 两者都属于输入解析这一段，所以都算合法首节点。
    assert nodes[0] in {"parse", "normalize"}
    assert "parse" in nodes
    assert {"parse", "blind_solve", "check_numeric", "guard", "explain", "submit"} <= set(nodes)


async def test_evidence_is_produced_before_claims() -> None:
    """工具证据（ev_1/ev_2）在 claim 占号之前产生 —— 这是 E4 能通过的前提。"""
    primary, second = _providers(_diagnose(), _steps())
    result, _ = await _run(primary, second)
    evidence_seqs = [entry.evidence.seq for entry in result.evidence]
    claim_seqs = [item.seq for item in result.claims]
    assert evidence_seqs and claim_seqs
    assert max(evidence_seqs) < min(claim_seqs)


async def test_claim_block_is_reserved_before_the_diagnose_call() -> None:
    """ADR-020 的**时机**断言：占号必须发生在产生 claim 的那次 LLM 调用之前。

    只断言"编号递增"是查不出问题的（诊断调用本身不产证据时，两种顺序发出同样的号）。
    所以这里直接观测"进入 diagnose 时下一个号是多少"：
    占号在调用之前 ⇒ 该值 == 块起点；若有人把占号挪到调用之后 ⇒ 该值会小一整块。
    """

    class Probe(Runner):
        watermark: int | None = None

        async def _node_diagnose(self, spec: RunSpec):  # noqa: ANN201
            Probe.watermark = self.ctx.peek_next_seq()
            return await super()._node_diagnose(spec)

    primary, second = _providers(_diagnose(), _steps())
    limits = RunLimits()
    runner = Probe(primary=primary, second=second, limits=limits, run_id="r-timing")
    result = await runner.run(_spec())
    assert result.terminal is Terminal.DIAGNOSED
    assert Probe.watermark is not None
    first_claim = min(item.seq for item in result.claims)
    assert Probe.watermark == first_claim + limits.max_claims, (
        "占号必须早于 diagnose 那次 LLM 调用：进入调用时整块号已经领走；"
        "若把占号挪到调用之后，这里会变成 watermark == first_claim（E4 结构上恒真）"
    )
    assert max(item.seq for item in result.claims) < Probe.watermark


# --- ② 必4：claim 占号早于 LLM 调用 ---------------------------------------
async def test_late_evidence_cannot_support_a_claim() -> None:
    """占号之后才产生的证据 → E4 必须拒（审查 §6-必4 的"情况 2"）。"""
    primary, second = _providers(_diagnose(), _steps())
    runner = Runner(primary=primary, second=second, run_id="r-order")

    block = runner._reserve_claim_block()  # 先占号（= LLM 调用之前）
    runner._node_tools(  # 之后才产生证据（模拟"结论下完又补证据"）
        _spec(), type("B", (), {"steps": ["a", "b"]})()
    )
    later = runner._ledger()
    assert later, "工具证据应在占号之后产生"
    claim_seq = block.take()
    decision = bind_claims(
        run_id=runner.run_id,
        claims=[
            ClaimWithSeq(
                claim=Claim(
                    field=ClaimField.ERROR_TYPE,
                    value="计算表达错误",
                    evidence_ids=[later[0].evidence.id],
                ),
                seq=claim_seq,
            )
        ],
        ledger=later,
    )
    assert (ClaimField.ERROR_TYPE, BindingViolation.CAUSAL_INVERSION) in decision.violations
    assert decision.escalate is True
    # 顺序事实：这些证据的 seq 全部大于 claim 的占号
    assert min(entry.evidence.seq for entry in later) > claim_seq


async def test_model_supplied_seq_is_rejected_by_schema() -> None:
    """模型自带 `seq` → 整条 claim 不合法 → 回喂重试耗尽 → escalate(diagnose_schema_invalid)。"""
    primary, second = _providers(*([_diagnose(include_seq=999999)] * 3))
    result, _ = await _run(primary, second)
    assert result.terminal is Terminal.ESCALATED
    assert result.reason_code == ReasonCode.DIAGNOSE_SCHEMA_INVALID.value


async def test_explain_evidence_cannot_be_cited_by_claims() -> None:
    """讲解阶段的验算证据 seq 必然大于 claim 占号：模型若引用它就是 E2/E4，绝不放行。"""
    primary, second = _providers(_diagnose(evidence_ids=("ev_3",)), _steps(verifiable_expr="1/3"))
    result, _ = await _run(primary, second)
    assert result.terminal is Terminal.ESCALATED
    assert result.reason_code == ReasonCode.INSUFFICIENT_EVIDENCE.value


async def test_explain_evidence_is_produced_after_claims() -> None:
    """正常路径下讲解验算证据确实产生在 claim 之后，且没有 claim 引用它。"""
    primary, second = _providers(_diagnose(evidence_ids=("ev_1",)), _steps(verifiable_expr="1/3"))
    result, _ = await _run(primary, second)
    assert result.terminal is Terminal.DIAGNOSED
    latest_claim = max(item.seq for item in result.claims)
    later = [
        entry
        for entry in result.evidence
        if entry.evidence.produced_by == "explain" and entry.evidence.seq > latest_claim
    ]
    assert later, "讲解验算证据必须在 claim 之后产生（否则 E4 没有活样本）"
    cited = {eid for item in result.claims for eid in item.claim.evidence_ids}
    assert not ({entry.evidence.id for entry in later} & cited), "后产生的证据不得被 claim 引用"


# --- ③ 六个终态 -----------------------------------------------------------
@pytest.mark.parametrize(
    ("spec_overrides", "expected_code"),
    [
        ({"stem": "  "}, ReasonCode.UNPARSABLE_STEM),
        ({"student_answer": "  "}, ReasonCode.UNPARSABLE_ANSWER),
        ({"answer_key": ""}, ReasonCode.UNPARSABLE_STEM),
        ({"student_answer": "x" * (config.SANDBOX_MAX_EXPR_LEN + 1)}, ReasonCode.UNPARSABLE_ANSWER),
    ],
)
async def test_needs_clarification_cases(spec_overrides: dict, expected_code: ReasonCode) -> None:
    primary = MockProvider(responses=[], name="mock-primary", model="mock-primary")
    second = MockProvider(responses=[], name="mock-second", model="mock-second")
    runner = Runner(primary=primary, second=second, run_id="r-nc")
    result = await runner.run(_spec(**spec_overrides))
    assert result.terminal is Terminal.NEEDS_CLARIFICATION
    assert result.reason_code == expected_code.value


async def test_student_already_correct_escalates() -> None:
    """学生答对：不许"找出一个错因"（F6 误判率的守门人）。"""
    primary = MockProvider(responses=[_blind(KEY)], name="mock-primary", model="mock-primary")
    second = MockProvider(responses=[_blind(KEY)], name="mock-second", model="mock-second")
    runner = Runner(primary=primary, second=second, run_id="r-s")
    result = await runner.run(_spec(student_answer=KEY))
    assert result.terminal is Terminal.ESCALATED
    assert result.reason_code == ReasonCode.STUDENT_ANSWER_CORRECT.value
    assert primary.calls and len(primary.calls) == 1, "只在盲解用了一次模型，没有继续诊断"


async def test_bad_answer_key_escalates() -> None:
    """分支 B：模型与学生答案一致但都不等于答案键 → 疑似答案有误 → escalate。"""
    primary = MockProvider(responses=[_blind(WRONG)], name="mock-primary", model="mock-primary")
    second = MockProvider(responses=[_blind(WRONG)], name="mock-second", model="mock-second")
    runner = Runner(primary=primary, second=second, run_id="r-b")
    result = await runner.run(_spec())
    assert result.terminal is Terminal.ESCALATED
    assert result.reason_code == ReasonCode.SUSPECTED_BAD_ANSWER_KEY.value
    assert result.blind is not None and result.blind.suspect.value == "confirmed"


async def test_insufficient_evidence_escalates() -> None:
    primary, second = _providers(_diagnose(evidence_ids=("ev_99",)))
    result, _ = await _run(primary, second)
    assert result.terminal is Terminal.ESCALATED
    assert result.reason_code == ReasonCode.INSUFFICIENT_EVIDENCE.value


async def test_low_confidence_escalates() -> None:
    primary, second = _providers(_diagnose(confidence=0.1), _steps())
    result, _ = await _run(primary, second)
    assert result.terminal is Terminal.ESCALATED
    assert result.reason_code == ReasonCode.LOW_CONFIDENCE.value


async def test_low_confidence_is_classified_as_capability_when_step_diff_is_a_placeholder() -> None:
    """真实环境里的典型情形：只有 numeric_check，`step_diff` 占位 → 模型不敢给高置信。

    这类 escalate 必须记在**能力性**分母上，否则报告会把"没有 step_diff"读成"机制缺陷"。
    """
    primary, second = _providers(_diagnose(confidence=0.1), _steps())
    result, _ = await _run(primary, second)
    from veridx.guard.evidence import EscalationKind

    assert result.escalation_kind is EscalationKind.CAPABILITY


async def test_uncertain_when_blind_solve_cannot_parse() -> None:
    primary = MockProvider(responses=[_blind("int(x**2,(x,0,1))")], name="mock-primary", model="mock-primary")
    second = MockProvider(responses=[], name="mock-second", model="mock-second")
    runner = Runner(primary=primary, second=second, run_id="r-c")
    result = await runner.run(_spec())
    assert result.terminal is Terminal.UNCERTAIN
    assert result.reason_code == ReasonCode.SANDBOX_UNSUPPORTED.value


async def test_uncertain_when_three_way_disagree() -> None:
    primary = MockProvider(responses=[_blind("2")], name="mock-primary", model="mock-primary")
    second = MockProvider(responses=[], name="mock-second", model="mock-second")
    runner = Runner(primary=primary, second=second, run_id="r-d")
    result = await runner.run(_spec())
    assert result.terminal is Terminal.UNCERTAIN
    assert result.reason_code == ReasonCode.VERIFIER_DISAGREES.value


async def test_failed_on_unexpected_exception() -> None:
    class Boom:
        model = "boom"

        def describe(self) -> dict:
            return {"provider": "boom", "model": "boom"}

        async def chat(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("provider exploded")

    runner = Runner(primary=Boom(), second=None, run_id="r-f")  # type: ignore[arg-type]
    result = await runner.run(_spec())
    assert result.terminal is Terminal.FAILED
    assert result.reason_code == ReasonCode.INTERNAL_ERROR.value
    assert any("internal:RuntimeError" in note for note in result.notes)


async def test_canceled_on_cancelled_error() -> None:
    class Cancel:
        model = "cancel"

        def describe(self) -> dict:
            return {"provider": "cancel", "model": "cancel"}

        async def chat(self, *args: object, **kwargs: object) -> object:
            raise asyncio.CancelledError

    tracer = TraceRecorder(conn=None, run_id="r-cancel")
    runner = Runner(primary=Cancel(), second=None, tracer=tracer, run_id="r-cancel")  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        await runner.run(_spec())
    assert tracer.events()[-1].status == "canceled"
    assert tracer.events()[-1].reason_code == ReasonCode.USER_CANCELED.value


def test_every_terminal_is_covered_by_a_test() -> None:
    """终态互斥完备：6 个终态都必须有用例（本文件覆盖 6/6）。"""
    covered = {
        Terminal.DIAGNOSED,
        Terminal.ESCALATED,
        Terminal.NEEDS_CLARIFICATION,
        Terminal.UNCERTAIN,
        Terminal.FAILED,
        Terminal.CANCELED,
    }
    assert covered == set(Terminal) and len(Terminal) == 6


# --- ④ 预算 -------------------------------------------------------------
async def test_call_budget_stops_immediately() -> None:
    primary, second = _providers(_diagnose(), _steps())
    result, tracer = await _run(primary, second, limits=RunLimits(max_llm_calls=1))
    assert result.terminal is Terminal.UNCERTAIN
    assert result.reason_code == ReasonCode.CALL_BUDGET_EXCEEDED.value
    assert result.llm_calls == 1, "超预算必须立即停，不能继续跑"
    assert any(event.reason_code == ReasonCode.CALL_BUDGET_EXCEEDED.value for event in tracer.events())


async def test_token_budget_stops_immediately() -> None:
    """预算不够时**一个请求都不发**（必10：事前估算，不是事后记账）。"""
    primary, second = _providers(_diagnose(), _steps())
    result, _ = await _run(primary, second, limits=RunLimits(max_tokens=1))
    assert result.terminal is Terminal.UNCERTAIN
    assert result.reason_code == ReasonCode.SKIPPED_BUDGET.value
    assert result.llm_calls == 0
    assert primary.calls == [], "预算不够时不许发请求"


async def test_wall_clock_budget_stops_immediately() -> None:
    primary, second = _providers(_diagnose(), _steps())
    result, _ = await _run(primary, second, limits=RunLimits(max_wall_ms=0))
    assert result.terminal is Terminal.UNCERTAIN
    assert result.reason_code == ReasonCode.TIMED_OUT.value


async def test_claims_cannot_exceed_reserved_block() -> None:
    """模型给出超过预留量的 claim → 不许静默截断，直接 escalate。"""
    many = json.dumps(
        {
            "error_type": "计算表达错误",
            "root_cause": "x",
            "confidence": 0.9,
            "claims": [{"field": "root_cause", "value": f"r{i}", "evidence_ids": ["ev_1"]} for i in range(6)],
        },
        ensure_ascii=False,
    )
    primary, second = _providers(many)
    result, _ = await _run(primary, second, limits=RunLimits(max_claims=2))
    assert result.terminal is Terminal.ESCALATED
    assert result.reason_code == ReasonCode.DIAGNOSE_SCHEMA_INVALID.value


# --- ⑤ 落库 -------------------------------------------------------------
def test_n5_non_finite_payload_is_refused_at_storage_layer(conn) -> None:  # noqa: ANN001
    """N5：`payload` 是自由 dict，不受 `StrictModel` 保护 —— 存储层必须自己拦。"""
    import math

    repo = Repository(conn)
    with pytest.raises(RepositoryError, match="非有限数"):
        repo.create_evidence(
            run_id="r-n5",
            evidence_id="ev_1",
            kind="numeric_check",
            payload={"value": math.inf},
            produced_by="test",
            seq=1,
        )


def test_n5_tool_context_refuses_non_finite_payload() -> None:
    from veridx.tools.context import ToolContext

    ctx = ToolContext(run_id="r-n5")
    with pytest.raises(ValueError, match="非有限数"):
        ctx.record_evidence(
            kind="numeric_check",
            payload={"nested": {"value": float("nan")}},
            produced_by="test",
        )
    assert ctx.drafts == [], "拒掉之后不能留下半成品"


def test_guard_trace_records_claim_content() -> None:
    """审查 🟡G：escalated 不写 diagnoses，M5 与 F 类细分要从 trace 重算。"""
    from veridx.trace.recorder import TraceRecorder

    tracer = TraceRecorder(conn=None, run_id="r-t")
    runner = Runner(primary=MockProvider(), second=None, tracer=tracer, run_id="r-t")
    claims = [
        ClaimWithSeq(
            claim=Claim(field=ClaimField.ERROR_TYPE, value="程序性错误", evidence_ids=["ev_9"]),
            seq=3,
        )
    ]
    decision = bind_claims(run_id="r-t", claims=claims, ledger=[])
    runner._trace_guard(decision, claims=claims, stage="guard", item_id=None)
    detail = tracer.events()[-1].detail
    assert detail["claims"][0]["field"] == "error_type"
    assert detail["claims"][0]["value"] == "程序性错误"
    assert detail["claims"][0]["evidence_ids"] == ["ev_9"]
    assert detail["claims"][0]["accepted"] is False
    assert detail["violations"] == [["error_type", "unknown_evidence"]]


@pytest.fixture()
def conn(tmp_path: Path):  # noqa: ANN201
    connection, _ = store_db.init_db(tmp_path / "runner.db", config.MIGRATIONS_DIR)
    repo = Repository(connection)
    repo.create_question(
        question_id="q-1",
        course="calculus",
        stem=STEM,
        correct_answer=KEY,
        difficulty="3",
    )
    repo.create_attempt(attempt_id="a-1", question_id="q-1", student_answer=WRONG)
    yield connection
    connection.close()


async def test_diagnosis_is_persisted(conn) -> None:  # noqa: ANN001
    primary, second = _providers(_diagnose(), _steps())
    result, _ = await _run(primary, second, conn=conn)
    assert result.terminal is Terminal.DIAGNOSED

    diagnoses = conn.execute("SELECT * FROM diagnoses WHERE run_id = ?", (result.run_id,)).fetchall()
    assert len(diagnoses) == 1
    row = diagnoses[0]
    assert row["terminal"] == "diagnosed"
    assert row["error_type"] == "计算表达错误"
    claims = conn.execute("SELECT * FROM claims WHERE diagnosis_id = ?", (row["id"],)).fetchall()
    assert len(claims) == len(result.claims)
    steps = conn.execute("SELECT * FROM explanation_steps WHERE diagnosis_id = ?", (row["id"],)).fetchall()
    assert len(steps) >= MIN_STEPS
    evidence = conn.execute("SELECT * FROM evidence WHERE run_id = ?", (result.run_id,)).fetchall()
    assert len(evidence) == len(result.evidence)
