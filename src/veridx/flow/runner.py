"""受控控制流（P1.6 / 设计文档 §2.3）：5 节点 · 6 终态 · 预算。

## 为什么是"受控"而不是"自主 Agent"

控制流由代码写死，模型只决定**内容**（错因、置信度、讲解），不决定**流程**。
于是每一步都有名字、有入参出参、能回放 —— 这是可审计与可消融的前提。

```
parse → blind_solve → check_numeric/step_diff → diagnose(claims) → explain → submit
```

## 三条结构性保证

1. **终态恰好一个**：所有终态都走 `_finish()`，它设一次值并抛 `_Done` 中断后续节点。
   任何未被 `_finish` 捕获的异常一律变 `failed`；`CancelledError` 变 `canceled`。
2. **claim 占号早于产生 claim 的 LLM 调用**（ADR-020）：`reserve_claim_seqs()` 写在
   `call_structured(diagnose)` **之前**。否则 E4 结构上恒真（审查 §6-必4）。
3. **预算三重闸**：LLM 调用次数 / token / wall-clock。超了立即 `uncertain`，
   **不是**"继续跑完再说"（§2.3.2）。

## 降级阶梯（§12.2）

| 档 | 触发 | 本文实现 |
|---|---|---|
| 1 | 讲解重试后仍不合法 | `Explanation.degraded=True`，终态仍 `diagnosed` |
| 2 | 证据不足 | `escalated` + `insufficient_evidence` |
| 4 | 预算/超时 | `uncertain` + `call_budget_exceeded` / `skipped_budget` / `timed_out` |
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, Field

from veridx import config
from veridx.api.schemas import Arm, EvidenceBinding, Terminal
from veridx.clock import new_id
from veridx.extract.extractor import run_extract
from veridx.extract.schema import FieldSource
from veridx.flow.blind_solve import BlindSolveOutcome, Branch, Suspect, blind_solve
from veridx.flow.explain import Explanation, explain
from veridx.guard.claim import Claim, ClaimField, Evidence
from veridx.guard.evidence import (
    BindingDecision,
    ClaimSeqBlock,
    ClaimWithSeq,
    EscalationKind,
    LedgerEntry,
    bind_claims,
    capability_gap,
    evidence_binding_of,
    reserve_claim_seqs,
)
from veridx.guard.injection import scan_untrusted
from veridx.guard.structured import Budget, StructuredOutputError, call_structured
from veridx.llm.base import LLMProvider
from veridx.normalize.gate import CONFLICT, normalize_side
from veridx.prompts import diagnose as diagnose_prompt
from veridx.sandbox import Verdict, compare, evaluate, static_check
from veridx.sandbox.verdict import Reason
from veridx.store.repository import Repository
from veridx.taxonomy import ErrorType
from veridx.tools.context import ToolContext
from veridx.tools.dispatch import dispatch
from veridx.tools.spec import ToolName
from veridx.trace.event import NodeName, NodeStatus, ReasonCode
from veridx.trace.recorder import TraceRecorder

__all__ = ["DiagnoseDraft", "RunLimits", "RunResult", "RunSpec", "Runner"]

DEFAULT_MAX_LLM_CALLS = 7
DEFAULT_MAX_CLAIMS = 5


class DiagnoseDraft(BaseModel):
    """诊断输出契约。

    `claims` 直接复用 `Claim`（`extra="forbid"`）——模型若自带 `seq`，整条 claim 会被
    pydantic 拒掉并触发回喂重试（这是 §6.1 的反作弊面）。

    长度上限（审查 必3）与盲解同理：输出是**不可信的大对象**。
    """

    error_type: ErrorType
    error_subtype: str | None = Field(default=None, max_length=64)
    root_cause: str = Field(default="", max_length=500)
    first_divergent_step: int | None = None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list, max_length=3)
    kp_ids: list[str] = Field(default_factory=list, max_length=6)
    claims: list[Claim] = Field(min_length=1, max_length=8)


@dataclass(frozen=True)
class RunSpec:
    """单题输入。`answer_key` 只在**程序侧**参与判等，盲解阶段看不到它俩（见 blind_solve）。"""

    stem: str
    student_answer: str
    answer_key: str
    attempt_id: str = "a-unknown"
    question_id: str = "q-unknown"
    symbols: frozenset[str] | None = None
    arm: str = Arm.D.value
    item_id: str | None = None
    #: C-38（方案 §2.1）：`structured` 直接进流程；`freetext` 先过隔离提取器
    input_mode: str = "structured"
    raw_input: str | None = None
    #: 出处（`FieldSource`）：判据字段**永远**不能是 `extracted`
    stem_source: str = "user_form"
    student_answer_source: str = "user_form"
    answer_key_source: str = "user_form"


@dataclass
class RunLimits:
    """可注入的预算上限（测试用：把上限压到 1 就能验证"立即 uncertain"）。"""

    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    max_tokens: int = config.PER_ITEM_TOKENS
    max_wall_ms: int = 120_000
    max_claims: int = DEFAULT_MAX_CLAIMS
    explain: bool = True


@dataclass
class RunResult:
    run_id: str
    terminal: Terminal
    reason_code: str | None
    diagnosis: dict[str, Any] | None = None
    claims: list[ClaimWithSeq] = field(default_factory=list)
    decision: BindingDecision | None = None
    blind: BlindSolveOutcome | None = None
    explanation: Explanation | None = None
    evidence: list[LedgerEntry] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    #: 实施单 §2：输入/输出分开（`tokens` 仍是总量，预算与旧调用方不变）
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    #: 终态是因为"能力边界"还是"机制缺陷"（报告口径，§6.2）
    escalation_kind: EscalationKind | None = None
    notes: list[str] = field(default_factory=list)


def _brief(value: str | list[str], limit: int = 200) -> str | list[str]:
    """trace 里的 claim 值留档：短值原样，长值截断（原文不入库，避免 trace 膨胀）。"""
    if isinstance(value, list):
        return [item[:limit] for item in value]
    return value[:limit]


def _was_accepted(decision: BindingDecision, item: ClaimWithSeq) -> bool:
    verdict = next((v for v in decision.verdicts if v.seq == item.seq), None)
    return bool(verdict and verdict.accepted)


class _Done(Exception):
    """内部信号：终态已确定，中断后续节点。**只有 `_finish` 能抛它。**"""


class Runner:
    """一次 run 的执行器（一个实例只服务一个 run）。"""

    def __init__(
        self,
        *,
        primary: LLMProvider,
        second: LLMProvider | None = None,
        extract: LLMProvider | None = None,
        conn: Any | None = None,
        limits: RunLimits | None = None,
        tracer: TraceRecorder | None = None,
        run_id: str | None = None,
        budget: Budget | None = None,
        clock: Callable[[], float] = time.monotonic,
        normalize: bool = True,
    ) -> None:
        self.run_id = run_id or new_id("r")
        self.limits = limits or RunLimits()
        self.conn = conn
        self.clock = clock
        self.primary = primary
        self.second = second
        #: 隔离提取器 / 模型归一化器（方案 §4）：**无工具、单轮、温度 0**，与主模型是两套凭据。
        #: `None` = 没有隔离模型可用 → `freetext` 模式 fail closed，归一化只能走确定性解析器。
        self.extract = extract
        #: 关掉归一化只用于对照实验与回放（默认开）
        self.normalize_enabled = normalize
        self.budget = budget or Budget(remaining_tokens=self.limits.max_tokens)
        self.tracer = tracer or TraceRecorder(conn=conn, run_id=self.run_id)
        self._started = clock()
        self._llm_calls = 0
        self._tokens = 0
        self._tokens_in = 0
        self._tokens_out = 0
        self._terminal: Terminal | None = None
        self._reason: str | None = None
        self._blind: BlindSolveOutcome | None = None
        self._escalation_kind: EscalationKind | None = None
        #: 诊断阶段的真实重试次数（`None` = 还没走到 diagnose，别用 0 假装"没重试过"）
        self._schema_retries: int | None = None
        #: 方案 §2.1：无标准答案也**合法**（纯盲解模式），不能报成"输入坏了"
        self._key_mode: str = "present"
        #: 归一化两侧的结论（落 trace、进报告头；`None` = 该侧没有参与归一化）
        self._normalized: dict[str, Any] = {}
        self.notes: list[str] = []
        self.ctx = ToolContext(run_id=self.run_id, conn=conn)

    # ---- 公共入口 ---------------------------------------------------------
    async def run(self, spec: RunSpec) -> RunResult:
        self.ctx.symbols = spec.symbols
        blind: BlindSolveOutcome | None = None
        decision: BindingDecision | None = None
        explanation: Explanation | None = None
        claims: list[ClaimWithSeq] = []
        diagnosis: dict[str, Any] | None = None
        try:
            spec = await self._node_parse(spec)
            blind = await self._node_blind(spec)
            self._node_tools(spec, blind)
            block = self._reserve_claim_block()
            draft = await self._node_diagnose(spec)
            claims, decision = self._node_guard(draft, block, item_id=spec.item_id)
            self._check_confidence(draft, decision)
            self._node_kb(draft.kp_ids, item_id=spec.item_id)
            explanation = await self._node_explain(spec, draft, decision)
            diagnosis = self._node_submit(spec, draft, claims, decision, blind, explanation)
            self._finish(Terminal.DIAGNOSED, None)
        except _Done:
            pass
        except asyncio.CancelledError:
            self.notes.append("canceled")
            self.tracer.event(
                NodeName.SUBMIT.value,
                status=NodeStatus.CANCELED,
                reason_code=ReasonCode.USER_CANCELED.value,
                detail={"canceled_at": self._llm_calls},
                item_id=spec.item_id,
            )
            self._terminal = Terminal.CANCELED
            self._reason = ReasonCode.USER_CANCELED.value
            raise
        except Exception as exc:  # noqa: BLE001 - 未捕获异常 → failed（可归因）
            self.tracer.event(
                NodeName.SUBMIT.value,
                status=NodeStatus.FAIL,
                reason_code=ReasonCode.INTERNAL_ERROR.value,
                detail={"error": f"{type(exc).__name__}: {exc}"[:200]},
                item_id=spec.item_id,
            )
            self.notes.append(f"internal:{type(exc).__name__}")
            # 这里不能再走 `_finish`：它靠抛 `_Done` 中断流程，而我们已经在异常处理里
            self._terminal = Terminal.FAILED
            self._reason = ReasonCode.INTERNAL_ERROR.value

        return RunResult(
            run_id=self.run_id,
            terminal=self._terminal or Terminal.FAILED,
            reason_code=self._reason,
            diagnosis=diagnosis,
            claims=claims,
            decision=decision,
            blind=blind or self._blind,
            explanation=explanation,
            evidence=list(self._ledger()),
            llm_calls=self._llm_calls,
            tokens=self._tokens,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            latency_ms=int((self.clock() - self._started) * 1000),
            escalation_kind=(
                decision.escalation_kind if decision and decision.escalation_kind else self._escalation_kind
            ),
            notes=self.notes,
        )

    # ---- 终态与预算 -------------------------------------------------------
    def _finish(self, terminal: Terminal, reason_code: str | None) -> None:
        """**唯一**的终态写入口：设一次值，然后中断后续节点。"""
        if terminal is Terminal.DIAGNOSED:
            # 成功路径不带 reason_code（设计文档 §8.4 的示例即 null）；
            # 其余 5 个终态必须可归因 —— 见交付文档的差异声明。
            pass
        elif not reason_code:
            raise AssertionError(f"{terminal.value} 必须带 reason_code（I4：每个终态可归因）")
        self._terminal = terminal
        self._reason = reason_code
        raise _Done

    def _check_budget(self, stage: str) -> None:
        """每次 LLM 调用前查三重预算；超了立即 uncertain，不继续跑。"""
        if self._llm_calls >= self.limits.max_llm_calls:
            self.tracer.event(
                stage,
                status=NodeStatus.SKIPPED,
                reason_code=ReasonCode.CALL_BUDGET_EXCEEDED.value,
                detail={"llm_calls": self._llm_calls, "limit": self.limits.max_llm_calls},
            )
            self._finish(Terminal.UNCERTAIN, ReasonCode.CALL_BUDGET_EXCEEDED.value)
        if self._tokens >= self.limits.max_tokens:
            self.tracer.event(
                stage,
                status=NodeStatus.SKIPPED,
                reason_code=ReasonCode.SKIPPED_BUDGET.value,
                detail={"tokens": self._tokens, "limit": self.limits.max_tokens},
            )
            self._finish(Terminal.UNCERTAIN, ReasonCode.SKIPPED_BUDGET.value)
        elapsed_ms = int((self.clock() - self._started) * 1000)
        if elapsed_ms >= self.limits.max_wall_ms:
            self.tracer.event(
                stage,
                status=NodeStatus.SKIPPED,
                reason_code=ReasonCode.TIMED_OUT.value,
                detail={"elapsed_ms": elapsed_ms, "limit": self.limits.max_wall_ms},
            )
            self._finish(Terminal.UNCERTAIN, ReasonCode.TIMED_OUT.value)

    def _spend(self, usage_total: int, *, usage_in: int = 0, usage_out: int = 0) -> None:
        self._llm_calls += 1
        self._tokens += max(usage_total, 0)
        # 实施单 §2：输入/输出分开记。以前两个字段都填 `_tokens`（总量），
        # 于是"这道题花了多少 token / 多少钱"的分析全是假的。
        self._tokens_in += max(usage_in, 0)
        self._tokens_out += max(usage_out, 0)
        self.budget.spend(max(usage_total, 0))

    # ---- 节点 0：隔离提取（只有 freetext 才走）----------------------------
    async def _node_extract(self, spec: RunSpec) -> RunSpec:
        """自由文本 → 结构化字段（方案 §1 ~ §4）。

        三条顺序不能换：**静态前置（不调模型）→ 隔离提取 → 出处核验**。
        任何一步失败都 fail closed，绝不"用原始文本硬上"。
        """
        raw = (spec.raw_input or "").strip()
        if not raw:
            self.tracer.event(
                NodeName.EXTRACT.value,
                status=NodeStatus.SKIPPED,
                reason_code=ReasonCode.UNPARSABLE_STEM.value,
                detail={"reason": "empty_raw_input"},
                item_id=spec.item_id,
            )
            self._finish(Terminal.NEEDS_CLARIFICATION, ReasonCode.UNPARSABLE_STEM.value)
        # ① 静态前置：长度与字符集，一个请求都不发就知道它不可能是题
        if len(raw) > config.STEM_MAX_CHARS * 2:
            self.tracer.event(
                NodeName.EXTRACT.value,
                status=NodeStatus.SKIPPED,
                reason_code=ReasonCode.EXTRACTION_FAILED.value,
                detail={"reason": "raw_too_long", "raw_chars": len(raw)},
                item_id=spec.item_id,
            )
            self._finish(Terminal.NEEDS_CLARIFICATION, ReasonCode.EXTRACTION_FAILED.value)
        if not any(ch.isprintable() for ch in raw):
            self._finish(Terminal.NEEDS_CLARIFICATION, ReasonCode.EXTRACTION_FAILED.value)

        # ② 隔离提取：没有隔离模型可用 → fail closed（不许退回主模型，那是特权混淆）
        self._check_budget(NodeName.EXTRACT.value)
        if self.extract is None:
            self.tracer.event(
                NodeName.EXTRACT.value,
                status=NodeStatus.SKIPPED,
                reason_code=ReasonCode.EXTRACTION_FAILED.value,
                detail={"reason": "no_extract_provider"},
                item_id=spec.item_id,
            )
            self._finish(Terminal.NEEDS_CLARIFICATION, ReasonCode.EXTRACTION_FAILED.value)
        outcome = await run_extract(self.extract, raw, tracer=self.tracer, budget=self.budget)
        if outcome.attempts:
            self._spend(
                outcome.tokens_in + outcome.tokens_out,
                usage_in=outcome.tokens_in,
                usage_out=outcome.tokens_out,
            )
        if not outcome.ok or outcome.draft is None:
            code = outcome.reason_code or ReasonCode.EXTRACTION_FAILED.value
            terminal = (
                Terminal.NEEDS_CLARIFICATION
                if code in {ReasonCode.EXTRACTION_FAILED.value, ReasonCode.PROVENANCE_MISMATCH.value}
                else Terminal.ESCALATED
            )
            self.notes.append(f"extract_failed:{code}")
            self._finish(terminal, code)

        # ③ 判据装配：**数据字段**可以来自提取器，**判据字段**永远不行（方案 §1 铁律 1）
        draft = outcome.draft
        spec = replace(
            spec,
            stem=draft.stem,
            stem_source=FieldSource.EXTRACTED.value,
            student_answer=draft.student_answer or "",
            student_answer_source=(
                FieldSource.EXTRACTED.value if draft.student_answer else FieldSource.DEFAULT.value
            ),
        )
        self.tracer.event(
            NodeName.EXTRACT.value,
            status=NodeStatus.OK,
            detail={
                "stage": "assemble",
                "stem_source": spec.stem_source,
                "student_answer_source": spec.student_answer_source,
                "answer_key_source": spec.answer_key_source,
                "detected_lang": draft.detected_lang,
                "layout_hint": draft.layout_hint,
                # v1.1：偏移是**服务端算的**（审计用），不是模型给的
                "segments": outcome.segments.to_detail() if outcome.segments else None,
                "verbatim": "ok",
            },
            item_id=spec.item_id,
        )
        return spec

    # ---- 节点 1：parse ---------------------------------------------------
    async def _node_parse(self, spec: RunSpec) -> RunSpec:
        if spec.input_mode == "freetext":
            spec = await self._node_extract(spec)

        stem = spec.stem.strip()
        answer = spec.student_answer.strip()
        if not stem:
            raise self._reject(NodeName.PARSE, ReasonCode.UNPARSABLE_STEM, "题干为空")
        if not answer:
            raise self._reject(NodeName.PARSE, ReasonCode.UNPARSABLE_ANSWER, "学生答案为空")
        if len(answer) > config.SANDBOX_MAX_EXPR_LEN:
            # 与 API 入口闸门同一口径（§8.5），控制流自己也守一道
            self._finish(Terminal.NEEDS_CLARIFICATION, ReasonCode.UNPARSABLE_ANSWER.value)
        if not spec.answer_key.strip():
            # 方案 §2.1：无标准答案**合法**（纯盲解模式）。这里不能报"输入坏了"。
            # 只有结构化模式（表单/数据集）才把缺答案键当输入问题。
            if spec.input_mode == "freetext":
                self._key_mode = "absent"
                self.notes.append("key_mode=absent")
            else:
                raise self._reject(NodeName.PARSE, ReasonCode.UNPARSABLE_STEM, "标准答案为空")

        spec = await self._node_normalize(spec)

        if spec.answer_key.strip():
            key_static = static_check(spec.answer_key)
            if key_static is not None and key_static[1].startswith(Reason.NOT_REPRESENTABLE.value):
                # 必6（X4）：**答案键本身**超出可表示范围 → 能力边界（映射 escalated），
                # 不能让它退化成"判不了/格式错"：那会把容量问题说成输入问题。
                # 静态判（0.2ms）→ 一个请求都不发，比等模型答完再说便宜得多。
                self.tracer.event(
                    NodeName.PARSE.value,
                    status=NodeStatus.SKIPPED,
                    reason_code=ReasonCode.ANSWER_UNREPRESENTABLE.value,
                    detail={"reason": key_static[1], "answer_key_len": len(spec.answer_key)},
                    item_id=spec.item_id,
                )
                self._finish(Terminal.ESCALATED, ReasonCode.ANSWER_UNREPRESENTABLE.value)
        self.tracer.event(
            NodeName.PARSE.value,
            status=NodeStatus.OK,
            detail={
                "stem_len": len(stem),
                "answer_len": len(answer),
                "input_mode": spec.input_mode,
                "key_mode": self._key_mode,
                "sources": {
                    "stem": spec.stem_source,
                    "student_answer": spec.student_answer_source,
                    "answer_key": spec.answer_key_source,
                },
                "normalized": {
                    side: {"verdict": item.verdict, "normalizer": item.normalizer}
                    for side, item in self._normalized.items()
                },
            },
            item_id=spec.item_id,
        )
        return spec

    async def _node_normalize(self, spec: RunSpec) -> RunSpec:
        """两侧**同时**归一化（方案 §3.4），并用双闸决定能不能拿去判等。

        任一侧 `conflict` → `escalated/normalization_conflict`；
        任一侧不可用 → `escalated/normalization_unverified`（宁可不答，见 §6.1.5）。
        纯 ASCII 输入两侧都是 `unchanged` → 一次模型都不调，行为与改造前完全一致。
        """
        if not self.normalize_enabled:
            return spec
        sides = (
            ("answer_key", spec.answer_key, "answer_key"),
            ("student_answer", spec.student_answer, "student_answer"),
        )
        updates: dict[str, str] = {}
        for label, value, attr in sides:
            if not value or not value.strip():
                continue
            outcome = await normalize_side(
                value,
                side=attr,  # type: ignore[arg-type]
                provider=self.extract,
                tracer=self.tracer,
                budget=self.budget,
            )
            self._normalized[label] = outcome
            if outcome.tokens_in or outcome.tokens_out:
                self._spend(
                    outcome.tokens_in + outcome.tokens_out,
                    usage_in=outcome.tokens_in,
                    usage_out=outcome.tokens_out,
                )
            if outcome.verdict == CONFLICT:
                self.notes.append(f"normalization_conflict:{label}")
                self._finish(Terminal.ESCALATED, ReasonCode.NORMALIZATION_CONFLICT.value)
            if not outcome.usable:
                self.notes.append(f"normalization_unverified:{label}:{','.join(outcome.notes[:2])}")
                self._finish(Terminal.ESCALATED, ReasonCode.NORMALIZATION_UNVERIFIED.value)
            # 用归一化后的值继续（**两侧都换**，否则误差会直接变成"判错"）
            if outcome.normalized != value:
                updates[attr] = outcome.normalized
        return replace(spec, **updates) if updates else spec

    def _reject(self, node: str, code: ReasonCode, message: str) -> _Done:
        self.tracer.event(
            node,
            status=NodeStatus.FAIL,
            reason_code=code.value,
            detail={"message": message},
        )
        self.notes.append(message)
        self._terminal = Terminal.NEEDS_CLARIFICATION
        self._reason = code.value
        return _Done()

    # ---- 节点 2：blind_solve --------------------------------------------
    async def _node_blind(self, spec: RunSpec) -> BlindSolveOutcome:
        self._check_budget(NodeName.BLIND_SOLVE.value)
        outcome = await blind_solve(
            self.primary,
            stem=spec.stem,
            correct_answer=spec.answer_key,
            student_answer=spec.student_answer,
            symbols=spec.symbols,
            tracer=self.tracer,
            budget=self.budget,
            second_provider=self.second,
            item_id=spec.item_id,
        )
        if "budget_insufficient_before_call" in outcome.notes:
            # 必10：**一个请求都没发**。不许把它记成一次 LLM 调用。
            self.notes.append("budget_insufficient_before_call")
            self._finish(Terminal.UNCERTAIN, ReasonCode.SKIPPED_BUDGET.value)
        self._spend(outcome.usage_total, usage_in=outcome.usage_in, usage_out=outcome.usage_out)
        self._blind = outcome

        if self._key_mode == "absent":
            # 方案 §2.1：没有标准答案时**不假装能判对错**。盲解答案留在 outcome 里
            # （报告可见），但终态是 `escalated` —— 判据缺失不是模型的错，也不是输入坏了。
            self.tracer.event(
                NodeName.BLIND_SOLVE.value,
                status=NodeStatus.SKIPPED,
                reason_code=ReasonCode.INSUFFICIENT_EVIDENCE.value,
                detail={
                    "reason": "key_mode_absent",
                    "model_answer_present": bool(outcome.model_answer),
                    "note": "无标准答案 → 纯盲解模式下不给出对错判定，转人工",
                },
                item_id=spec.item_id,
            )
            self._escalation_kind = EscalationKind.CAPABILITY
            self._finish(Terminal.ESCALATED, ReasonCode.INSUFFICIENT_EVIDENCE.value)

        if outcome.branch is Branch.S_STUDENT_CORRECT:
            # 学生本来就对：不许"找出一个错因"（F6 误判率的守门人）
            self._finish(Terminal.ESCALATED, ReasonCode.STUDENT_ANSWER_CORRECT.value)
        if outcome.branch is Branch.B_BAD_KEY:
            # 双闸未过时只记 candidate，对外一律"已提交人工复核"，绝不说"答案错"
            self._finish(Terminal.ESCALATED, ReasonCode.SUSPECTED_BAD_ANSWER_KEY.value)
        if outcome.branch is Branch.C_UNCERTAIN:
            # 判不了 ≠ 判错：能判等但超能力 → sandbox_unsupported；压根没答出来 → solve_schema_invalid
            code = (
                ReasonCode.SANDBOX_UNSUPPORTED
                if "cannot_parse" in outcome.notes
                else ReasonCode.SOLVE_SCHEMA_INVALID
            )
            self._finish(Terminal.UNCERTAIN, code.value)
        if outcome.branch is Branch.D_DISAGREE:
            self._finish(Terminal.UNCERTAIN, ReasonCode.VERIFIER_DISAGREES.value)
        return outcome

    # ---- 节点 3：程序证据（工具层）---------------------------------------
    def _node_tools(self, spec: RunSpec, blind: BlindSolveOutcome) -> None:
        observation = dispatch(
            ToolName.CHECK_NUMERIC,
            {"expr": spec.answer_key, "value": spec.student_answer},
            self.ctx,
        )
        self.tracer.event(
            NodeName.CHECK_NUMERIC.value,
            status=NodeStatus.OK if observation.get("status") == "ok" else NodeStatus.SKIPPED,
            detail={k: v for k, v in observation.items() if k != "hint"},
            item_id=spec.item_id,
        )
        if blind.steps:
            dispatch(
                ToolName.STEP_DIFF,
                {"model_steps": blind.steps, "student_steps": [spec.student_answer]},
                self.ctx,
            )

    def _node_kb(self, kp_ids: list[str], *, item_id: str | None) -> None:
        """诊断后查表校验知识点（产 `kb_lookup` 证据 + 建 kp_links）。

        注意：这一步的证据 `seq` 必然大于 claim 占号 —— 所以**不能**用它的证据去支撑
        已经发过号的 claim（E4 会拒）。它在流水线里的位置只用于"落库与关联"。
        """
        if not kp_ids or self.conn is None:
            return
        observation = dispatch(ToolName.KB_LOOKUP, {"kp_ids": kp_ids}, self.ctx)
        self.tracer.event(
            NodeName.KB_LOOKUP.value,
            status=NodeStatus.OK if observation.get("status") == "ok" else NodeStatus.SKIPPED,
            detail={k: v for k, v in observation.items() if k != "found"},
            item_id=item_id,
        )

    # ---- 节点 4：claim 占号 + diagnose -----------------------------------
    def _reserve_claim_block(self) -> ClaimSeqBlock:
        """ADR-020：**必须在下面那次 LLM 调用之前**占号。"""
        return reserve_claim_seqs(self.ctx.next_seq, self.limits.max_claims)

    async def _node_diagnose(self, spec: RunSpec) -> DiagnoseDraft:
        self._check_budget(NodeName.DIAGNOSE.value)
        body, nonce = diagnose_prompt.render(
            stem=spec.stem,
            answer_key=spec.answer_key,
            student_answer=spec.student_answer,
            evidence_block=self._evidence_block(),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是数学错题诊断专家，只依据给定证据下结论。\n\n" + diagnose_prompt.FENCE_INSTRUCTION
                ),
            },
            {"role": "user", "content": body},
        ]
        self.tracer.event(
            NodeName.DIAGNOSE.value,
            status=NodeStatus.OK,
            detail={
                "l0": {"nonce": nonce, "fenced": True, "prompt_chars": len(body)},
                "l1_hits": scan_untrusted("\n".join([spec.stem, spec.answer_key, spec.student_answer])),
            },
            item_id=spec.item_id,
        )
        try:
            result = await call_structured(
                self.primary,
                messages,
                DiagnoseDraft,
                tracer=self.tracer,
                budget=self.budget,
                node=NodeName.DIAGNOSE.value,
            )
        except StructuredOutputError as exc:
            self.notes.append(f"diagnose_failed:{exc.errors[:2]}")
            self._finish(Terminal.ESCALATED, ReasonCode.DIAGNOSE_SCHEMA_INVALID.value)
        # 真实重试次数 = 尝试次数 - 1（trace 里每次尝试都是独立事件，这里只是给 SSE 一个数）
        self._schema_retries = max(len(result.attempts) - 1, 0)
        self._spend(result.usage_total, usage_in=result.usage_in, usage_out=result.usage_out)
        return result.model if isinstance(result.model, DiagnoseDraft) else DiagnoseDraft(**result.data)

    def _evidence_block(self) -> str:
        lines = []
        for entry in self._ledger():
            evidence = entry.evidence
            lines.append(
                f"- {evidence.id} kind={evidence.kind.value} status={evidence.status.value}"
                f" produced_by={evidence.produced_by} seq={evidence.seq}"
            )
        return "\n".join(lines)

    # ---- 节点 5：证据绑定 -------------------------------------------------
    def _node_guard(
        self, draft: DiagnoseDraft, block: ClaimSeqBlock, *, item_id: str | None
    ) -> tuple[list[ClaimWithSeq], BindingDecision]:
        claims: list[ClaimWithSeq] = []
        for claim in draft.claims:
            try:
                seq = block.take()
            except IndexError:
                self.notes.append("too_many_claims")
                self._finish(Terminal.ESCALATED, ReasonCode.DIAGNOSE_SCHEMA_INVALID.value)
            claims.append(ClaimWithSeq(claim=claim, seq=seq))

        decision = bind_claims(run_id=self.run_id, claims=claims, ledger=self._ledger())
        self._trace_guard(decision, claims=claims, stage="guard", item_id=item_id)
        if decision.escalate:
            self._finish(Terminal.ESCALATED, ReasonCode.INSUFFICIENT_EVIDENCE.value)
        return claims, decision

    def _guard_detail(
        self, decision: BindingDecision, *, claims: list[ClaimWithSeq], stage: str
    ) -> dict[str, Any]:
        """guard 事件的 detail：**三个判据各写各的**，SSE 只做搬运（B-1）。

        `_trace_guard` 与 `_check_confidence` 都走这里 —— 两处各写一份就会出现
        "低置信度那次 guard 事件没有 accepted/rejected"，界面于是把它当成"通过"。
        """
        accepted = len(decision.accepted)
        rejected = len(decision.rejected)
        return {
            "stage": stage,
            "escalate": decision.escalate,
            "escalation_kind": decision.escalation_kind.value if decision.escalation_kind else None,
            "accepted": accepted,
            "rejected": rejected,
            "schema": "retry" if (self._schema_retries or 0) > 0 else "pass",
            "schema_retries": self._schema_retries,
            "evidence_binding": evidence_binding_of(decision),
            "violations": [[field.value, reason.value] for field, reason in decision.violations],
            "confidence_cap": decision.confidence_cap,
            "needs_review": decision.needs_review,
            "claim_seqs": [item.seq for item in claims],
            "evidence_seqs": [entry.evidence.seq for entry in self._ledger()],
            # 审查 🟡G：escalated 不写 diagnoses 行，M5（证据绑定通过率）与 F3/F4
            # 细分必须能从 trace 重算 —— 所以 claims 的内容也要落 trace
            "claims": [
                {
                    "field": item.claim.field.value,
                    "value": _brief(item.claim.value),
                    "evidence_ids": list(item.claim.evidence_ids),
                    "seq": item.seq,
                    "accepted": _was_accepted(decision, item),
                }
                for item in claims
            ],
        }

    def _trace_guard(
        self,
        decision: BindingDecision,
        *,
        claims: list[ClaimWithSeq],
        stage: str,
        item_id: str | None,
    ) -> None:
        self.tracer.event(
            NodeName.GUARD.value,
            status=NodeStatus.OK if not decision.escalate else NodeStatus.FAIL,
            reason_code=ReasonCode.INSUFFICIENT_EVIDENCE.value if decision.escalate else None,
            detail=self._guard_detail(decision, claims=claims, stage=stage),
            item_id=item_id,
        )

    def _check_confidence(self, draft: DiagnoseDraft, decision: BindingDecision) -> None:
        effective = min(draft.confidence, decision.confidence_cap)
        if effective < config.CONFIDENCE_THETA:
            self.notes.append(f"low_confidence:{effective:.2f}")
            # 低置信度也分口径：因为"没有 step_diff / 判不了值"而低 = 能力边界
            self._escalation_kind = (
                EscalationKind.CAPABILITY
                if capability_gap(self._ledger(), self.run_id)
                else EscalationKind.MECHANISM
            )
            self.tracer.event(
                NodeName.GUARD.value,
                status=NodeStatus.FAIL,
                reason_code=ReasonCode.LOW_CONFIDENCE.value,
                detail={
                    **self._guard_detail(decision, claims=[], stage="confidence"),
                    "confidence": effective,
                    "theta": config.CONFIDENCE_THETA,
                    "escalation_kind": self._escalation_kind.value,
                },
            )
            self._finish(Terminal.ESCALATED, ReasonCode.LOW_CONFIDENCE.value)

    # ---- 节点 6：讲解（P1.7）---------------------------------------------
    async def _node_explain(
        self, spec: RunSpec, draft: DiagnoseDraft, decision: BindingDecision
    ) -> Explanation | None:
        if not self.limits.explain:
            return None
        self._check_budget(NodeName.EXPLAIN.value)
        accepted_fields = {v.claim.field.value for v in decision.verdicts if v.accepted}
        explanation = await explain(
            self.primary,
            stem=spec.stem,
            student_answer=spec.student_answer,
            error_type=draft.error_type.value,
            error_subtype=draft.error_subtype,
            root_cause=draft.root_cause,
            tracer=self.tracer,
            budget=self.budget,
            verify=self._verify_step,
            accepted_fields=accepted_fields,
            item_id=spec.item_id,
        )
        if "budget_insufficient_before_call" in (explanation.reason or ""):
            self.notes.append("budget_insufficient_before_call")
        self._spend(
            explanation.usage_total,
            usage_in=explanation.usage_in,
            usage_out=explanation.usage_out,
        )
        if explanation.degraded:
            # 降级阶梯第 1 档：只给结构化诊断，不报错
            self.notes.append(f"explain_degraded:{explanation.reason}")
        return explanation

    def _verify_step(self, expr: str) -> bool | None:
        """讲解步骤的复核。

        支持两种写法：
        * `A == B` → 用 `compare` 判等（**这是模型最自然的写法**，直接丢给沙箱会得到
          `unsupported_node:Compare`，于是"可验证"永远降级成"教学性表述"）；
        * 纯表达式 → 能求值且结果有限即算通过。

        语义仍是"这一步的数学结论可被程序核验"，**不是**"这一步是对的"。
        """
        text = expr.strip()
        left, _, right = text.partition("==")
        if right.strip():
            comparison = compare(left.strip(), right.strip(), symbols=self.ctx.symbols)
            passed = comparison.equal is True
            payload: dict[str, object] = {"expr": text, "equal": comparison.equal}
        else:
            result = evaluate(text, symbols=self.ctx.symbols)
            passed = result.verdict is Verdict.OK
            payload = {"expr": text, "verdict": result.verdict.value, "reason": result.reason}
        self.ctx.record_evidence(
            kind="numeric_check",
            payload=payload,
            produced_by=NodeName.EXPLAIN.value,
            status="ok" if passed else "skipped",
        )
        return True if passed else None

    # ---- 节点 7：submit ---------------------------------------------------
    def _node_submit(
        self,
        spec: RunSpec,
        draft: DiagnoseDraft,
        claims: list[ClaimWithSeq],
        decision: BindingDecision,
        blind: BlindSolveOutcome,
        explanation: Explanation | None,
    ) -> dict[str, Any] | None:
        error_type_claim = next(
            (item for item in claims if item.claim.field is ClaimField.ERROR_TYPE),
            None,
        )
        error_type = (
            str(error_type_claim.claim.value) if error_type_claim is not None else draft.error_type.value
        )
        confidence = min(draft.confidence, decision.confidence_cap)
        diagnosis: dict[str, Any] | None = None

        if self.conn is not None:
            repo = Repository(self.conn)
            diagnosis = repo.create_diagnosis(
                attempt_id=spec.attempt_id,
                run_id=self.run_id,
                error_type=error_type,
                error_subtype=draft.error_subtype,
                first_divergent_step=draft.first_divergent_step,
                locator="model",
                root_cause=draft.root_cause or None,
                hypotheses=draft.hypotheses,
                kp_ids=draft.kp_ids,
                confidence=confidence,
                needs_review=decision.needs_review,
                terminal=Terminal.DIAGNOSED.value,
                arm=spec.arm,
                prompt_version_id=diagnose_prompt.VERSION,
                model=getattr(self.primary, "model", "unknown"),
                temperature=config.DEFAULT_TEMPERATURE,
                evidence_binding=EvidenceBinding.PASS.value,
                verify_agrees=blind.suspect is Suspect.CONFIRMED,
                suspected_bad_answer_key=blind.suspect.value,
                input_tokens=self._tokens_in,
                output_tokens=self._tokens_out,
                latency_ms=int((self.clock() - self._started) * 1000),
            )
            for item in decision.accepted:
                repo.create_claim(
                    diagnosis_id=diagnosis["id"],
                    field=item.claim.field.value,
                    value=item.claim.value,
                    evidence_ids=list(item.claim.evidence_ids),
                    seq=item.seq,
                )
            for step in explanation.steps if explanation else []:
                repo.create_explanation_step(
                    diagnosis_id=diagnosis["id"],
                    seq=step.seq,
                    title=step.title,
                    content=step.content,
                    key_point=step.key_point,
                    targets_error=step.targets_error,
                    verifiable=step.verifiable,
                )

        self.tracer.event(
            NodeName.SUBMIT.value,
            status=NodeStatus.OK,
            detail={
                "terminal": Terminal.DIAGNOSED.value,
                "error_type": error_type,
                "confidence": confidence,
                "needs_review": decision.needs_review,
                "steps": len(explanation.steps) if explanation else 0,
                "verify_agrees": blind.suspect is Suspect.CONFIRMED,
                "suspected_bad_answer_key": blind.suspect.value,
                "arm": spec.arm,
                "prompt_version": diagnose_prompt.VERSION,
                "prompt_hash": diagnose_prompt.prompt_hash(),
            },
            item_id=spec.item_id,
        )
        return diagnosis

    # ---- 台账（从 DB 或内存构造，供 E2 使用）-----------------------------
    def _ledger(self) -> list[LedgerEntry]:
        if self.conn is None:
            return [
                LedgerEntry(
                    run_id=self.run_id,
                    evidence=Evidence(
                        id=record["id"],
                        kind=record["kind"],
                        payload=record["payload"],
                        status=record["status"],
                        produced_by=record["produced_by"],
                        seq=record["seq"],
                    ),
                )
                for record in self.ctx.drafts
            ]
        rows = Repository(self.conn).list_evidence(self.run_id)
        entries: list[LedgerEntry] = []
        for row in rows:
            entries.append(
                LedgerEntry(
                    run_id=self.run_id,
                    evidence=Evidence(
                        id=row["id"],
                        kind=row["kind"],
                        payload=row["payload"],
                        status=row["status"],
                        produced_by=row["produced_by"],
                        seq=row["seq"],
                    ),
                )
            )
        return entries
