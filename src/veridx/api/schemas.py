"""REST + SSE 契约（gate G0 冻结，设计文档第 8 章）。

冻结规则（TODO §0.7）：
* 只允许**加字段**，不允许改含义
* 改含义 = `schema_version` +1，旧 runs 标记 legacy，不参与对比
* 变更必须在 `doc/CHANGELOG-contract.md` 记一行

13 个端点（§8.1）+ 6 类 SSE 事件（§8.2）的 request/response 模型全部在此。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from veridx import config
from veridx.config import SeedPolicy
from veridx.guard.claim import Claim, Evidence, EvidenceKind
from veridx.taxonomy import ErrorType
from veridx.trace.event import TraceEvent

API_PREFIX = "/api/v1"


# ---------------------------------------------------------------------------
# 终态 / 状态（§2.3：互斥完备，4 业务 + 2 工程）
# ---------------------------------------------------------------------------
class Terminal(StrEnum):
    DIAGNOSED = "diagnosed"
    ESCALATED = "escalated"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNCERTAIN = "uncertain"
    CANCELED = "canceled"
    FAILED = "failed"


BUSINESS_TERMINALS: frozenset[Terminal] = frozenset(
    {Terminal.DIAGNOSED, Terminal.ESCALATED, Terminal.NEEDS_CLARIFICATION, Terminal.UNCERTAIN}
)
ENGINEERING_TERMINALS: frozenset[Terminal] = frozenset({Terminal.CANCELED, Terminal.FAILED})


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DIAGNOSED = "diagnosed"
    ESCALATED = "escalated"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNCERTAIN = "uncertain"
    CANCELED = "canceled"
    FAILED = "failed"


class EvidenceBinding(StrEnum):
    PASS = "pass"
    FAIL_AFTER_RETRY = "fail_after_retry"
    SKIPPED = "skipped"


class Locator(StrEnum):
    PROGRAM = "program"
    MODEL = "model"


class SuspectedBadAnswerKey(StrEnum):
    NONE = "none"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"


class Arm(StrEnum):
    A = "A"  # zero-shot 直接诊断
    B = "B"  # CoT
    C = "C"  # 盲解
    D = "D"  # 盲解 + 程序验算 + 证据绑定（本项目主张）
    E = "E"  # D + self-critique


class Split(StrEnum):
    TRAIN = "train"
    HELDOUT = "heldout"
    REDTEAM = "redteam"
    MR2 = "mr2"
    CI_SUBSET = "ci_subset"


# ---------------------------------------------------------------------------
# 题库与作答（FR-1）
# ---------------------------------------------------------------------------
class QuestionCreate(BaseModel):
    course: str = Field(min_length=1, max_length=64)
    stem: str = Field(min_length=1)
    #: 入口闸门（§8.5）：标准答案会被送进沙箱求值，长度上限与 SANDBOX_MAX_EXPR_LEN 同源
    correct_answer: str = Field(min_length=1, max_length=config.SANDBOX_MAX_EXPR_LEN)
    qtype: str | None = Field(default=None, max_length=32)
    difficulty: str | None = Field(default=None, max_length=8)
    kp_ids: list[str] = Field(default_factory=list)
    source: str | None = Field(default=None, max_length=128)


class QuestionOut(BaseModel):
    id: str
    course: str
    stem: str
    correct_answer: str
    qtype: str | None = None
    difficulty: str | None = None
    kp_ids: list[str] = Field(default_factory=list)
    source: str | None = None
    created_at: str


class AttemptCreate(BaseModel):
    """录一次作答：可内联新题，也可引用已有题目（两者恰择一）。

    `student_answer` 有**入口闸门**（§8.5）：长度 ≤ `SANDBOX_MAX_EXPR_LEN`。
    它挡掉的是"用超长字符串刷 CPU"这条最廉价的路 ——
    **入口校验 ≠ 修改沙箱判据**（后者才是刻意保留的隐患）。
    """

    question: QuestionCreate | None = None
    question_id: str | None = None
    student_answer: str = Field(min_length=1, max_length=config.SANDBOX_MAX_EXPR_LEN)

    @model_validator(mode="after")
    def _exactly_one_question(self) -> AttemptCreate:
        if (self.question is None) == (self.question_id is None):
            raise ValueError("question 与 question_id 必须且只能提供一个")
        return self


class AttemptOut(BaseModel):
    attempt_id: str
    question_id: str
    #: wire 名沿用 §8.4 冻结的 `fingerprint`；落地列名是 `attempts.answer_digest`（C-14）
    fingerprint: str


class ImportRowError(BaseModel):
    row: int = Field(ge=1)
    error: str


class ImportPayload(BaseModel):
    """批量导入（FR-1.2）：原始文本 + 格式，逐行报错不中断。"""

    format: Literal["jsonl", "csv"] = "jsonl"
    content: str = Field(min_length=1)


class ImportResult(BaseModel):
    total: int = Field(ge=0)
    created: int = Field(ge=0)
    errors: list[ImportRowError] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 诊断任务（FR-2 / §2.5 时序）
# ---------------------------------------------------------------------------
class DiagnoseResponse(BaseModel):
    """202 = 新建任务；200 = 幂等命中已有结果。"""

    task_id: str
    status: TaskStatus
    estimated_ms: int | None = None
    idempotent_hit: bool = False
    diagnosis_id: str | None = None
    terminal: Terminal | None = None
    run_id: str | None = None


class TaskOut(BaseModel):
    task_id: str
    attempt_id: str
    status: TaskStatus
    terminal: Terminal | None = None
    reason_code: str | None = None
    run_id: str | None = None
    created_at: str
    updated_at: str


class CancelOut(BaseModel):
    task_id: str
    status: TaskStatus
    already_terminal: bool = False


# ---------------------------------------------------------------------------
# 诊断结果全文（§8.4）
# ---------------------------------------------------------------------------
class Hypothesis(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class ExplanationStepOut(BaseModel):
    seq: int = Field(ge=1)
    title: str
    content: str
    key_point: str | None = None
    targets_error: str | None = None
    verifiable: bool = False


class SimilarQuestionOut(BaseModel):
    id: str
    stem: str
    answer: str | None = None
    solution: str | None = None
    check_level: str | None = None
    check_passed: bool = False


class KpOut(BaseModel):
    id: str
    name: str


class DiagnosisOut(BaseModel):
    id: str
    attempt_id: str
    run_id: str
    terminal: Terminal
    reason_code: str | None = None

    error_type: ErrorType | None = None
    error_subtype: str | None = None
    first_divergent_step: int | None = None
    locator: Locator | None = None
    root_cause: str | None = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    kp_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    needs_review: bool = False

    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    steps: list[ExplanationStepOut] = Field(default_factory=list)
    similar_questions: list[SimilarQuestionOut] = Field(default_factory=list)
    kps: list[KpOut] = Field(default_factory=list)

    arm: Arm
    prompt_version_id: str
    model: str
    temperature: float
    seed: int | None = None
    evidence_binding: EvidenceBinding
    guard_retries: int = 0
    cascade_level: int = 0
    verify_agrees: bool | None = None
    suspected_bad_answer_key: SuspectedBadAnswerKey = SuspectedBadAnswerKey.NONE

    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    created_at: str


# ---------------------------------------------------------------------------
# 回放（§11.3）
# ---------------------------------------------------------------------------
class TraceEventOut(BaseModel):
    """`TraceEvent` 的传输镜像；字段一致性由 tests/test_contracts.py 守护。"""

    run_id: str
    item_id: str | None = None
    node: str
    seq: int
    parent_seq: int | None = None
    start_ms: int
    end_ms: int | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    status: str
    reason_code: str | None = None
    detail: dict = Field(default_factory=dict)

    @classmethod
    def from_event(cls, event: TraceEvent) -> TraceEventOut:
        return cls(**event.to_dict())


class ReplayOut(BaseModel):
    run_id: str
    item_id: str | None = None
    events: list[TraceEventOut] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 知识点薄弱聚合（FR-5.2）
# ---------------------------------------------------------------------------
class WeakKpOut(BaseModel):
    kp_id: str
    name: str
    error_count: int = Field(ge=0)
    last_error_at: str | None = None


class WeakKpList(BaseModel):
    total: int = Field(ge=0)
    items: list[WeakKpOut] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 评测（FR-7）
# ---------------------------------------------------------------------------
class EvalRunCreate(BaseModel):
    arm: Arm = Arm.D
    split: Split = Split.HELDOUT
    dataset_hash: str | None = None
    k: int = Field(default=1, ge=1, le=10)
    limit: int | None = Field(default=None, ge=1)
    seed_policy: SeedPolicy = SeedPolicy.FIXED
    prompt_version_id: str | None = None


class EvalRunAccepted(BaseModel):
    eval_run_id: str
    status: TaskStatus
    queued_items: int = Field(ge=0)


class EvalReportOut(BaseModel):
    eval_run_id: str
    format: Literal["json", "markdown"] = "json"
    content: str
    generated_at: str


# ---------------------------------------------------------------------------
# 健康检查（含 provider 可达性）
# ---------------------------------------------------------------------------
class ProviderHealth(BaseModel):
    mode: str
    model: str
    reachable: bool | None = None
    second_model: str | None = None


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    schema_version: int
    provider: ProviderHealth


# ---------------------------------------------------------------------------
# SSE 事件协议（§8.2：6 类，`done` 恰好发送一次）
# ---------------------------------------------------------------------------
class SSEEventName(StrEnum):
    STAGE = "stage"
    EVIDENCE = "evidence"
    CLAIM = "claim"
    TOKEN = "token"
    GUARD = "guard"
    DONE = "done"


class StageEventData(BaseModel):
    stage: str
    # ⚠️ 与 trace 的 NodeStatus 同构（§8.2 契约要求 2）；`canceled` 是 P0.6 补的（C-13），
    # 否则取消时前端收不到"这一步被取消"
    status: Literal["ok", "retry", "fail", "skipped", "canceled"]
    ms: int = Field(ge=0)
    seq: int = Field(ge=1)


class EvidenceEventData(BaseModel):
    """§8.2 的扁平样例（如 `equal`）统一收进 `payload`，见 CHANGELOG C-08。"""

    id: str = Field(pattern=r"^ev_\d+$")
    kind: EvidenceKind
    status: Literal["ok", "skipped", "failed"] = "ok"
    seq: int = Field(ge=1)
    payload: dict = Field(default_factory=dict)


class ClaimEventData(BaseModel):
    field: str
    value: str | list[str]
    evidence_ids: list[str] = Field(min_length=1)


class TokenEventData(BaseModel):
    text: str


class GuardEventData(BaseModel):
    schema_: Literal["pass", "fail", "retry"] = Field(alias="schema")
    evidence_binding: EvidenceBinding
    retries: int = Field(default=0, ge=0)

    model_config = ConfigDict(populate_by_name=True)


class TokenUsage(BaseModel):
    input_tokens: int = Field(alias="in", ge=0)
    output_tokens: int = Field(alias="out", ge=0)

    model_config = ConfigDict(populate_by_name=True)


class DoneEventData(BaseModel):
    terminal: Terminal
    reason_code: str | None = None
    run_id: str
    latency_ms: int = Field(ge=0)
    tokens: TokenUsage


EventData = (
    StageEventData | EvidenceEventData | ClaimEventData | TokenEventData | GuardEventData | DoneEventData
)

_EVENT_MODELS: dict[SSEEventName, type[BaseModel]] = {
    SSEEventName.STAGE: StageEventData,
    SSEEventName.EVIDENCE: EvidenceEventData,
    SSEEventName.CLAIM: ClaimEventData,
    SSEEventName.TOKEN: TokenEventData,
    SSEEventName.GUARD: GuardEventData,
    SSEEventName.DONE: DoneEventData,
}


class SSEFrame(BaseModel):
    """一条 SSE 帧：`event: <name>` + `data: <json>`；解析与生成都用它。"""

    event: SSEEventName
    data: EventData

    def to_text(self) -> str:
        return f"event: {self.event.value}\ndata: {self.data.model_dump_json(by_alias=True)}\n\n"

    @classmethod
    def parse(cls, event: str, data_json: str) -> SSEFrame:
        name = SSEEventName(event)
        model = _EVENT_MODELS[name]
        return cls(event=name, data=model.model_validate_json(data_json))  # type: ignore[arg-type]
