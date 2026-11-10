"""分步讲解生成（P1.7 / `flow/explain.py`）。

## 三件事

1. **至少 3 步，且每步有 `targets_error`** —— 没有"针对哪个错误动作"的步骤不算讲解，
   只是复述。所以这一步的 schema 强制 `min_length=3` + `targets_error` 非空，
   再过一遍**代码侧**校验（不信模型自称合规）。
2. **每步标注"可验证 / 教学性"** —— `verifiable=True` 的步骤必须附带 `verifiable_expr`，
   由沙箱复核；复核不过就**降级为教学性表述**（`verifiable=False`）。
   教学性表述不进证据链，也不得被当作事实（I3）。
3. **讲解缺失时降级为"只给诊断"** —— 结构化输出重试耗尽不许抛到上层，
   返回 `degraded=True` + 原因，终态仍然是 `diagnosed`。

> 注意副作用：`verifiable` 步骤的沙箱复核发生在 **claim 占号之后**，
> 它产生的证据 `seq` 一定大于 claim 的 `seq` —— 于是"引用它的 claim"会被 E4 拒。
> 这是**正确**行为，也是 P1.6 里 E4 唯一非构造性的活样本。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from veridx.guard.structured import Budget, StructuredOutputError, call_structured
from veridx.llm.base import LLMProvider
from veridx.prompts import explain as prompt
from veridx.trace.event import NodeName, NodeStatus, ReasonCode
from veridx.trace.recorder import TraceRecorder

__all__ = ["Explanation", "ExplanationStep", "ExplainDraft", "StepDraft", "explain", "MIN_STEPS"]

#: 讲解步数下限（P1.7 验收）
MIN_STEPS = 3


class StepDraft(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1)
    key_point: str | None = None
    targets_error: str = Field(min_length=1, description="这一步针对学生的哪个错误动作")
    verifiable: bool = False
    verifiable_expr: str | None = None


class ExplainDraft(BaseModel):
    # schema 层就要求 ≥3 步：少一步直接判不合法 → 回喂重试
    steps: list[StepDraft] = Field(min_length=MIN_STEPS)


@dataclass
class ExplanationStep:
    seq: int
    title: str
    content: str
    key_point: str | None
    targets_error: str
    verifiable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "title": self.title,
            "content": self.content,
            "key_point": self.key_point,
            "targets_error": self.targets_error,
            "verifiable": self.verifiable,
        }


@dataclass
class Explanation:
    steps: list[ExplanationStep] = field(default_factory=list)
    #: 讲解缺失/不足 → 降级为"只给诊断"（终态不变，不报错）
    degraded: bool = False
    reason: str | None = None
    usage_total: int = 0

    @property
    def teaching_only(self) -> list[ExplanationStep]:
        return [step for step in self.steps if not step.verifiable]

    @property
    def verifiable(self) -> list[ExplanationStep]:
        return [step for step in self.steps if step.verifiable]


#: 复核器：给一个表达式，返回 True=验算通过 / False=不通过 / None=判不了
Verifier = Callable[[str], bool | None]


async def explain(
    provider: LLMProvider,
    *,
    stem: str,
    student_answer: str,
    error_type: str,
    error_subtype: str | None,
    root_cause: str,
    tracer: TraceRecorder | None = None,
    budget: Budget | None = None,
    verify: Verifier | None = None,
    item_id: str | None = None,
) -> Explanation:
    """生成分步讲解。**永不抛异常**：失败 → `degraded=True`，由调用方降级。"""
    messages = [
        {"role": "system", "content": "你是数学老师，只讲补救步骤，不下诊断结论。"},
        {
            "role": "user",
            "content": prompt.render(
                stem=stem,
                student_answer=student_answer,
                error_type=error_type,
                error_subtype=error_subtype,
                root_cause=root_cause,
            ),
        },
    ]

    try:
        result = await call_structured(
            provider,
            messages,
            ExplainDraft,
            tracer=tracer,
            budget=budget,
            node=NodeName.EXPLAIN.value,
        )
    except StructuredOutputError as exc:
        _trace_degraded(tracer, item_id=item_id, reason="explain_failed", detail=exc.errors[:3])
        return Explanation(degraded=True, reason="explain_failed")

    draft = result.model if isinstance(result.model, ExplainDraft) else ExplainDraft(**result.data)
    steps: list[ExplanationStep] = []
    dropped = 0
    for raw in draft.steps:
        if not raw.targets_error.strip():
            dropped += 1
            continue
        verifiable = False
        if raw.verifiable and raw.verifiable_expr and verify is not None:
            # 只有程序说"成立"才允许标记为可验证；判不了 ≠ 成立（同 compare 的三层保护）
            verifiable = verify(raw.verifiable_expr) is True
        steps.append(
            ExplanationStep(
                seq=len(steps) + 1,
                title=raw.title.strip(),
                content=raw.content.strip(),
                key_point=(raw.key_point or "").strip() or None,
                targets_error=raw.targets_error.strip(),
                verifiable=verifiable,
            )
        )

    if len(steps) < MIN_STEPS:
        _trace_degraded(
            tracer,
            item_id=item_id,
            reason="explain_too_few_steps",
            detail={"kept": len(steps), "dropped": dropped},
        )
        return Explanation(
            steps=steps,
            degraded=True,
            reason="explain_too_few_steps",
            usage_total=result.usage_total,
        )

    if tracer is not None:
        tracer.event(
            NodeName.EXPLAIN.value,
            status=NodeStatus.OK,
            detail={
                "steps": len(steps),
                "verifiable": len([s for s in steps if s.verifiable]),
                "teaching_only": len([s for s in steps if not s.verifiable]),
                "dropped": dropped,
                "prompt_version": prompt.VERSION,
                "prompt_hash": prompt.prompt_hash(),
            },
            item_id=item_id,
        )
    return Explanation(steps=steps, usage_total=result.usage_total)


def _trace_degraded(
    tracer: TraceRecorder | None, *, item_id: str | None, reason: str, detail: object
) -> None:
    """降级本身也要落 trace（不许静默返回"只给诊断"）。"""
    if tracer is None:
        return
    tracer.event(
        NodeName.EXPLAIN.value,
        status=NodeStatus.SKIPPED,
        reason_code=ReasonCode.SCHEMA_INVALID.value if "failed" in reason else None,
        detail={"degraded": True, "reason": reason, "detail": detail},
        item_id=item_id,
    )
