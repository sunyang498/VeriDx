"""盲解交叉验证（设计文档 §3.1 / P1.5）。

## 要解决的问题是**信息集污染**，不是采样多样性

一次调用里同时给出题干、标准答案、学生答案，模型必然"找出一个错因"（顺从性偏差）。
所以盲解**只把题干**交给模型，比对在程序侧完成：

```
① blind_solve（看不到学生答案）→ (模型答案, 标准步骤)
② sympy 判等：model ↔ key、model ↔ student、student ↔ key
③ 分支裁决
```

独立性来自**信息隔离**，所以"问两遍""换温度"都不算交叉验证（共享同一污染上下文）。

## 分支（§3.1.2）

| 分支 | 条件 | 动作 |
|---|---|---|
| A 正常错 | model == key，且 model != student | 进入 diagnose |
| B 标准答案可疑 | model == student，且 student != key | **必须过双闸**才置位 |
| C 模型不确定 | 任一侧 `equal is None`（判不了） | 降置信度 / escalate |
| D 三者不一致 | 其余 | `uncertain`，不许猜 |
| S 学生本来就对 | student == key | **不许诊断**（F6 误判率的守门人） |

> `S` 是设计文档四分支表**没有覆盖**的一格：那张表的前提是"学生被判错"。
> 可一旦 `student == key`，三者其实一致，A/B/D 的条件都不成立。
> 不补这一格，系统会在这个最常见的"学生做对了"场景里走进 D 或强行诊断。

## 分支 B 的双闸（§3.1.5）

学生与模型可能犯**同一种**常见误解 → 答案相同 → 若直接置位就是"质疑出题人"，比误判学生更伤公信力。

* **闸 1**：所有判等由 sympy 给出（`equal` 只可能来自沙箱）。模型自述"我和学生一样"**不生效**。
* **闸 2**：**第二个模型**（不同厂商）独立求解，答案也必须等于学生答案。
* 过闸 1 未过闸 2 → 只记 `candidate`（对外只说"已提交人工复核"，**不说答案错**）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from veridx.guard.structured import Budget, StructuredOutputError, call_structured
from veridx.llm.base import LLMProvider
from veridx.prompts import blind_solve as prompt
from veridx.sandbox import Comparison, compare
from veridx.trace.event import NodeName, NodeStatus, ReasonCode
from veridx.trace.recorder import TraceRecorder

__all__ = [
    "BlindDraft",
    "BlindSolveOutcome",
    "Branch",
    "Suspect",
    "blind_solve",
    "build_messages",
]

#: 盲解阶段的学生可见内容：**只有题干**
SYSTEM_PROMPT = "你在做一道独立的数学求解题。你不知道也不需要知道任何人的作答。"


class BlindDraft(BaseModel):
    """盲解输出。`answer` 允许空串（多解/无法唯一确定），此时按 C 分支处理。"""

    answer: str = ""
    steps: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    note: str = ""


class Branch(StrEnum):
    A_WRONG = "A"  # 正常错：model == key 且 model != student
    B_BAD_KEY = "B"  # 标准答案可疑（需双闸）
    C_UNCERTAIN = "C"  # 模型不确定 / 判不了
    D_DISAGREE = "D"  # 三者不一致
    S_STUDENT_CORRECT = "S"  # 学生本来就对（设计表未覆盖，见模块 docstring）


class Suspect(StrEnum):
    """`suspected_bad_answer_key` 的三取值（§3.1.5）。"""

    NONE = "none"
    CANDIDATE = "candidate"  # 只过闸 1：进人工复核，不对外称"答案错"
    CONFIRMED = "confirmed"  # 闸 1 + 闸 2 都过


@dataclass
class BlindSolveOutcome:
    branch: Branch
    model_answer: str = ""
    steps: list[str] = field(default_factory=list)
    suspect: Suspect = Suspect.NONE
    gate1_passed: bool = False
    gate2_passed: bool = False
    second_answer: str | None = None
    student_matches_key: bool = False
    #: 证据/日志用：每一步判等的实际值（None = 判不了 ≠ 不相等）
    equal_model_key: bool | None = None
    equal_model_student: bool | None = None
    equal_student_key: bool | None = None
    notes: list[str] = field(default_factory=list)
    usage_total: int = 0
    prompt_version: str = prompt.VERSION
    prompt_hash: str = ""

    @property
    def suggests_diagnosis(self) -> bool:
        """只有 A 分支才应该继续诊断（S 与 D 都不许"找出一个错因"）。"""
        return self.branch is Branch.A_WRONG


def build_messages(stem: str) -> list[dict[str, Any]]:
    """盲解阶段的对话。**只含题干**，没有任何人（包括学生）的作答。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt.render(stem)},
    ]


def _equal(left: str, right: str, symbols: frozenset[str] | None) -> bool | None:
    if not left.strip() or not right.strip():
        return None
    result: Comparison = compare(left, right, symbols=symbols)
    return result.equal


async def blind_solve(
    provider: LLMProvider,
    *,
    stem: str,
    correct_answer: str,
    student_answer: str,
    symbols: frozenset[str] | None = None,
    tracer: TraceRecorder | None = None,
    budget: Budget | None = None,
    second_provider: LLMProvider | None = None,
    item_id: str | None = None,
) -> BlindSolveOutcome:
    """独立求解 → sympy 判等 → 分支裁决（分支 B 走双闸）。"""
    messages = build_messages(stem)
    outcome = BlindSolveOutcome(branch=Branch.C_UNCERTAIN, prompt_hash=prompt.prompt_hash())
    _assert_no_leak(messages, stem=stem, student_answer=student_answer)

    try:
        result = await call_structured(
            provider,
            messages,
            BlindDraft,
            tracer=tracer,
            budget=budget,
            node=NodeName.BLIND_SOLVE.value,
        )
    except StructuredOutputError as exc:
        outcome.notes.append(f"blind_solve_failed:{exc.errors[:2]}")
        if tracer is not None:
            tracer.event(
                NodeName.BLIND_SOLVE.value,
                status=NodeStatus.FAIL,
                reason_code=ReasonCode.SCHEMA_INVALID.value,
                detail={"branch": Branch.C_UNCERTAIN.value, "attempts": len(exc.attempts)},
                item_id=item_id,
            )
        return outcome

    draft = result.model
    if not isinstance(draft, BlindDraft):  # pragma: no cover - 防御
        draft = BlindDraft(**result.data)
    outcome.model_answer = draft.answer.strip()
    outcome.steps = list(draft.steps)
    outcome.usage_total = result.usage_total

    outcome.equal_model_key = _equal(outcome.model_answer, correct_answer, symbols)
    outcome.equal_model_student = _equal(outcome.model_answer, student_answer, symbols)
    outcome.equal_student_key = _equal(student_answer, correct_answer, symbols)
    outcome.branch = _decide(outcome, draft=draft)

    if outcome.branch is Branch.B_BAD_KEY:
        await _run_gates(
            outcome,
            second_provider=second_provider,
            stem=stem,
            symbols=symbols,
            tracer=tracer,
            budget=budget,
            item_id=item_id,
        )

    if tracer is not None:
        tracer.event(
            NodeName.BLIND_SOLVE.value,
            status=NodeStatus.OK,
            detail={
                "branch": outcome.branch.value,
                "suspect": outcome.suspect.value,
                "gate1": outcome.gate1_passed,
                "gate2": outcome.gate2_passed,
                "equal_model_key": outcome.equal_model_key,
                "equal_model_student": outcome.equal_model_student,
                "equal_student_key": outcome.equal_student_key,
                "prompt_version": outcome.prompt_version,
                "prompt_hash": outcome.prompt_hash,
                "answer_len": len(outcome.model_answer),
                "steps": len(outcome.steps),
            },
            item_id=item_id,
        )
    return outcome


def _decide(outcome: BlindSolveOutcome, *, draft: BlindDraft) -> Branch:
    """分支裁决。**判不了的（None）一律不许当成"不相等"。**"""
    if not outcome.model_answer:
        outcome.notes.append("model_answer_empty" if not draft.note else draft.note[:120])
        return Branch.C_UNCERTAIN
    if outcome.equal_model_key is None or outcome.equal_student_key is None:
        outcome.notes.append("cannot_parse")
        return Branch.C_UNCERTAIN
    if outcome.equal_student_key:
        # 学生答案本来就等于标准答案：这不是"错因诊断"场景（F6 的守门人）
        outcome.student_matches_key = True
        return Branch.S_STUDENT_CORRECT
    if outcome.equal_model_key and outcome.equal_model_student is False:
        return Branch.A_WRONG
    if outcome.equal_model_student is True:
        # 闸 1 在这里已经是"sympy 给的"，模型自述不参与
        outcome.gate1_passed = True
        outcome.suspect = Suspect.CANDIDATE
        return Branch.B_BAD_KEY
    return Branch.D_DISAGREE


async def _run_gates(
    outcome: BlindSolveOutcome,
    *,
    second_provider: LLMProvider | None,
    stem: str,
    symbols: frozenset[str] | None,
    tracer: TraceRecorder | None,
    budget: Budget | None,
    item_id: str | None,
) -> None:
    """闸 2：第二个模型（不同厂商）独立求解，答案也必须等于学生答案。"""
    if second_provider is None:
        outcome.notes.append("gate2_skipped:no_second_provider")
        return
    try:
        second = await call_structured(
            second_provider,
            build_messages(stem),
            BlindDraft,
            tracer=tracer,
            budget=budget,
            node=NodeName.BLIND_SOLVE.value,
        )
    except StructuredOutputError as exc:
        outcome.notes.append(f"gate2_failed:{exc.errors[:2]}")
        return

    draft = second.model if isinstance(second.model, BlindDraft) else BlindDraft(**second.data)
    outcome.usage_total += second.usage_total
    outcome.second_answer = draft.answer.strip() or None
    # 闸 2 也必须是 sympy 判等：第二个模型说"一样"不算
    agreed = _equal(outcome.second_answer or "", outcome.model_answer, symbols)
    outcome.gate2_passed = agreed is True
    if outcome.gate2_passed:
        outcome.suspect = Suspect.CONFIRMED
    else:
        outcome.notes.append("gate2_disagreed")


def _assert_no_leak(messages: list[dict[str, Any]], *, stem: str, student_answer: str) -> None:
    """学生答案若出现在 prompt 里，而题干与模板里都没有它 —— 信息隔离被破坏。

    真正的保证是 `prompts.blind_solve.render` **只收题干**；这里是第二道网：
    万一有人把学生答案拼进上下文（补丁、重写、模板注入），这里会直接炸。

    已知盲区：若答案字符串本身出现在模板的示例里（如 `1/3`），子串检测无法区分
    "注入"与"撞上示例"，此时不报。所以它只做兜底，不能替代签名约束。
    """
    answer = student_answer.strip()
    if not answer or answer in stem or answer in prompt.template():
        return
    joined = "\n".join(str(message.get("content", "")) for message in messages)
    if answer in joined:
        raise ValueError("盲解 prompt 泄露了学生答案（信息集隔离被破坏）")
