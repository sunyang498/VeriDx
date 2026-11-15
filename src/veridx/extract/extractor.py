"""隔离提取器（方案 §4）：自由文本 → 结构化字段。

隔离约束**写在这里，不只是写在文档里**：

| # | 约束 | 落点 |
|---|---|---|
| 1 | 不提供任何工具/函数 | 只发 `messages`，从不传 tools |
| 2 | 单轮、`temperature=0`、`response_format=json`、`max_tokens=800` | `call_structured(max_tokens=…)` |
| 3 | 输入用既有的 L0 围栏（随机 nonce）包住 | `extract.prompt.render` |
| 4 | schema `extra="forbid"` + 全字段上界 + 枚举 | `extract.schema.ExtractDraft` |
| 5 | 失败重试 ≤1 | `EXTRACT_MAX_RETRIES` |
| 6 | 任何失败 → **fail closed** | 返回 `ExtractOutcome(ok=False)`，绝不"用原始文本硬上" |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from veridx import config
from veridx.extract import prompt as prompt_mod
from veridx.extract.schema import ExtractDraft
from veridx.extract.span import SegmentCheck, check_segments, verify_verbatim
from veridx.guard.structured import Budget, StructuredOutputError, call_structured
from veridx.llm.base import LLMProvider
from veridx.trace.event import NodeName, ReasonCode
from veridx.trace.recorder import TraceRecorder

__all__ = ["ExtractOutcome", "run_extract"]

SYSTEM = (
    "你是「题目结构提取器」。你只做一件事：从给定文本里**逐字摘录**出题干与学生答案，并给出字符坐标。\n"
    "你不需要、也不允许做任何超出摘录的事情。\n\n" + prompt_mod.FENCE_INSTRUCTION
)


@dataclass
class ExtractOutcome:
    ok: bool
    reason_code: str | None = None
    draft: ExtractDraft | None = None
    nonce: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    attempts: int = 0
    notes: list[str] = field(default_factory=list)
    #: §2.4 的机械检查结果（M1 拒绝时 `ok=False`；M2/M3 只记录）
    segments: SegmentCheck | None = None


async def run_extract(
    provider: LLMProvider,
    raw: str,
    *,
    tracer: TraceRecorder | None = None,
    budget: Budget | None = None,
    max_retry: int | None = None,
) -> ExtractOutcome:
    """跑一次隔离提取。**永不抛异常**：失败一律落到 `reason_code`（fail closed）。"""
    if not isinstance(raw, str) or not raw.strip():
        return ExtractOutcome(ok=False, reason_code=ReasonCode.EXTRACTION_FAILED.value, notes=["empty_input"])

    body, nonce = prompt_mod.render(raw)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": body},
    ]
    retries = config.EXTRACT_MAX_RETRIES if max_retry is None else max_retry

    if tracer is not None:
        tracer.event(
            NodeName.EXTRACT.value,
            status="ok",
            detail={
                "stage": "request",
                "prompt_version": prompt_mod.VERSION,
                "prompt_hash": prompt_mod.prompt_hash(),
                "model": provider.model,
                "l0": {"nonce": nonce, "fenced": True, "prompt_chars": len(body), "raw_chars": len(raw)},
                "max_retry": retries,
                "max_tokens": config.EXTRACT_MAX_TOKENS,
                "tools": [],  # 空列表是**事实**，不是装饰：隔离靠"没有工具"这件事
            },
        )

    try:
        result = await call_structured(
            provider,
            messages,
            ExtractDraft,
            tracer=tracer,
            budget=budget,
            node=NodeName.EXTRACT.value,
            max_retry=retries,
            max_tokens=config.EXTRACT_MAX_TOKENS,
        )
    except StructuredOutputError as exc:
        outcome = ExtractOutcome(
            ok=False,
            reason_code=ReasonCode.EXTRACTION_FAILED.value,
            nonce=nonce,
            attempts=len(exc.attempts),
            notes=[str(item)[:120] for item in exc.errors[:3]],
        )
        _trace_result(tracer, outcome)
        return outcome

    draft = result.model
    # ---- 出处核验（v1.1）：每段文本必须能在原文里**逐字定位**到 ----
    # 模型只输出文本，偏移由我们算（实测它报偏移的准确率只有 2/5）
    stem_located = verify_verbatim(raw, draft.stem)
    if not stem_located.ok:
        outcome = ExtractOutcome(
            ok=False,
            reason_code=ReasonCode.PROVENANCE_MISMATCH.value,
            draft=draft,
            nonce=nonce,
            tokens_in=result.usage_in,
            tokens_out=result.usage_out,
            attempts=len(result.attempts),
            notes=["stem_not_verbatim"],
        )
        _trace_result(tracer, outcome)
        return outcome

    answer_located = None
    if draft.student_answer is not None:
        # M2 消歧：避开题干占用的那段
        answer_located = verify_verbatim(raw, draft.student_answer, avoid=stem_located.span)
        if not answer_located.ok:
            outcome = ExtractOutcome(
                ok=False,
                reason_code=ReasonCode.PROVENANCE_MISMATCH.value,
                draft=draft,
                nonce=nonce,
                tokens_in=result.usage_in,
                tokens_out=result.usage_out,
                attempts=len(result.attempts),
                notes=["student_answer_not_verbatim"],
            )
            _trace_result(tracer, outcome)
            return outcome

    # ---- §2.4 三条机械检查：M1 会拒，M2/M3 只记录 ----
    segments = check_segments(raw, stem=stem_located, student_answer=answer_located)
    if not segments.ok:
        outcome = ExtractOutcome(
            ok=False,
            reason_code=ReasonCode.PROVENANCE_MISMATCH.value,
            draft=draft,
            nonce=nonce,
            tokens_in=result.usage_in,
            tokens_out=result.usage_out,
            attempts=len(result.attempts),
            notes=list(segments.flags),
        )
        _trace_result(tracer, outcome)
        return outcome

    outcome = ExtractOutcome(
        ok=True,
        draft=draft,
        nonce=nonce,
        tokens_in=result.usage_in,
        tokens_out=result.usage_out,
        attempts=len(result.attempts),
        notes=["verbatim_ok", f"stem_how:{stem_located.how}", *segments.flags],
        segments=segments,
    )
    _trace_result(tracer, outcome)
    return outcome


def _trace_result(tracer: TraceRecorder | None, outcome: ExtractOutcome) -> None:
    if tracer is None:
        return
    tracer.event(
        NodeName.EXTRACT.value,
        status="ok" if outcome.ok else "fail",
        reason_code=outcome.reason_code,
        tokens_in=outcome.tokens_in,
        tokens_out=outcome.tokens_out,
        detail={
            "stage": "result",
            "ok": outcome.ok,
            "verbatim_ok": outcome.ok,
            "attempt_index": max(outcome.attempts - 1, 0),
            "attempts": outcome.attempts,
            "nonce": outcome.nonce,
            "notes": outcome.notes,
            "student_answer_present": bool(outcome.draft and outcome.draft.student_answer),
            "detected_lang": outcome.draft.detected_lang if outcome.draft else None,
            "layout_hint": outcome.draft.layout_hint if outcome.draft else None,
            # 服务端算出来的定位区间（审计用；**不是模型给的**）
            "segments": outcome.segments.to_detail() if outcome.segments else None,
        },
    )
