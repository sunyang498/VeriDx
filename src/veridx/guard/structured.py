"""结构化守卫：容错解析 → schema 校验 → **错误回喂**重试（≤2）→ 降级。

## 为什么"回喂"有效（面试会被追问）

模型第一次失败通常是**约束未被执行**，而不是**不知道正确的约束**。
把校验器的具体判据回灌，等于把"生成问题"降级成"**修复问题**"——
修复比生成简单，这是它在第 2 次达峰的原因。第 3 次以后收益趋零：
同一上下文连续失败说明这是**能力边界**而不是随机波动，此时应换模型（级联）而不是继续重试。

## 三条硬约束

1. **每次重试是独立 trace 事件**（带 `attempt_index`、原始坏文本留档），不覆盖 —— I4；
2. **重试前查预算**：重试是"放大结构"（1 次 → 3 次 = 3 倍 token）。预算不够就降级，
   不许"先烧完再报错"；
3. **降级本身也要写 trace**（`status=skipped` + `reason_code`），不许静默返回"只给诊断"。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from veridx import config
from veridx.guard.schema import ParseOutcome, tolerant_parse
from veridx.llm.base import ChatResult, LLMProvider, ProviderError
from veridx.trace.event import NodeName, NodeStatus, ReasonCode
from veridx.trace.recorder import TraceRecorder
from veridx.trace.secrets import scrub

TModel = TypeVar("TModel", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """重试耗尽仍不合法。上层据此降级（`escalate` 或"只给诊断不含讲解"）。"""

    def __init__(self, errors: list[str], *, attempts: list[dict[str, Any]]) -> None:
        super().__init__("; ".join(errors)[:300])
        self.errors = errors
        self.attempts = attempts


@dataclass
class Budget:
    """token 预算闸门（重试前必查）。"""

    remaining_tokens: int
    #: 每次重试的估算成本（保守值）
    estimated_call_tokens: int = 1500

    def can_afford(self, calls: int = 1) -> bool:
        return self.remaining_tokens >= self.estimated_call_tokens * calls

    def spend(self, tokens: int) -> None:
        self.remaining_tokens -= max(tokens, 0)


@dataclass
class StructuredResult:
    data: dict[str, Any]
    model: TModel
    attempts: list[dict[str, Any]] = field(default_factory=list)
    usage_total: int = 0
    parse: ParseOutcome | None = None
    degraded: bool = False


def _feedback(errors: list[str]) -> str:
    return (
        "上一次输出不合法，具体问题是：\n- "
        + "\n- ".join(errors)
        + "\n请修正后重新**只输出 JSON**，字段名与取值范围必须完全符合要求。"
    )


def _archive(raw: str) -> dict[str, Any]:
    """原始坏文本留档：前 N 字符 + sha1（超长时只留摘要，但 sha 始终留）。"""
    cleaned = scrub(raw)
    return {
        "raw_head": cleaned[: config.RAW_TEXT_KEEP_CHARS],
        "raw_len": len(raw),
        "raw_sha": hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:16],
        "truncated": len(cleaned) > config.RAW_TEXT_KEEP_CHARS,
    }


async def call_structured(
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    model_cls: type[TModel],
    *,
    tracer: TraceRecorder | None = None,
    budget: Budget | None = None,
    max_retry: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    node: str = NodeName.DIAGNOSE.value,
) -> StructuredResult:
    """调结构化输出。成功返回 `StructuredResult`；失败抛 `StructuredOutputError`。"""
    retries = config.GUARD_MAX_RETRIES if max_retry is None else max_retry
    conversation = list(messages)
    attempts: list[dict[str, Any]] = []
    usage_total = 0
    errors: list[str] = []

    # provider 选择依据必须可观察（否则"三种可切换"是假能力）
    if tracer is not None:
        tracer.event(
            NodeName.PARSE.value,
            status=NodeStatus.OK,
            detail={"provider": provider.describe(), "node": node, "max_retry": retries},
        )

    for attempt_index in range(retries + 1):
        if budget is not None and attempt_index > 0 and not budget.can_afford():
            if tracer is not None:
                tracer.event(
                    node,
                    status=NodeStatus.SKIPPED,
                    reason_code=ReasonCode.SKIPPED_BUDGET.value,
                    detail={"attempt_index": attempt_index, "reason": "budget_exhausted_before_retry"},
                )
            errors.append("budget_exhausted_before_retry")
            raise StructuredOutputError(errors, attempts=attempts)

        try:
            result: ChatResult = await provider.chat(
                conversation,
                temperature=config.DEFAULT_TEMPERATURE if temperature is None else temperature,
                seed=seed,
                response_format="json",
            )
        except ProviderError as exc:
            usage_total += 0
            attempts.append({"attempt_index": attempt_index, "errors": [str(exc)], "stage": "provider"})
            if tracer is not None:
                tracer.event(
                    node,
                    status=NodeStatus.FAIL,
                    reason_code=ReasonCode.PROVIDER_5XX.value
                    if exc.retryable
                    else ReasonCode.INTERNAL_ERROR.value,
                    detail={
                        "attempt_index": attempt_index,
                        "error": scrub(str(exc)),
                        "retryable": exc.retryable,
                    },
                )
            if not exc.retryable:
                raise StructuredOutputError([str(exc)], attempts=attempts) from exc
            errors = [str(exc)]
            continue

        usage_total += result.usage.total
        if budget is not None:
            budget.spend(result.usage.total)

        parsed = tolerant_parse(result.text)
        attempt_errors: list[str] = []
        data: dict[str, Any] | None = None
        if parsed.value is None:
            attempt_errors = parsed.errors or ["unparseable"]
        elif not isinstance(parsed.value, dict):
            attempt_errors = [f"not_an_object:{type(parsed.value).__name__}"]
        else:
            try:
                data = model_cls(**parsed.value).model_dump()
            except ValidationError as exc:
                attempt_errors = [
                    f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in exc.errors()
                ]

        attempts.append(
            {
                "attempt_index": attempt_index,
                "errors": attempt_errors,
                "stage": "validate",
                "usage": result.usage.total,
                "transformations": parsed.transformations,
                "ambiguous": parsed.ambiguous,
                **_archive(result.text),
            }
        )

        if tracer is not None:
            tracer.event(
                node,
                status=NodeStatus.RETRY if attempt_errors else NodeStatus.OK,
                reason_code=ReasonCode.SCHEMA_INVALID.value if attempt_errors else None,
                tokens_in=result.usage.input_tokens,
                tokens_out=result.usage.output_tokens,
                detail={
                    "attempt_index": attempt_index,
                    "errors": attempt_errors,
                    "transformations": parsed.transformations,
                    "ambiguous": parsed.ambiguous,
                    **_archive(result.text),
                },
            )

        if data is not None:
            model = model_cls(**data)
            return StructuredResult(
                data=data,
                model=model,
                attempts=attempts,
                usage_total=usage_total,
                parse=parsed,
            )

        errors = attempt_errors
        conversation = [
            *conversation,
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": _feedback(attempt_errors)},
        ]

    if tracer is not None:
        tracer.event(
            node,
            status=NodeStatus.FAIL,
            reason_code=ReasonCode.SCHEMA_INVALID.value,
            detail={"attempts": len(attempts), "final_errors": errors, "degraded": True},
        )
    raise StructuredOutputError(errors, attempts=attempts)


__all__ = ["Budget", "StructuredOutputError", "StructuredResult", "call_structured"]
