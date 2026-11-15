"""双闸与三档结论（方案 §3.2 / §3.3 / §3.4）。

```
闸 1 · 双路径一致性   确定性解析器 × 模型归一化 → 都产出就必须一致（否则 conflict）
闸 2 · 沙箱等价性     compare(raw_side, normalized_side) → 判出"不相等"即 conflict
```

三档结论 **不许混用**：

| 结论 | 含义 | 能否用于判对错 |
|---|---|---|
| `verified` | 双闸都过（或**确定性重写**，零随机） | ✅ 可以 |
| `conflict` | 两条路径不一致 / 沙箱判出不等 | ❌ → `escalated` |
| `unverified` | **只有模型产出**、无法交叉验证 | ❌ → 需要判等时 `escalated` |

> ⚠️ 一条必须写清楚的口径（方案 §3.2 的字面读法会让 LaTeX 完全不可用）：
> 闸 2 要求 `compare(raw, normalized) == equal`，但**原文是 LaTeX 时沙箱根本解析不了**，
> 闸 2 只能返回"判不出"。若把"判不出"一律记成 `unverified`，`\\frac{5}{7}` 就永远不能用于判等
> —— 而这正是本节存在的理由。因此本实现的判据是：
>
> * **确定性解析器**的成功是**构造性证明**（纯字符串重写、零随机、无外部输入）→ `verified`；
>   但**只要沙箱能判**且判出"不相等"，立刻降级 `conflict`（说明解析器有 bug，必须被看见）。
> * **模型产出**必须由确定性解析器交叉验证（闸 1）才算 `verified`；没有交叉源 → `unverified`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from veridx.normalize.model import normalize_with_model
from veridx.normalize.parser import to_expr
from veridx.normalize.static import static_parse_ok
from veridx.sandbox import compare
from veridx.trace.event import NodeName
from veridx.trace.recorder import TraceRecorder

__all__ = [
    "CONFLICT",
    "UNCHANGED",
    "UNVERIFIED",
    "VERIFIED",
    "NormalizeOutcome",
    "both_sides_verified",
    "judge_safe",
    "normalize_side",
]

VERIFIED = "verified"
CONFLICT = "conflict"
UNVERIFIED = "unverified"
#: 原文本来就是普通表达式、一个字都没改 —— 与 `verified` 等价，单列便于统计"多少题真需要归一化"
UNCHANGED = "unchanged"

Side = Literal["answer_key", "student_answer"]


@dataclass
class NormalizeOutcome:
    side: Side
    raw: str
    normalized: str
    normalizer: Literal["parser", "llm", "none"]
    verdict: str
    #: 闸 1 的结论：两条路径是否一致（`None` = 没有第二条路径可交叉）
    agree: bool | None = None
    #: 闸 2 的结论：沙箱能否判出"原始 = 归一化"（`None` = 判不出）
    sandbox_agree: bool | None = None
    notes: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def usable(self) -> bool:
        """能不能拿去做判等（方案 §3.3 + 审查 §3.5）.

        **两条都要满足**：
        ① 结论是 `verified` / `unchanged`；
        ② 归一化结果**能被沙箱静态解析** —— 否则 `verified` 只是"解析器自己说成功"。
        """
        return self.verdict in {VERIFIED, UNCHANGED} and static_parse_ok(self.normalized)

    def to_detail(self) -> dict[str, object]:
        return {
            "side": self.side,
            "raw": self.raw[:200],
            "normalized": self.normalized[:200],
            "normalizer": self.normalizer,
            "verdict": self.verdict,
            "agree": self.agree,
            "sandbox_agree": self.sandbox_agree,
            "notes": self.notes,
        }


def _sandbox_equal(left: str, right: str) -> bool | None:
    """闸：沙箱能否判出"两侧相等"。`None` = 判不出（**不是**"不相等"）。"""
    if left.strip() == right.strip():
        return True
    try:
        return compare(left, right).equal
    except Exception:  # noqa: BLE001 - 归一化层不许因为沙箱抖动把整题炸掉
        return None


async def normalize_side(
    raw: str,
    *,
    side: Side,
    provider: object | None = None,
    tracer: TraceRecorder | None = None,
    budget: object | None = None,
    allow_model: bool = True,
    cross_check: bool = False,
) -> NormalizeOutcome:
    """归一化**一侧**。

    * `allow_model=False`：只跑确定性解析器（回归测试 / "省钱"的证明）；
    * `cross_check=True`：**即使解析器成功也要再跑一次模型**，然后走闸 1 双向比对。
      默认关闭（方案 §3.1：解析成功就不用模型，省钱且零风险）；
      preflight 与"模型给错"的验收用例打开它 —— 那正是闸 1 存在的场景。
    """
    parsed = to_expr(raw)
    model_value: tuple[str, int, int] | None = None

    needs_model = (not parsed.ok) or cross_check
    if needs_model and allow_model and provider is not None:
        modeled = await normalize_with_model(
            provider,  # type: ignore[arg-type]
            raw,
            path=side,
            tracer=tracer,
            budget=budget,  # type: ignore[arg-type]
        )
        if modeled is not None:
            expr, _nonce, tokens_in, tokens_out = modeled
            model_value = (expr, tokens_in, tokens_out)

    # ---- 路径 A：确定性解析器成功 ----
    if parsed.ok:
        outcome = NormalizeOutcome(
            side=side,
            raw=raw,
            normalized=parsed.text,
            normalizer="parser",
            verdict=UNCHANGED if not parsed.changed else VERIFIED,
            notes=list(parsed.notes),
        )
        # 闸 2：沙箱能判就判。判出**不相等**说明确定性重写改坏了值 → 必须降级
        sandbox = _sandbox_equal(raw, parsed.text)
        outcome.sandbox_agree = sandbox
        if sandbox is False:
            outcome.verdict = CONFLICT
            outcome.notes.append("parser_changed_value")
            _trace(tracer, outcome)
            return outcome

        if model_value is None:
            _trace(tracer, outcome)
            return outcome

        # 闸 1（只在 cross_check 时走到）：两条路径必须一致
        expr, tokens_in, tokens_out = model_value
        outcome.tokens_in, outcome.tokens_out = tokens_in, tokens_out
        cross = _sandbox_equal(parsed.text, expr)
        outcome.agree = cross
        if cross is False:
            outcome.verdict = CONFLICT
            outcome.normalized = parsed.text  # 保留解析器结果供人核对
            outcome.notes.append("paths_disagree")
        elif cross is True:
            outcome.notes.append("cross_checked")
        else:
            outcome.verdict = UNVERIFIED
            outcome.notes.append("cross_check_unavailable")
        _trace(tracer, outcome)
        return outcome

    # ---- 路径 B：解析器没把握 ----
    if model_value is None:
        outcome = NormalizeOutcome(
            side=side,
            raw=raw,
            normalized=parsed.text or raw,
            normalizer="none",
            verdict=UNVERIFIED,
            notes=[*parsed.notes, "no_model_output"],
        )
        _trace(tracer, outcome)
        return outcome

    expr, tokens_in, tokens_out = model_value
    outcome = NormalizeOutcome(
        side=side,
        raw=raw,
        normalized=expr,
        normalizer="llm",
        verdict=UNVERIFIED,
        notes=list(parsed.notes),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    # 闸 2：模型产出也必须过沙箱等价性（能判就判）
    sandbox = _sandbox_equal(raw, expr)
    outcome.sandbox_agree = sandbox
    if sandbox is False:
        outcome.verdict = CONFLICT
        outcome.notes.append("model_changed_value")
        _trace(tracer, outcome)
        return outcome

    # 闸 1：与确定性解析器的**候选**交叉验证（解析器说"没把握" ≠ "没候选"）
    if parsed.text and parsed.text != raw.strip():
        cross = _sandbox_equal(parsed.text, expr)
        outcome.agree = cross
        if cross is False:
            outcome.verdict = CONFLICT
            outcome.notes.append("paths_disagree")
            _trace(tracer, outcome)
            return outcome
        if cross is True:
            outcome.verdict = VERIFIED
            outcome.notes.append("cross_checked")
            _trace(tracer, outcome)
            return outcome

    outcome.notes.append("no_cross_check")
    _trace(tracer, outcome)
    return outcome


def both_sides_verified(key: NormalizeOutcome, student: NormalizeOutcome) -> bool:
    """方案 §3.4：**任一侧不可用 → equal 不可信**（只归一化一侧会把误差直接变成"判错"）。"""
    return key.usable and student.usable


def judge_safe(key: NormalizeOutcome, student: NormalizeOutcome) -> bool:
    """能不能进入 `compare()` 的判等结论（方案 §3.3 的纪律）。"""
    if CONFLICT in {key.verdict, student.verdict}:
        return False
    return both_sides_verified(key, student)


def _trace(tracer: TraceRecorder | None, outcome: NormalizeOutcome) -> None:
    if tracer is None:
        return
    tracer.event(
        NodeName.NORMALIZE.value,
        status="ok" if outcome.usable else "skipped",
        detail={"stage": "result", **outcome.to_detail()},
    )
