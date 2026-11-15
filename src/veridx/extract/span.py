"""出处核验（方案 v1.1 §2.3 / §2.4）：机械比对，**没有模型、没有"语义相似"放宽**。

安全性质**只有一条**：

> 提取出的每一段文本，必须能在原文里被"**逐字定位**"到。定位不到 = 它改了字/编了内容 = 拒绝。

模型给不给偏移**不影响安全性** —— 所以 v1.1 让模型只输出文本，偏移由服务端算（`locate`），
并落 trace 供审计。丢掉的是"同一片段出现多次"的消歧能力，用 §2.4 的三条机械检查补：

| # | 检查 | 违规后果 |
|---|---|---|
| **M1 不重叠** | `stem` 与 `student_answer` 的定位区间不许相交 | `ambiguous_segmentation`（**唯一会拒的**） |
| **M2 消歧** | 多处匹配：优先与另一字段不重叠的那个；仍多个 → 取最左并记 `ambiguous_span` | 只记录 |
| **M3 覆盖度** | 提取总长 / 原文长 < 0.3 且原文 > 20 字 → 记 `low_coverage` | 只记录 |

> M2/M3 只记录，因为按 §6.1.5，切分质量属于"数据质量"不是安全问题 ——
> 我们要的是**可观测**，不是替用户改题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "COVERAGE_MIN",
    "COVERAGE_MIN_CHARS",
    "LocateResult",
    "SegmentCheck",
    "check_segments",
    "locate",
    "locate_all",
    "verify_verbatim",
    "ws_normalize",
]

_FULLWIDTH_SPACE = "\u3000"

#: M3 的阈值（方案 §2.4）：低于它就记 `low_coverage`
COVERAGE_MIN = 0.3
COVERAGE_MIN_CHARS = 20


def ws_normalize(text: str) -> str:
    """只做空白归一：去首尾、折叠连续空白、统一全角/半角空格。"""
    if not isinstance(text, str):
        return ""
    collapsed = re.sub(r"\s+", " ", text.replace(_FULLWIDTH_SPACE, " "))
    return collapsed.strip()


@dataclass
class LocateResult:
    ok: bool
    span: tuple[int, int] | None = None
    how: str = "none"
    occurrences: int = 0
    ambiguous: bool = False


def locate_all(original: str, value: str) -> list[tuple[int, int]]:
    """`value`（去首尾空白后）在原文里的**全部**匹配区间。"""
    if not isinstance(original, str) or not isinstance(value, str):
        return []
    target = value.strip()
    if not target:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        index = original.find(target, start)
        if index < 0:
            break
        spans.append((index, index + len(target)))
        start = index + 1
    return spans


def locate(original: str, value: str) -> tuple[int, int] | None:
    """在一段原文里**精确定位**一个逐字子串（取最左），找不到返回 `None`。"""
    spans = locate_all(original, value)
    return spans[0] if spans else None


def verify_verbatim(original: str, value: str, *, avoid: tuple[int, int] | None = None) -> LocateResult:
    """`ws_normalize(value)` 必须在 `ws_normalize(original)` 里被定位到。

    先按原文逐字找；找不到再试"空白归一后"的等价定位（只放宽空白，别的都不放宽）。
    `avoid` 给 M2 用：优先选与另一字段不重叠的那个匹配位置。
    """
    if not isinstance(original, str) or not isinstance(value, str):
        return LocateResult(ok=False)
    spans = locate_all(original, value)
    how = "exact"

    if not spans:
        target = ws_normalize(value)
        if not target:
            return LocateResult(ok=False)
        compact, index_map = _compact(original)
        found = compact.find(target)
        if found < 0:
            return LocateResult(ok=False, how="none")
        start = index_map[found]
        end = index_map[found + len(target) - 1] + 1
        spans = [(start, end)]
        how = "ws_normalized"

    chosen, ambiguous = _disambiguate(spans, avoid)
    return LocateResult(
        ok=True,
        span=chosen,
        how=how,
        occurrences=len(spans),
        ambiguous=ambiguous,
    )


def _compact(text: str) -> tuple[str, list[int]]:
    """压掉多余空白，同时记下每个压缩后字符对应的原文下标。"""
    chars: list[str] = []
    index_map: list[int] = []
    last_space = False
    for index, char in enumerate(text):
        is_space = char.isspace() or char == _FULLWIDTH_SPACE
        if is_space:
            if last_space:
                continue
            chars.append(" ")
            index_map.append(index)
            last_space = True
        else:
            chars.append(char)
            index_map.append(index)
            last_space = False
    return "".join(chars), index_map


def _disambiguate(
    spans: list[tuple[int, int]], avoid: tuple[int, int] | None
) -> tuple[tuple[int, int], bool]:
    """M2：优先与另一字段不重叠的那个；仍多个 → 取最左并标记 `ambiguous`。"""
    if len(spans) == 1:
        return spans[0], False
    if avoid is not None:
        clean = [span for span in spans if not _overlaps(span, avoid)]
        if len(clean) == 1:
            return clean[0], False
        if len(clean) > 1:
            return clean[0], True
        return spans[0], True
    return spans[0], True


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


@dataclass
class SegmentCheck:
    """§2.4 的机械检查结果。**M1 会拒，M2/M3 只记录。**"""

    ok: bool
    flags: list[str] = field(default_factory=list)
    coverage: float = 1.0
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)

    def to_detail(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "flags": list(self.flags),
            "coverage": round(self.coverage, 3),
            "spans": {name: list(span) for name, span in self.spans.items()},
        }


def check_segments(
    original: str,
    *,
    stem: LocateResult,
    student_answer: LocateResult | None,
) -> SegmentCheck:
    """M1/M2/M3 一起跑。输入必须是已经 `verify_verbatim` 通过的定位结果。"""
    spans: dict[str, tuple[int, int]] = {}
    if stem.span is not None:
        spans["stem"] = stem.span
    if student_answer is not None and student_answer.span is not None:
        spans["student_answer"] = student_answer.span

    flags: list[str] = []

    # M1：两个字段抢同一段文字 → 一定是切错了（唯一会拒的）
    if "stem" in spans and "student_answer" in spans and _overlaps(spans["stem"], spans["student_answer"]):
        return SegmentCheck(ok=False, flags=["ambiguous_segmentation"], spans=spans)

    # M2：歧义只记录
    if stem.ambiguous:
        flags.append("ambiguous_span:stem")
    if student_answer is not None and student_answer.ambiguous:
        flags.append("ambiguous_span:student_answer")

    # M3：覆盖度过低只记录
    total = sum(span[1] - span[0] for span in spans.values())
    length = len(original)
    coverage = total / length if length else 0.0
    if length > COVERAGE_MIN_CHARS and coverage < COVERAGE_MIN:
        flags.append("low_coverage")

    return SegmentCheck(ok=True, flags=flags, coverage=coverage, spans=spans)
