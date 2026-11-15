"""出处核验（方案 §2.3）：机械比对，**没有模型、没有"语义相似"放宽**。

这条是防"洗白"的核心：**提取器可以说"这段是题干"，但不能说"题干是这些字"**。
"""

from __future__ import annotations

import re

__all__ = ["locate", "resolve_span", "verify_span", "ws_normalize"]

#: 全角空格 → 半角（`\u3000`）；连续空白折叠成单个空格
_FULLWIDTH_SPACE = "\u3000"


def ws_normalize(text: str) -> str:
    """只做空白归一：去首尾、折叠连续空白、统一全角/半角空格。"""
    if not isinstance(text, str):
        return ""
    collapsed = re.sub(r"\s+", " ", text.replace(_FULLWIDTH_SPACE, " "))
    return collapsed.strip()


def verify_span(original: str, value: str, span: tuple[int, int] | None) -> bool:
    """`original[a:b]` 与 `value` 在空白归一后必须**逐字相等**。

    * `span` 越界 / 倒置 → False（不是"按 0 处理"）；
    * `value` 为空 → False（空字段应当用 `None`，不该用空串占位）。
    """
    if span is None:
        return False
    if not isinstance(value, str) or not value:
        return False
    a, b = span
    if not isinstance(a, int) or not isinstance(b, int):
        return False
    if a < 0 or b > len(original) or a >= b:
        return False
    return ws_normalize(original[a:b]) == ws_normalize(value)


def locate(original: str, value: str) -> tuple[int, int] | None:
    """在一段原文里**精确定位**一个逐字子串，返回 `[start, end)`；找不到返回 `None`。

    为什么需要它（实测驱动）：`qwen3.7-flash` 报字符偏移的准确率只有 **2/5**，
    含换行时几乎必错。让它算偏移，等于把"能不能用"押在它的算术上。

    但**判据不能放宽**：这里用的是 `str.find`，找不到就是改了字 —— 与 `verify_span` 同一把尺子。
    分工因此变成：**模型只负责"选哪一段"，偏移由我们算**。
    """
    if not isinstance(original, str) or not isinstance(value, str):
        return None
    target = value.strip()
    if not target:
        return None
    index = original.find(target)
    if index < 0:
        return None
    return (index, index + len(target))


def resolve_span(
    original: str, value: str, span: tuple[int, int] | None
) -> tuple[bool, tuple[int, int] | None, str]:
    """先信模型给的 span；不成立就用 `locate` 重新定位。

    返回 `(ok, span, how)`，`how ∈ {"model", "computed", "none"}`：
    * `model`    —— 模型给的偏移**本身就是对的**；
    * `computed` —— 模型偏移错了，但它的取值仍是原文的**逐字子串**，由我们算准位置；
    * `none`     —— 找不到 → 说明它改写了内容 → 调用方必须 fail closed。
    """
    if verify_span(original, value, span):
        return True, span, "model"
    found = locate(original, value)
    if found is not None:
        return True, found, "computed"
    return False, None, "none"
