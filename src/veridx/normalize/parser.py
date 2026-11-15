"""确定性归一化器（方案 §3.1）：LaTeX / 常见记法 → 沙箱能算的 Python 表达式。

三条纪律：

1. **零随机**：纯字符串重写，没有模型、没有网络、没有沙箱调用；
2. **只做可交叉验证的转换**：拿不准就标 `ok=False`，交给模型归一化器；
3. **不猜语义**：`1,5` 是欧洲小数还是中文千分位无法区分 → 不碰它（留给闸 2 拦）。

为什么要它：公开数据集（ProcessBench / GSM8K / MATH）的答案写着 `\\frac{5}{7}`、`\\boxed{32}`、
`1.5\\times10^{3}`，而沙箱只吃 Python 表达式 —— 不归一化，这些题会全部掉出评测分母。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["NORMALIZER_VERSION", "ParseResult", "to_expr"]

#: 归一化器版本（进报告头；跨 run 对比时必须相同，方案 §6.2 第 3 条）
NORMALIZER_VERSION = "norm-parser-v1"

_WRAPPERS = (
    ("$$", "$$"),
    ("\\[", "\\]"),
    ("\\(", "\\)"),
    ("$", "$"),
)

#: `\left(` / `\right)` 只是排版，去掉不影响含义
_DROP_TEX = (
    "\\left",
    "\\right",
    "\\!",
    "\\,",
    "\\;",
    "\\:",
    "\\ ",
    "\\quad",
    "\\qquad",
    "\\displaystyle",
    "\\limits",
)

_EXACT_REPLACEMENTS = (
    ("\\times", "*"),
    ("\\cdot", "*"),
    ("\\div", "/"),
    ("\\ast", "*"),
    ("\\pi", "pi"),
    ("\\infty", "inf"),
    ("\\leq", "<="),
    ("\\geq", ">="),
    ("\\neq", "!="),
    ("\\lt", "<"),
    ("\\gt", ">"),
)

#: 中文/英文单位后缀：`12 厘米` → `12`（单位不是数值的一部分；**不做**单位换算）
_UNIT_SUFFIX = re.compile(
    r"(?:平方|立方)?\s*(?:厘米|米|毫米|千米|公分|平方米|立方米|cm|mm|km|m|s|kg|g|元|个|人|小时|分钟)\s*$"
)

_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\s*(?:\[([^\]]*)\])?\s*\{([^{}]*)\}")
#: 数值盒子：`\boxed{32}`、`\mathbf{1.5}` —— 内容一定是答案本体，剥掉壳
_BOXED = re.compile(r"\\(?:boxed|mathbf)\s*\{([^{}]*)\}")
#: 排版盒子：`\text{cm}`、`\mathrm{d}` —— 内容**不一定是数值**，剥壳后要复核
_TYPESET = re.compile(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}")
_SUP = re.compile(r"\^\{([^{}]*)\}")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*\\?%")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
#: 归一化后允许出现的字符（沙箱能解析的子集）。出现别的字符 → 判"没把握"。
_PLAUSIBLE = re.compile(r"^[\d\s.+\-*/()^_a-zA-Z]+$")


@dataclass
class ParseResult:
    """`ok=False` 表示"没把握" —— 调用方应交给模型归一化器，而不是硬用 `text`。"""

    text: str
    ok: bool
    changed: bool = False
    notes: list[str] = field(default_factory=list)


def _strip_wrappers(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    for left, right in _WRAPPERS:
        if stripped.startswith(left) and stripped.endswith(right) and len(stripped) > len(left) + len(right):
            return stripped[len(left) : len(stripped) - len(right)].strip(), True
    return stripped, False


def to_expr(raw: str) -> ParseResult:
    """把一段答案文本转成沙箱表达式。**不调用沙箱、不调用模型。**"""
    if not isinstance(raw, str) or not raw.strip():
        return ParseResult(text=raw if isinstance(raw, str) else "", ok=False, notes=["empty"])

    text = raw
    notes: list[str] = []

    text, unwrapped = _strip_wrappers(text)
    if unwrapped:
        notes.append("unwrapped")

    # `\boxed{32}` / `\mathbf{1.5}` → 内容（盒子只是排版）
    for _ in range(4):
        swapped = _BOXED.sub(lambda m: m.group(1), text)
        if swapped == text:
            break
        text = swapped
        notes.append("unboxed")

    # `\text{cm}` / `\mathrm{d}` → 内容，但**只有内容是纯数值时才可信**
    typeset_ok = True
    for match in _TYPESET.finditer(text):
        if not re.fullmatch(r"[-+]?[\d.]+", match.group(1).strip()):
            typeset_ok = False
    for _ in range(4):
        swapped = _TYPESET.sub(lambda m: m.group(1), text)
        if swapped == text:
            break
        text = swapped
        notes.append("untypeset")

    for needle in _DROP_TEX:
        if needle in text:
            text = text.replace(needle, "")
            notes.append("dropped_tex_spacing")

    # `\frac{a}{b}` 可能嵌套（`\frac{\frac{1}{2}}{3}`），反复展开到不动点
    for _ in range(6):
        swapped = _FRAC.sub(lambda m: f"(({m.group(1)})/({m.group(2)}))", text)
        if swapped == text:
            break
        text = swapped
        notes.append("frac")

    for _ in range(6):
        swapped = _SQRT.sub(
            lambda m: f"(({m.group(2)})**(1/({m.group(1)})))" if m.group(1) else f"sqrt({m.group(2)})",
            text,
        )
        if swapped == text:
            break
        text = swapped
        notes.append("sqrt")

    # `x^{10}` → `x**(10)`（必须在 frac 之后：`\frac` 的花括号已经被吃掉）
    if _SUP.search(text):
        text = _SUP.sub(lambda m: f"**({m.group(1)})", text)
        notes.append("sup_braced")

    for needle, replacement in _EXACT_REPLACEMENTS:
        if needle in text:
            text = text.replace(needle, replacement)
            notes.append(f"replaced:{needle}")

    if _PERCENT.search(text):
        text = _PERCENT.sub(lambda m: f"(({m.group(1)})/100)", text)
        notes.append("percent")

    # 千分位：只吃掉"严谨的三位分组"（`1,234` / `12,345,678`）。
    # `1,5`（欧洲小数）与 `1,5`（中文千分位）无法区分 → **不动**，由闸 2 拦。
    if _THOUSANDS.search(text):
        text = _THOUSANDS.sub("", text)
        notes.append("thousands_sep")

    text = text.replace("{", "(").replace("}", ")").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=\d)\s+(?=[+\-*/)])", "", text)
    text = re.sub(r"(?<=[+\-*/])\s+(?=\d)", "", text)

    if not text:
        return ParseResult(text="", ok=False, notes=[*notes, "empty_after_normalize"])

    # 还剩反斜杠 = 有我们没覆盖的 TeX 命令 → 不硬来
    if "\\" in text:
        return ParseResult(text=text, ok=False, notes=[*notes, "residual_tex"])

    # `\text{cm}` 这类排版盒子剥出来的不是数值 → 不硬来（单位换算不在本层）
    if not typeset_ok:
        return ParseResult(text=text, ok=False, notes=[*notes, "typeset_non_numeric"])

    # 数字之间的逗号还在（如 `1,5`）= 欧洲小数 vs 千分位无法区分 → 不硬来
    if re.search(r"\d,\d", text):
        return ParseResult(text=text, ok=False, notes=[*notes, "ambiguous_comma"])

    # 纯数值 + 单位后缀时才剥单位，避免把 `12米+3米` 这类算式改坏
    unitless = _UNIT_SUFFIX.sub("", text).strip()
    if unitless != text and re.fullmatch(r"[-+]?[\d.\s]+", unitless):
        text = unitless
        notes.append("unit_suffix")

    if not _PLAUSIBLE.match(text):
        return ParseResult(text=text, ok=False, notes=[*notes, "implausible_chars"])

    return ParseResult(text=text, ok=True, changed=text != raw.strip(), notes=notes)
