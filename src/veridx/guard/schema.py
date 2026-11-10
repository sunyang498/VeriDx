"""容错解析（`tolerant_parse`）：坏 JSON 分类表 F1–F12 的唯一实现。

## 三条纪律（来自 P1.1.1 审查教训）

1. **判据绑语义，不绑形状**。不许"用正则抠 `{...}` 就当结果"——
   必须走"切出候选文本 → `json.loads` → 交给 schema"。正则遇到字符串里的 `}` 就错，
   而**"错"与"绕过"是同一件事**。
2. **宽容解析的唯一合法目标是"不改值"**。不许全局替换中文引号：
   `"a": "他说“好”"` 里的中文引号是**内容**，替换就改了值。
   只在**结构位置**（不在任何 ASCII 双引号字符串内）替换。
3. **永不抛异常**。失败一律返回带 `errors` 的结果——与 `evaluate()` 的 I3 同一条不变量。

## 有序候选（命中即停，不"再猜一次"）

| 步 | 处理 | 对应分类 |
|---|---|---|
| 1 | 原文 `json.loads` | 干净 JSON |
| 2 | 剥 ```` ```json ```` / ```` ``` ```` 围栏 | F1 |
| 3 | 截取最外层平衡 `{}` / `[]`（**按字符串状态机扫描**，不是正则） | F2 / F10 |
| 4 | 去尾逗号（同样在字符串外） | F3 |
| 5 | 结构位置的中文引号 → ASCII | F5（**绝不动 F6 的值内部**） |
| 6 | 单引号对象 → `ast.literal_eval` | F4 |
| 7 | 双重编码：`"{\\"a\\":1}"` 解一层再解 | F8 |
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from veridx import config

_FENCE = re.compile(r"^```(?:json|JSON)?\s*\n?(?P<body>.*?)\n?```\s*$", re.S)

#: 中文引号（结构位置替换用）
_CN_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}


@dataclass
class ParseOutcome:
    value: Any | None = None
    errors: list[str] = field(default_factory=list)
    #: 走过哪些修复步骤（审计用；空 = 原文即合法）
    transformations: list[str] = field(default_factory=list)
    #: 出现了多个顶层 JSON 时置 True（取第一个，**不合并**）
    ambiguous: bool = False
    #: 原文摘要（脱敏后由调用方决定存不存全文）
    source_sha: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.errors


def tolerant_parse(text: object, *, max_chars: int | None = None) -> ParseOutcome:
    """把"模型可能吐出的各种畸形 JSON"变成 dict/list，或给出结构化错误。**永不抛。**"""
    try:
        return _parse(text, max_chars if max_chars is not None else config.TOLERANT_PARSE_MAX_CHARS)
    except BaseException as exc:  # noqa: BLE001 - 契约：永不抛
        return ParseOutcome(errors=[f"internal:{type(exc).__name__}"], source_sha=_sha(text))


def _parse(text: object, max_chars: int) -> ParseOutcome:
    if not isinstance(text, str):
        return ParseOutcome(errors=[f"not_a_string:{type(text).__name__}"], source_sha=_sha(text))
    source_sha = _sha(text)
    stripped = text.strip()
    if not stripped:
        return ParseOutcome(errors=["empty"], source_sha=source_sha)
    if len(stripped) > max_chars:
        # 超长：解析成本不可控，直接拒（对应 F12）
        return ParseOutcome(errors=[f"too_long:{len(stripped)}"], source_sha=source_sha)

    outcome = ParseOutcome(source_sha=source_sha)

    # 1 原文
    direct = _try_json(stripped)
    if direct is not None:
        if isinstance(direct, str):
            # 双重编码（F8）：`"{\"a\":1}"` —— 解一层字符串再解
            nested = _try_json(direct.strip())
            if isinstance(nested, (dict, list)):
                outcome.transformations.append("double_decode")
                return _finish(outcome, nested)
        return _finish(outcome, direct)

    # 2 剥代码块围栏（F1）
    fenced = _FENCE.match(stripped)
    if fenced:
        outcome.transformations.append("strip_fence")
        body = fenced.group("body").strip()
        direct = _try_json(body)
        if direct is not None:
            return _finish(outcome, direct)
        stripped = body

    # 3 截取最外层平衡片段（F2 / F10）
    extracted, ambiguous = _extract_outermost(stripped)
    if extracted is not None and extracted != stripped:
        outcome.transformations.append("extract_outermost")
        outcome.ambiguous = ambiguous
        direct = _try_json(extracted)
        if direct is not None:
            return _finish(outcome, direct)
        stripped = extracted

    # 4 去尾逗号（F3）
    no_trailing = _strip_trailing_commas(stripped)
    if no_trailing != stripped:
        outcome.transformations.append("strip_trailing_comma")
        direct = _try_json(no_trailing)
        if direct is not None:
            return _finish(outcome, direct)
        stripped = no_trailing

    # 5 结构位置的中文引号（F5；F6 的值内部不动）
    ascii_quoted = _normalize_structural_quotes(stripped)
    if ascii_quoted != stripped:
        outcome.transformations.append("normalize_structural_quotes")
        direct = _try_json(ascii_quoted)
        if direct is not None:
            return _finish(outcome, direct)
        stripped = ascii_quoted

    # 6 单引号对象（F4）
    if "'" in stripped:
        literal = _try_literal_eval(stripped)
        if literal is not None:
            outcome.transformations.append("literal_eval")
            return _finish(outcome, literal)

    # 7 双重编码（F8）
    double_encoded = _try_double_encoded(stripped)
    if double_encoded is not None:
        outcome.transformations.append("double_decode")
        return _finish(outcome, double_encoded)

    outcome.errors.append(_diagnose(stripped))
    return outcome


def _finish(outcome: ParseOutcome, value: Any) -> ParseOutcome:
    if not isinstance(value, (dict, list)):
        outcome.errors.append(f"not_an_object:{type(value).__name__}")
        return outcome
    outcome.value = value
    return outcome


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None


def _try_literal_eval(text: str) -> Any | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, (dict, list)) else None


def _try_double_encoded(text: str) -> Any | None:
    inner = _try_json(text)
    if not isinstance(inner, str):
        return None
    return _try_json(inner.strip())


def _sha(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8", "replace")).hexdigest()[:16]


def _scan(text: str) -> list[tuple[int, bool]]:
    """把文本按"是否在 ASCII 双引号字符串内"标注每个字符的状态。

    这是 F5/F6 能用同一份逻辑的前提：**只有 `in_string=False` 的位置才允许改引号**。
    正确处理反斜杠转义（`\\"` 不算结束符）。
    """
    states: list[tuple[int, bool]] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        states.append((index, in_string))
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
    return states


def _extract_outermost(text: str) -> tuple[str | None, bool]:
    """找到第一个平衡的 `{...}` 或 `[...]`（字符串感知），并判断后面是否还有第二个。"""
    states = dict(_scan(text))
    start = None
    opener = ""
    depth = 0
    for index, char in enumerate(text):
        if states.get(index, False):
            continue
        if start is None and char in "{[":
            start = index
            opener = char
            depth = 1
            continue
        if start is None:
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                rest = text[index + 1 :].strip()
                ambiguous = bool(rest) and rest.startswith(("{", "["))
                return candidate, ambiguous
    if start is not None and opener:
        return text[start:], False  # 截断（F7）：交给上层报结构化错误
    return None, False


def _strip_trailing_commas(text: str) -> str:
    states = dict(_scan(text))
    out: list[str] = []
    for index, char in enumerate(text):
        if char == "," and not states.get(index, False):
            rest = text[index + 1 :].lstrip()
            if rest[:1] in ("}", "]", ""):
                continue  # 丢掉这个逗号
        out.append(char)
    return "".join(out)


def _normalize_structural_quotes(text: str) -> str:
    """只替换**字符串外**的中文引号（F5）；值内部的（F6）逐字节保留。"""
    states = dict(_scan(text))
    out = []
    for index, char in enumerate(text):
        if char in _CN_QUOTES and not states.get(index, False):
            replacement = _CN_QUOTES[char]
            # 中文单引号不替换成 ASCII 单引号（会破坏 Python 里的 `'` 语义），只换成双引号更稳
            out.append('"' if replacement == "'" else replacement)
        else:
            out.append(char)
    return "".join(out)


def _diagnose(text: str) -> str:
    """给失败一个可读原因（不猜、不补）。"""
    if not text:
        return "empty"
    if text.count("{") + text.count("[") > text.count("}") + text.count("]"):
        return "truncated"  # F7
    return "unparseable"
