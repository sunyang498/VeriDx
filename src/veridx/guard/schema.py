"""容错解析（`tolerant_parse`）：坏 JSON 分类表 F1–F12 的唯一实现。

## 三条纪律（来自 P1.1.1 审查教训）

1. **判据绑语义，不绑形状**。不许"用正则抠 `{...}` 就当结果"——
   必须走"切出候选文本 → `json.loads` → 交给 schema"。正则遇到字符串里的 `}` 就错，
   而**"错"与"绕过"是同一件事**。
2. **宽容解析的唯一合法目标是"不改值"**。不许全局替换中文引号：
   `"a": "他说“好”"` 里的中文引号是**内容**，替换就改了值。
   只在**结构位置**（不在任何字符串内）替换。
3. **永不抛异常**。失败一律返回带 `errors` 的结果——与 `evaluate()` 的 S3 同一条不变量。

## 引号族（P1.3 审查 🟠A / 🟡D）

`_scan` 同时跟踪四种引号族：ASCII 双引号、ASCII 单引号、中文双引号 `“”`、中文单引号 `‘’`。
同一时刻只有**一个**"当前字符串族"，因此别的族的引号落在字符串内时只是内容。

* 中文引号没有转义机制，**同形重入**（`“a“b””`）意味着边界不可判 →
  报 `ambiguous_quotes` + `ambiguous=True`，**不许取第一种解释**。
* ASCII 单引号与中文引号只有处在**结构位置**（行首或紧跟 `{ [ , : (` 或空白）才算开引号，
  否则 `It's done` 这类正文会把后面整段吞进"字符串"。

## 不静默丢值（P1.3 审查 🟠B / 🟠C）

`json.loads` 的默认行为有两处会**悄悄改语义**：

* 重复键取最后一个值（模型自相矛盾时被抹平）
* 接受 `NaN` / `Infinity` / `-Infinity`（下游 pydantic 拿到非有限数）

两者都是**终局判定**，报 `duplicate_key:<key>` / `non_finite:<literal>` 并停止后续"洗"。

## 有序候选（命中即停，不"再猜一次"）

| 步 | 处理 | 对应分类 |
|---|---|---|
| 1 | 原文 | — |
| 2 | 剥 ```` ``` ```` 围栏 | F1 |
| 3 | 截取最外层平衡片段 | F2 / F10 |
| 4 | 去尾逗号 | F3 |
| 5 | 结构位置中文引号 → ASCII | F5 |
| 6 | 单引号对象（`literal_eval`） | F4 / F6 |
| 7 | 双重编码解一层 | F8 |
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from veridx import config

__all__ = ["ParseOutcome", "tolerant_parse"]

_FENCE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*\n(?P<body>.*?)(?:\n\s*```\s*)?$", re.DOTALL)

#: 中文引号 → ASCII 双引号（只用于结构位置）
_CN_QUOTES = {"“": '"', "”": '"', "‘": '"', "’": '"'}

#: 各引号族：开 → (闭, 族名, 是否支持反斜杠转义)
_QUOTE_FAMILY: dict[str, tuple[str, str, bool]] = {
    '"': ('"', "ascii", True),
    "'": ("'", "ascii", True),
    "“": ("”", "cn", False),
    "‘": ("’", "cn", False),
}

#: 非标准引号只有紧跟这些字符（或位于行首）才算开引号
_OPENER_CONTEXT = frozenset(" \t\r\n{[,(:")


@dataclass
class ParseOutcome:
    value: Any | None = None
    errors: list[str] = field(default_factory=list)
    #: 走过哪些修复步骤（审计用；空 = 原文即合法）
    transformations: list[str] = field(default_factory=list)
    #: 有歧义时置 True：多个顶层 JSON（取第一个），或同形引号嵌套（直接失败）
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
    direct, fatal = _try_json(stripped)
    if fatal is not None:
        outcome.errors.append(fatal)
        return outcome
    if direct is not None:
        if isinstance(direct, str):
            # 双重编码（F8）：`"{\"a\":1}"` —— 解一层字符串再解
            nested, nested_fatal = _try_json(direct.strip())
            if nested_fatal is not None:
                outcome.errors.append(nested_fatal)
                return outcome
            if isinstance(nested, (dict, list)):
                outcome.transformations.append("double_decode")
                return _finish(outcome, nested)
        return _finish(outcome, direct)

    # 2 剥代码块围栏（F1）
    fenced = _FENCE.match(stripped)
    if fenced:
        outcome.transformations.append("strip_fence")
        body = fenced.group("body").strip()
        direct, fatal = _try_json(body)
        if fatal is not None:
            outcome.errors.append(fatal)
            return outcome
        if direct is not None:
            return _finish(outcome, direct)
        stripped = body

    # 3 截取最外层平衡片段（F2 / F10）
    extracted, ambiguous = _extract_outermost(stripped)
    if extracted is not None and extracted != stripped:
        outcome.transformations.append("extract_outermost")
        outcome.ambiguous = ambiguous
        direct, fatal = _try_json(extracted)
        if fatal is not None:
            outcome.errors.append(fatal)
            return outcome
        if direct is not None:
            return _finish(outcome, direct)
        stripped = extracted

    # 4 去尾逗号（F3）
    no_trailing = _strip_trailing_commas(stripped)
    if no_trailing != stripped:
        outcome.transformations.append("strip_trailing_comma")
        direct, fatal = _try_json(no_trailing)
        if fatal is not None:
            outcome.errors.append(fatal)
            return outcome
        if direct is not None:
            return _finish(outcome, direct)
        stripped = no_trailing

    # 5 结构位置的中文引号（F5；F6 的值内部不动）
    _, ambiguous_quotes = _scan(stripped)
    if ambiguous_quotes:
        # 同形引号嵌套：任何"标准化"都是猜测
        outcome.ambiguous = True
        outcome.errors.append("ambiguous_quotes")
        return outcome
    ascii_quoted = _normalize_structural_quotes(stripped)
    if ascii_quoted != stripped:
        outcome.transformations.append("normalize_structural_quotes")
        direct, fatal = _try_json(ascii_quoted)
        if fatal is not None:
            outcome.errors.append(fatal)
            return outcome
        if direct is not None:
            return _finish(outcome, direct)
        stripped = ascii_quoted

    # 6 单引号对象（F4）
    if "'" in stripped:
        literal = _try_literal_eval(stripped)
        if literal is not None:
            outcome.transformations.append("literal_eval")
            overflow = _find_non_finite(literal)
            if overflow is not None:
                # `{'a': 1e999}` 走的是 literal_eval，不经过 json hook
                outcome.errors.append(f"non_finite:{overflow}")
                return outcome
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


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non_finite:{name}")


def _reject_overfloat(text: str) -> float:
    """`1e999` 是**合法 JSON 数字**，`json` 解析时静默溢出成 `inf`（审查 §6-必2）。

    `parse_constant` 只拦字面量 `NaN/Infinity`，拦不到溢出——
    非有限数的第三次穿透（前两次：沙箱值域、字面量）。
    """
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non_finite:overflow:{text}")
    return value


def _find_non_finite(value: Any) -> str | None:
    """递归找非有限浮点（覆盖 `literal_eval` 路径：`{'a': 1e999}` → `inf`）。"""
    if isinstance(value, float):
        return None if math.isfinite(value) else "overflow"
    if isinstance(value, dict):
        for key, item in value.items():
            found = _find_non_finite(key) or _find_non_finite(item)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            found = _find_non_finite(item)
            if found:
                return found
    return None


def _try_json(text: str) -> tuple[Any, str | None]:
    """严格 JSON：拒非有限数（字面量与溢出）、拒重复键。

    返回 ``(value, fatal)``：

    * ``fatal is None`` 且 ``value is None`` → 只是不合法 JSON（后续步骤可以继续洗）
    * ``fatal`` 非空 → 终局判定（``non_finite:*`` / ``duplicate_key:*``），必须终止
    """
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = val
        return out

    try:
        value = json.loads(
            text,
            object_pairs_hook=hook,
            parse_constant=_reject_constant,
            parse_float=_reject_overfloat,
        )
    except json.JSONDecodeError:
        return None, None
    except ValueError as exc:
        message = str(exc)
        if message.startswith("non_finite"):
            return None, message
        return None, None
    except RecursionError:
        return None, None
    if duplicates:
        # 静默取最后一个值会掩盖模型自相矛盾（审查 🟠B）
        return None, f"duplicate_key:{duplicates[0]}"
    return value, None


def _try_literal_eval(text: str) -> Any | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, (dict, list)) else None


def _try_double_encoded(text: str) -> Any | None:
    inner, _ = _try_json(text)
    if not isinstance(inner, str):
        return None
    value, fatal = _try_json(inner.strip())
    return None if fatal is not None else value


def _sha(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8", "replace")).hexdigest()[:16]


def _can_open(text: str, index: int, char: str) -> bool:
    """非标准引号（单引号 / 中文引号）只在结构位置才算开引号。"""
    if char == '"':
        return True
    return index == 0 or text[index - 1] in _OPENER_CONTEXT


def _scan(text: str) -> tuple[list[tuple[int, bool]], bool]:
    """标注每个字符是否在字符串内，并报告中文引号同形嵌套。

    这是 F5/F6 能用同一份逻辑的前提：**只有 `in_string=False` 的位置才允许改引号**。
    正确处理反斜杠转义（`\\"` 不算结束符）。
    """
    states: list[tuple[int, bool]] = []
    family: str | None = None
    closer = ""
    backslash = False
    escaped = False
    ambiguous = False

    for index, char in enumerate(text):
        if family is None:
            states.append((index, False))
            opened = _QUOTE_FAMILY.get(char)
            if opened is not None and _can_open(text, index, char):
                closer, family, backslash = opened
                escaped = False
            continue
        if escaped:
            states.append((index, True))
            escaped = False
            continue
        if backslash and char == "\\":
            states.append((index, True))
            escaped = True
            continue
        if char == closer:
            # 闭引号视为结构位置（可被 F5 标准化）
            states.append((index, False))
            family = None
            closer = ""
            backslash = False
            continue
        states.append((index, True))
        if family == "cn" and char in _CN_QUOTES:
            # 中文引号无转义、无方向冗余：同形重入即边界不可判（审查 🟠A）
            ambiguous = True

    return states, ambiguous


def _extract_outermost(text: str) -> tuple[str | None, bool]:
    """找到第一个平衡的 `{...}` 或 `[...]`（字符串感知），并判断后面是否还有第二个。"""
    states = dict(_scan(text)[0])
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
    states = dict(_scan(text)[0])
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
    states = dict(_scan(text)[0])
    out = []
    for index, char in enumerate(text):
        if char in _CN_QUOTES and not states.get(index, False):
            # 中文单引号统一换成 ASCII 双引号：换成 `'` 会破坏后续 literal_eval 语义
            out.append('"')
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
