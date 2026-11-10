"""坏 JSON 分类表 F1–F12 的样本集（先写表，再写样本）。

**每格至少一条**，每条带 `why`。参考 §8.2：

| K | 类型 | 应有结果 |
|---|---|---|
| F1 | ```` ```json ```` 围栏 | 剥壳成功 |
| F2 | 前后废话 | 截取最外层 |
| F3 | 尾逗号 | 接受 |
| F4 | 单引号 | 接受 |
| F5 | 中文引号（结构位置） | 接受 |
| F6 | 中文引号（**值内部**） | 接受且**逐字节不改值** |
| F7 | 截断 | 结构化错误，不许猜补 |
| F8 | 双重编码 | 解一层再解 |
| F9 | 类型错（schema） | 由上层校验负责，解析层应成功 |
| F10 | 多个 JSON 并存 | 取第一个 + `ambiguous=True` |
| F11 | 空 / 纯空白 / 非字符串 | 结构化错误 |
| F12 | 超长 | 结构化错误，**不解析** |
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JsonSample:
    kind: str
    raw: object
    parse_ok: bool
    expect_transform: str | None
    why: str
    #: 解析成功后必须逐字节保留的字符串值（F6 的守门人）
    must_preserve: str | None = None
    ambiguous: bool = False


GOOD = '{"error_type": "程序性错误", "confidence": 0.8}'

SAMPLES: tuple[JsonSample, ...] = (
    JsonSample("F1_fence", f"```json\n{GOOD}\n```", True, "strip_fence", "最常见的包裹形式"),
    JsonSample("F1_fence_bare", f"```\n{GOOD}\n```", True, "strip_fence", "无语言标记"),
    JsonSample(
        "F2_preamble", f"好的，以下是结果：\n{GOOD}\n希望有帮助。", True, "extract_outermost", "前后废话"
    ),
    JsonSample("F3_trailing_comma", '{"error_type": "程序性错误",}', True, "strip_trailing_comma", "尾逗号"),
    JsonSample("F4_single_quote", "{'error_type': '程序性错误'}", True, "literal_eval", "Python 单引号"),
    JsonSample(
        "F5_cn_quotes",
        "{“error_type”: “程序性错误”}",
        True,
        "normalize_structural_quotes",
        "结构位置中文引号",
    ),
    JsonSample(
        "F6_cn_quotes_in_value",
        '{"root_cause": "他说“这步符号写反了”"}',
        True,
        None,
        "值内部的中文引号是内容，**绝不能替换**",
        must_preserve="他说“这步符号写反了”",
    ),
    JsonSample(
        "F6_brace_in_value",
        '{"root_cause": "集合 {x|x>0} 与 } 混排"}',
        True,
        None,
        "值内部的 } 不能干扰外层截取",
        must_preserve="集合 {x|x>0} 与 } 混排",
    ),
    JsonSample("F7_truncated", '{"error_type": "程序性错误", "confidence":', False, None, "截断：不许猜补"),
    JsonSample(
        "F8_double_encoded", '"{\\"error_type\\": \\"程序性错误\\"}"', True, "double_decode", "双重编码"
    ),
    JsonSample("F9_wrong_type", '{"confidence": "高"}', True, None, "解析应成功，类型错由 schema 层报"),
    JsonSample(
        "F10_two_objects",
        f'{GOOD}\n{{"error_type": "其他"}}',
        True,
        "extract_outermost",
        "多个 JSON：取第一个且标 ambiguous，不许合并",
        ambiguous=True,
    ),
    JsonSample("F11_empty", "", False, None, "空串"),
    JsonSample("F11_blank", "   \n\t ", False, None, "纯空白"),
    JsonSample("F11_non_string", None, False, None, "非字符串"),
    JsonSample("F11_not_object", "123", False, None, "不是对象/数组"),
    JsonSample("F12_too_long", '{"a": "' + "x" * 9000 + '"}', False, None, "超长：不解析"),
)

#: 参数化用的"任意坏输入"（断言永不抛）
NEVER_RAISE_INPUTS: tuple[object, ...] = (
    None,
    123,
    b"{}",
    [],
    {},
    "",
    "   ",
    "{",
    "}",
    "[]",
    '"unterminated',
    "{{{{{{",
    "```json\n```",
    '{"a":}',
    "\x00",
    "😀",
    "{" * 200,
    '{"a": 1} trailing',
    '"just a string"',
    "true",
    "null",
)


def kinds() -> set[str]:
    return {sample.kind.split("_")[0] for sample in SAMPLES}
