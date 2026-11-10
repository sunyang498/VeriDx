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

另有两条**终局判定**（`json.loads` 默认会静默改语义，必须显式拒绝），
它们不是"修复分类"而是错误码，因此不带 F 编号：

| 错误码 | 触发 | 报什么 |
|---|---|---|
| `non_finite` | `NaN` / `Infinity` / `-Infinity` | `non_finite:<literal>` |
| `duplicate_key` | 同一对象里重复键 | `duplicate_key:<key>` |

F5×F6 的交界（P1.3 审查 🟠A）单独成格：中文引号在结构位置与值内部同时出现时，
`_scan` 必须同时知道两种引号族；同形嵌套则边界不可判，报 `ambiguous_quotes`。
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
    #: 失败时 errors[0] 必须以它开头（终局判定的错误码）
    expect_error: str | None = None


GOOD = '{"error_type": "程序性错误", "confidence": 0.8}'

SAMPLES: tuple[JsonSample, ...] = (
    JsonSample("F1_fence", f"```json\n{GOOD}\n```", True, "strip_fence", "最常见的包裹形式"),
    JsonSample("F1_fence_bare", f"```\n{GOOD}\n```", True, "strip_fence", "无语言标记"),
    JsonSample(
        "F2_preamble", f"好的，以下是结果：\n{GOOD}\n希望有帮助。", True, "extract_outermost", "前后废话"
    ),
    JsonSample(
        "F2_apostrophe_preamble",
        'It\'s the result: {"a": 1}. Done!',
        True,
        "extract_outermost",
        "正文里的撇号不能吞掉后面的对象（审查 🟡D 的对称风险）",
    ),
    JsonSample("F3_trailing_comma", '{"error_type": "程序性错误",}', True, "strip_trailing_comma", "尾逗号"),
    JsonSample("F4_single_quote", "{'error_type': '程序性错误'}", True, "literal_eval", "Python 单引号"),
    JsonSample(
        "F4_brace_in_single_value",
        "{'a': '}'}",
        True,
        "literal_eval",
        "单引号字符串里的 } 不能提前闭合外层（审查 🟡D）",
        must_preserve="}",
    ),
    JsonSample(
        "F5_cn_quotes",
        "{“error_type”: “程序性错误”}",
        True,
        "normalize_structural_quotes",
        "结构位置中文引号",
    ),
    JsonSample(
        "F5_cn_single_quotes",
        "{‘a’: ‘b’}",
        True,
        "normalize_structural_quotes",
        "中文单引号同样是结构引号（审查 🟡D）",
    ),
    JsonSample(
        "F5_cn_nested_ambiguous",
        "{“root_cause”: “他说“好””}",
        False,
        None,
        "同形引号嵌套：边界不可判，必须报 ambiguous_quotes（审查 🟠A）",
        ambiguous=True,
        expect_error="ambiguous_quotes",
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
        "F6_cn_in_single_quoted_value",
        "{'root_cause': '他说“好”'}",
        True,
        "literal_eval",
        "F5×F6 交界：单引号包裹的值里有中文引号，仍须逐字节保留（审查 🟠A）",
        must_preserve="他说“好”",
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
        "F9_non_finite_nan",
        '{"confidence": NaN}',
        False,
        None,
        "json.loads 默认接受 NaN，必须拒（审查 🟠C）",
        expect_error="non_finite:NaN",
    ),
    JsonSample(
        "F9_non_finite_inf",
        '{"confidence": Infinity}',
        False,
        None,
        "json.loads 默认接受 Infinity，必须拒（审查 🟠C）",
        expect_error="non_finite:Infinity",
    ),
    JsonSample(
        "F9_non_finite_neg_inf",
        '{"confidence": -Infinity}',
        False,
        None,
        "-Infinity 同样不许穿透（审查 🟠C）",
        expect_error="non_finite:-Infinity",
    ),
    JsonSample(
        "F9_duplicate_key",
        '{"confidence": 0.9, "confidence": 0.1}',
        False,
        None,
        "重复键不许静默取最后一个（审查 🟠B）",
        expect_error="duplicate_key:confidence",
    ),
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
    '{"a": NaN}',
    '{"a": Infinity}',
    '{"a": 1, "a": 2}',
    "{“a“b””}",
    "{'a': '",
    "It's done",
    "{“a”: ”b“}",
)


def kinds() -> set[str]:
    return {sample.kind.split("_")[0] for sample in SAMPLES}
