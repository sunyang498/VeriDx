"""R3：复数字面量归一化（`1j` → `1*I`）。

## 为什么只做这一条

| 规则 | 现在做？ | 理由 |
|---|---|---|
| **R3** `1j → 1*I` | ✅ | 替换的是 **Python 复数字面量**，不接受任何"数字与标识符相邻"的歧义判断 |
| R4 `<数字><标识符> → 插入 *`（`3I`、`2x`） | ❌ **延后到 P1.8** | 见下 |

### R4 为什么必须延后（这是本轮最重要的一条教训）

最初的 R4 规则写的是"`e` 后必须紧跟数字才算指数"。反例：

```
1e-5   →  e 后面是 '-' 不是数字  →  被当成 "1 * e - 5"  →  值从 1e-05 变成 -2.28
```

**不报错、不崩溃、不拒绝，只是悄悄换了一个值。** 它会直接喂进 `check_numeric` 生成一条
`equal=false` 的"证据"，把一次正确的学生答案判成错误——**改值比报错可怕一百倍**。

同一族还有 `1e+5`、`.5e3`、`1_000`、`2e-3`。要覆盖它们需要 tokenizer 级规格表 + 属性测试，
这是 P1.8（与符号等价、步骤解析同批）的工作量。

**更重要的非技术理由**：R4 会**替学生改错**。学生写 `2x` 可能是隐式乘法，也可能是变量名写错；
系统自动补 `*`，就把一个可能的书写错误变成"合法表达式"——**错因诊断的输入被篡改了**。
这不是沙箱问题，是产品语义问题。
"""

from __future__ import annotations

import re

#: 只匹配"数字紧邻 j/J"——Python 复数字面量的写法
_COMPLEX_LITERAL = re.compile(r"(?<![\w.])(?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?P<j>[jJ])(?!\w)")


def normalize_complex_literal(expr: str) -> str | None:
    """把 `1j`/`2.5j`/`1e3j` 改写成 `1*I`/`2.5*I`/`1e3*I`。

    返回 `None` 表示"没有改写"（调用方据此把 `normalized` 字段留空）。
    """
    if not isinstance(expr, str) or ("j" not in expr and "J" not in expr):
        return None
    rewritten = _COMPLEX_LITERAL.sub(r"\g<num>*I", expr)
    return None if rewritten == expr else rewritten
