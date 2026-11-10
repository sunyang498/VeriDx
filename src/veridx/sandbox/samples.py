"""攻防与边界样本集（内容 v3.2：87 条 / 6 组 / K1–K21）。

> ⚠️ **这是"已知攻击与已知边界的清单"，不是安全性证明。**
> "逃逸 0"只说明"这些条挡住了"。真正的安全性论证是**分类完备性**：
> 任何不属于算术的语法都进不了执行层，任何进得去的都在预算内。
> 对应可执行断言见 `tests/test_invariants.py` 的 I4。

## 分组

| 组 | 条数 | 覆盖 | 判据 |
|---|---|---|---|
| A 攻击 | 20 | K12–K17 | 逃逸 = 0（全部 `rejected` 且 reason ∈ `CAPABILITY_REASONS`） |
| B 正常算术 | 30 | K1/K2 | 误杀 = 0（全部 `ok`） |
| C 合法不支持 | 15 | K3/K6–K10 | 归类正确，且**不计入 M17** |
| D 值域边界 | 10 | K3/K4/K5 | 绝不判 `ok` |
| E 输入形态 | 8 | K18 | 永不抛异常 |
| F 运行时故障 | 4 | K19–K21 | `error`/`timeout`，不进 rejected |

每条样本都带 `why`：测试失败时直接打印"这条在证明什么"，避免"加了一条不知道测什么"。
"""

from __future__ import annotations

from dataclasses import dataclass

from veridx.sandbox.verdict import Verdict


@dataclass(frozen=True)
class Sample:
    expr: object
    expect: Verdict
    reason_prefix: str
    why: str
    k: str  # 分类表编号（K1–K21）


#: 制造"超硬上限"（>4096）的表达式
_LONG_CHAIN = "1" + "+1" * 500_000  # 1,000,001 字符
_PAREN_BOMB = "(" * 2500 + "1" + ")" * 2500  # 5001 字符（> 硬上限 4096）


# ---------------------------------------------------------------------------
# A 攻击 · 20 条（全部具备能力：能逃逸，或能烧资源）
# ---------------------------------------------------------------------------
ATTACKS: tuple[Sample, ...] = (
    # K12 超硬上限（2）
    Sample(_LONG_CHAIN, Verdict.REJECTED, "expr_too_long", "1e6 字符运算链：解析前就拦，DoS 面", "K12"),
    Sample(_PAREN_BOMB, Verdict.REJECTED, "expr_too_long", "5001 字符括号炸弹：超硬上限 4096", "K12"),
    # K13 节点爆炸（1）——255 字符但 >200 个 AST 节点，长度不是判据，节点数才是
    Sample(
        "1" + "+1" * 127,
        Verdict.REJECTED,
        "too_many_nodes",
        "255 字符 / 383 节点：证明长度与节点数是两个独立上限",
        "K13",
    ),
    # K14 幂塔与指数（4）
    Sample("9**9**9", Verdict.REJECTED, "nested_pow", "指数塔：三个字面量都不大，只有嵌套能挡", "K14"),
    Sample("2**10**8", Verdict.REJECTED, "nested_pow", "同上，另一种写法", "K14"),
    Sample("2**201", Verdict.REJECTED, "exponent_too_large", "单层但指数 201 > 200", "K14"),
    Sample("1**1**1**1", Verdict.REJECTED, "nested_pow", "值无害但结构是塔：判据是结构不是值", "K14"),
    # K15 逃逸结构（5）
    Sample("(1).__class__", Verdict.REJECTED, "forbidden_syntax", "属性访问：逃逸主通道", "K15"),
    Sample("x[0]", Verdict.REJECTED, "forbidden_syntax", "下标访问", "K15"),
    Sample("[i for i in range(3)]", Verdict.REJECTED, "forbidden_syntax", "推导式", "K15"),
    Sample("lambda: 1", Verdict.REJECTED, "forbidden_syntax", "lambda", "K15"),
    Sample("x if x else y", Verdict.REJECTED, "forbidden_syntax", "条件表达式（非算术）", "K15"),
    # K16 非白名单调用（6）
    Sample("open('a')", Verdict.REJECTED, "forbidden_call", "文件系统", "K16"),
    Sample("eval('1')", Verdict.REJECTED, "forbidden_call", "代码执行", "K16"),
    Sample("exec('x=1')", Verdict.REJECTED, "forbidden_call", "代码执行", "K16"),
    Sample("globals()", Verdict.REJECTED, "forbidden_call", "内省", "K16"),
    Sample("getattr(x,'y')", Verdict.REJECTED, "forbidden_call", "内省 + 属性", "K16"),
    Sample("factorial(3)", Verdict.REJECTED, "forbidden_call", "不在白名单：既是 DoS 面又非必需", "K16"),
    # K17 dunder（2）
    Sample("__builtins__", Verdict.REJECTED, "forbidden_name", "内省入口", "K17"),
    Sample("__class__", Verdict.REJECTED, "forbidden_name", "属性链起点", "K17"),
)

# ---------------------------------------------------------------------------
# B 正常算术 · 30 条（误杀 = 0）
# ---------------------------------------------------------------------------
NORMAL: tuple[Sample, ...] = tuple(
    Sample(expr, Verdict.OK, "", why, "K1")
    for expr, why in (
        ("e-1", "自然常数减法：题目最常见形态"),
        ("1/6", "分数"),
        ("0.96", "小数"),
        ("-6", "负整数"),
        ("-1/3", "负分数"),
        ("sqrt(2)", "无理数"),
        ("log(2)", "对数"),
        ("1e-3", "科学计数法（负指数）"),
        ("1e+5", "科学计数法（正指数）"),
        (".5e3", "小数前导点的科学计数法"),
        ("2**10", "小指数幂"),
        ("e**2", "常数幂"),
        ("sin(pi/2)", "三角函数"),
        ("cos(0)", "三角函数"),
        ("tan(0)", "三角函数"),
        ("exp(0)", "指数函数"),
        ("ln(e)", "对数与常数互逆"),
        ("log(100, 10)", "两参数对数"),
        ("Abs(-3)", "绝对值"),
        ("floor(2.7)", "向下取整"),
        ("ceil(2.1)", "向上取整"),
        ("(1+2)*3", "括号"),
        ("3.14", "小数"),
        ("37%5", "% 取模（曾误杀）"),
        ("7//2", "// 整除（曾误杀）"),
        ("-7//2", "负数整除：Python 向下取整语义"),
        ("I**2", "复数运算但结果落到实轴 → -1"),
        ("exp(I*pi)", "欧拉公式 → -1"),
        ("Abs(3+4*I)", "复数模 → 5"),
        ("Abs(3+4j)", "同上，Python 字面量写法（R3 归一化）"),
    )
)

# ---------------------------------------------------------------------------
# C 合法但不支持 · 15 条（归类正确，**不计入 M17**）
# ---------------------------------------------------------------------------
UNSUPPORTED: tuple[Sample, ...] = (
    Sample("2<x<3", Verdict.UNSUPPORTED, "unsupported_node", "链式比较：合法数学", "K7"),
    Sample("-inf<x<inf", Verdict.UNSUPPORTED, "unsupported_node", "区间", "K7"),
    Sample("x<2 or x>3", Verdict.UNSUPPORTED, "unsupported_node", "布尔组合", "K7"),
    Sample("[0,1]", Verdict.UNSUPPORTED, "unsupported_node", "区间字面量", "K7"),
    Sample("x∈(0,1)", Verdict.UNSUPPORTED, "parse_error", "非 Python 语法：解析即失败，不会执行", "K10"),
    Sample("1+", Verdict.UNSUPPORTED, "parse_error", "残缺表达式", "K10"),
    Sample("x+1", Verdict.UNSUPPORTED, "symbolic", "含自由符号：数值层不做符号等价", "K6"),
    Sample("x%2", Verdict.UNSUPPORTED, "symbolic", "% 对符号：能力边界不是攻击", "K6"),
    Sample("sin(x)", Verdict.UNSUPPORTED, "symbolic", "符号三角", "K6"),
    Sample("foo+1", Verdict.UNSUPPORTED, "unknown_symbol", "拼写错误 ≠ 攻击", "K8"),
    Sample("sqrt+1", Verdict.UNSUPPORTED, "function_as_value", "把函数当数用：写错，没有能力", "K11"),
    Sample("y=1", Verdict.UNSUPPORTED, "parse_error", "赋值不是表达式", "K10"),
    Sample(
        "+".join(["x"] * 179),
        Verdict.UNSUPPORTED,
        "oversize",
        "357 字符合法多项式：超容量但不是攻击",
        "K9",
    ),
    Sample(
        "+".join(["1"] * 127) + "+",
        Verdict.UNSUPPORTED,
        "parse_error",
        "255 字符残式：先过长度再解析失败",
        "K10",
    ),
    Sample("2+3I", Verdict.UNSUPPORTED, "parse_error", "隐式乘法：R4 延后，保持 parse_error 而不改值", "K10"),
)

# ---------------------------------------------------------------------------
# D 值域边界 · 10 条（绝不判 ok）
# ---------------------------------------------------------------------------
DOMAIN: tuple[Sample, ...] = (
    Sample("1j", Verdict.UNSUPPORTED, "complex_result", "Python 复数字面量（归一化为 1*I）", "K3"),
    Sample("3*I", Verdict.UNSUPPORTED, "complex_result", "纯虚数", "K3"),
    Sample("I", Verdict.UNSUPPORTED, "complex_result", "虚数单位", "K3"),
    Sample("sqrt(-1)", Verdict.UNSUPPORTED, "complex_result", "负开方", "K3"),
    Sample("2**1e9", Verdict.UNSUPPORTED, "non_finite", "溢出成 inf：不能算 ok", "K4"),
    Sample("1e400", Verdict.UNSUPPORTED, "non_finite", "浮点字面量直接溢出", "K4"),
    Sample("exp(1000)", Verdict.UNSUPPORTED, "non_finite", "指数溢出", "K4"),
    Sample("inf", Verdict.UNSUPPORTED, "non_finite", "inf 合法化后统一为值域问题", "K4"),
    Sample("0/0", Verdict.UNSUPPORTED, "non_finite", "nan：比 inf 更危险（会静默判不相等）", "K4"),
    Sample("1/0", Verdict.UNSUPPORTED, "undefined", "zoo：数学未定义", "K5"),
)

# ---------------------------------------------------------------------------
# E 输入形态 · 8 条（永不抛）
# ---------------------------------------------------------------------------
INPUT_FORMS: tuple[Sample, ...] = (
    Sample(None, Verdict.UNSUPPORTED, "not_a_string", "None：调用方类型 bug", "K18"),
    Sample(123, Verdict.UNSUPPORTED, "not_a_string", "int", "K18"),
    Sample(b"1+1", Verdict.UNSUPPORTED, "not_a_string", "bytes", "K18"),
    Sample({}, Verdict.UNSUPPORTED, "not_a_string", "dict", "K18"),
    Sample([], Verdict.UNSUPPORTED, "not_a_string", "list", "K18"),
    Sample("", Verdict.UNSUPPORTED, "empty_expr", "空串", "K18"),
    Sample("    ", Verdict.UNSUPPORTED, "empty_expr", "纯空白", "K18"),
    Sample("😀", Verdict.UNSUPPORTED, "parse_error", "emoji：非 Python 语法", "K18"),
)

#: 容差判等配对（§4.8.2）
TOLERANCE_PAIRS: tuple[tuple[str, str, bool], ...] = (
    ("1/3", "0.333333", True),
    ("1/6", "0.166", False),
    ("sqrt(2)", "1.4142135", True),
)

#: 非判等配对：`equal` 必须是 `None`（"判不了"），**不能**是 `False`
NON_COMPARABLE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("I", "0", "复数不参与数值判等"),
    ("2**1e9", "1e308", "溢出：历史上会误判 equal=True"),
    ("0/0", "5", "nan：历史上会误判 equal=False（把'判不了'当'不相等'）"),
    ("x+1", "2", "符号表达式"),
    ("2<x<3", "2", "区间"),
)


# ---------------------------------------------------------------------------
# F 运行时故障 · 4 条（需要注入，走单独测试；计入 87 条总数）
# ---------------------------------------------------------------------------
RUNTIME_FAULTS: tuple[Sample, ...] = (
    Sample("1+1", Verdict.TIMEOUT, "eval_timeout", "计算预算压到 0 秒（确定性超时）", "K21"),
    Sample("1+1", Verdict.ERROR, "spawn_failed", "无 __main__ 守卫 / spawn 失败", "K19"),
    Sample("1+1", Verdict.ERROR, "eval_error", "子进程返回 ok=false", "K20"),
    Sample("1+1", Verdict.ERROR, "internal", "最外层兜底捕获漏网异常", "K20"),
)

ALL_SAMPLES: tuple[Sample, ...] = ATTACKS + NORMAL + UNSUPPORTED + DOMAIN + INPUT_FORMS


def group_counts() -> dict[str, int]:
    return {
        "A_attacks": len(ATTACKS),
        "B_normal": len(NORMAL),
        "C_unsupported": len(UNSUPPORTED),
        "D_domain": len(DOMAIN),
        "E_input_forms": len(INPUT_FORMS),
        "F_runtime_faults": len(RUNTIME_FAULTS),
        "total": len(ALL_SAMPLES) + len(RUNTIME_FAULTS),
    }
