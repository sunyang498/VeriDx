"""攻防样本集（设计文档 §4.8）。

* **攻击集 20 条 / 6 类** → 要求逃逸率 = 0
* **回归集 25 条**（20 正常 + 5 合法不支持）→ 要求误杀率 = 0

> `"("*200 + "1" + ")"*200` 在 §4.8.1 里同时属于"资源耗尽"和"超长/超深"，
> 这里只保留一条，另外补两条不同的超长形态，保证 20 条互不重复。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackSample:
    category: str
    expr: str


def _long_chain(pairs: int = 500_000) -> str:
    """`1+1+1+…`，长度 2*pairs+1（> 256 → 触发长度上限，不进入解析）。"""
    return "1" + "+1" * pairs


ATTACK_SAMPLES: tuple[AttackSample, ...] = (
    # 1 代码执行（4）
    AttackSample("code_exec", "__import__('os').getcwd()"),
    AttackSample("code_exec", 'eval("1+1")'),
    AttackSample("code_exec", 'exec("x=1")'),
    AttackSample("code_exec", "getattr(__import__('os'),'system')('dir')"),
    # 2 文件系统（3）
    AttackSample("filesystem", "open('models.py').read()"),
    AttackSample("filesystem", "__import__('pathlib').Path('.').iterdir()"),
    AttackSample("filesystem", "open('/etc/passwd').read()"),
    # 3 属性逃逸（3）
    AttackSample("attribute", "(1).__class__.__bases__"),
    AttackSample("attribute", "().__class__.__mro__"),
    AttackSample("attribute", "(1).__class__.__subclasses__()"),
    # 4 内省（3）
    AttackSample("introspection", "globals()"),
    AttackSample("introspection", "locals()"),
    AttackSample("introspection", "getattr(__builtins__,'eval')"),
    # 5 资源耗尽（4）
    AttackSample("resource", "9**9**9"),
    AttackSample("resource", "2**10**8"),
    AttackSample("resource", "factorial(10**7)"),
    AttackSample("resource", "(" * 200 + "1" + ")" * 200),
    # 6 超长/超深（3）
    AttackSample("length", "+".join(["1"] * 150)),  # 299 字符
    AttackSample("length", _long_chain()),  # 1,000,001 字符
    AttackSample("length", "1**1**1**1"),  # 多层幂（深度 2）
)

#: 20 条正常表达式（必须 `ok` 且能算出数）
REGRESSION_OK: tuple[str, ...] = (
    "e-1",
    "1/6",
    "0.96",
    "-6",
    "-1/3",
    "sqrt(2)",
    "log(2)",
    "1e-3",
    "2**10",
    "e**2",
    "sin(pi/2)",
    "cos(0)",
    "tan(0)",
    "exp(0)",
    "ln(e)",
    "Abs(-3)",
    "floor(2.7)",
    "ceil(2.1)",
    "(1+2)*3",
    "3.14",
)

#: 5 条"合法数学但不支持"（必须 `unsupported`，**不能**是 rejected）
REGRESSION_UNSUPPORTED: tuple[str, ...] = (
    "2<x<3",
    "-inf<x<inf",
    "x<2 or x>3",
    "[0,1]",
    "x∈(0,1)",
)

#: 容差判等用例（§4.8.2 的配对样本）
TOLERANCE_PAIRS: tuple[tuple[str, str, bool], ...] = (
    ("1/3", "0.333333", True),
    ("1/6", "0.166", False),
    ("sqrt(2)", "1.4142135", True),
)


def attack_categories() -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in ATTACK_SAMPLES:
        counts[sample.category] = counts.get(sample.category, 0) + 1
    return counts
