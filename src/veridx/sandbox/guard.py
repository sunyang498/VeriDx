"""第 1–3 层防御：AST 白名单 → 名字白名单 → 静态资源上限。

**只判定、不执行**。执行在常驻 worker 进程里（`pool.py` / `_worker.py`）。

三个必须显式列举、不能用父类的地方：

* **运算符不能用 `ast.operator` 当白名单**——`BitXor`（`^`）、`LShift`（`<<`）、`Pow`
  都是它的子类，用父类会把 `x^2`、`1<<10**8` 一起放行；
* **`ast.Pow` 不是 `BinOp`**，它是运算符节点；
* **危险结构必须优先于"不支持"**——`().__class__.__mro__` 同时含 `Tuple`（不支持）
  与 `Attribute`（危险），顺序反了会把一条属性逃逸洗成"合法数学"。

## 判定顺序（不可调换）

```
长度（硬上限 → rejected；容量 → unsupported:oversize）
  → 解析（失败 → unsupported:parse_error，因为解析失败什么都不会执行）
  → 节点数（→ rejected:too_many_nodes）
  → 危险结构（→ rejected:forbidden_syntax）
  → 合法但不支持的结构（→ unsupported:unsupported_node）
  → 其余非白名单节点（→ rejected:forbidden_syntax）
  → 语义（幂/字面量/名字/调用）
```
"""

from __future__ import annotations

import ast

from veridx import config
from veridx.sandbox.verdict import Reason, Verdict

#: 函数白名单。**不含 `factorial`**：既是 DoS 面（`factorial(10**7)`）又非必需。
#: 函数名**只能**出现在 `Call.func`；当值使用 → `unsupported:function_as_value`。
FUNCTION_NAMES: tuple[str, ...] = (
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "log",
    "ln",
    "exp",
    "Abs",
    "floor",
    "ceil",
)

#: 符号白名单（含常数 `e`/`E`/`pi`/`I`/`inf`/`oo`）
#: `inf` / `oo` 合法化：避免"同一个无穷三种待遇"（`-inf` rejected、`1e400` ok、`-inf<x<inf` unsupported）
SYMBOL_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "n",
    "a",
    "b",
    "c",
    "e",
    "pi",
    "E",
    "I",
    "inf",
    "oo",
)

#: 常数（覆盖符号表里的同名符号）
CONSTANT_NAMES: tuple[str, ...] = ("e", "E", "pi", "I", "inf", "oo")

#: 只由算术构成的语法节点（**逐项显式列举**，含运算符节点）
_ALLOWED_NODE_TYPES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Call,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,  # %
    ast.FloorDiv,  # //
    ast.Pow,
    ast.UAdd,
    ast.USub,
)

#: 危险结构（**优先于** unsupported 判定）
_DANGEROUS_NODE_TYPES: tuple[type, ...] = (
    ast.Attribute,
    ast.Subscript,
    ast.Lambda,
    ast.Await,
    ast.Starred,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.NamedExpr,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Yield,
    ast.YieldFrom,
    ast.IfExp,
    ast.Import,
    ast.ImportFrom,
)

#: 合法数学但不支持
_UNSUPPORTED_NODE_TYPES: tuple[type, ...] = (
    ast.Compare,
    ast.BoolOp,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
)

#: dunder 名（内省主通道）——能力型
_DUNDER_PREFIX = "__"


def _type_name(node: ast.AST) -> str:
    return type(node).__name__


def _pow_depth(node: ast.AST) -> int:
    depth = 0
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        depth += 1
        node = node.right
    return depth


def _check_pow(node: ast.BinOp) -> tuple[Verdict, str] | None:
    right = node.right
    if (
        isinstance(right, ast.Constant)
        and isinstance(right.value, int)
        and not isinstance(right.value, bool)
        and abs(right.value) > config.SANDBOX_MAX_EXPONENT
    ):
        return (Verdict.REJECTED, f"{Reason.EXPONENT_TOO_LARGE.value}:{right.value}")
    if _pow_depth(right) >= config.SANDBOX_MAX_NESTED_POW:
        return (Verdict.REJECTED, Reason.NESTED_POW.value)
    return None


def _check_semantics(
    node: ast.AST, symbols: frozenset[str], called: frozenset[int]
) -> tuple[Verdict, str] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return _check_pow(node)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, complex):
            # R3 应该在 core 里归一化掉；走到这里说明归一化没覆盖到
            return (Verdict.UNSUPPORTED, f"{Reason.COMPLEX_RESULT.value}:literal")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # 字符串/布尔字面量：不是攻击，只是"不是数"
            return (Verdict.UNSUPPORTED, f"{Reason.NON_NUMERIC.value}:constant:{type(value).__name__}")
        if isinstance(value, int) and abs(value) > config.SANDBOX_MAX_INT_LITERAL:
            return (Verdict.REJECTED, f"{Reason.INT_LITERAL_TOO_LARGE.value}:{abs(value)}")

    if isinstance(node, ast.Name):
        name = node.id
        if name.startswith(_DUNDER_PREFIX):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_NAME.value}:{name}")
        if name in symbols:
            return None
        if id(node) in called:
            # 它是某个 Call 的函数名，已在 Call 分支完成白名单校验
            return None
        if name in FUNCTION_NAMES:
            # `sqrt + 1`：把函数当数用。**不是**攻击（没有能力），只是写错
            return (Verdict.UNSUPPORTED, f"{Reason.FUNCTION_AS_VALUE.value}:{name}")
        return (Verdict.UNSUPPORTED, f"{Reason.UNKNOWN_SYMBOL.value}:{name}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_CALL.value}:{_type_name(node.func)}")
        if node.func.id not in FUNCTION_NAMES:
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_CALL.value}:{node.func.id}")
        if node.keywords:
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_CALL.value}:keyword")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_CALL.value}:starred")
        if len(node.args) > 2:  # log(x, base) 最多两个
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_CALL.value}:arity")
    return None


def static_check(expr: str, symbols: frozenset[str] | None = None) -> tuple[Verdict, str] | None:
    """返回 `(verdict, reason)` 表示"不需要执行就能判定"；`None` 表示可进入执行层。"""
    if not isinstance(expr, str):
        return (Verdict.UNSUPPORTED, f"{Reason.NOT_A_STRING.value}:{type(expr).__name__}")
    if not expr.strip():
        return (Verdict.UNSUPPORTED, Reason.EMPTY_EXPR.value)

    length = len(expr)
    if length > config.SANDBOX_HARD_EXPR_LEN:
        return (Verdict.REJECTED, f"{Reason.EXPR_TOO_LONG.value}:{length}")
    if length > config.SANDBOX_MAX_EXPR_LEN:
        # 长答案不等于攻击：这是能力边界
        return (Verdict.UNSUPPORTED, f"{Reason.OVERSIZE.value}:{length}")

    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        # 解析失败 = 什么都不会执行，因此不是攻击；`x∈(0,1)`、`1+` 走这里
        return (Verdict.UNSUPPORTED, Reason.PARSE_ERROR.value)

    nodes = list(ast.walk(tree))
    if len(nodes) > config.SANDBOX_MAX_AST_NODES:
        return (Verdict.REJECTED, f"{Reason.TOO_MANY_NODES.value}:{len(nodes)}")

    for node in nodes:
        if isinstance(node, _DANGEROUS_NODE_TYPES):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_SYNTAX.value}:{_type_name(node)}")

    for node in nodes:
        if isinstance(node, _UNSUPPORTED_NODE_TYPES):
            return (Verdict.UNSUPPORTED, f"{Reason.UNSUPPORTED_NODE.value}:{_type_name(node)}")

    for node in nodes:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_SYNTAX.value}:{_type_name(node)}")

    known = symbols if symbols is not None else frozenset(SYMBOL_NAMES)
    called = frozenset(id(c.func) for c in nodes if isinstance(c, ast.Call) and isinstance(c.func, ast.Name))
    for node in nodes:
        bad = _check_semantics(node, known, called)
        if bad is not None:
            return bad
    return None
