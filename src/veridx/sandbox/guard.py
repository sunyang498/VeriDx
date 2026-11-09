"""第 1–3 层防御：AST 白名单 → 名字白名单 → 静态资源上限。

**四层防御里只有这一层是"拒绝之前"**，所以它必须只做判定、不做执行。
执行在 `_worker.py`（子进程 + 超时）。

两个必须显式列举、不能用父类的地方（§4.7 坑 1/坑 2）：

* **运算符不能用 `ast.operator` 当白名单**——`BitXor`（`^`）、`LShift`（`<<`）、
  `Pow` 都是它的子类，用父类会把 `x^2`、`1<<10**8` 一起放行；
* **`ast.Pow` 不是 `BinOp`**，它是运算符节点。判"是不是幂"要写
  `isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)`。
"""

from __future__ import annotations

import ast

from veridx import config
from veridx.sandbox.verdict import Reason, Verdict

#: 函数白名单。**不含 `factorial`**：它既是 DoS 面（`factorial(10**7)`）又非必需（§4.4 层 2）
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

#: 符号白名单。注意 `e`/`E`/`pi`/`I` 是常量，其余是自由符号
SYMBOL_NAMES: tuple[str, ...] = ("x", "y", "z", "n", "a", "b", "c", "e", "pi", "E", "I")

#: 显式列举"只由算术构成"的语法节点（含运算符节点本身，因为 `ast.walk` 会访问它们）
_ALLOWED_NODE_TYPES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Call,  # 函数调用本身合法，**调用目标**在语义层按白名单判（`forbidden_call`）
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)

#: **危险结构优先于"不支持"**：`().__class__.__mro__` 同时含 Tuple（不支持）与 Attribute（危险），
#: 必须判 rejected —— 否则攻击样本会因为"顺带写了个元组"而被洗成 unsupported
_DANGEROUS_NODE_TYPES: tuple[type, ...] = (
    ast.Attribute,  # .  属性访问（逃逸主通道）
    ast.Subscript,  # [] 下标
    ast.Lambda,
    ast.Await,
    ast.Starred,
    ast.JoinedStr,  # f-string
    ast.FormattedValue,
    ast.NamedExpr,  # :=
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

#: 合法数学但我们不支持的结构 → `unsupported`（**不是** rejected，否则误杀率失真）
_UNSUPPORTED_NODE_TYPES: tuple[type, ...] = (
    ast.Compare,  # 2<x<3
    ast.BoolOp,  # x<2 or x>3
    ast.List,  # 区间 [0,1]
    ast.Tuple,
    ast.Set,
    ast.Dict,
)


def _type_name(node: ast.AST) -> str:
    return type(node).__name__


def _pow_depth(node: ast.AST) -> int:
    """指数上还嵌着几层幂（`a**b**c` → 1）。"""
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
        return (Verdict.REJECTED, f"{Reason.EXPONENT_TOO_LARGE}:{right.value}")
    if _pow_depth(right) >= config.SANDBOX_MAX_NESTED_POW:
        # 9**9**9、2**10**8：三个字面量都不大，只有"幂嵌套"能挡住（§4.7 坑 3）
        return (Verdict.REJECTED, Reason.NESTED_POW.value)
    return None


def _check_semantics(node: ast.AST, known_names: frozenset[str]) -> tuple[Verdict, str] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return _check_pow(node)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_SYNTAX}:Constant:{type(value).__name__}")
        if isinstance(value, int) and abs(value) > config.SANDBOX_MAX_INT_LITERAL:
            return (Verdict.REJECTED, f"{Reason.INT_LITERAL_TOO_LARGE}:{abs(value)}")

    if isinstance(node, ast.Name) and node.id not in known_names:
        return (Verdict.REJECTED, f"{Reason.UNKNOWN_NAME}:{node.id}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_CALL}:{_type_name(node.func)}")
        if node.func.id not in FUNCTION_NAMES:
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_CALL}:{node.func.id}")
        if node.keywords:
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_CALL}:keyword")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_CALL}:starred")
    return None


def static_check(expr: str, symbols: frozenset[str] | None = None) -> tuple[Verdict, str] | None:
    """返回 `(verdict, reason)` 表示"不需要执行就能判定"；`None` 表示可进入第 4 层。"""
    if not isinstance(expr, str) or not expr.strip():
        return (Verdict.UNSUPPORTED, Reason.EMPTY_EXPR.value)

    if len(expr) > config.SANDBOX_MAX_EXPR_LEN:
        # 先判长度：`"("*200+"1"+")"*200` 这类样本在 parse 阶段会以各种形式失败，
        # 长度是唯一稳定、可解释的判据（§4.8.1 的"超长/超深"三类都靠它）
        return (Verdict.REJECTED, f"{Reason.EXPR_TOO_LONG}:{len(expr)}")

    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        # 解析不了 = 什么都不会执行，因此不是攻击；`x∈(0,1)`、`x<2 或 x>3` 走这里
        return (Verdict.UNSUPPORTED, Reason.PARSE_ERROR.value)

    nodes = list(ast.walk(tree))
    if len(nodes) > config.SANDBOX_MAX_AST_NODES:
        return (Verdict.REJECTED, f"{Reason.TOO_MANY_NODES}:{len(nodes)}")

    # 危险结构优先于"不支持"：`().__class__.__mro__` 同时含 Tuple 与 Attribute，必须判 rejected
    for node in nodes:
        if isinstance(node, _DANGEROUS_NODE_TYPES):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_SYNTAX}:{_type_name(node)}")

    for node in nodes:
        if isinstance(node, _UNSUPPORTED_NODE_TYPES):
            return (Verdict.UNSUPPORTED, f"{Reason.UNSUPPORTED_NODE}:{_type_name(node)}")

    for node in nodes:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            return (Verdict.REJECTED, f"{Reason.FORBIDDEN_SYNTAX}:{_type_name(node)}")

    known = frozenset(FUNCTION_NAMES) | (symbols if symbols is not None else frozenset(SYMBOL_NAMES))
    for node in nodes:
        bad = _check_semantics(node, known)
        if bad is not None:
            return bad
    return None
