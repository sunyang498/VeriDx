"""归一化层：把"不可信的自由文本答案"变成"沙箱能算且经过验证的表达式"。

模块分工：

* `parser` —— 确定性重写（零随机、无网络、无沙箱），覆盖 LaTeX 与常见记法；
* `model`  —— 隔离小模型（无工具、单轮、temperature=0、输出 200 token 封顶）；
* `gate`   —— 双闸与三档结论，**唯一**决定"能不能拿去判等"的地方。
"""

from __future__ import annotations

from veridx.normalize.gate import (
    CONFLICT,
    UNCHANGED,
    UNVERIFIED,
    VERIFIED,
    NormalizeOutcome,
    both_sides_verified,
    judge_safe,
    normalize_side,
)
from veridx.normalize.parser import NORMALIZER_VERSION, ParseResult, to_expr

__all__ = [
    "CONFLICT",
    "NORMALIZER_VERSION",
    "NormalizeOutcome",
    "ParseResult",
    "UNCHANGED",
    "UNVERIFIED",
    "VERIFIED",
    "both_sides_verified",
    "judge_safe",
    "normalize_side",
    "to_expr",
]
