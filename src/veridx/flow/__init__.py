"""控制流：5 节点受控管线 + 终态选择 + 步数/调用预算。

不写死 prompt 字符串、不解析 JSON（设计文档 §2.2 / §2.3）。
提示词一律来自 `veridx.prompts.*`（版本化 + hash）。

P1.5 已实现：`blind_solve`（信息集隔离 + sympy 判等 + 分支裁决 + 双闸）。
管线编排（parse → diagnose → verify → explain → submit）在 P1.6。
"""

from __future__ import annotations

from veridx.flow.blind_solve import (
    BlindDraft,
    BlindSolveOutcome,
    Branch,
    Suspect,
    blind_solve,
    build_messages,
)

__all__ = [
    "BlindDraft",
    "BlindSolveOutcome",
    "Branch",
    "Suspect",
    "blind_solve",
    "build_messages",
]
