"""工具分发：**只认白名单**，**永不抛异常**。

两条硬约束（P1.2 验收）：

1. `dispatch("__import__", ...)` → `unknown_tool`，**绝不执行**；
2. 任何工具异常都转成结构化 observation，不冒到上层（否则一个坏参数就是 500）。
"""

from __future__ import annotations

from typing import Any

from veridx.tools.context import ToolContext
from veridx.tools.impl import check_numeric, escalate, kb_lookup, step_diff
from veridx.tools.spec import TOOL_NAMES, TOOL_SCHEMAS, ToolName

_REGISTRY = {
    ToolName.CHECK_NUMERIC: check_numeric,
    ToolName.STEP_DIFF: step_diff,
    ToolName.KB_LOOKUP: kb_lookup,
    ToolName.ESCALATE: escalate,
}


def dispatch(name: object, args: object, ctx: ToolContext) -> dict[str, Any]:
    """执行工具。返回 dict，**不抛异常**。"""
    if not isinstance(name, str) or name not in TOOL_NAMES:
        return {
            "status": "tool_error",
            "error": "unknown_tool",
            "message": f"工具 {name!r} 不在白名单内",
            "allowed": sorted(TOOL_NAMES),
        }
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {
            "status": "tool_error",
            "error": "invalid_arguments",
            "message": f"参数必须是对象，收到 {type(args).__name__}",
        }

    handler = _REGISTRY[name]
    try:
        result = handler(args, ctx)
    except BaseException as exc:  # noqa: BLE001 - 错误也是 observation
        return {
            "status": "tool_error",
            "error": "tool_exception",
            "message": f"{type(exc).__name__}: {exc}"[:200],
            "tool": name,
        }
    if not isinstance(result, dict):
        return {
            "status": "tool_error",
            "error": "invalid_result",
            "message": f"工具返回了 {type(result).__name__}",
            "tool": name,
        }
    return result


__all__ = ["TOOL_SCHEMAS", "dispatch"]
