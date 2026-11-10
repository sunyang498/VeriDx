"""工具层：`check_numeric` / `step_diff` / `kb_lookup` / `escalate` + 白名单 dispatch。

**不调用 LLM 做判断**（避免递归与自证，设计文档 §2.2）：
判等由沙箱给，步骤定位由 sympy 给（P1.8），知识点只做查表。
"""

from __future__ import annotations

from veridx.tools.context import ToolContext
from veridx.tools.dispatch import dispatch
from veridx.tools.spec import ALLOWED_TOOLS, TOOL_NAMES, TOOL_SCHEMAS, ToolName

__all__ = [
    "ALLOWED_TOOLS",
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
    "ToolContext",
    "ToolName",
    "dispatch",
]
