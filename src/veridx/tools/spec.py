"""工具契约：白名单 + JSON Schema（给模型看的那一份）。

`description` 里必须写清"何时该用 / 何时不该用"——模型选路全靠这句话（设计文档 §2.2）。
"""

from __future__ import annotations

from typing import Any


class ToolName:
    """**白名单**：dispatch 只认这几个名字。"""

    CHECK_NUMERIC = "check_numeric"
    STEP_DIFF = "step_diff"
    KB_LOOKUP = "kb_lookup"
    ESCALATE = "escalate"


ALLOWED_TOOLS: tuple[str, ...] = (
    ToolName.CHECK_NUMERIC,
    ToolName.STEP_DIFF,
    ToolName.KB_LOOKUP,
    ToolName.ESCALATE,
)

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": ToolName.CHECK_NUMERIC,
            "description": (
                "用程序判定两个数学表达式是否相等（数值，相对误差见 config.REL_TOL）。"
                "当学生答案是数字或可计算的表达式时使用。"
                "返回 equal=true/false 或 cannot_parse（符号/复数/区间/无穷等无法比较）。"
                "cannot_parse 时**不要重试**，直接依据解题过程判断。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "标准答案或你算出的正确值，如 'e-1'"},
                    "value": {"type": "string", "description": "学生答案，如 '1'"},
                },
                "required": ["expr", "value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ToolName.STEP_DIFF,
            "description": (
                "逐步比对两套解法，定位第一个偏离的步骤。"
                "**当前为占位实现**（P1.8 才做真算法），返回 locator=model 表示无法程序定位。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_steps": {"type": "array", "items": {"type": "string"}},
                    "student_steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["model_steps", "student_steps"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ToolName.KB_LOOKUP,
            "description": "按 id 查知识点名称，用于把 kp_ids 对齐到有限知识点表。只读。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kp_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["kp_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ToolName.ESCALATE,
            "description": (
                "转人工。用于：学生答案其实是对的、没有足够证据下结论、超出能力边界。"
                "**宁可转人工也不要编造错因。** 这是终态工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason_code": {
                        "type": "string",
                        "description": "§12.1 错误码，如 insufficient_evidence",
                    },
                },
                "required": ["reason_code"],
                "additionalProperties": False,
            },
        },
    },
)

TOOL_NAMES: frozenset[str] = frozenset(ALLOWED_TOOLS)
