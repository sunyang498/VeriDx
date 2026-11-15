"""输入隔离层：把不可信的自由文本变成"有出处、可核对"的结构化输入（方案 §1 ~ §4）。

* `schema`    —— `ExtractDraft`（严格 schema）/ `CanonicalProblem`（唯一输入形态 + 判据来源校验）
* `span`      —— 出处核验（机械比对，无模型）
* `prompt`    —— L0 围栏内的提取提示词
* `extractor` —— 隔离调用（无工具、单轮、温度 0、成本封顶、失败 fail closed）
"""

from __future__ import annotations

from veridx.extract.extractor import ExtractOutcome, run_extract
from veridx.extract.schema import CanonicalProblem, ExtractDraft, FieldSource, UntrustedJudgedInput
from veridx.extract.span import locate, resolve_span, verify_span, ws_normalize

__all__ = [
    "CanonicalProblem",
    "ExtractDraft",
    "ExtractOutcome",
    "FieldSource",
    "UntrustedJudgedInput",
    "locate",
    "resolve_span",
    "run_extract",
    "verify_span",
    "ws_normalize",
]
