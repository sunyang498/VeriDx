"""分步讲解提示词（版本化 + hash）。

讲解是**最后一个** LLM 节点，只负责"怎么补救"，不参与"错没错"的裁决
（判据在 `guard.evidence`，不由讲解改变）。

三条硬要求（对应 P1.7 验收）：
1. 至少 3 步；
2. **每步都要有 `targets_error`**（这一步针对哪个错误动作），否则这一步不算有效讲解；
3. 每步标注 `verifiable`：`true` 表示该步结论可以被程序验算（附带 `verifiable_expr`），
   `false` 表示是**教学性表述**——教学性表述不进证据链，也不得被当作事实。
"""

from __future__ import annotations

import hashlib

__all__ = ["VERSION", "prompt_hash", "render", "template"]

VERSION = "pv_explain_v1"

_TEMPLATE = """你是数学老师。针对下面这次错误，给出**分步补救讲解**。

【题干】
{stem}

【学生答案】
{student_answer}

【已确认的错因】
大类：{error_type}
子类：{error_subtype}
根因：{root_cause}

【要求】
1. 至少 3 步，每步一句话标题 + 一段正文；
2. 每步必须写 `targets_error`：这一步**针对学生的哪个错误动作**（不要写"认真审题"这类空话）；
3. 若这一步的结论可以被程序验算（例如"3/6+2/6 等于 5/6"），填 `verifiable: true` 并给
   `verifiable_expr`：写成 `左边 == 右边`（如 `1/2+1/3 == 5/6`）或一个纯数值表达式，
   **不要写自然语言、不要写单位**。系统会用沙箱复核；
   否则填 `verifiable: false` —— 它是**教学性表述**，不会被当成事实；
4. 不要重复诊断结论，只讲"接下来怎么改"。

【输出格式】只输出一段 JSON：
{{"steps": [{{"title": "", "content": "", "key_point": "", "targets_error": "",
  "verifiable": false, "verifiable_expr": null}}]}}"""


def template() -> str:
    return _TEMPLATE


def render(
    *,
    stem: str,
    student_answer: str,
    error_type: str,
    error_subtype: str | None,
    root_cause: str,
) -> str:
    return _TEMPLATE.format(
        stem=stem.strip(),
        student_answer=student_answer.strip(),
        error_type=error_type,
        error_subtype=error_subtype or "（未细分）",
        root_cause=root_cause,
    )


def prompt_hash(body: str | None = None) -> str:
    return hashlib.sha256((body if body is not None else _TEMPLATE).encode("utf-8")).hexdigest()[:16]
