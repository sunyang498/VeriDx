"""盲解提示词（版本化；**只接收题干**，结构上不可能带进学生答案）。

版本号与 hash 一起入 `prompt_versions`（设计文档 §2.7 的指纹需要它）：
改一个字，`prompt_version` 与 `prompt_hash` 都会变，缓存与幂等指纹随之失效。

v2（审查 必2）：题干包进**随机 nonce** 的 `<<DATA id=…>>` 区块。
固定字面量挡不住"题干自己写闭合标签"，只有每次重生成的 nonce 才挡得住。
"""

from __future__ import annotations

import hashlib

from veridx.prompts.fence import FENCE_INSTRUCTION, fenced, new_nonce

__all__ = ["VERSION", "prompt_hash", "render", "template"]

VERSION = "pv_blind_solve_v2"

_TEMPLATE = """你是一名严谨的数学老师。请**独立求解**下面这道题。

【不可信输入边界】
{instruction}

【要求】
1. 只依据 `<<DATA…>>` 区块里的题干作答。**不要猜测学生写了什么**，也不要假设有人在这道题上犯过错；
   你没有、也不会看到任何人的作答。
2. `answer` 给出你的最终答案，尽量写成可计算的表达式（如 3/4、sqrt(2)、x=2、1/3*x^2）。
3. `steps` 给出关键步骤，字符串数组，最多 8 条，每条一句话。
4. 若答案不唯一（多解 / 需要额外条件 / 区间），把 `answer` 留空字符串，并在 `note` 里说明原因。

{fenced}

【输出格式】只输出一段 JSON，不要任何解释文字：
{{"answer": "", "steps": [], "confidence": 0.0, "note": ""}}"""


def template() -> str:
    return _TEMPLATE


def render(stem: str, *, nonce: str | None = None) -> tuple[str, str]:
    """渲染提示词，返回 `(prompt, nonce)`。

    **签名里没有学生答案**——这是信息集隔离的结构性保证。
    `nonce` 每次渲染重新生成（调用方可注入以便测试）。
    """
    token = nonce or new_nonce()
    body = _TEMPLATE.format(
        instruction=FENCE_INSTRUCTION,
        fenced=fenced(stem.strip(), token, label="field=stem"),
    )
    return body, token


def prompt_hash(body: str | None = None) -> str:
    return hashlib.sha256((body if body is not None else _TEMPLATE).encode("utf-8")).hexdigest()[:16]
