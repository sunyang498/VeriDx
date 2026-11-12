"""诊断提示词（版本化 + hash）。

与盲解的区别：**诊断阶段可以看到全部信息**（题干、标准答案、学生答案、程序证据）。
信息隔离已经在盲解阶段完成，这里要的是"解释为什么错"，不是"判断谁错"。

⚠️ 这里的每条 `claims[]` 都必须带 `evidence_ids`，且**不得自带 `seq`**
（`Claim` 的 `extra="forbid"` 会拒掉它——序号由服务端发号）。
"""

from __future__ import annotations

import hashlib

from veridx.prompts.fence import FENCE_INSTRUCTION, fenced, new_nonce

__all__ = ["VERSION", "prompt_hash", "render", "template"]

VERSION = "pv_diagnose_v2"

_TEMPLATE = """你是数学错题诊断专家。请**基于程序证据**给出错因诊断。

【不可信输入边界】
{instruction}

下面的题干 / 标准答案 / 学生答案 / 程序证据**全部**放在 `<<DATA id=…>>` 区块里。

【硬性要求】
1. `claims[]` 里每一条都必须引用**上面出现过的** `evidence_id`，**空引用会被拒绝**；
2. `claims[]` 的 `field` **只能是这五个之一**：`error_type` / `error_subtype` / `root_cause` /
   `first_divergent_step` / `kp_ids`；**每个字段最多一条**。
   `hypotheses` 与 `confidence` 有各自的顶层字段，**不要**塞进 `claims`；
3. 不要在 `claims[]` 里写 `seq` 字段（序号由服务端分配，写了会被整条拒绝）；
4. 必须有一条 `field = "error_type"` 的 claim —— 没有它整份诊断会被判"证据不足"；
5. 错因大类只能是：概念性错误 / 程序性错误 / 审题性错误 / 计算表达错误 / 其他；
6. 若声明「审题性错误」，`error_subtype` 必须是 misread_condition / unit_confusion / question_misread 之一；
7. 「概念性 / 程序性 / 计算表达」必须有**强程序证据**（numeric_check 或 step_diff）才成立；
   证据不足以支撑时，**如实降低 confidence**，不要编造；
8. `confidence` ∈ [0,1]，表示"这条诊断可信"的程度，不是"学生做错"的程度；
9. `kp_ids` **只放在顶层字段**，不要为它写 claim（除非证据列表里已有 `kb_lookup`）。

{fenced}

【输出格式】只输出一段 JSON：
{{"error_type": "", "error_subtype": null, "root_cause": "", "first_divergent_step": null,
  "confidence": 0.0, "hypotheses": [{{"label": "", "confidence": 0.0}}], "kp_ids": [],
  "claims": [{{"field": "error_type", "value": "", "evidence_ids": ["ev_1"]}}]}}"""


def template() -> str:
    return _TEMPLATE


def render(
    *,
    stem: str,
    answer_key: str,
    student_answer: str,
    evidence_block: str,
    nonce: str | None = None,
) -> tuple[str, str]:
    """渲染诊断提示词，返回 `(prompt, nonce)`。

    **四个不可信输入全部进同一个 nonce 区块**：题干、标准答案、学生答案，
    以及程序证据（它的 payload 里含模型自己生成的字段）。
    """
    token = nonce or new_nonce()
    untrusted = "\n\n".join(
        [
            fenced(stem.strip(), token, label="field=stem"),
            fenced(answer_key.strip(), token, label="field=answer_key"),
            fenced(student_answer.strip(), token, label="field=student_answer"),
            fenced(evidence_block.strip() or "（无可用证据）", token, label="field=evidence"),
        ]
    )
    body = _TEMPLATE.format(instruction=FENCE_INSTRUCTION, fenced=untrusted)
    return body, token


def prompt_hash(body: str | None = None) -> str:
    return hashlib.sha256((body if body is not None else _TEMPLATE).encode("utf-8")).hexdigest()[:16]
