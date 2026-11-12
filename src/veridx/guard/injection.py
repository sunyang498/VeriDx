"""L1 输入侧检测：**只打标，不拦截**（审查 必14）。

## 为什么不拦截

拦截等于"按意图判据"：应试题干里写"忽略""假设""```json"极其正常，
关键词过滤会在真实数据上大面积误杀（推高 M19），而攻击者换一个同义词就绕过去了。
**先只观测，拿到真实命中分布，再决定要不要拦。**

## 它挡不住什么（写清楚，避免又是"文档里的防线"）

* 挡不住「计算斐波那契数列第 1000000 项」——这句话里**没有任何注入模式**，
  它就是一道正常题目。那类攻击靠的是静态位数闸 + 预算闸（§6.5），不是这条。
"""

from __future__ import annotations

import re

__all__ = ["PATTERNS", "scan_untrusted"]

#: 命中即打标。**这些都是"看起来像指令"的形状，不是"必然是攻击"的判据。**
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_instructions", re.compile(r"忽略(以上|上面|之前|前面|上述)[^。\n]{0,12}(指令|要求|规则|设置)")),
    ("system_prompt", re.compile(r"system\s*prompt|系统提示|你是一个", re.IGNORECASE)),
    ("role_switch", re.compile(r"(你现在是|从此以后你是|扮演|act as)\s*\w*")),
    ("code_fence", re.compile(r"```")),
    ("json_shape", re.compile(r"\{\s*\"[a-z_]+\"\s*:")),
    ("template_hole", re.compile(r"\{\{|\$\{")),
    ("fence_forge", re.compile(r"</?DATA\s+id=", re.IGNORECASE)),
    ("output_format", re.compile(r"【\s*(输出格式|要求|题干)\s*】")),
)


def scan_untrusted(text: str) -> list[str]:
    """返回命中的标签名（去重、稳定顺序）。**不改变任何 verdict 或终态。**"""
    if not isinstance(text, str) or not text:
        return []
    hits: list[str] = []
    for label, pattern in PATTERNS:
        if pattern.search(text) and label not in hits:
            hits.append(label)
    return hits
