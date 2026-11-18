"""judge 唯一的 prompt（T2 错因归类）。

纪律（方案 v1.1 §5）：**只给结论与标准答案，不给推理过程**。
被 agent 的解释说服是最隐蔽的污染源，所以 trace / notes / 置信度一律不进 prompt。
"""

from __future__ import annotations

import hashlib
import json

__all__ = ["CLASSES", "PROMPT_VERSION", "SYSTEM", "build_user", "prompt_hash"]

PROMPT_VERSION = "judge-t2-v1"

#: 六个错因类目（已拍板；**不加「粗心」**——不可操作，且会吸走所有难判样本）
CLASSES: tuple[str, ...] = (
    "概念混淆",
    "运算错误",
    "漏步或多步",
    "读题或条件误用",
    "单位与量纲",
    "无法归类",
)

SYSTEM = """你是数学错题诊断的**判分员**。你只做一件事：把「被测系统给出的错因主张」归到 6 个类目之一。

类目定义（只能选其一，写全名）：
1. 概念混淆 —— 用错定义 / 定理 / 公式
2. 运算错误 —— 思路对，算错
3. 漏步或多步 —— 少做或多做了一步
4. 读题或条件误用 —— 看错条件、用错已知量
5. 单位与量纲 —— 换算或单位错误
6. 无法归类 —— 主张太笼统、无法判断，或不属于以上任何一类

要求：
- 判断的是**错因本身**，不是措辞。用词相近但错因不同，必须判不同类；
- 不要因为主张写得详细就倾向判对，也不要因为它短就判「无法归类」；
- 老师的原话只是参考背景，**不要照抄老师的措辞来迎合**；
- 只输出 JSON，且只有两个键。

输出格式：
{"class": "<上面 6 个类目之一，一字不改>", "reason": "<20 字以内>"}"""


def build_user(*, stem: str, student: str, teacher: str, claim: str) -> str:
    return "\n".join(
        [
            f"【题目】{stem}",
            f"【学生答案】{student}",
            f"【老师的原话描述】{teacher or '（无）'}",
            f"【被测系统的错因主张】{claim}",
            "",
            "请把「被测系统的错因主张」归到 6 个类目之一，只输出 JSON。",
        ]
    )


def prompt_hash() -> str:
    payload = SYSTEM + json.dumps(list(CLASSES), ensure_ascii=False) + PROMPT_VERSION
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
