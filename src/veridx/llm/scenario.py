"""按提示词类型作答的脚本 provider：**离线也能跑完整演示**。

`MockProvider` 是按顺序吐固定字符串——适合单测"第 2 次成功"这类场景，但跑不动
五节点流水线（盲解/诊断/讲解三次调用需要**三种不同结构**的回复）。

本模块按**提示词特征**分派：它不知道自己在流程的哪一步，只看用户消息里有没有
"独立求解 / 错因诊断 / 分步补救讲解"。于是：

* `LLM_MODE=mock` 时整条链路真跑通（不需要任何 key、不发一个网络请求）；
* CI 里跑端到端不需要录制真实响应；
* 诊断回复会**引用证据块里真实出现的 `ev_N`**——这让 E2 在离线状态下也是有内容的检查，
  而不是"随便编一个 id 然后被拒"。

⚠️ 它**不模拟模型的智力**：错因、讲解都是模板文本。它的用途是验证**机制**（流程、
证据绑定、终态、SSE），不是验证诊断质量。任何质量指标都必须在真模型上跑。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from veridx.llm.base import ChatResult, Usage

__all__ = ["ScenarioMockProvider"]

_EVIDENCE_LINE = re.compile(r"-\s+(ev_\d+)\s+kind=(\w+)\s+status=(\w+)")
_STEM_BLOCK = re.compile(r"<<DATA id=\w+ field=stem>>\s*\n(.+?)\n<</DATA", re.DOTALL)
_LEGACY_STEM = re.compile(r"【题干】\s*\n(.+)")
_RAW_BLOCK = re.compile(r"<<DATA id=\w+ field=raw_input>>\s*\n(.+?)\n<</DATA", re.DOTALL)
_ANY_BLOCK = re.compile(r"<<DATA id=\w+ field=\w+>>\s*\n(.+?)\n<</DATA", re.DOTALL)

#: 离线演示用的中文数字表（只为让 mock 能跑通"四分之三"，不是识别器）
_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_CN_FRACTION = re.compile(r"([零一二两三四五六七八九十]+)分之([零一二两三四五六七八九十]+)")


@dataclass
class ScenarioMockProvider:
    name: str = "scenario-mock"
    model: str = "scenario-mock"
    #: 脚本化的"盲解答案"。`None` = 尽量从题干里算出算术子串（见 `_extract_expected`）；
    #: 显式给值只用于离线 smoke（`veridx run --mock`）——题干不是纯算式时，
    #: 让脚本 provider 扮演"会做题的模型"，而不是假装算出了积分。
    blind_answer: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "mode": "mock"}

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        response_format: str | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "response_format": response_format,
                "max_tokens": max_tokens,
            }
        )
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        text = self._reply(prompt)
        usage = Usage(input_tokens=max(len(prompt) // 4, 1), output_tokens=max(len(text) // 4, 1))
        self.total_usage = self.total_usage + usage
        return ChatResult(
            text=text,
            usage=usage,
            meta={"provider": self.name, "model": self.model, "mode": "mock"},
        )

    def _reply(self, prompt: str) -> str:
        if "题目结构提取器" in prompt:
            return _extract_reply(prompt)
        if "数学表达式转写器" in prompt:
            return _normalize_reply(prompt)
        if "独立求解" in prompt:
            return json.dumps(
                {
                    "answer": self.blind_answer or _extract_expected(prompt),
                    "steps": ["先化简", "再代入"],
                    "confidence": 0.9,
                    "note": "",
                },
                ensure_ascii=False,
            )
        if "分步补救讲解" in prompt:
            return json.dumps(
                {
                    "steps": [
                        {
                            "title": "回到定义",
                            "content": "把这一步的公式重新写一遍，确认指数与分母的位置。",
                            "key_point": "幂函数积分指数加一",
                            "targets_error": "root_cause：把 x^2 的指数当成分母",
                            "verifiable": False,
                            "verifiable_expr": None,
                        },
                        {
                            "title": "代入上下限",
                            "content": "先求原函数，再代上限减下限，别漏掉下限那一项。",
                            "key_point": "牛顿-莱布尼茨公式",
                            "targets_error": "root_cause：代入时只算了上限",
                            "verifiable": False,
                            "verifiable_expr": None,
                        },
                        {
                            "title": "自查量级",
                            "content": "积分结果应与被积函数在区间上的量级相当，用估值快速自查。",
                            "key_point": "量级自查",
                            "targets_error": "error_type：算完不做合理性检查",
                            "verifiable": False,
                            "verifiable_expr": None,
                        },
                    ]
                },
                ensure_ascii=False,
            )
        return self._diagnose(prompt)

    def _diagnose(self, prompt: str) -> str:
        ok_ids = [
            match.group(1)
            for match in _EVIDENCE_LINE.finditer(prompt)
            if match.group(3) == "ok" and match.group(2) in {"numeric_check", "step_diff"}
        ]
        evidence_ids = ok_ids[:1] or ["ev_1"]
        return json.dumps(
            {
                "error_type": "计算表达错误",
                "error_subtype": None,
                "root_cause": "被积函数的指数处理错了：把 x^2 当成需要除的因子。",
                "first_divergent_step": None,
                "confidence": 0.72,
                "hypotheses": [{"label": "幂函数积分公式记错", "confidence": 0.6}],
                "kp_ids": [],
                "claims": [
                    {
                        "field": "error_type",
                        "value": "计算表达错误",
                        "evidence_ids": evidence_ids,
                    },
                    {"field": "root_cause", "value": "指数处理错误", "evidence_ids": evidence_ids},
                ],
            },
            ensure_ascii=False,
        )


_STEM_BLOCK = re.compile(r"<<DATA id=\w+ field=stem>>\s*\n(.+?)\n<</DATA", re.DOTALL)
_LEGACY_STEM = re.compile(r"【题干】\s*\n(.+)")


def _cn_value(text: str) -> int | None:
    if not text:
        return None
    if text == "十":
        return 10
    if "十" in text:  # 十一 / 二十 / 二十三
        head, _, tail = text.partition("十")
        tens = _CN_DIGITS.get(head, 1) if head else 1
        ones = _CN_DIGITS.get(tail, 0) if tail else 0
        return tens * 10 + ones
    if len(text) == 1:
        return _CN_DIGITS.get(text)
    return None


def _extract_reply(prompt: str) -> str:
    """脚本提取器：按"学生答案："把原文切成题干与作答。

    v1.1：**不返回偏移**（模型不该数字符），只返回逐字片段。
    """
    match = _RAW_BLOCK.search(prompt)
    raw = match.group(1) if match else ""
    stem, student = raw, None
    for marker in ("学生答案：", "学生答：", "学生答案:", "答案：", "Student answer:"):
        index = raw.find(marker)
        if index >= 0:
            stem = raw[:index]
            student = raw[index + len(marker) :]
            break
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in raw)
    return json.dumps(
        {
            "stem": stem.strip(),
            "student_answer": student.strip() if student is not None and student.strip() else None,
            "detected_lang": "zh" if has_cjk else "en",
            "layout_hint": "with_answer_line" if student is not None else "single",
        },
        ensure_ascii=False,
    )


def _normalize_reply(prompt: str) -> str:
    """脚本归一化器：只处理"X分之Y"这一种中文分数，其它一律 `needs_human`。"""
    match = _RAW_BLOCK.search(prompt) or _ANY_BLOCK.search(prompt)
    raw = (match.group(1) if match else "").strip()
    fraction = _CN_FRACTION.search(raw)
    if fraction:
        denominator = _cn_value(fraction.group(1))
        numerator = _cn_value(fraction.group(2))
        if denominator and numerator is not None:
            return json.dumps({"expr": f"(({numerator})/({denominator}))", "needs_human": False})
    return json.dumps({"expr": "0", "needs_human": True})


def _extract_expected(prompt: str) -> str:
    """从题干里**算**出答案：脚本 provider 也能扮演"会做题的模型"。

    只认纯算术子串（`12*3`、`1/2+1/3`），交给沙箱求值；失败就退回 `1/3`。
    这样离线演示里 `numeric_check` 能真的判等成功，而不是永远"判不了"。
    """
    match = _STEM_BLOCK.search(prompt) or _LEGACY_STEM.search(prompt)
    stem = match.group(1) if match else ""
    for candidate in re.findall(r"[-+*/^().\d\s]{3,}", stem):
        expr = candidate.strip()
        if not re.search(r"[-+*/^]", expr):
            continue
        value = _safe_eval(expr)
        if value is not None:
            return value
    return "1/3"


def _safe_eval(expr: str) -> str | None:
    from veridx.sandbox import Verdict, evaluate

    result = evaluate(expr)
    if result.verdict is Verdict.OK and result.value is not None:
        return repr(float(result.value))
    return None
