"""TD4（保留缺陷）的可测量化：`collision_prob` 的口径 + 一条 strict xfail 特征测试。

背景（设计文档 §3.1.6）：`_assert_no_leak` 用子串匹配判断"学生答案有没有被拼进 prompt"，
而 prompt 里有一段 16 位随机十六进制防伪码。答案若全 hex 且很短（`7` / `35`），
就有概率正好落在防伪码里 → 抛 `ValueError` → 整题 `failed/internal_error`。

**本文件不修它**（已裁定保留到 P4），只做两件事：
1. 把"多大概率"算成数（`collision_prob`）；
2. 用一条 `xfail(strict=True)` 把它钉住 —— 修好那天它会 XPASS → 变红 → 强制更新记录。
"""

from __future__ import annotations

import json

import pytest

from veridx.flow.blind_solve import collision_prob
from veridx.flow.runner import Runner, RunSpec
from veridx.llm.mock import MockProvider
from veridx.prompts import blind_solve as blind_prompt

STEM = "计算 2+3 的值"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("7", 0.644), ("5", 0.644), ("35", 0.057), ("-3", 0.0), ("e-1", 0.0)],
)
def test_collision_prob_self_check_values(answer: str, expected: float) -> None:
    """实施单 §1 给的自检值：容差 0.005。"""
    assert collision_prob(answer, STEM) == pytest.approx(expected, abs=0.005)


def test_collision_prob_early_exits() -> None:
    """三种早退：空答案 / 答案本来就在题干里 / 比防伪码还长。"""
    assert collision_prob("", STEM) == 0.0
    assert collision_prob("2+3", STEM) == 0.0, "答案在题干里 → `_assert_no_leak` 本来就早退"
    assert collision_prob("0123456789abcdef0", STEM) == 0.0, "L>16 不可能成为 nonce 的子串"


def test_collision_prob_matches_simulated_rate() -> None:
    """口径必须和"真的随机 nonce 撞一次"的实测频率一致（误差 0.03 以内）。

    这一步是防止公式写对了、方向却反了（比如把 17-L 写成 L）。
    """
    import secrets

    answer = "35"
    trials = 2000
    hits = sum(1 for _ in range(trials) if answer in secrets.token_hex(8))
    rate = hits / trials
    assert abs(rate - collision_prob(answer, STEM)) < 0.03, (
        f"实测 {rate:.3f} vs 预测 {collision_prob(answer, STEM):.3f}"
    )


@pytest.mark.xfail(strict=True, reason="TD4：反泄漏断言对短 hex 答案假阳性（设计文档 §3.1.6）")
async def test_td4_short_hex_answer_should_not_kill_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 nonce 固定成含 `7` 的串，答案 `7` → 现在**必然** `failed/internal_error`。

    断言写的是**期望行为**（应该还能诊断完）：今天它失败 → xfail；
    修好 TD4 那天它会 XPASS → strict 让套件变红 → 逼着更新这条记录。
    """
    monkeypatch.setattr(blind_prompt, "new_nonce", lambda: "0123456789abcdef")

    blind = json.dumps({"answer": "5", "steps": ["相加"], "confidence": 0.9, "note": ""}, ensure_ascii=False)
    diagnose = json.dumps(
        {
            "error_type": "计算表达错误",
            "root_cause": "个位加法进位错",
            "confidence": 0.9,
            "claims": [{"field": "error_type", "value": "计算表达错误", "evidence_ids": ["ev_1"]}],
        },
        ensure_ascii=False,
    )
    steps = json.dumps(
        {
            "steps": [
                {
                    "title": f"第 {i + 1} 步",
                    "content": "先算个位再算十位",
                    "key_point": "进位",
                    "targets_error": "root_cause：个位加法进位错",
                    "verifiable": False,
                    "verifiable_expr": None,
                }
                for i in range(3)
            ]
        },
        ensure_ascii=False,
    )
    primary = MockProvider(responses=[blind, diagnose, steps], name="p", model="p")
    runner = Runner(primary=primary, second=None, run_id="r-td4")
    result = await runner.run(
        RunSpec(stem=STEM, student_answer="7", answer_key="5", attempt_id="a-1", question_id="q-1")
    )

    # 期望行为：防泄漏断言不该把"答案恰好是 hex 字符"当成泄漏
    assert result.terminal.value == "diagnosed", f"实际 {result.terminal.value}/{result.reason_code}"


def test_collision_prob_is_zero_for_the_answers_used_in_the_rest_of_the_suite() -> None:
    """回归护栏：其余用例用的答案都不该有误伤概率。

    反例也写进来：`"20"` 看起来像普通数字，其实是**全 hex**（2 与 0 都在 `0-9a-f` 里），
    所以它有 5.7% 的误伤概率 —— 这正是 TD4 难被直觉发现的原因。
    """
    for answer in ("e-1", "5/7", "1/4", "-3"):
        assert collision_prob(answer, "求 ∫_0^1 x^2 dx 的值。") == 0.0, answer
    assert collision_prob("20", "求 2^10 的值。") == pytest.approx(0.057, abs=0.005)
