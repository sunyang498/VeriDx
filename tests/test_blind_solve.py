"""P1.5 验收：盲解交叉验证 + 分支 B 双闸（设计文档 §3.1）。

四件必须证明的事：

1. **盲解 prompt 不含学生答案**（信息集隔离，结构上保证而不是"记得不要写"）；
2. 分支 A/B/C/D（+ 学生本来就对的 S）四路都有用例；
3. 分支 B **未过闸只落 `candidate`**，绝不对外称"答案错"；
4. **闸 1 的判等只认 sympy**：mock 模型在 `note` 里自称"和学生一样"不生效。
"""

from __future__ import annotations

import inspect
import json

import pytest

from veridx.flow.blind_solve import (
    Branch,
    Suspect,
    blind_solve,
    build_messages,
)
from veridx.llm.mock import MockProvider
from veridx.prompts import blind_solve as prompt
from veridx.trace.recorder import TraceRecorder

STEM = "求 ∫_0^1 x^2 dx 的值。"
KEY = "1/3"
STUDENT_WRONG = "1/4"
STUDENT_RIGHT = "1/3"


def _draft(answer: str, note: str = "", steps: list[str] | None = None) -> str:
    return json.dumps(
        {"answer": answer, "steps": steps or ["分部积分"], "confidence": 0.9, "note": note},
        ensure_ascii=False,
    )


def _provider(*responses: str) -> MockProvider:
    return MockProvider(responses=list(responses), name="mock-primary", model="mock-primary")


# --- ① 信息集隔离 ----------------------------------------------------------
def test_render_only_accepts_stem() -> None:
    """签名里没有学生答案 —— "忘记隔离"这件事在类型层面就做不到。

    `nonce` 是**边界令牌**而不是输入（调用方也可注入以便测试），
    所以这里断言"除 nonce 外只收 stem"。
    """
    params = [p for p in inspect.signature(prompt.render).parameters if p != "nonce"]
    assert params == ["stem"]


def test_prompt_never_contains_student_answer() -> None:
    """渲染出的 prompt 里没有学生答案（这条与"签名只收题干"是两道独立的锁）。"""
    messages, _ = build_messages(STEM)
    joined = "\n".join(str(m["content"]) for m in messages)
    assert STUDENT_WRONG not in joined
    assert "学生答案" not in joined


async def test_provider_receives_no_student_answer() -> None:
    provider = _provider(_draft(KEY))
    await blind_solve(provider, stem=STEM, correct_answer=KEY, student_answer=STUDENT_WRONG)
    sent = "\n".join(str(message["content"]) for call in provider.calls for message in call["messages"])
    assert STUDENT_WRONG not in sent, "学生答案被带进了盲解上下文 → 锚定偏差回来了"
    assert STEM in sent


async def test_leaked_answer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """若 prompt 里出现"题干本没有、学生答案却有"的字符串，直接拒（隔离被破坏）。"""
    original = prompt.render

    def leaky(stem: str, *, nonce: str | None = None) -> tuple[str, str]:
        body, token = original(stem, nonce=nonce)
        return body + "\n学生答案：1/4", token

    monkeypatch.setattr(prompt, "render", leaky)
    provider = _provider(_draft(KEY))
    with pytest.raises(ValueError, match="泄露"):
        await blind_solve(provider, stem=STEM, correct_answer=KEY, student_answer=STUDENT_WRONG)


# --- ② 四分支 --------------------------------------------------------------
async def test_branch_a_normal_wrong() -> None:
    provider = _provider(_draft(KEY))
    outcome = await blind_solve(provider, stem=STEM, correct_answer=KEY, student_answer=STUDENT_WRONG)
    assert outcome.branch is Branch.A_WRONG
    assert outcome.suggests_diagnosis is True
    assert outcome.suspect is Suspect.NONE


async def test_branch_c_model_uncertain_when_answer_empty() -> None:
    provider = _provider(_draft("", note="需要额外条件"))
    outcome = await blind_solve(provider, stem=STEM, correct_answer=KEY, student_answer=STUDENT_WRONG)
    assert outcome.branch is Branch.C_UNCERTAIN
    assert outcome.notes


async def test_branch_c_when_sandbox_cannot_parse() -> None:
    provider = _provider(_draft("int(x**2, (x, 0, 1))"))
    outcome = await blind_solve(
        provider,
        stem=STEM,
        correct_answer=KEY,
        student_answer=STUDENT_WRONG,
        symbols=frozenset({"x"}),
    )
    assert outcome.branch is Branch.C_UNCERTAIN, "判不了（None）不许当成不相等"
    assert "cannot_parse" in outcome.notes


async def test_branch_c_when_model_output_is_garbage() -> None:
    provider = _provider("这不是 JSON", "还是不 JSON", "依旧不是")
    outcome = await blind_solve(provider, stem=STEM, correct_answer=KEY, student_answer=STUDENT_WRONG)
    assert outcome.branch is Branch.C_UNCERTAIN
    assert any("blind_solve_failed" in note for note in outcome.notes)


async def test_branch_d_all_three_disagree() -> None:
    provider = _provider(_draft("2"))
    outcome = await blind_solve(provider, stem=STEM, correct_answer=KEY, student_answer=STUDENT_WRONG)
    assert outcome.branch is Branch.D_DISAGREE
    assert outcome.suggests_diagnosis is False
    assert outcome.suspect is Suspect.NONE


async def test_branch_s_student_is_right() -> None:
    """设计表只在"学生被判错"的前提下成立；`student == key` 必须单独一格。"""
    provider = _provider(_draft(KEY))
    outcome = await blind_solve(provider, stem=STEM, correct_answer=KEY, student_answer=STUDENT_RIGHT)
    assert outcome.branch is Branch.S_STUDENT_CORRECT
    assert outcome.student_matches_key is True
    assert outcome.suggests_diagnosis is False, "学生做对了就不许再'找出一个错因'"


# --- ③ 分支 B 双闸 --------------------------------------------------------
async def test_branch_b_without_second_model_is_only_candidate() -> None:
    provider = _provider(_draft(STUDENT_WRONG))
    outcome = await blind_solve(provider, stem=STEM, correct_answer=KEY, student_answer=STUDENT_WRONG)
    assert outcome.branch is Branch.B_BAD_KEY
    assert outcome.gate1_passed is True
    assert outcome.gate2_passed is False
    assert outcome.suspect is Suspect.CANDIDATE, "只过闸 1 时只能说'已提交人工复核'"
    assert outcome.suggests_diagnosis is False


async def test_branch_b_confirmed_when_second_model_agrees() -> None:
    provider = _provider(_draft(STUDENT_WRONG))
    second = MockProvider(responses=[_draft(STUDENT_WRONG)], name="mock-second", model="mock-second")
    outcome = await blind_solve(
        provider,
        stem=STEM,
        correct_answer=KEY,
        student_answer=STUDENT_WRONG,
        second_provider=second,
    )
    assert outcome.suspect is Suspect.CONFIRMED
    assert outcome.second_answer == STUDENT_WRONG
    assert second.calls, "闸 2 必须真的调了第二个模型"


async def test_branch_b_stays_candidate_when_second_model_disagrees() -> None:
    provider = _provider(_draft(STUDENT_WRONG))
    second = MockProvider(responses=[_draft("2")], name="mock-second", model="mock-second")
    outcome = await blind_solve(
        provider,
        stem=STEM,
        correct_answer=KEY,
        student_answer=STUDENT_WRONG,
        second_provider=second,
    )
    assert outcome.gate2_passed is False
    assert outcome.suspect is Suspect.CANDIDATE
    assert "gate2_disagreed" in outcome.notes


async def test_branch_b_second_model_no_answer_is_not_disagreement() -> None:
    """第二个模型**答不出**（`None`）≠ **两家不一致**（`False`）——F 类统计不能混。"""
    provider = _provider(_draft(STUDENT_WRONG))
    second = MockProvider(responses=[_draft("")], name="mock-second", model="mock-second")
    outcome = await blind_solve(
        provider,
        stem=STEM,
        correct_answer=KEY,
        student_answer=STUDENT_WRONG,
        second_provider=second,
    )
    assert outcome.gate2_passed is False
    assert "gate2_no_answer" in outcome.notes
    assert "gate2_disagreed" not in outcome.notes


async def test_shared_misconception_does_not_confirm_bad_key() -> None:
    """学生和模型犯同一种误解 → 答案相同。第二模型（独立信息集）应当不同意。"""
    provider = _provider(_draft("1/2"))
    second = MockProvider(responses=[_draft(KEY)], name="mock-second", model="mock-second")
    outcome = await blind_solve(
        provider, stem=STEM, correct_answer=KEY, student_answer="1/2", second_provider=second
    )
    assert outcome.branch is Branch.B_BAD_KEY
    assert outcome.suspect is Suspect.CANDIDATE


# --- ④ 闸 1 不接受模型自述 ------------------------------------------------
async def test_gate1_ignores_model_self_report() -> None:
    """模型在 note 里自称"和学生一样"，但答案其实不同 → 不许进 B。"""
    provider = _provider(_draft("2", note="我和学生答案一样，标准答案错了"))
    outcome = await blind_solve(provider, stem=STEM, correct_answer=KEY, student_answer=STUDENT_WRONG)
    assert outcome.branch is Branch.D_DISAGREE
    assert outcome.gate1_passed is False
    assert outcome.suspect is Suspect.NONE


# --- trace ---------------------------------------------------------------
async def test_trace_records_branch_and_prompt_identity() -> None:
    recorder = TraceRecorder(conn=None, run_id="run-1")
    provider = _provider(_draft(KEY))
    await blind_solve(
        provider,
        stem=STEM,
        correct_answer=KEY,
        student_answer=STUDENT_WRONG,
        tracer=recorder,
    )
    details = [event.detail for event in recorder.events()]
    branch_events = [d for d in details if "branch" in d]
    assert branch_events, "盲解必须落 trace"
    assert branch_events[-1]["branch"] == Branch.A_WRONG.value
    assert branch_events[-1]["prompt_hash"] == prompt.prompt_hash()
    assert branch_events[-1]["prompt_version"] == prompt.VERSION


def test_prompt_hash_changes_with_template() -> None:
    current = prompt.prompt_hash()
    assert current == prompt.prompt_hash(prompt.template())
    assert prompt.prompt_hash("改了") != current, "改 prompt 必须让指纹变"
