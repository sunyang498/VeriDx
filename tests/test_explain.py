"""P1.7 验收：分步讲解（`flow/explain.py`）。

三条 DoD：
1. steps ≥ 3 且每步有 `targets_error`（schema 强制 + 代码侧再校验）；
2. 每步能选"可验证 / 教学性"，可验证的必须由程序复核通过；
3. 讲解缺失时**降级为只给诊断**（不报错、不抛异常）。
"""

from __future__ import annotations

import json
from collections.abc import Callable

from veridx.flow.explain import MIN_STEPS, explain
from veridx.llm.mock import MockProvider


def _step(
    title: str,
    *,
    targets: str = "把上限抄成 2",
    verifiable: bool = False,
    expr: str | None = None,
) -> dict:
    return {
        "title": title,
        "content": f"{title} 的具体做法",
        "key_point": "关键点",
        "targets_error": targets,
        "verifiable": verifiable,
        "verifiable_expr": expr,
    }


def _payload(steps: list[dict]) -> str:
    return json.dumps({"steps": steps}, ensure_ascii=False)


def _provider(*responses: str) -> MockProvider:
    return MockProvider(responses=list(responses), name="mock-primary", model="mock-primary")


async def _run(
    provider: MockProvider,
    *,
    verify: Callable[[str], bool | None] | None = None,
):  # noqa: ANN202
    return await explain(
        provider,
        stem="求 ∫_0^1 x^2 dx",
        student_answer="1/4",
        error_type="计算表达错误",
        error_subtype=None,
        root_cause="把指数的分母当成了 2",
        verify=verify,
    )


# --- ① 正常路径 -----------------------------------------------------------
async def test_three_steps_each_target_an_error() -> None:
    provider = _provider(_payload([_step("第一步"), _step("第二步"), _step("第三步")]))
    result = await _run(provider)
    assert result.degraded is False
    assert len(result.steps) >= MIN_STEPS
    assert all(step.targets_error for step in result.steps), "每步都必须针对一个具体错误动作"
    assert [step.seq for step in result.steps] == [1, 2, 3]
    assert result.teaching_only == result.steps, "未声明可验证 → 全部是教学性表述"


async def test_verifiable_step_must_pass_program_check() -> None:
    provider = _provider(
        _payload(
            [
                _step("第一步", verifiable=True, expr="1/3"),
                _step("第二步"),
                _step("第三步"),
            ]
        )
    )
    checked: list[str] = []

    def verify(expr: str) -> bool | None:
        checked.append(expr)
        return True

    result = await _run(provider, verify=verify)
    assert checked == ["1/3"], "只在 verifiable=True 且有表达式时才复核"
    assert [step.verifiable for step in result.steps] == [True, False, False]


async def test_equality_expression_is_verified_with_compare() -> None:
    """`1/2+1/3 == 5/6` 必须用 `compare` 判等，而不是丢给求值器。

    把它直接丢给沙箱会得到 `unsupported_node:Compare` → 所有"可验证"步骤
    都会被静默降级成"教学性表述"（真实演示里出现过，属于假降级）。
    """
    from veridx.flow.runner import Runner

    verified = Runner(primary=MockProvider(), run_id="r-verify")._verify_step("1/2+1/3 == 5/6")
    assert verified is True

    wrong = Runner(primary=MockProvider(), run_id="r-verify")._verify_step("1/2+1/3 == 5/7")
    assert wrong is None, "不成立的等式不能算'可验证'"


async def test_failed_verification_degrades_to_teaching_step() -> None:
    """程序说"判不了/不成立"时，这一步只能当教学性表述 —— 不许冒充事实。"""
    provider = _provider(
        _payload(
            [
                _step("第一步", verifiable=True, expr="x^2"),
                _step("第二步"),
                _step("第三步"),
            ]
        )
    )
    result = await _run(provider, verify=lambda expr: None)
    assert result.steps[0].verifiable is False
    assert result.degraded is False


async def test_verifiable_without_expr_is_teaching_only() -> None:
    provider = _provider(
        _payload(
            [
                _step("第一步", verifiable=True, expr=None),
                _step("第二步"),
                _step("第三步"),
            ]
        )
    )
    result = await _run(provider, verify=lambda expr: True)
    assert result.steps[0].verifiable is False


# --- ② 降级（不报错）------------------------------------------------------
async def test_two_steps_is_invalid_then_degrades() -> None:
    provider = _provider(_payload([_step("只有两步"), _step("第二步")]) * 3)
    result = await _run(provider)
    assert result.degraded is True
    assert result.reason == "explain_failed"
    assert result.steps == []


async def test_garbage_output_degrades_without_raising() -> None:
    provider = _provider("这不是 JSON", "还不是", "依然不是")
    result = await _run(provider)
    assert result.degraded is True
    assert result.steps == []


async def test_whitespace_targets_error_is_dropped_then_degrades() -> None:
    """`targets_error="   "` 能过 schema（长度 3），必须由代码侧再拦一次。"""
    provider = _provider(
        _payload(
            [
                _step("第一步", targets="   "),
                _step("第二步"),
                _step("第三步"),
            ]
        )
    )
    result = await _run(provider)
    assert result.degraded is True, "去掉无效步骤后不足 3 步 → 降级"
    assert result.reason == "explain_too_few_steps"
    assert len(result.steps) == 2


async def test_empty_targets_error_fails_schema_and_degrades() -> None:
    """空字符串在 schema 层就不合法 → 回喂重试耗尽 → 降级（不抛）。"""
    provider = _provider(_payload([_step("第一步", targets=""), _step("第二步"), _step("第三步")]) * 3)
    result = await _run(provider)
    assert result.degraded is True
    assert result.reason == "explain_failed"


async def test_degradation_is_traced() -> None:
    from veridx.trace.recorder import TraceRecorder

    recorder = TraceRecorder(conn=None, run_id="run-1")
    provider = _provider("坏", "坏", "坏")
    await explain(
        provider,
        stem="x",
        student_answer="1",
        error_type="其他",
        error_subtype=None,
        root_cause="x",
        tracer=recorder,
    )
    events = recorder.events()
    assert events, "降级也必须落 trace（不许静默返回）"
    assert events[-1].detail.get("degraded") is True
