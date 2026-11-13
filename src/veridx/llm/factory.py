"""Provider 工厂：`auto` 模式下没有 key 就退化为 mock（保证永远能跑通）。

`VERIDX_LLM_MODE`：
* `real`  —— 必须真 key，没有就报错（CI 里显式要求真实调用时用）
* `mock`  —— 只用脚本 provider
* `auto`  —— 有 key 用真、没 key 用 mock（本地默认）
"""

from __future__ import annotations

from typing import Any

from veridx import config
from veridx.llm.base import LLMProvider
from veridx.llm.openai_compat import OpenAICompatProvider
from veridx.llm.scenario import ScenarioMockProvider


class ProviderConfigError(RuntimeError):
    pass


def _settings_for(role: str) -> tuple[str, str, str, str]:
    """返回 ``(mode, model, base_url, api_key)``；`role` 决定用哪一组（审查 🟡F1）。"""
    if role == "second":
        return (
            config.SECOND_MODE or config.LLM_MODE,
            config.SECOND_MODEL,
            config.SECOND_BASE_URL,
            config.SECOND_API_KEY,
        )
    return config.LLM_MODE, config.MODEL, config.BASE_URL, config.API_KEY


def resolve_mode(*, role: str = "primary", mode: str | None = None) -> str:
    """把 `auto` 落地成 `real` / `mock`，并让 `REQUIRE_REAL_LLM` 能 fail-closed。

    `auto` 静默退化为 mock 会让"评测跑的是真模型"变成一句无法验证的声明，
    所以给出这道硬开关：设了 `VERIDX_REQUIRE_REAL_LLM=1` 就必须真调。
    """
    configured, model, _base_url, api_key = _settings_for(role)
    requested = (mode or configured or "auto").lower()
    if requested not in {"auto", "real", "mock"}:
        raise ProviderConfigError(f"未知 LLM 模式：{requested}")
    if requested == "mock":
        return "mock"
    if config.has_key(api_key or "") and bool(model):
        return "real"
    if requested == "real" or config.REQUIRE_REAL_LLM:
        raise ProviderConfigError(f"{role} provider 需要 key，但配置为空（LLM_MODE={requested}）")
    return "mock"


def effective_mode(role: str = "primary") -> str:
    """健康检查用：不抛异常，拿不到真模型时报 `unavailable`。"""
    try:
        return resolve_mode(role=role)
    except ProviderConfigError:
        return "unavailable"


def build_provider(
    *,
    role: str = "primary",
    mode: str | None = None,
    responses: list[str] | None = None,
    transport: Any = None,
    blind_answer: str | None = None,
) -> LLMProvider:
    """构造 provider。`role="second"` 走第二模型（分支 B 闸 2 / judge）。

    `blind_answer` 只对 mock 生效：离线 smoke 时把"盲解"脚本成答案键
    （题干不是纯算式时，脚本 provider 本来就算不出来）。
    """
    resolved_mode = resolve_mode(role=role, mode=mode)
    _mode, model, base_url, api_key = _settings_for(role)

    if resolved_mode == "mock":
        # `mock` 模式下用**按提示词分派**的 provider：整条流水线能真跑通（离线演示/CI）
        return ScenarioMockProvider(
            name=f"scenario-mock:{role}", model=f"mock-{role}", blind_answer=blind_answer
        )

    return OpenAICompatProvider(
        model=model,
        base_url=base_url,
        api_key=api_key,
        name=f"openai_compat:{role}",
        # 每个 role 有自己的温度（`TEMPERATURE` / `TEMPERATURE_KIMI`）
        temperature=config.DEFAULT_TEMPERATURE if role == "primary" else config.SECOND_TEMPERATURE,
        transport=transport,
    )


__all__ = ["ProviderConfigError", "build_provider", "effective_mode", "resolve_mode"]
