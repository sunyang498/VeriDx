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
from veridx.llm.mock import MockProvider
from veridx.llm.openai_compat import OpenAICompatProvider


class ProviderConfigError(RuntimeError):
    pass


def build_provider(
    *,
    role: str = "primary",
    mode: str | None = None,
    responses: list[str] | None = None,
    transport: Any = None,
) -> LLMProvider:
    """构造 provider。`role="second"` 走第二模型（分支 B 闸 2 / judge）。"""
    resolved_mode = (mode or config.LLM_MODE).lower()

    model = config.MODEL if role == "primary" else config.SECOND_MODEL
    base_url = config.BASE_URL if role == "primary" else config.SECOND_BASE_URL
    api_key = config.API_KEY if role == "primary" else config.SECOND_API_KEY
    has_key = config.has_key(api_key or "") and bool(model)

    if resolved_mode == "mock":
        return MockProvider(responses=list(responses or []), model=f"mock-{role}")
    if resolved_mode == "real" and not has_key:
        raise ProviderConfigError(f"{role} provider 需要 key，但配置为空（LLM_MODE=real）")
    if not has_key:
        return MockProvider(responses=list(responses or []), model=f"mock-{role}")

    return OpenAICompatProvider(
        model=model,
        base_url=base_url,
        api_key=api_key,
        name=f"openai_compat:{role}",
        transport=transport,
    )


__all__ = ["ProviderConfigError", "build_provider"]
