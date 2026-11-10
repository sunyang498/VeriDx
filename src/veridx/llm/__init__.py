"""Provider 抽象：全 async，`describe()` 必进 trace。"""

from __future__ import annotations

from veridx.llm.base import ChatResult, LLMProvider, ProviderError, ToolCall, Usage
from veridx.llm.factory import ProviderConfigError, build_provider
from veridx.llm.mock import MockProvider
from veridx.llm.openai_compat import OpenAICompatProvider

__all__ = [
    "ChatResult",
    "LLMProvider",
    "MockProvider",
    "OpenAICompatProvider",
    "ProviderConfigError",
    "ProviderError",
    "ToolCall",
    "Usage",
    "build_provider",
]
