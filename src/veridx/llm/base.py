"""Provider 契约（NFR-8：换 Provider 只改配置，不改调用方代码）。

三条设计决定：

1. **全 async**。同步 SDK 的 HTTP 请求无法被 `asyncio.Task.cancel()` 中断 ——
   用户关掉页面后仍在烧额度（§5.3）。所以这里只暴露 `async def`。
2. **不引 `openai` SDK**，直接用 `httpx.AsyncClient`：少一层封装，且超时/重试策略可见。
3. **`describe()` 必须进 trace**。否则"三种 provider 可切换"就是假能力 ——
   **判据：这个能力在 trace 里能被观察到吗？观察不到 = 假能力。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ChatResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()
    #: provider 的自述信息，**必须进 trace**
    meta: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Provider 层错误（可重试与不可重试由 `retryable` 区分）。"""

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def describe(self) -> dict[str, Any]:
        """返回可落 trace 的元信息（**绝不包含密钥**）。"""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        seed: int | None = None,
        response_format: str | None = None,
    ) -> ChatResult: ...


__all__ = ["ChatResult", "LLMProvider", "ProviderError", "ToolCall", "Usage"]
