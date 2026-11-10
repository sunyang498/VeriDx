"""脚本化 Provider：CI 与本地无 key 时的一等公民。

两个用途：

* **测试**：给定一串回复，按顺序返回；`fail_times` 可以让前 N 次抛可重试错误，
  用来验证退避与"重试是独立 trace 事件"。
* **离线开发**：`LLM_MODE=mock` 时整条链路仍能跑通（不依赖网络）。

它必须和真 provider **同构**：同样 async、同样 `describe()`、同样记账
（`usage` 累加），否则"用 mock 跑通"不能证明"真 provider 也能跑通"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from veridx.llm.base import ChatResult, ProviderError, Usage


@dataclass
class MockProvider:
    responses: list[str] = field(default_factory=list)
    name: str = "mock"
    model: str = "mock-model"
    fail_times: int = 0
    #: 每次调用记录收到什么（断言"盲解看不到学生答案"这类性质时用）
    calls: list[dict[str, Any]] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)
    delay_seconds: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "mode": "mock",
            "responses_left": len(self.responses),
        }

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        seed: int | None = None,
        response_format: str | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "temperature": temperature,
                "seed": seed,
                "response_format": response_format,
            }
        )
        if self.delay_seconds:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        if self.fail_times > 0:
            self.fail_times -= 1
            raise ProviderError("mock transient failure", retryable=True, status_code=503)

        text = self.responses.pop(0) if self.responses else "{}"
        usage = Usage(input_tokens=len(str(messages)) // 4, output_tokens=max(len(text) // 4, 1))
        self.total_usage = self.total_usage + usage
        return ChatResult(
            text=text,
            usage=usage,
            meta={"provider": self.name, "model": self.model, "mode": "mock"},
        )


__all__ = ["MockProvider"]
