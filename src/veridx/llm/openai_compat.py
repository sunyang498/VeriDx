"""OpenAI 兼容 Provider（GLM / DeepSeek / Qwen / Ollama 同一协议）。

## 三个必须做对的地方

1. **用 `httpx.AsyncClient`**：同步 SDK 的请求无法被 `Task.cancel()` 中断。
2. **重试只为可重试错误**（5xx / 429 / 超时）；4xx 不重试 —— 参数错重试 100 次也还是错。
3. **异常消息必须脱敏**：`httpx` 可能把完整 URL（含 `?key=`）拼进消息，直接落 trace 就是事故。

## 时间预算

* 单次请求 `SANDBOX` 之外的独立预算：`LLM_REQUEST_TIMEOUT_SECONDS`（默认 30s）
* 退避 `LLM_BACKOFF_BASE_SECONDS × 2^n`，最多 `LLM_MAX_ATTEMPTS` 次
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from veridx import config
from veridx.llm.base import ChatResult, ProviderError, Usage
from veridx.trace.secrets import safe_exception


class OpenAICompatProvider:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        name: str | None = None,
        temperature: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("API key 为空：拒绝构造 provider（避免把空 key 发出去）")
        self.name = name or "openai_compat"
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._temperature = config.DEFAULT_TEMPERATURE if temperature is None else temperature
        self._transport = transport
        #: 记账（评测成本口径）
        self.total_usage = Usage()
        self.last_attempts = 0

    # ---- 元信息（绝不返回密钥）----
    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "base_url_host": _host_of(self._base_url),
            "mode": "real",
        }

    async def aclose(self) -> None:
        pass

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        response_format: str | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: ProviderError | None = None
        self.last_attempts = 0
        for attempt in range(config.LLM_MAX_ATTEMPTS):
            self.last_attempts = attempt + 1
            try:
                async with httpx.AsyncClient(
                    timeout=config.LLM_REQUEST_TIMEOUT_SECONDS, transport=self._transport
                ) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions", json=payload, headers=headers
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = ProviderError(f"transport_error: {safe_exception(exc)}", retryable=True)
            else:
                if response.status_code < 400:
                    return self._parse(response)
                retryable = response.status_code >= 500 or response.status_code == 429
                last_error = ProviderError(
                    f"http_{response.status_code}: {safe_exception(Exception(response.text[:200]))}",
                    retryable=retryable,
                    status_code=response.status_code,
                )
                if not retryable:
                    raise last_error

            if attempt < config.LLM_MAX_ATTEMPTS - 1:
                await asyncio.sleep(config.LLM_BACKOFF_BASE_SECONDS * (2**attempt))

        assert last_error is not None
        raise last_error

    def _parse(self, response: httpx.Response) -> ChatResult:
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"invalid_json_from_provider: {safe_exception(exc)}", retryable=False
            ) from exc

        choices = body.get("choices") or []
        message = (choices[0] or {}).get("message") if choices else {}
        text = (message or {}).get("content") or ""
        raw_usage = body.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("prompt_tokens") or 0),
            output_tokens=int(raw_usage.get("completion_tokens") or 0),
        )
        self.total_usage = self.total_usage + usage
        return ChatResult(
            text=text,
            usage=usage,
            meta={
                "provider": self.name,
                "model": self.model,
                "attempts": self.last_attempts,
                "response_id": body.get("id"),
            },
        )


def _host_of(url: str) -> str:
    try:
        return httpx.URL(url).host
    except Exception:  # noqa: BLE001
        return "invalid"


__all__ = ["OpenAICompatProvider"]
