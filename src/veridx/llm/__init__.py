"""Provider 抽象：chat / chat_json / chat_tools（OpenAI 兼容，httpx 异步）。

不引 openai SDK：§5.3 要求 HTTP 可被 asyncio 真正中断。
"""

from __future__ import annotations
