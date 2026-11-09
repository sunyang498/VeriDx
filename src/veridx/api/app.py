"""FastAPI 应用装配。

`create_app(db_path)` 是唯一入口：测试用 tmp 库，线上用 `VERIDX_DB_PATH`。
启动时补齐迁移（幂等），因此 `veridx serve` 不需要先手工 `db init`。

两个与"可运维"有关的实现细节：

* **Request-ID 必须包在整条中间件栈之外**。Starlette 的 `ServerErrorMiddleware` 在所有
  用户中间件之外；未捕获异常会穿过 `@app.middleware("http")`，于是那行
  `response.headers[...]` 根本没机会执行（P0.6 审查 §3.2 实测：body 有、header 没有）。
  所以这里用 `FastAPI.__call__` 的子类，直接包住 ASGI 应用。
* **拒绝裸 `:memory:`**：每请求一条连接，lifespan 建的内存库随连接关闭即销毁，
  后续请求拿到空库（实测 `no such table: questions`）。宁可明确报错，不留假能力。
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import Message, Receive, Scope, Send

from veridx import __version__, config
from veridx.api.errors import install_error_handlers
from veridx.api.routes import router
from veridx.clock import new_id
from veridx.store import db as store_db

MEMORY_DB_ERROR = (
    "不支持 ':memory:'：本项目每请求开一条连接，内存库会随 lifespan 结束而销毁。"
    "请用临时文件（tests 用 tmp_path，本地用 runtime/veridx.db）。"
)


class VeriDxFastAPI(FastAPI):
    """把 Request-ID 放在 `ServerErrorMiddleware` 之外，保证 500 也有 `X-Request-Id`。"""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return

        request_id = Headers(scope=scope).get("x-request-id") or new_id("req")
        # Request.state 读的就是 scope["state"]，先写进去，处理器里才拿得到
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("X-Request-Id", request_id)
            await send(message)

        await super().__call__(scope, receive, send_with_request_id)


def create_app(db_path: str | Path | None = None) -> FastAPI:
    resolved = str(db_path or config.get_settings().db_path)
    if resolved == ":memory:":
        raise ValueError(MEMORY_DB_ERROR)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        conn, _ = store_db.init_db(resolved, config.MIGRATIONS_DIR)
        try:
            app.state.schema_version = store_db.current_version(conn)
        finally:
            conn.close()
        yield

    app = VeriDxFastAPI(
        title="VeriDx",
        version=__version__,
        summary="Verification-first diagnosis for student mistakes - every claim carries evidence.",
        description=(
            "错题诊断 Agent：模型只解释'为什么错'，对错由程序裁决。"
            "每个诊断字段必须绑定程序产生的可执行证据（设计文档 §3.3）。"
        ),
        lifespan=lifespan,
    )
    app.state.db_path = Path(resolved)

    install_error_handlers(app)
    app.include_router(router)
    return app


def openapi_spec(app: FastAPI | None = None) -> dict:
    """导出 OpenAPI。

    默认**不指向真实库**：`openapi()` 本身不跑 lifespan，但"看起来无副作用、实际指向生产库"
    的默认值是隐患（P0.6 审查 §3.4），所以用一个临时路径。
    """
    if app is not None:
        return app.openapi()
    return create_app(db_path=Path(tempfile.gettempdir()) / "veridx-openapi.db").openapi()
