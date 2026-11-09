"""FastAPI 应用装配。

`create_app(db_path)` 是唯一入口：测试用 tmp 库，线上用 `VERIDX_DB_PATH`。
启动时补齐迁移（幂等），因此 `veridx serve` 不需要先手工 `db init`。

⚠️ 拒绝裸 `:memory:`：我们用"每请求一条连接"，而 lifespan 里建的 memory 库在连接关闭后
即销毁 —— 后续请求拿到的是空库（实测 `no such table: questions`）。
宁可明确报错，也不留一个"看起来支持"的参数。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from veridx import __version__, config
from veridx.api.errors import install_error_handlers
from veridx.api.routes import router
from veridx.clock import new_id
from veridx.store import db as store_db

MEMORY_DB_ERROR = (
    "不支持 ':memory:'：本项目每请求开一条连接，内存库会随 lifespan 结束而销毁。"
    "请用临时文件（tests 用 tmp_path，本地用 runtime/veridx.db）。"
)


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

    app = FastAPI(
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

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        # 500 的定位靠它：run_id 只有在控制流跑起来之后才存在，而 500 多半在之前
        request_id = request.headers.get("X-Request-Id") or new_id("req")
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    install_error_handlers(app)
    app.include_router(router)
    return app


def openapi_spec(app: FastAPI | None = None) -> dict:
    """导出 OpenAPI。不连库：`openapi()` 不触发 lifespan。"""
    return (app or create_app()).openapi()
