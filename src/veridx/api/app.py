"""FastAPI 应用装配。

`create_app(db_path)` 是唯一入口：测试用 tmp 库，线上用 `VERIDX_DB_PATH`。
启动时补齐迁移（幂等），因此 `veridx serve` 不需要先手工 `db init`。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from veridx import __version__, config
from veridx.api.errors import install_error_handlers
from veridx.api.routes import router
from veridx.store import db as store_db


def create_app(db_path: str | Path | None = None) -> FastAPI:
    resolved = Path(db_path or config.get_settings().db_path)

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
    app.state.db_path = resolved
    install_error_handlers(app)
    app.include_router(router)
    return app


def openapi_spec(app: FastAPI | None = None, **kwargs: Any) -> dict[str, Any]:
    return (app or create_app(**kwargs)).openapi()
