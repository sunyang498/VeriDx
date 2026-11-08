"""FastAPI 依赖：每请求一条 SQLite 连接。

不做连接池：SQLite 单写者 + WAL（NFR-9），每请求开连接（~1ms）比自造池更不容易错，
且天然线程安全（uvicorn 的同步端点跑在线程池里）。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request

from veridx.store import db as store_db
from veridx.store.repository import Repository


def get_repo(request: Request) -> Iterator[Repository]:
    conn = store_db.connect(request.app.state.db_path)
    try:
        yield Repository(conn)
    finally:
        conn.close()


RepoDep = Annotated[Repository, Depends(get_repo)]
