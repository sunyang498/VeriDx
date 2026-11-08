"""时间与标识符约定（全局唯一实现，避免各模块各写一套）。

* 时间戳：UTC ISO8601，秒级，形如 `2026-11-08T15:22:31Z`
* ID：`<前缀>-<8 hex>`；证据 id 由运行内序号决定（`ev_<seq>`），不用随机串
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

PREFIX_QUESTION = "q"
PREFIX_ATTEMPT = "a"
PREFIX_TASK = "t"
PREFIX_DIAGNOSIS = "d"
PREFIX_RUN = "r"

_ID_BYTES = 4


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(_ID_BYTES)}"


def evidence_id(seq: int) -> str:
    return f"ev_{seq}"
