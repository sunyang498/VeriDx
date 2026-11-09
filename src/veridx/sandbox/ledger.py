"""沙箱台账：**拒绝也要落痕**（§4.9 的 L3 与 L1/L2 的分界线）。

落哪些？**所有非 `ok` 的判定**都落（`unsupported` / `rejected` / `timeout` / `error`），
因为：

* M17（沙箱拒绝率）只统计 `verdict='rejected'`；
* 但"合法数学被误杀"要看 `unsupported` 的分布，"预算不足"要看 `timeout` 的分布；
* 台账带 `verdict` 列，统计时按需过滤，比事后补一层日志便宜。

不落 `ok`：正常求值每条都写，会把台账变成第二份 trace。
"""

from __future__ import annotations

import sqlite3

from veridx.clock import new_id, now_utc
from veridx.sandbox.verdict import SandboxResult, Verdict


def record_verdict(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    result: SandboxResult,
) -> str | None:
    """把非 `ok` 的判定写入 `sandbox_rejects`，返回台账 id（`ok` 返回 None）。"""
    if result.verdict is Verdict.OK:
        return None
    record_id = new_id("sr")
    conn.execute(
        "INSERT INTO sandbox_rejects (id, run_id, verdict, reason, expr_digest, expr_len, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            record_id,
            run_id,
            result.verdict.value,
            result.reason,
            result.expr_digest,
            result.expr_len,
            now_utc(),
        ),
    )
    return record_id


def rejection_rate(conn: sqlite3.Connection, *, run_id: str | None = None) -> dict[str, float | int]:
    """M17 的原料：`rejected / (ok + unsupported + rejected)`。

    注意分母是**求值总数**，而台账里没有 `ok` 行 —— 所以这里只能统计台账内部的比例，
    真正的 M17 需要运行侧同时计数（P1.4 的 trace 会提供）。这里给出的是**原因分布**。
    """
    where = " WHERE run_id = ?" if run_id else ""
    params: tuple[object, ...] = (run_id,) if run_id else ()
    rows = conn.execute(
        f"SELECT verdict, COUNT(*) AS n FROM sandbox_rejects{where} GROUP BY verdict", params
    ).fetchall()
    counts = {str(r["verdict"]): int(r["n"]) for r in rows}
    total = sum(counts.values())
    reasons = conn.execute(
        f"SELECT reason, COUNT(*) AS n FROM sandbox_rejects{where} GROUP BY reason ORDER BY n DESC",
        params,
    ).fetchall()
    return {
        "total": total,
        "counts": counts,
        "shared_rejected": (counts.get("rejected", 0) / total) if total else 0.0,
        "reasons": {str(r["reason"]): int(r["n"]) for r in reasons},
    }
