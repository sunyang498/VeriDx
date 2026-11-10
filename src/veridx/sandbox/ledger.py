"""沙箱台账：**拒绝也要落痕**（§4.9 的 L3 与 L1/L2 的分界线）。

落哪些？**所有非 `ok` 的判定**都落（`unsupported` / `rejected` / `timeout` / `error`）：

* M17（沙箱拒绝率）**只统计 `CAPABILITY_REASONS`**（能力型），不按 verdict 数；
* 但"合法数学被误杀"要看 `unsupported` 的分布，"预算不足"要看 `timeout` 的分布；
* 台账带 `verdict` 列，统计时按需过滤，比事后补一层日志便宜。

不落 `ok`：正常求值每条都写，会把台账变成第二份 trace。

## M17 的口径（一处定义、两处引用）

```
分子 = count(capability)        ← 只认 CAPABILITY_REASONS
分母 = 求值总次数（含 ok）      ← 来自 trace（P1.4），台账给不出
```

台账只能提供**原因分布**，给不出分母——这一点写在 `rejection_rate()` 的返回里，
避免有人拿"台账内比例"冒充 M17。
"""

from __future__ import annotations

import sqlite3

from veridx.clock import new_id, now_utc
from veridx.sandbox.verdict import (
    CAPABILITY_REASONS,
    SandboxResult,
    Verdict,
    is_capability,
    reason_prefix,
)


def record_verdict(conn: sqlite3.Connection, *, run_id: str, result: SandboxResult) -> str | None:
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


def rejection_rate(conn: sqlite3.Connection, *, run_id: str | None = None) -> dict[str, object]:
    """台账视角的统计。

    `capability_share` 是**台账内**能力型占非 ok 的比例 —— **不是 M17**
    （M17 的分母是求值总次数，含 ok，只有 trace 才知道）。
    """
    where = " WHERE run_id = ?" if run_id else ""
    params: tuple[object, ...] = (run_id,) if run_id else ()

    counts = {
        str(row["verdict"]): int(row["n"])
        for row in conn.execute(
            f"SELECT verdict, COUNT(*) AS n FROM sandbox_rejects{where} GROUP BY verdict", params
        )
    }
    by_verdict: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        f"SELECT verdict, reason, COUNT(*) AS n FROM sandbox_rejects{where}"
        " GROUP BY verdict, reason ORDER BY n DESC",
        params,
    ):
        by_verdict.setdefault(str(row["verdict"]), {})[str(row["reason"])] = int(row["n"])

    total = sum(counts.values())
    capability = sum(
        n for reasons in by_verdict.values() for reason, n in reasons.items() if is_capability(reason)
    )
    return {
        "total_non_ok": total,
        "counts": counts,
        "capability_count": capability,
        "capability_share": (capability / total) if total else 0.0,
        "reasons_by_verdict": by_verdict,
        "capability_reasons": sorted(CAPABILITY_REASONS),
        "note": "M17 的分母是求值总次数（含 ok），来自 trace；本函数只给台账视角",
    }


def capability_reasons_seen(conn: sqlite3.Connection, *, run_id: str | None = None) -> set[str]:
    """台账里出现过的能力型原因前缀（用于与 `CAPABILITY_REASONS` 交叉校验）。"""
    where = " WHERE run_id = ?" if run_id else ""
    params: tuple[object, ...] = (run_id,) if run_id else ()
    seen: set[str] = set()
    for row in conn.execute(f"SELECT reason FROM sandbox_rejects{where}", params):
        prefix = reason_prefix(str(row["reason"]))
        if prefix in CAPABILITY_REASONS:
            seen.add(prefix)
    return seen
