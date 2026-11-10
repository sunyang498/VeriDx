"""四个工具的已知副作用清单（防止"工具层偷偷写库"）：

| 工具 | 读 | 写 |
|---|---|---|
| `check_numeric` | — | `evidence`（numeric_check）+ `sandbox_rejects`（非 ok 时） |
| `step_diff` | — | `evidence`（step_diff，占位时 status=skipped） |
| `kb_lookup` | `kp` | — |
| `escalate` | — | — （终态由控制流写 `finish_task`） |
"""

from __future__ import annotations

from typing import Any

from veridx.sandbox import Verdict, compare
from veridx.sandbox.verdict import Reason, SandboxResult
from veridx.store.repository import RepositoryError
from veridx.tools.context import ToolContext
from veridx.tools.spec import ToolName
from veridx.trace.event import ReasonCode


def _bad(message: str) -> dict[str, Any]:
    return {"status": "tool_error", "message": message}


# ---------------------------------------------------------------------------
# check_numeric
# ---------------------------------------------------------------------------
def check_numeric(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    expr = args.get("expr")
    value = args.get("value")
    if not isinstance(expr, str) or not expr.strip():
        return _bad("expr 必须是非空字符串")
    if not isinstance(value, str) or not value.strip():
        return _bad("value 必须是非空字符串")

    comparison = compare(expr, value, symbols=ctx.symbols)
    left: SandboxResult = comparison.left
    right: SandboxResult = comparison.right

    # 非 ok 的判定要落台账（M17 原料）；同时把原因回灌给模型
    ctx.record_sandbox(left)
    ctx.record_sandbox(right)

    comparable = comparison.equal is not None
    payload: dict[str, Any] = {
        "expr": expr,
        "student_answer": value,
        "equal": comparison.equal,
        "comparable": comparable,
        "left": left.to_payload(),
        "right": right.to_payload(),
    }
    if not comparable:
        payload["blocked_reason"] = comparison.reason

    evidence = ctx.record_evidence(
        kind="numeric_check",
        payload=payload,
        produced_by=ToolName.CHECK_NUMERIC,
        # 无法判定时 status=skipped：E2 只接受 ok，所以引用它的 claim 会被拒
        status="ok" if comparable else "skipped",
    )

    observation: dict[str, Any] = {
        "status": "ok" if comparable else "cannot_parse",
        "evidence_id": evidence["id"],
        "equal": comparison.equal,
    }
    if comparable:
        observation["expr_value"] = left.value
        observation["student_value"] = right.value
    else:
        observation["reason"] = _blocked_reason(left, right, comparison.reason)
        observation["hint"] = "无法数值比较（符号/复数/区间/无穷/超界）。不要重试，依据解题过程判断。"
    return observation


def _blocked_reason(left: SandboxResult, right: SandboxResult, reason: str) -> str:
    for result in (left, right):
        if result.verdict is not Verdict.OK:
            return result.reason or result.verdict.value
    return reason


# ---------------------------------------------------------------------------
# step_diff（P1.8 之前是占位）
# ---------------------------------------------------------------------------
def step_diff(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    model_steps = args.get("model_steps")
    student_steps = args.get("student_steps")
    if not isinstance(model_steps, list) or not model_steps:
        return _bad("model_steps 必须是非空数组")
    if not isinstance(student_steps, list) or not student_steps:
        return _bad("student_steps 必须是非空数组")
    if not all(isinstance(step, str) for step in (*model_steps, *student_steps)):
        return _bad("steps 的元素必须是字符串")

    payload = {
        "status": "not_implemented",
        "model_steps": len(model_steps),
        "student_steps": len(student_steps),
        "first_divergence": None,
        "locator": "model",
        "note": "P1.8 实现 sympy 逐步对齐；在此之前 first_divergent_step 只能由模型判断",
    }
    evidence = ctx.record_evidence(
        kind="step_diff",
        payload=payload,
        produced_by=ToolName.STEP_DIFF,
        status="skipped",
    )
    return {
        "status": "unsupported",
        "locator": "model",
        "evidence_id": evidence["id"],
        "reason": Reason.NON_NUMERIC.value,
        "note": payload["note"],
    }


# ---------------------------------------------------------------------------
# kb_lookup
# ---------------------------------------------------------------------------
def kb_lookup(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    kp_ids = args.get("kp_ids")
    if not isinstance(kp_ids, list) or not kp_ids:
        return _bad("kp_ids 必须是非空数组")
    if not all(isinstance(kp_id, str) and kp_id.strip() for kp_id in kp_ids):
        return _bad("kp_ids 的元素必须是非空字符串")
    if ctx.conn is None:
        return {"status": "unavailable", "note": "无数据库连接", "found": []}

    placeholders = ",".join("?" * len(kp_ids))
    try:
        rows = ctx.conn.execute(
            f"SELECT id, name, parent_id FROM kp WHERE id IN ({placeholders})", tuple(kp_ids)
        ).fetchall()
    except Exception as exc:  # noqa: BLE001 - 工具层不抛
        return _bad(f"查询知识点失败：{type(exc).__name__}")

    found = [dict(row) for row in rows]
    missing = sorted(set(kp_ids) - {item["id"] for item in found})
    return {"status": "ok", "found": found, "missing": missing}


# ---------------------------------------------------------------------------
# escalate
# ---------------------------------------------------------------------------
def escalate(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    reason_code = args.get("reason_code")
    if not isinstance(reason_code, str) or not reason_code.strip():
        return _bad("reason_code 必须是非空字符串")
    valid = {code.value for code in ReasonCode}
    if reason_code not in valid:
        # 手写漂移会让失败分类 F1–F8 统计不出来，且不会报错
        return _bad(f"reason_code 不在枚举内：{reason_code}；允许值：{sorted(valid)}")
    return {"status": "escalated", "terminal": "escalated", "reason_code": reason_code}


def unknown_kp_error(exc: Exception) -> dict[str, Any]:
    return _bad(f"{type(exc).__name__}: {exc}")


__all__ = [
    "RepositoryError",
    "check_numeric",
    "escalate",
    "kb_lookup",
    "step_diff",
    "unknown_kp_error",
]
