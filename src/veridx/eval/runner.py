"""批量跑题（实施单 §3）：**不走 HTTP**，直接用 `Runner` —— 与网页同一条代码路径。

产出不是"系统好不好"，而是"卡在哪一环"。所以每一条都要能回答：
终态是什么、证据绑定到哪一档、讲解能不能用、花了多少 token、failed 时到底谁抛的异常。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veridx import config
from veridx.clock import new_id
from veridx.flow.blind_solve import collision_prob
from veridx.guard.evidence import BINDING_FAIL, BINDING_PARTIAL, BINDING_PASS
from veridx.llm import factory as llm_factory
from veridx.store import db as store_db
from veridx.trace.recorder import TraceRecorder

__all__ = [
    "SUPPORTED_ARMS",
    "EvalCase",
    "ItemResult",
    "dataset_hash",
    "load_suite",
    "run_suite",
]

#: 本次只支持完整系统臂 A；结构上预留"关掉某个机制"的对照臂（实施单 §3 扩展点）
SUPPORTED_ARMS: tuple[str, ...] = ("A",)

#: 讲解"可用"的口径（与 `flow.explain.MIN_STEPS` 保持一致，但不 import 它避免耦合）
MIN_USABLE_STEPS = 3


@dataclass(frozen=True)
class EvalCase:
    id: str
    stem: str
    correct: str
    student: str
    course: str = "calculus"
    student_steps: list[str] | None = None
    note: str = ""


@dataclass
class ItemResult:
    id: str
    k_index: int
    run_id: str
    terminal: str
    reason_code: str | None
    binding: str
    accepted: int | None
    rejected: int | None
    steps: int
    steps_usable: bool
    tokens_in: int
    tokens_out: int
    llm_calls: int
    latency_ms: int | None
    collision_prob: float
    failure_attribution: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "k_index": self.k_index,
            "run_id": self.run_id,
            "terminal": self.terminal,
            "reason_code": self.reason_code,
            "binding": self.binding,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "steps": self.steps,
            "steps_usable": self.steps_usable,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "llm_calls": self.llm_calls,
            "latency_ms": self.latency_ms,
            "collision_prob": round(self.collision_prob, 6),
            "failure_attribution": self.failure_attribution,
            "notes": self.notes,
        }


def dataset_hash(path: Path) -> str:
    """对文件**内容**做 sha256 前 16 位：同一份数据换台机器也必须得到同一个值。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_suite(path: Path) -> list[EvalCase]:
    """读 JSONL：一行一题。坏行直接报错并指出行号 —— 静默跳过会让分母悄悄变小。"""
    if not path.exists():
        raise FileNotFoundError(f"数据集不存在：{path}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("//"):
            continue
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"{path.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name} 第 {lineno} 行不是 JSON 对象")
        missing = [key for key in ("id", "stem", "correct", "student") if not raw.get(key)]
        if missing:
            raise ValueError(f"{path.name} 第 {lineno} 行缺字段：{missing}")
        case_id = str(raw["id"])
        if case_id in seen:
            raise ValueError(f"{path.name} 第 {lineno} 行 id 重复：{case_id}")
        seen.add(case_id)
        steps = raw.get("student_steps")
        cases.append(
            EvalCase(
                id=case_id,
                stem=str(raw["stem"]),
                correct=str(raw["correct"]),
                student=str(raw["student"]),
                course=str(raw.get("course") or "calculus"),
                student_steps=[str(item) for item in steps] if isinstance(steps, list) else None,
                note=str(raw.get("note") or ""),
            )
        )
    return cases


def _failure_attribution(conn: sqlite3.Connection, run_id: str) -> str | None:
    """`failed` 必须逐条归因：从 trace 里把异常类型 + 原文摘要读出来。

    不分归因就会把"我的断言打人"（TD4 的 `ValueError`）读成"模型能力差"。
    """
    row = conn.execute(
        "SELECT detail FROM trace_events WHERE run_id = ? AND reason_code = 'internal_error'"
        " ORDER BY seq DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        detail = json.loads(row["detail"] or "{}")
    except ValueError:
        return None
    error = str(detail.get("error") or "").strip()
    if not error:
        return None
    head, _, tail = error.partition(":")
    return f"{head.strip()}: {tail.strip()[:160]}" if tail else error[:160]


def _binding_of(result: Any) -> tuple[str, int | None, int | None]:
    decision = getattr(result, "decision", None)
    if decision is None:
        # 没走到 guard（escalate 在更早的节点）→ 只能是 `skipped`，不许假装 pass
        return "skipped", None, None
    accepted = len(decision.accepted)
    rejected = len(decision.rejected)
    if decision.escalate:
        return BINDING_FAIL, accepted, rejected
    if rejected == 0:
        return BINDING_PASS, accepted, rejected
    return BINDING_PARTIAL, accepted, rejected


def _steps_usable(result: Any) -> tuple[int, bool]:
    explanation = getattr(result, "explanation", None)
    steps = list(getattr(explanation, "steps", []) or [])
    usable = len(steps) >= MIN_USABLE_STEPS and all(bool(getattr(s, "targets_error", "")) for s in steps)
    return len(steps), usable


def _build_providers(mode: str | None) -> tuple[Any, Any | None, Any | None]:
    primary = llm_factory.build_provider(role="primary", mode=mode)
    second: Any | None = None
    try:
        second = llm_factory.build_provider(role="second", mode=mode)
    except llm_factory.ProviderConfigError:
        second = None
    extract: Any | None = None
    try:
        extract = llm_factory.build_provider(role="extract", mode=mode)
    except llm_factory.ProviderConfigError:
        extract = None
    return primary, second, extract


async def _run_one(
    case: EvalCase,
    *,
    conn: sqlite3.Connection,
    mode: str | None,
    explain: bool,
    k_index: int,
) -> ItemResult:
    from veridx.cli_ops import Case, ensure_rows
    from veridx.flow.runner import RunLimits, Runner, RunSpec

    question, attempt = ensure_rows(
        conn,
        Case(
            id=case.id,
            stem=case.stem,
            answer_key=case.correct,
            student=case.student,
            course=case.course,
        ),
    )
    primary, second, extract = _build_providers(mode)
    run_id = new_id("r")
    runner = Runner(
        primary=primary,
        second=second,
        extract=extract,
        conn=conn,
        limits=RunLimits(explain=explain),
        tracer=TraceRecorder(conn=conn, run_id=run_id),
        run_id=run_id,
    )
    result = await runner.run(
        RunSpec(
            stem=question["stem"],
            student_answer=attempt["student_answer"],
            answer_key=question["correct_answer"],
            attempt_id=attempt["id"],
            question_id=question["id"],
        )
    )
    binding, accepted, rejected = _binding_of(result)
    steps, usable = _steps_usable(result)
    terminal = result.terminal.value
    return ItemResult(
        id=case.id,
        k_index=k_index,
        run_id=run_id,
        terminal=terminal,
        reason_code=result.reason_code,
        binding=binding,
        accepted=accepted,
        rejected=rejected,
        steps=steps,
        steps_usable=usable,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        llm_calls=result.llm_calls,
        latency_ms=result.latency_ms if result.latency_ms > 0 else None,
        collision_prob=collision_prob(case.student, case.stem),
        failure_attribution=_failure_attribution(conn, run_id) if terminal == "failed" else None,
        notes=list(result.notes),
    )


async def run_suite(
    cases: Sequence[EvalCase],
    *,
    db_path: Path,
    arms: Sequence[str] = ("A",),
    k: int = 1,
    mode: str | None = None,
    explain: bool = True,
    limit: int | None = None,
    reuse: Sequence[ItemResult] = (),
    on_progress: Any = None,
) -> list[ItemResult]:
    """跑完 `cases × k`，返回逐题明细。**已完成过的 `(id, k_index)` 直接复用**。"""
    if not cases:
        return []
    if not set(arms) <= set(SUPPORTED_ARMS):
        raise ValueError(
            f"暂不支持的臂：{sorted(set(arms) - set(SUPPORTED_ARMS))}，当前只有 {SUPPORTED_ARMS}"
        )
    if k < 1:
        raise ValueError("k 必须 ≥ 1")
    selected = list(cases)[: limit if limit is not None else None]
    cached = {(item.id, item.k_index): item for item in reuse}

    conn, _applied = store_db.init_db(db_path)
    results: list[ItemResult] = []
    try:
        for case in selected:
            for k_index in range(k):
                hit = cached.get((case.id, k_index))
                if hit is not None:
                    results.append(hit)
                    if on_progress is not None:
                        on_progress(f"复用 {case.id}#{k_index} {hit.terminal}")
                    continue
                item = await _run_one(case, conn=conn, mode=mode, explain=explain, k_index=k_index)
                results.append(item)
                if on_progress is not None:
                    on_progress(f"{case.id}#{k_index} {item.terminal} {item.reason_code or '-'}")
    finally:
        conn.close()
    return results


def default_db_path(digest: str, k: int) -> Path:
    """`eval/tmp/<dataset_hash>-k<k>.db`：同一份数据集重跑共用 trace，便于复用与回放。"""
    return config.PROJECT_ROOT / "eval" / "tmp" / f"{digest}-k{k}.db"
