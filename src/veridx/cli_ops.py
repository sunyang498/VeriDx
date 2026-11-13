"""`veridx run` / `veridx replay`：把**已有的** Runner 与 trace 表包成两个壳。

这两个命令存在的唯一理由是"可离线复现"：

* `run`    —— 一条命令跑完一题，打印终态 / reason / 证据表 / trace 位置；
* `replay` —— 把某次 run 的 trace 按人读顺序打出来（`--at N` 看单步完整输入）。

**不重新设计流水线，不新增事件源**：`run` 直接构造 `Runner`，`replay` 直接读
`trace_events` 表 —— 与 SSE 派生事件的那份数据是同一份。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veridx import config
from veridx.clock import new_id
from veridx.flow.runner import RunLimits, Runner, RunSpec
from veridx.trace.event import NodeStatus, ReasonCode
from veridx.trace.recorder import TraceRecorder

__all__ = ["Case", "cmd_replay", "cmd_run", "ensure_rows", "load_case"]


@dataclass(frozen=True)
class Case:
    """一题一答。`cases.json` 与数据库 `questions` 两种来源收敛成这一个形状。"""

    id: str
    stem: str
    answer_key: str
    student: str
    course: str = "calculus"


def _iter_dataset_cases() -> list[Case]:
    """扫 `datasets/**/*.json`。支持 `{"cases": [...]}`、`[...]`、单个对象三种写法。"""
    cases: list[Case] = []
    if not config.DATASETS_DIR.exists():
        return cases
    for path in sorted(config.DATASETS_DIR.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items = data.get("cases", []) if isinstance(data, dict) else data
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            stem = item.get("stem") or item.get("question")
            answer_key = item.get("correct") or item.get("correct_answer")
            student = item.get("student") or item.get("student_answer")
            if not isinstance(stem, str) or not isinstance(answer_key, str) or not isinstance(student, str):
                continue  # 不是"题+答"的样本（例如只标了 gold label）：跳过，不猜
            cases.append(
                Case(
                    id=str(item["id"]),
                    stem=stem,
                    answer_key=answer_key,
                    student=student,
                    course=str(item.get("course") or "calculus"),
                )
            )
    return cases


def load_case(case_id: str, conn: sqlite3.Connection | None = None) -> Case | None:
    """按 id 找题：**先 datasets 后数据库**（数据集是可提交的、可复现的那份）。"""
    for case in _iter_dataset_cases():
        if case.id == case_id:
            return case
    if conn is None:
        return None
    row = conn.execute("SELECT * FROM questions WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    attempt = conn.execute(
        "SELECT student_answer FROM attempts WHERE question_id = ? ORDER BY created_at DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    return Case(
        id=str(row["id"]),
        stem=str(row["stem"]),
        answer_key=str(row["correct_answer"]),
        student=str(attempt["student_answer"]) if attempt else "",
        course=str(row["course"]),
    )


def ensure_rows(conn: sqlite3.Connection, case: Case) -> tuple[dict[str, Any], dict[str, Any]]:
    """跑之前先把 question / attempt 落库：`diagnoses.attempt_id` 有外键，不能凭空写。"""
    from veridx.store.repository import Repository

    repo = Repository(conn)
    question = repo.get_question(case.id)
    if question is None:
        question = repo.create_question(
            course=case.course,
            stem=case.stem,
            correct_answer=case.answer_key,
            qtype="numeric",
            question_id=case.id,
        )
    attempt = repo.find_attempt(question["id"], case.student)
    if attempt is None:
        attempt = repo.create_attempt(question_id=question["id"], student_answer=case.student)
    return question, attempt


def _build_pair(mode: str | None, *, blind_answer: str | None = None) -> tuple[Any, Any | None]:
    from veridx.llm import factory as llm_factory

    primary = llm_factory.build_provider(role="primary", mode=mode, blind_answer=blind_answer)
    second: Any | None = None
    with contextlib.suppress(llm_factory.ProviderConfigError):
        second = llm_factory.build_provider(role="second", mode=mode)
    return primary, second


def _evidence_lines(rows: list[Any]) -> list[str]:
    lines = []
    for row in rows:
        payload = json.loads(row["payload"] or "{}")
        brief = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(brief) > 100:
            brief = brief[:97] + "..."
        lines.append(f"  {row['id']}  {row['kind']:<16} {row['status']:<8} seq={row['seq']}  {brief}")
    return lines


def _print_result(result: Any, *, db_path: Path, out: Path | None) -> None:
    print(f"run_id:     {result.run_id}")
    print(f"terminal:   {result.terminal.value}")
    print(f"reason:     {result.reason_code or '-'}")
    print(f"latency_ms: {result.latency_ms}")
    print(f"llm_calls:  {result.llm_calls}")
    print(f"tokens:     {result.tokens}")
    print(f"notes:      {', '.join(result.notes) if result.notes else '-'}")
    print("evidence:")
    if result.evidence:
        for line in _evidence_lines(
            [
                {
                    "id": entry.evidence.id,
                    "kind": entry.evidence.kind,
                    "status": entry.evidence.status,
                    "seq": entry.evidence.seq,
                    "payload": json.dumps(entry.evidence.payload, ensure_ascii=False),
                }
                for entry in result.evidence
            ]
        ):
            print(line)
    else:
        print("  （无）")
    print(f"trace:      db={db_path}  →  veridx replay {result.run_id}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "terminal": result.terminal.value,
                    "reason_code": result.reason_code,
                    "latency_ms": result.latency_ms,
                    "llm_calls": result.llm_calls,
                    "tokens": result.tokens,
                    "notes": result.notes,
                    "evidence": [
                        {
                            "id": entry.evidence.id,
                            "kind": entry.evidence.kind,
                            "status": entry.evidence.status,
                            "seq": entry.evidence.seq,
                            "payload": entry.evidence.payload,
                        }
                        for entry in result.evidence
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"out:        {out}")


async def _run_case(
    case: Case,
    *,
    conn: sqlite3.Connection,
    mode: str | None,
    explain: bool,
) -> Any:
    question, attempt = ensure_rows(conn, case)
    # 离线 smoke：脚本 provider 扮演"会做题的模型"（盲解答案 = 答案键）。
    # 真模型下不给提示，`blind_answer` 被忽略。
    primary, second = _build_pair(mode, blind_answer=case.answer_key if mode == "mock" else None)
    run_id = new_id("r")
    runner = Runner(
        primary=primary,
        second=second,
        conn=conn,
        limits=RunLimits(explain=explain),
        tracer=TraceRecorder(conn=conn, run_id=run_id),
        run_id=run_id,
    )
    spec = RunSpec(
        stem=question["stem"],
        student_answer=attempt["student_answer"],
        answer_key=question["correct_answer"],
        attempt_id=attempt["id"],
        question_id=question["id"],
    )
    try:
        result = await runner.run(spec)
    except Exception as exc:  # noqa: BLE001 - 连 Runner 都挂了也必须留痕，不许静默
        TraceRecorder(conn=conn, run_id=run_id).event(
            "submit",
            status=NodeStatus.FAIL,
            reason_code=ReasonCode.INTERNAL_ERROR.value,
            detail={"error": f"{type(exc).__name__}: {exc}"[:200], "stage": "cli_run"},
        )
        raise
    return result


def cmd_run(
    *,
    case_id: str | None,
    db_path: str | None,
    mock: bool,
    real: bool,
    student: str | None,
    out: str | None,
    explain: bool,
) -> int:
    from veridx.store import db as store_db

    mode = "mock" if mock else ("real" if real else None)
    path = Path(db_path) if db_path else config.get_settings().db_path
    conn, _applied = store_db.init_db(path)
    try:
        if case_id:
            case = load_case(case_id, conn)
            if case is None:
                print(f"找不到题：{case_id}（datasets/ 与 {path} 里都没有）")
                return 1
        else:
            print("缺少 --case（或 --stem/--student/--answer-key）")
            return 1
        if student is not None:
            case = Case(
                id=case.id,
                stem=case.stem,
                answer_key=case.answer_key,
                student=student,
                course=case.course,
            )
        try:
            result = asyncio.run(_run_case(case, conn=conn, mode=mode, explain=explain))
        except Exception as exc:  # noqa: BLE001 - 失败也要可排查
            print("terminal:   failed")
            print(f"reason:     {ReasonCode.INTERNAL_ERROR.value}")
            print(f"error:      {(type(exc).__name__ + ': ' + str(exc))[:240]}")
            print(f"trace:      db={path}（已落 submit/fail 事件，可用 veridx replay 查）")
            return 1
        _print_result(result, db_path=path, out=Path(out) if out else None)
        # 退出码只表示"这次 run 有没有跑完并落盘"，不表示"诊断成功"：
        # 把 `uncertain` 映射成非 0 会让 CI 把"能力边界"当成"命令坏了"。
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def _load_events(conn: sqlite3.Connection, run_id: str) -> list[Any]:
    return conn.execute("SELECT * FROM trace_events WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()


def _short(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _print_timeline(rows: list[Any]) -> None:
    for row in rows:
        reason = row["reason_code"] or "-"
        print(
            f"{row['seq']:>3}  {row['node']:<12} {row['status']:<8} {reason:<28}"
            f" {int(row['end_ms'] or 0):>6}ms  {_short(row['detail'] or '{}')}"
        )


def _diff(a: list[Any], b: list[Any]) -> int:
    """两次 run 的差异：先比终态口径，再比节点序列。**只报事实，不做判定。**"""
    signature = [(row["node"], row["status"], row["reason_code"]) for row in a]
    other = [(row["node"], row["status"], row["reason_code"]) for row in b]
    print(f"events: {len(a)} vs {len(b)}")
    for index, (left, right) in enumerate(zip(signature, other, strict=False), start=1):
        if left != right:
            print(f"  #{index}: {left} != {right}")
    for extra in signature[len(other) :]:
        print(f"  only-in-first: {extra}")
    for extra in other[len(signature) :]:
        print(f"  only-in-second: {extra}")
    last_a = signature[-1] if signature else None
    last_b = other[-1] if other else None
    print(f"final: {last_a} vs {last_b}")
    return 0


def cmd_replay(*, run_id: str, db_path: str | None, at: int | None, diff: str | None) -> int:
    from veridx.store import db as store_db

    path = Path(db_path) if db_path else config.get_settings().db_path
    if not path.exists():
        print(f"数据库不存在：{path}")
        return 1
    conn = store_db.connect(path)
    try:
        rows = _load_events(conn, run_id)
        if not rows:
            print(f"找不到 run：{run_id}")
            return 1
        if at is not None:
            target = next((row for row in rows if int(row["seq"]) == at), None)
            if target is None:
                print(f"run {run_id} 没有 seq={at} 的事件（共 {len(rows)} 条）")
                return 1
            print(f"run {run_id} · seq={at} · node={target['node']} · status={target['status']}")
            print(f"reason: {target['reason_code'] or '-'}")
            print("input/detail:")
            print(json.dumps(json.loads(target["detail"] or "{}"), ensure_ascii=False, indent=2))
            children = [row for row in rows if row["parent_seq"] is not None and int(row["parent_seq"]) == at]
            if children:
                print(f"children ({len(children)}):")
                _print_timeline(children)
            return 0
        if diff:
            other = _load_events(conn, diff)
            if not other:
                print(f"找不到对照 run：{diff}")
                return 1
            return _diff(rows, other)
        print(f"run {run_id} · {len(rows)} events")
        _print_timeline(rows)
        return 0
    finally:
        conn.close()
