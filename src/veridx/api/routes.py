"""13 个 REST 端点（设计文档 §8.1）。

P0 状态：G0 桩。**能真跑的部分**（题库/作答/任务状态/取消/健康）直连 SQLite；
**依赖 Agent 内核的部分**（诊断全文/SSE/回放/知识点/评测）返回 501 并标注实现它的卡，
避免"假数据被当成真结果"。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import StreamingResponse

from veridx import __version__, config
from veridx.api import schemas as S
from veridx.api.deps import RepoDep
from veridx.api.errors import ApiError, ErrorCode, not_implemented
from veridx.api.jobs import JobManager, job_spec_from
from veridx.api.stream import sse_frame
from veridx.clock import new_id
from veridx.sched.fingerprint import fingerprint
from veridx.store import db as store_db
from veridx.store.repository import RepositoryError, map_integrity_error

router = APIRouter(prefix=S.API_PREFIX)


# --- 1. 建题 ---------------------------------------------------------------
@router.post("/questions", response_model=S.QuestionOut, status_code=201, tags=["question"])
def create_question(payload: S.QuestionCreate, repo: RepoDep) -> S.QuestionOut:
    record = repo.create_question(
        course=payload.course,
        stem=payload.stem,
        correct_answer=payload.correct_answer,
        qtype=payload.qtype,
        difficulty=payload.difficulty,
        kp_ids=payload.kp_ids,
        source=payload.source,
    )
    return S.QuestionOut.model_validate(record)


# --- 2. 录一次作答 ---------------------------------------------------------
@router.post("/attempts", response_model=S.AttemptOut, status_code=201, tags=["attempt"])
def create_attempt(payload: S.AttemptCreate, repo: RepoDep) -> S.AttemptOut:
    """建题 + 建作答是一个动作：两步必须同事务，否则会留下孤儿题。"""
    try:
        with repo.transaction():
            if payload.question is not None:
                question = repo.create_question(
                    course=payload.question.course,
                    stem=payload.question.stem,
                    correct_answer=payload.question.correct_answer,
                    qtype=payload.question.qtype,
                    difficulty=payload.question.difficulty,
                    kp_ids=payload.question.kp_ids,
                    source=payload.question.source,
                )
            else:
                question = repo.get_question(payload.question_id or "")
                if question is None:
                    raise ApiError(ErrorCode.NOT_FOUND, f"题目不存在：{payload.question_id}")

            existing = repo.find_attempt(question["id"], payload.student_answer)
            if existing is not None:
                raise ApiError(
                    ErrorCode.DUPLICATE_ATTEMPT,
                    f"该作答已存在：{existing['id']}（同题同答视为重复样本，避免给某道题偷偷加权）",
                )

            digest = fingerprint(
                question=question["stem"],
                correct=question["correct_answer"],
                student_answer=payload.student_answer,
            )
            attempt = repo.create_attempt(
                question_id=question["id"], student_answer=payload.student_answer, answer_digest=digest
            )
    except sqlite3.IntegrityError as exc:  # 并发下的兜底：DB 唯一约束才是真相
        reason = map_integrity_error(exc)
        if reason == "duplicate_attempt":
            raise ApiError(ErrorCode.DUPLICATE_ATTEMPT, f"该作答已存在（{reason}）") from exc
        # 不要把外键失败说成"重复作答"：错误信息指错方向比不报错更糟
        raise ApiError(ErrorCode.INVALID_INPUT, f"数据约束不满足（{reason}）") from exc
    except RepositoryError as exc:
        raise ApiError(ErrorCode.INVALID_INPUT, str(exc)) from exc

    return S.AttemptOut(attempt_id=attempt["id"], question_id=question["id"], fingerprint=digest)


# --- 3. 批量导入 -----------------------------------------------------------
@router.post("/attempts/import", response_model=S.ImportResult, tags=["attempt"])
def import_attempts(payload: S.ImportPayload) -> S.ImportResult:
    # P2.2 实现；行级错误映射已就绪（map_integrity_error → duplicate_attempt），不冒 500
    raise not_implemented("P2.2")


# --- 4. 触发诊断（幂等）----------------------------------------------------
@router.post(
    "/attempts/{attempt_id}/diagnose", response_model=S.DiagnoseResponse, status_code=202, tags=["task"]
)
async def diagnose(
    attempt_id: str,
    request: Request,
    response: Response,
    repo: RepoDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> S.DiagnoseResponse:
    attempt = repo.get_attempt(attempt_id)
    if attempt is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"作答不存在：{attempt_id}")

    if idempotency_key:
        existing = repo.find_task_by_idempotency_key(idempotency_key)
        if existing is not None:
            # 幂等键是"这一次请求的身份证"：同一把钥匙开另一扇门必须报错，
            # 否则评测跑批会静默拿到上一题的结果（不报错，但结论全错）。
            if existing["attempt_id"] != attempt_id:
                raise ApiError(
                    ErrorCode.IDEMPOTENCY_KEY_CONFLICT,
                    f"Idempotency-Key 已被用于另一条作答（{existing['attempt_id']}）",
                )
            response.status_code = 200
            return S.DiagnoseResponse(
                task_id=existing["id"],
                status=S.TaskStatus(existing["status"]),
                estimated_ms=existing["estimated_ms"],
                idempotent_hit=True,
                terminal=S.Terminal(existing["terminal"]) if existing["terminal"] else None,
                run_id=existing["run_id"],
            )

    question = repo.get_question(attempt["question_id"])
    if question is None:  # pragma: no cover - 外键保证不会发生
        raise ApiError(ErrorCode.NOT_FOUND, f"题目不存在：{attempt['question_id']}")

    task = repo.create_task(
        attempt_id=attempt_id,
        estimated_ms=config.ESTIMATED_LATENCY_MS,
        idempotency_key=idempotency_key,
    )
    # 真正的诊断在后台跑（202 的语义）：立刻返回 task_id，用 /tasks/{id}/stream 追进度
    manager: JobManager = request.app.state.jobs
    run_id = new_id("r")
    manager.submit(
        task_id=task["id"],
        run_id=run_id,
        spec=job_spec_from(attempt, question),
    )
    return S.DiagnoseResponse(
        task_id=task["id"],
        status=S.TaskStatus(task["status"]),
        estimated_ms=task["estimated_ms"],
        run_id=run_id,
    )


# --- 5. 查任务状态（SSE 不可用时的轮询兼容）--------------------------------
@router.get("/tasks/{task_id}", response_model=S.TaskOut, tags=["task"])
def get_task(task_id: str, repo: RepoDep) -> S.TaskOut:
    task = repo.get_task(task_id)
    if task is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"任务不存在：{task_id}")
    return S.TaskOut(
        task_id=task["id"],
        attempt_id=task["attempt_id"],
        status=S.TaskStatus(task["status"]),
        terminal=S.Terminal(task["terminal"]) if task["terminal"] else None,
        reason_code=task["reason_code"],
        run_id=task["run_id"],
        created_at=task["created_at"],
        updated_at=task["updated_at"],
        diagnosis_id=find_diagnosis_id(repo.conn, task["run_id"]),
    )


def find_diagnosis_id(conn: sqlite3.Connection, run_id: str | None) -> str | None:
    """task → diagnosis 的映射不建表：`run_id` 就是那根线（诊断行带同一个 run_id）。"""
    if not run_id:
        return None
    row = conn.execute("SELECT id FROM diagnoses WHERE run_id = ?", (run_id,)).fetchone()
    return row["id"] if row else None


# --- 6. SSE 事件流 ---------------------------------------------------------
@router.get(
    "/tasks/{task_id}/stream",
    tags=["task"],
    responses={
        200: {
            "description": "SSE 帧流：6 类事件（stage/evidence/claim/token/guard/done），`done` 恰好一次",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
def stream_task(task_id: str, request: Request) -> StreamingResponse:
    """把 trace 表当作事件源：**不另造一套内存事件总线**。

    `trace_events` 本来就要为每个节点落一条（不变量 I4），从它派生 SSE 有三好处：
    ① 前后端看到的进度与落盘事实**同一份**，不会漂移；
    ② 客户端断线重连只要带上 `Last-Event-ID`（这里用 `?after=`）就能续；
    ③ 后台作业不需要知道"有没有人在看"。
    """
    db_path: Path = request.app.state.db_path
    conn = store_db.connect(db_path)
    task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if task is None:
        conn.close()
        raise ApiError(ErrorCode.NOT_FOUND, f"任务不存在：{task_id}")
    after = int(request.query_params.get("after", "0") or 0)

    def frames() -> Iterator[str]:
        try:
            yield from _trace_frames(conn, task, after=after)
        finally:
            conn.close()

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


#: trace 的 node → SSE 的 stage 名（其余 node 走专门的事件类型）
_STAGE_NODES = {
    "parse",
    "blind_solve",
    "check_numeric",
    "step_diff",
    "kb_lookup",
    "diagnose",
    "explain",
    "submit",
}

_STREAM_POLL_SECONDS = 0.25
_STREAM_HEARTBEAT_SECONDS = 10.0
_STREAM_MAX_SECONDS = 180.0


def _trace_frames(conn: sqlite3.Connection, task: Any, *, after: int) -> Iterator[str]:
    """轮询 trace/evidence/claims，把增量翻成 SSE 帧。"""
    import time

    started = time.monotonic()
    last_heartbeat = started
    last_seq = after
    seen_evidence = 0
    emitted_claims = False
    seq_counter = 0

    def next_seq() -> int:
        nonlocal seq_counter
        seq_counter += 1
        return seq_counter

    while True:
        run_id = task["run_id"]
        if run_id:
            rows = conn.execute(
                "SELECT * FROM trace_events WHERE run_id = ? AND seq > ? ORDER BY seq",
                (run_id, last_seq),
            ).fetchall()
            for row in rows:
                last_seq = row["seq"]
                yield from _event_frames(row, next_seq)

            new_evidence = conn.execute(
                "SELECT * FROM evidence WHERE run_id = ? AND seq > ? ORDER BY seq",
                (run_id, seen_evidence),
            ).fetchall()
            for row in new_evidence:
                seen_evidence = row["seq"]
                yield sse_frame(
                    S.SSEEventName.EVIDENCE,
                    S.EvidenceEventData(
                        id=row["id"],
                        kind=row["kind"],
                        status=row["status"],
                        seq=row["seq"],
                        payload=json.loads(row["payload"]),
                    ),
                )

            terminal = task["terminal"]
            if terminal and not emitted_claims:
                diagnosis_id = find_diagnosis_id(conn, run_id)
                if diagnosis_id:
                    for row in conn.execute(
                        "SELECT * FROM claims WHERE diagnosis_id = ? ORDER BY seq",
                        (diagnosis_id,),
                    ):
                        yield sse_frame(
                            S.SSEEventName.CLAIM,
                            S.ClaimEventData(
                                field=row["field"],
                                value=row["value"],
                                evidence_ids=json.loads(row["evidence_ids"]),
                                seq=row["seq"],
                            ),
                        )
                emitted_claims = True

            task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task["id"],)).fetchone()
            if task is None or task["terminal"]:
                yield sse_frame(
                    S.SSEEventName.DONE,
                    S.DoneEventData(
                        terminal=S.Terminal(task["terminal"]),
                        reason_code=task["reason_code"],
                        run_id=run_id,
                        latency_ms=int((time.monotonic() - started) * 1000),
                        tokens=S.TokenUsage(
                            input_tokens=task["tokens_in"] or 0,
                            output_tokens=task["tokens_out"] or 0,
                        ),
                    ),
                )
                return

        now = time.monotonic()
        if now - last_heartbeat >= _STREAM_HEARTBEAT_SECONDS:
            last_heartbeat = now
            yield sse_frame(
                S.SSEEventName.STAGE,
                S.StageEventData(
                    stage="heartbeat",
                    status="ok",
                    ms=int((now - started) * 1000),
                    seq=next_seq(),
                ),
            )
        if now - started >= _STREAM_MAX_SECONDS:
            # 不假装成功：超时就把 done 发成 failed，客户端能归因
            yield sse_frame(
                S.SSEEventName.DONE,
                S.DoneEventData(
                    terminal=S.Terminal.FAILED,
                    reason_code="stream_timeout",
                    run_id=task["run_id"],
                    latency_ms=int((now - started) * 1000),
                    tokens=S.TokenUsage(input_tokens=0, output_tokens=0),
                ),
            )
            return
        time.sleep(_STREAM_POLL_SECONDS)


def _event_frames(row: Any, next_seq: Callable[[], int]) -> Iterator[str]:
    detail = json.loads(row["detail"] or "{}")
    if row["node"] == "guard":
        escalate = detail.get("escalate")
        yield sse_frame(
            S.SSEEventName.GUARD,
            S.GuardEventData(
                schema="fail" if escalate else "pass",
                evidence_binding="fail_after_retry" if escalate else "pass",
                retries=int(detail.get("retries", 0) or 0),
            ),
        )
        return
    # `parse` 节点有两种事件：真正的输入解析，和"这次用了哪个 provider"的元信息。
    # 后者不是流程阶段，前端不该看到一个多出来的 parse。
    if row["node"] == "parse" and "provider" in detail:
        return
    if row["node"] in _STAGE_NODES:
        yield sse_frame(
            S.SSEEventName.STAGE,
            S.StageEventData(
                stage=row["node"],
                status=row["status"],
                ms=int(row["end_ms"] or 0),
                seq=next_seq(),
            ),
        )
    # 不伪造 `token` 帧：模型调用是非流式的（`call_structured` 一次拿完整 JSON），
    # 逐字增量根本不存在。编一串假 text 会让"流式"看起来实现了，实际是假的。


# --- 7. 取消（幂等）--------------------------------------------------------
@router.post("/tasks/{task_id}/cancel", response_model=S.CancelOut, tags=["task"])
def cancel_task(task_id: str, repo: RepoDep, request: Request) -> S.CancelOut:
    task = repo.get_task(task_id)
    if task is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"任务不存在：{task_id}")
    if task["terminal"] is not None:
        return S.CancelOut(task_id=task_id, status=S.TaskStatus(task["status"]), already_terminal=True)

    manager: JobManager = request.app.state.jobs
    manager.cancel(task_id)
    updated = repo.finish_task(task_id, S.Terminal.CANCELED.value, reason_code="user_canceled")
    return S.CancelOut(task_id=task_id, status=S.TaskStatus(updated["status"]))


# --- 8. 取诊断全文 ---------------------------------------------------------
@router.get("/diagnoses/{diagnosis_id}", response_model=S.DiagnosisOut, tags=["diagnosis"])
def get_diagnosis(diagnosis_id: str, repo: RepoDep) -> S.DiagnosisOut:
    conn = repo.conn
    row = conn.execute("SELECT * FROM diagnoses WHERE id = ?", (diagnosis_id,)).fetchone()
    if row is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"诊断不存在：{diagnosis_id}")
    return build_diagnosis(conn, row)


def build_diagnosis(conn: sqlite3.Connection, row: Any) -> S.DiagnosisOut:
    """把行拼成 `DiagnosisOut`。claims/evidence/steps 全部来自库，不重新推断。"""
    claims = [
        S.ClaimOut(
            field=claim["field"],
            value=_as_value(claim["value"]),
            evidence_ids=json.loads(claim["evidence_ids"] or "[]"),
            seq=claim["seq"],
        )
        for claim in conn.execute("SELECT * FROM claims WHERE diagnosis_id = ? ORDER BY seq", (row["id"],))
    ]
    evidence = [
        S.Evidence(
            id=item["id"],
            kind=item["kind"],
            payload=json.loads(item["payload"]),
            status=item["status"],
            produced_by=item["produced_by"],
            seq=item["seq"],
        )
        for item in conn.execute("SELECT * FROM evidence WHERE run_id = ? ORDER BY seq", (row["run_id"],))
    ]
    steps = [
        S.ExplanationStepOut(
            seq=step["seq"],
            title=step["title"],
            content=step["content"],
            key_point=step["key_point"],
            targets_error=step["targets_error"],
            verifiable=bool(step["verifiable"]),
        )
        for step in conn.execute(
            "SELECT * FROM explanation_steps WHERE diagnosis_id = ? ORDER BY seq", (row["id"],)
        )
    ]
    kps = [
        S.KpOut(id=kp["id"], name=kp["name"])
        for kp in conn.execute(
            "SELECT kp.id, kp.name FROM kp_links JOIN kp ON kp.id = kp_links.kp_id"
            " WHERE kp_links.diagnosis_id = ?",
            (row["id"],),
        )
    ]
    hypotheses = [
        S.Hypothesis(label=item.get("label", ""), confidence=float(item.get("confidence", 0.0)))
        for item in json.loads(row["hypotheses"] or "[]")
        if isinstance(item, dict)
    ]
    return S.DiagnosisOut(
        id=row["id"],
        attempt_id=row["attempt_id"],
        run_id=row["run_id"],
        terminal=S.Terminal(row["terminal"]),
        reason_code=row["reason_code"],
        error_type=row["error_type"],
        error_subtype=row["error_subtype"],
        first_divergent_step=row["first_divergent_step"],
        locator=row["locator"],
        root_cause=row["root_cause"],
        hypotheses=hypotheses,
        kp_ids=json.loads(row["kp_ids"] or "[]"),
        confidence=row["confidence"],
        needs_review=bool(row["needs_review"]),
        claims=claims,
        evidence=evidence,
        steps=steps,
        kps=kps,
        arm=row["arm"],
        prompt_version_id=row["prompt_version_id"],
        model=row["model"],
        temperature=row["temperature"],
        seed=row["seed"],
        evidence_binding=row["evidence_binding"],
        guard_retries=row["guard_retries"],
        cascade_level=row["cascade_level"],
        verify_agrees=bool(row["verify_agrees"]) if row["verify_agrees"] is not None else None,
        suspected_bad_answer_key=row["suspected_bad_answer_key"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        latency_ms=row["latency_ms"],
        created_at=row["created_at"],
    )


def _as_value(raw: str) -> str | list[str]:
    """`claims.value` 里数组是 JSON 编码存的；单值就是原字符串。"""
    text = raw or ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except ValueError:
            return text
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return text


# --- 9. 回放 ---------------------------------------------------------------
@router.get("/runs/{run_id}/replay", response_model=S.ReplayOut, tags=["replay"])
def replay_run(run_id: str) -> S.ReplayOut:
    raise not_implemented("P3.5")


# --- 10. 薄弱知识点 --------------------------------------------------------
@router.get("/kp/weak", response_model=S.WeakKpList, tags=["kp"])
def weak_kp() -> S.WeakKpList:
    raise not_implemented("P2.5")


# --- 11. 起一次实验 --------------------------------------------------------
@router.post("/eval/runs", response_model=S.EvalRunAccepted, status_code=202, tags=["eval"])
def create_eval_run(payload: S.EvalRunCreate) -> S.EvalRunAccepted:
    raise not_implemented("P4.1")


# --- 12. 取评测报告 --------------------------------------------------------
@router.get("/eval/runs/{eval_run_id}/report", response_model=S.EvalReportOut, tags=["eval"])
def get_eval_report(eval_run_id: str, report_format: str = "json") -> S.EvalReportOut:
    raise not_implemented("P4.3")


# --- 13. 健康检查 ----------------------------------------------------------
@router.get("/health", response_model=S.HealthOut, tags=["ops"])
def health(repo: RepoDep) -> S.HealthOut:
    from veridx.llm import factory as llm_factory
    from veridx.store import db as store_db

    # 声明模式（配置）与落地模式（工厂真实选择）必须都能看见：
    # `auto` 静默退化成 mock 时，只有后者能揭穿（审查 🟡F1）
    primary_mode = llm_factory.effective_mode("primary")
    has_second = config.has_second_model()
    provider = S.ProviderHealth(
        mode=config.LLM_MODE,
        effective_mode=primary_mode,
        model=config.MODEL,
        reachable=None,  # 不做网络探测：健康检查不能因为上游抖动而挂（NFR-6）
        second_model=config.SECOND_MODEL if has_second else None,
        second_effective_mode=llm_factory.effective_mode("second") if has_second else None,
        require_real_llm=config.REQUIRE_REAL_LLM,
    )
    # 要求真模型却拿不到 = 降级；mode=mock 是有意为之，仍算 ok
    needs_key = config.LLM_MODE in {"real", "auto"} and primary_mode != "real"
    return S.HealthOut(
        status="degraded" if needs_key else "ok",
        version=__version__,
        schema_version=store_db.current_version(repo.conn),
        provider=provider,
    )
