"""13 个 REST 端点（设计文档 §8.1）。

P0 状态：G0 桩。**能真跑的部分**（题库/作答/任务状态/取消/健康）直连 SQLite；
**依赖 Agent 内核的部分**（诊断全文/SSE/回放/知识点/评测）返回 501 并标注实现它的卡，
避免"假数据被当成真结果"。
"""

from __future__ import annotations

from fastapi import APIRouter, Header, Response

from veridx import __version__, config
from veridx.api import schemas as S
from veridx.api.deps import RepoDep
from veridx.api.errors import ApiError, ErrorCode, not_implemented
from veridx.sched.fingerprint import fingerprint

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

    fp = fingerprint(
        question=question["stem"],
        correct=question["correct_answer"],
        student_answer=payload.student_answer,
    )
    attempt = repo.create_attempt(
        question_id=question["id"], student_answer=payload.student_answer, fingerprint=fp
    )
    return S.AttemptOut(attempt_id=attempt["id"], question_id=question["id"], fingerprint=fp)


# --- 3. 批量导入 -----------------------------------------------------------
@router.post("/attempts/import", response_model=S.ImportResult, tags=["attempt"])
def import_attempts(payload: S.ImportPayload) -> S.ImportResult:
    raise not_implemented("P2.2")


# --- 4. 触发诊断（幂等）----------------------------------------------------
@router.post(
    "/attempts/{attempt_id}/diagnose", response_model=S.DiagnoseResponse, status_code=202, tags=["task"]
)
def diagnose(
    attempt_id: str,
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
            response.status_code = 200
            return S.DiagnoseResponse(
                task_id=existing["id"],
                status=S.TaskStatus(existing["status"]),
                estimated_ms=existing["estimated_ms"],
                idempotent_hit=True,
                terminal=S.Terminal(existing["terminal"]) if existing["terminal"] else None,
                run_id=existing["run_id"],
            )

    task = repo.create_task(
        attempt_id=attempt_id,
        estimated_ms=config.ESTIMATED_LATENCY_MS,
        idempotency_key=idempotency_key,
    )
    return S.DiagnoseResponse(
        task_id=task["id"], status=S.TaskStatus(task["status"]), estimated_ms=task["estimated_ms"]
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
    )


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
def stream_task(task_id: str) -> None:
    raise not_implemented("P2.1")


# --- 7. 取消（幂等）--------------------------------------------------------
@router.post("/tasks/{task_id}/cancel", response_model=S.CancelOut, tags=["task"])
def cancel_task(task_id: str, repo: RepoDep) -> S.CancelOut:
    task = repo.get_task(task_id)
    if task is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"任务不存在：{task_id}")
    if task["status"] in set(S.Terminal):
        return S.CancelOut(task_id=task_id, status=S.TaskStatus(task["status"]), already_terminal=True)
    updated = repo.set_task_status(
        task_id, "canceled", terminal=S.Terminal.CANCELED.value, reason_code="user_canceled"
    )
    return S.CancelOut(task_id=task_id, status=S.TaskStatus(updated["status"]))


# --- 8. 取诊断全文 ---------------------------------------------------------
@router.get("/diagnoses/{diagnosis_id}", response_model=S.DiagnosisOut, tags=["diagnosis"])
def get_diagnosis(diagnosis_id: str) -> S.DiagnosisOut:
    raise not_implemented("P1.6")


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
    from veridx.store import db as store_db

    schema_version = store_db.current_version(repo.conn)
    provider = S.ProviderHealth(
        mode=config.LLM_MODE,
        model=config.MODEL,
        reachable=None,  # 不做网络探测：健康检查不能因为上游抖动而挂（NFR-6）
        second_model=config.SECOND_MODEL if config.has_second_model() else None,
    )
    return S.HealthOut(
        status="ok" if provider.model else "degraded",
        version=__version__,
        schema_version=schema_version,
        provider=provider,
    )
