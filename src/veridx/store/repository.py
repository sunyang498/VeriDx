"""仓储层：纯 CRUD + 事务，不含业务规则（设计文档 §2.2）。

两条 P0.6 纪律（来自审查）：

* **终态只有一个写入口** `finish_task()`。允许转移表写死在 `ALLOWED_TRANSITIONS` 里，
  不在表里的转移一律 `RepositoryError` —— "终态不可逆"是可归因（`reason_code` 有意义）的前提。
* 插入后查不到 → `raise`，**不要** `or {}`。静默返回空字典会让"写失败"伪装成"没数据"。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from veridx.clock import new_id, now_utc
from veridx.trace.event import ReasonCode

_REASON_CODES = frozenset(r.value for r in ReasonCode)

QUEUED = "queued"
RUNNING = "running"
TERMINAL_STATUSES = (
    "diagnosed",
    "escalated",
    "needs_clarification",
    "uncertain",
    "canceled",
    "failed",
)
TASK_STATUSES = (QUEUED, RUNNING, *TERMINAL_STATUSES)

#: 允许的状态转移。终态不在键里 = 不可再转移。
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    QUEUED: frozenset({RUNNING, *TERMINAL_STATUSES}),
    RUNNING: frozenset(TERMINAL_STATUSES),
}


class RepositoryError(RuntimeError):
    """仓储层错误（资源不存在 / 状态非法 / 未标注必填字段）。"""


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Repository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ---- 事务 ----
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """显式事务。

        注意：连接是 `isolation_level=None`（autocommit），此时 `with conn:` **不会**开事务，
        必须自己 `BEGIN`/`COMMIT`（P0.6 审查 §1.1 实测确认）。
        """
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    # ---- questions ----
    def create_question(
        self,
        *,
        course: str,
        stem: str,
        correct_answer: str,
        qtype: str | None = None,
        difficulty: str | None = None,
        kp_ids: list[str] | None = None,
        source: str | None = None,
        question_id: str | None = None,
    ) -> dict[str, Any]:
        question_id = question_id or new_id("q")
        self.conn.execute(
            "INSERT INTO questions"
            " (id, course, stem, correct_answer, qtype, difficulty, kp_ids, source, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                question_id,
                course,
                stem,
                correct_answer,
                qtype,
                difficulty,
                json.dumps(kp_ids or [], ensure_ascii=False),
                source,
                now_utc(),
            ),
        )
        question = self.get_question(question_id)
        if question is None:
            raise RepositoryError(f"题目写入后不可读：{question_id}")
        return question

    def get_question(self, question_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
        record = _row_to_dict(row)
        if record is not None:
            record["kp_ids"] = json.loads(record["kp_ids"])
        return record

    # ---- attempts ----
    def create_attempt(
        self,
        *,
        question_id: str,
        student_answer: str,
        answer_digest: str | None = None,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        attempt_id = attempt_id or new_id("a")
        self.conn.execute(
            "INSERT INTO attempts (id, question_id, student_answer, answer_digest, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (attempt_id, question_id, student_answer, answer_digest, now_utc()),
        )
        attempt = self.get_attempt(attempt_id)
        if attempt is None:
            raise RepositoryError(f"作答写入后不可读：{attempt_id}")
        return attempt

    def get_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()
        return _row_to_dict(row)

    def find_attempt(self, question_id: str, student_answer: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM attempts WHERE question_id = ? AND student_answer = ?",
            (question_id, student_answer),
        ).fetchone()
        return _row_to_dict(row)

    # ---- evidence ----
    def create_evidence(
        self,
        *,
        run_id: str,
        evidence_id: str,
        kind: str,
        payload: dict[str, Any],
        produced_by: str,
        seq: int,
        status: str = "ok",
    ) -> dict[str, Any]:
        self.conn.execute(
            "INSERT INTO evidence"
            " (id, run_id, kind, payload, status, produced_by, seq, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                evidence_id,
                run_id,
                kind,
                json.dumps(payload, ensure_ascii=False),
                status,
                produced_by,
                seq,
                now_utc(),
            ),
        )
        row = self.conn.execute(
            "SELECT * FROM evidence WHERE run_id = ? AND id = ?", (run_id, evidence_id)
        ).fetchone()
        record = _row_to_dict(row)
        if record is None:
            raise RepositoryError(f"证据写入后不可读：{run_id}/{evidence_id}")
        record["payload"] = json.loads(record["payload"])
        return record

    def list_evidence(self, run_id: str, evidence_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """**必须带 `run_id`**：`ev_N` 只在 run 内唯一。

        P0.6 把主键改成 `(run_id, id)` 之后，只按 id 查会**跨 run 串证据**；
        如果串到的恰好是 `status=ok`，E2 校验会静默通过（比原主键 bug 更隐蔽）。
        """
        if evidence_ids is None:
            rows = self.conn.execute(
                "SELECT * FROM evidence WHERE run_id = ? ORDER BY seq", (run_id,)
            ).fetchall()
        elif not evidence_ids:
            return []
        else:
            placeholders = ",".join("?" * len(evidence_ids))
            rows = self.conn.execute(
                f"SELECT * FROM evidence WHERE run_id = ? AND id IN ({placeholders}) ORDER BY seq",
                (run_id, *evidence_ids),
            ).fetchall()
        records = [dict(r) for r in rows]
        for record in records:
            record["payload"] = json.loads(record["payload"])
        return records

    # ---- tasks ----
    def create_task(
        self,
        *,
        attempt_id: str,
        status: str = QUEUED,
        idempotency_key: str | None = None,
        estimated_ms: int | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        if status not in TASK_STATUSES:
            raise RepositoryError(f"未知任务状态：{status}")
        task_id = task_id or new_id("t")
        ts = now_utc()
        self.conn.execute(
            "INSERT INTO tasks"
            " (id, attempt_id, run_id, status, idempotency_key, estimated_ms, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, attempt_id, run_id, status, idempotency_key, estimated_ms, ts, ts),
        )
        task = self.get_task(task_id)
        if task is None:
            raise RepositoryError(f"任务写入后不可读：{task_id}")
        return task

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _row_to_dict(row)

    def find_task_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE idempotency_key = ?", (key,)).fetchone()
        return _row_to_dict(row)

    def start_task(self, task_id: str, *, run_id: str | None = None) -> dict[str, Any]:
        return self._transition(task_id, RUNNING, run_id=run_id)

    def finish_task(self, task_id: str, terminal: str, *, reason_code: str | None = None) -> dict[str, Any]:
        """**唯一**的终态写入口。终态不可再转移。"""
        if terminal not in TERMINAL_STATUSES:
            raise RepositoryError(f"非终态不能由 finish_task 写入：{terminal}")
        return self._transition(task_id, terminal, reason_code=reason_code)

    def _transition(
        self,
        task_id: str,
        target: str,
        *,
        reason_code: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        task = self.get_task(task_id)
        if task is None:
            raise RepositoryError(f"任务不存在：{task_id}")
        if reason_code is not None and reason_code not in _REASON_CODES:
            # 自由字符串会让 F1–F8 失败分类统计不出来，而且不会报错
            raise RepositoryError(f"未知 reason_code：{reason_code}")
        current = task["status"]
        allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise RepositoryError(f"非法的状态转移：{current} -> {target}")

        is_terminal = target in TERMINAL_STATUSES
        self.conn.execute(
            "UPDATE tasks SET status = ?, terminal = ?, reason_code = COALESCE(?, reason_code),"
            " run_id = COALESCE(?, run_id), updated_at = ? WHERE id = ?",
            (target, target if is_terminal else None, reason_code, run_id, now_utc(), task_id),
        )
        updated = self.get_task(task_id)
        if updated is None:
            raise RepositoryError(f"任务更新后不可读：{task_id}")
        return updated

    # ---- gold_labels ----
    def create_gold_label(
        self,
        *,
        attempt_id: str,
        error_type_major: str,
        labeler: str,
        student_is_correct: bool | None,
        answer_key_correct: bool | None,
        key_steps: list[str] | None = None,
        first_wrong_step: int | None = None,
        round: int = 1,
        label_id: str | None = None,
    ) -> dict[str, Any]:
        """人工标注入库。

        `student_is_correct` 是 M18 的**分母来源**，必须显式给值：
        留空等于让这条样本静默退出分母，指标会好看但失真。
        """
        if student_is_correct is None:
            raise RepositoryError("gold_label 必须显式给出 student_is_correct（M18 分母不能靠默认值）")
        label_id = label_id or new_id("gl")
        self.conn.execute(
            "INSERT INTO gold_labels (id, attempt_id, error_type_major, key_steps, first_wrong_step,"
            " labeler, labeled_at, round, student_is_correct, answer_key_correct)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                label_id,
                attempt_id,
                error_type_major,
                json.dumps(key_steps or [], ensure_ascii=False),
                first_wrong_step,
                labeler,
                now_utc(),
                round,
                int(student_is_correct),
                None if answer_key_correct is None else int(answer_key_correct),
            ),
        )
        row = self.conn.execute("SELECT * FROM gold_labels WHERE id = ?", (label_id,)).fetchone()
        record = _row_to_dict(row)
        if record is None:
            raise RepositoryError(f"标注写入后不可读：{label_id}")
        return record


DUPLICATE_ATTEMPT_CONSTRAINT = "uq_attempts_question_answer"


def map_integrity_error(exc: sqlite3.IntegrityError) -> str:
    """把 DB 约束失败翻译成行级错误码（P2.2 批量导入用，绝不冒到 500）。"""
    message = str(exc)
    # SQLite 报的是「列名」而不是「索引名」，所以按列判
    if "attempts.question_id" in message and "attempts.student_answer" in message:
        return "duplicate_attempt"
    if "ck_tasks_status" in message or "ck_tasks_terminal" in message:
        return "invalid_task_status"
    if "FOREIGN KEY" in message:
        return "references_missing"
    return "constraint_violation"
