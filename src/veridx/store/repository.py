"""仓储层：纯 CRUD + 事务，不含业务规则（设计文档 §2.2）。

P0 只覆盖 G0 桩需要的三张表（questions / attempts / tasks）；
其余表在 P1–P4 各自的卡里补仓储方法。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from veridx.clock import new_id, now_utc

TASK_STATUSES = (
    "queued",
    "running",
    "diagnosed",
    "escalated",
    "needs_clarification",
    "uncertain",
    "canceled",
    "failed",
)


class RepositoryError(RuntimeError):
    """仓储层错误（资源不存在 / 状态非法）。"""


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Repository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

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
        return self.get_question(question_id) or {}

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
        fingerprint: str | None = None,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        attempt_id = attempt_id or new_id("a")
        self.conn.execute(
            "INSERT INTO attempts (id, question_id, student_answer, fingerprint, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (attempt_id, question_id, student_answer, fingerprint, now_utc()),
        )
        return self.get_attempt(attempt_id) or {}

    def get_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()
        return _row_to_dict(row)

    # ---- tasks ----
    def create_task(
        self,
        *,
        attempt_id: str,
        status: str = "queued",
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
        return self.get_task(task_id) or {}

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _row_to_dict(row)

    def find_task_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE idempotency_key = ?", (key,)).fetchone()
        return _row_to_dict(row)

    def set_task_status(
        self,
        task_id: str,
        status: str,
        *,
        terminal: str | None = None,
        reason_code: str | None = None,
    ) -> dict[str, Any]:
        if status not in TASK_STATUSES:
            raise RepositoryError(f"未知任务状态：{status}")
        self.conn.execute(
            "UPDATE tasks SET status = ?, terminal = COALESCE(?, terminal),"
            " reason_code = COALESCE(?, reason_code), updated_at = ? WHERE id = ?",
            (status, terminal, reason_code, now_utc(), task_id),
        )
        task = self.get_task(task_id)
        if task is None:
            raise RepositoryError(f"任务不存在：{task_id}")
        return task
