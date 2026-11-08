"""存储层验收：迁移幂等 / PRAGMA / 表齐备 / CRUD 往返 / checksum 保护。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from veridx import config
from veridx.store import db as store_db
from veridx.store.repository import Repository, RepositoryError

# §7.1 的 16 张业务表 + P0 补齐的 tasks + 元表 schema_version
EXPECTED_TABLES = {
    "questions",
    "attempts",
    "diagnoses",
    "evidence",
    "claims",
    "explanation_steps",
    "similar_questions",
    "kp",
    "kp_links",
    "prompt_versions",
    "eval_runs",
    "eval_items",
    "trace_events",
    "sandbox_rejects",
    "redteam_cases",
    "gold_labels",
    "tasks",
    "schema_version",
}


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection, _ = store_db.init_db(tmp_path / "veridx.db", config.MIGRATIONS_DIR)
    yield connection
    connection.close()


def test_migrations_create_all_tables(conn: sqlite3.Connection) -> None:
    assert set(store_db.table_names(conn)) == EXPECTED_TABLES


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "veridx.db"
    first_conn, first = store_db.init_db(path, config.MIGRATIONS_DIR)
    assert first == [1]
    assert store_db.current_version(first_conn) == 1
    first_conn.close()

    second_conn, second = store_db.init_db(path, config.MIGRATIONS_DIR)
    assert second == []
    assert store_db.current_version(second_conn) == 1
    second_conn.close()


def test_pragmas_applied(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == config.DB_BUSY_TIMEOUT_MS
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_modified_migration_is_rejected(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    original = migrations / "001_init.sql"
    original.write_text("CREATE TABLE demo (id TEXT PRIMARY KEY);", encoding="utf-8")

    conn, applied = store_db.init_db(tmp_path / "v.db", migrations)
    assert applied == [1]

    original.write_text("CREATE TABLE demo (id TEXT PRIMARY KEY, extra TEXT);", encoding="utf-8")
    with pytest.raises(store_db.MigrationError, match="已被修改"):
        store_db.apply_migrations(conn, migrations)
    conn.close()


def test_missing_migration_file_is_rejected(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_init.sql").write_text("CREATE TABLE demo (id TEXT);", encoding="utf-8")
    conn, _ = store_db.init_db(tmp_path / "v.db", migrations)
    (migrations / "001_init.sql").unlink()
    with pytest.raises(store_db.MigrationError, match="缺失"):
        store_db.apply_migrations(conn, migrations)
    conn.close()


def test_question_attempt_task_roundtrip(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    question = repo.create_question(
        course="calculus",
        stem="求 ∫_0^1 x·e^x dx",
        correct_answer="1",
        qtype="numeric",
        difficulty="中",
        kp_ids=["kp_int_by_parts"],
    )
    assert question["id"].startswith("q-")
    assert question["kp_ids"] == ["kp_int_by_parts"]

    attempt = repo.create_attempt(
        question_id=question["id"], student_answer="e-1", fingerprint="3f9c1a7e0d4b2c88"
    )
    assert attempt["id"].startswith("a-")
    assert attempt["question_id"] == question["id"]

    task = repo.create_task(attempt_id=attempt["id"], idempotency_key="idem-1", estimated_ms=9000)
    assert task["status"] == "queued"
    assert repo.find_task_by_idempotency_key("idem-1")["id"] == task["id"]

    updated = repo.set_task_status(task["id"], "canceled", terminal="canceled", reason_code="user_canceled")
    assert updated["status"] == "canceled"
    assert updated["reason_code"] == "user_canceled"


def test_foreign_key_enforced(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_attempt(question_id="q-missing", student_answer="1")


def test_unknown_task_status_rejected(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    question = repo.create_question(course="calculus", stem="s", correct_answer="1")
    attempt = repo.create_attempt(question_id=question["id"], student_answer="2")
    with pytest.raises(RepositoryError):
        repo.create_task(attempt_id=attempt["id"], status="not_a_status")
