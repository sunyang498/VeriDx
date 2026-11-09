"""P0.6 回归测试：每条对应审查里的一个真实缺陷。

1. evidence 主键作用域（#1）
2. 连接跨线程（#2）
3. 迁移原子性 + 外键开关恢复（#3 / §5.3）
4. `complete_statement` 切分（§7）
5. 幂等键跨 attempt（#4）
6. 同题同答重复 vs 同题不同答（#5）
7. status/terminal 矛盾（#7）
8. 终态不可逆（§4.B）
9. 表重建后子表外键仍指向 attempts（§3.2 的回归测试）
10. gold_labels 必须显式给 student_is_correct（#6 / §5.1）
11. 500 固定文案 + request_id（#8 / §3.3）
12. 拒绝 :memory:（#9）
13. db version 不建库（#10）
14. health 无 key 时降级（#11）
15. stage.status 含 canceled（#12）
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from veridx import config
from veridx.api.app import MEMORY_DB_ERROR, create_app
from veridx.api.errors import INTERNAL_ERROR_MESSAGE
from veridx.api.schemas import StageEventData
from veridx.store import db as store_db
from veridx.store.repository import Repository, RepositoryError, map_integrity_error

QUESTION = {"course": "calculus", "stem": "求 ∫_0^1 x·e^x dx", "correct_answer": "1"}


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection, _ = store_db.init_db(tmp_path / "p06.db", config.MIGRATIONS_DIR)
    yield connection
    connection.close()


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(db_path=tmp_path / "p06_api.db")) as test_client:
        yield test_client


# 1 -------------------------------------------------------------------------
def test_evidence_id_is_unique_per_run(conn: sqlite3.Connection) -> None:
    for run_id in ("r-1", "r-2"):
        conn.execute(
            "INSERT INTO evidence (id, run_id, kind, status, produced_by, seq, created_at)"
            " VALUES ('ev_1', ?, 'numeric_check', 'ok', 'check_numeric', 1, 't')",
            (run_id,),
        )
    rows = conn.execute("SELECT run_id FROM evidence WHERE id='ev_1' ORDER BY run_id").fetchall()
    assert [r["run_id"] for r in rows] == ["r-1", "r-2"]

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO evidence (id, run_id, kind, status, produced_by, seq, created_at)"
            " VALUES ('ev_1', 'r-1', 'numeric_check', 'ok', 'check_numeric', 2, 't')"
        )


# 2 -------------------------------------------------------------------------
def test_connection_usable_across_threads(conn: sqlite3.Connection) -> None:
    errors: list[str] = []

    def use_in_other_thread() -> None:
        try:
            Repository(conn).create_question(course="c", stem="s", correct_answer="1")
        except Exception as exc:  # noqa: BLE001 - 测试要看到任何异常
            errors.append(f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=use_in_other_thread)
    thread.start()
    thread.join()
    assert errors == []


# 3 -------------------------------------------------------------------------
def test_failed_migration_leaves_no_half_schema_and_restores_fk(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_ok.sql").write_text("CREATE TABLE a (id TEXT PRIMARY KEY);", encoding="utf-8")
    (migrations / "002_bad.sql").write_text(
        "CREATE TABLE b (id TEXT PRIMARY KEY);\nCREATE TABLE b (id TEXT PRIMARY KEY);", encoding="utf-8"
    )

    conn = store_db.connect(tmp_path / "atomic.db")
    with pytest.raises(sqlite3.OperationalError):
        store_db.apply_migrations(conn, migrations)

    assert store_db.table_names(conn) == ["a", "schema_version"]
    assert store_db.current_version(conn) == 1
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1  # 失败路径也要恢复

    # 修好后重跑仍然能成功（没有"永远起不来"的状态）
    (migrations / "002_bad.sql").write_text("CREATE TABLE b (id TEXT PRIMARY KEY);", encoding="utf-8")
    assert store_db.apply_migrations(conn, migrations) == [2]
    conn.close()


# 4 -------------------------------------------------------------------------
def test_split_statements_handles_semicolons_in_comments_and_strings() -> None:
    sql = """
    -- 行注释里有个分号 ;
    CREATE TABLE a (id TEXT); /* 块注释里也有 ; */
    INSERT INTO a VALUES (';');
    CREATE TABLE b (id TEXT);
    -- 结尾注释，不应被当成未结束语句 ;
    """
    statements = store_db.split_statements(sql)
    assert len(statements) == 3
    assert statements[2].strip().startswith("CREATE TABLE b")


def test_split_statements_rejects_unterminated_tail() -> None:
    with pytest.raises(store_db.MigrationError, match="未结束"):
        store_db.split_statements("CREATE TABLE a (id TEXT)")


# 5 -------------------------------------------------------------------------
def test_idempotency_key_is_scoped_by_attempt(client: TestClient) -> None:
    a1 = client.post("/api/v1/attempts", json={"question": {**QUESTION, "stem": "s1"}, "student_answer": "2"})
    a2 = client.post("/api/v1/attempts", json={"question": {**QUESTION, "stem": "s2"}, "student_answer": "3"})
    assert a1.status_code == a2.status_code == 201
    id1, id2 = a1.json()["attempt_id"], a2.json()["attempt_id"]

    first = client.post(f"/api/v1/attempts/{id1}/diagnose", headers={"Idempotency-Key": "K"})
    assert first.status_code == 202

    # 同一把钥匙开另一扇门 -> 409，而不是返回别人的 task
    conflict = client.post(f"/api/v1/attempts/{id2}/diagnose", headers={"Idempotency-Key": "K"})
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_key_conflict"

    # 同 attempt 同 key 仍然是 200 幂等
    again = client.post(f"/api/v1/attempts/{id1}/diagnose", headers={"Idempotency-Key": "K"})
    assert again.status_code == 200
    assert again.json()["task_id"] == first.json()["task_id"]


# 6 -------------------------------------------------------------------------
def test_duplicate_attempt_rejected_but_different_answer_allowed(client: TestClient) -> None:
    question = client.post("/api/v1/questions", json=QUESTION).json()
    body = {"question_id": question["id"], "student_answer": "e-1"}

    assert client.post("/api/v1/attempts", json=body).status_code == 201
    duplicate = client.post("/api/v1/attempts", json=body)
    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == "duplicate_attempt"

    # 一题多错答是合法用法（FR-1.3）
    assert client.post("/api/v1/attempts", json={**body, "student_answer": "e"}).status_code == 201


def test_duplicate_attempt_maps_to_row_level_error_for_import(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    question = repo.create_question(course="c", stem="s", correct_answer="1")
    repo.create_attempt(question_id=question["id"], student_answer="9")
    with pytest.raises(sqlite3.IntegrityError) as excinfo:
        repo.create_attempt(question_id=question["id"], student_answer="9")
    assert map_integrity_error(excinfo.value) == "duplicate_attempt"


# 7 -------------------------------------------------------------------------
def test_status_and_terminal_cannot_contradict(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    question = repo.create_question(course="c", stem="s", correct_answer="1")
    attempt = repo.create_attempt(question_id=question["id"], student_answer="2")

    with pytest.raises(sqlite3.IntegrityError, match="ck_tasks_terminal"):
        conn.execute(
            "INSERT INTO tasks (id, attempt_id, status, terminal, created_at, updated_at)"
            " VALUES ('t-bad', ?, 'queued', 'diagnosed', 't', 't')",
            (attempt["id"],),
        )


# 8 -------------------------------------------------------------------------
def test_terminal_state_is_irreversible(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    question = repo.create_question(course="c", stem="s", correct_answer="1")
    attempt = repo.create_attempt(question_id=question["id"], student_answer="2")
    task = repo.create_task(attempt_id=attempt["id"])

    running = repo.start_task(task["id"], run_id="r-1")
    assert running["status"] == "running" and running["terminal"] is None

    done = repo.finish_task(task["id"], "diagnosed")
    assert done["status"] == done["terminal"] == "diagnosed"

    with pytest.raises(RepositoryError, match="非法"):
        repo.start_task(task["id"])
    with pytest.raises(RepositoryError, match="非法"):
        repo.finish_task(task["id"], "canceled")


# 9 -------------------------------------------------------------------------
def test_child_tables_still_reference_attempts_after_rebuild(conn: sqlite3.Connection) -> None:
    """§3.2 的关键回归：表重建若"先改名旧表"，子表外键会被改写成 attempts_old。"""
    for child in ("tasks", "diagnoses", "eval_items", "gold_labels"):
        targets = {row["table"] for row in conn.execute(f"PRAGMA foreign_key_list({child})")}
        assert "attempts" in targets, f"{child} 的外键没指向 attempts：{targets}"
        assert not any(t.startswith("attempts_") for t in targets), f"{child} 的外键悬空：{targets}"


# 10 ------------------------------------------------------------------------
def test_gold_label_requires_explicit_student_is_correct(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    question = repo.create_question(course="c", stem="s", correct_answer="1")
    attempt = repo.create_attempt(question_id=question["id"], student_answer="2")

    with pytest.raises(RepositoryError, match="student_is_correct"):
        repo.create_gold_label(
            attempt_id=attempt["id"],
            error_type_major="无错",
            labeler="me",
            student_is_correct=None,
            answer_key_correct=True,
        )

    label = repo.create_gold_label(
        attempt_id=attempt["id"],
        error_type_major="无错",
        labeler="me",
        student_is_correct=True,
        answer_key_correct=False,
    )
    assert label["student_is_correct"] == 1
    assert label["answer_key_correct"] == 0


# 11 ------------------------------------------------------------------------
def test_500_does_not_leak_exception_and_carries_request_id(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "boom.db")

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail: sk-do-not-leak")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/boom", headers={"X-Request-Id": "req-test-1"})

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["message"] == INTERNAL_ERROR_MESSAGE
    assert body["request_id"] == "req-test-1"
    assert "sk-do-not-leak" not in response.text
    assert "RuntimeError" not in response.text


def test_request_id_header_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.headers["X-Request-Id"].startswith("req-")


# 12 ------------------------------------------------------------------------
def test_in_memory_database_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match=MEMORY_DB_ERROR.split("：")[0]):
        create_app(db_path=":memory:")


# 13 ------------------------------------------------------------------------
def test_db_version_does_not_create_file(tmp_path: Path) -> None:
    from veridx.cli import main

    target = tmp_path / "nope" / "absent.db"
    assert main(["db", "version", "--db", str(target)]) == 0
    assert not target.exists()
    assert not target.parent.exists()


# 14 ------------------------------------------------------------------------
def test_health_degrades_when_real_model_has_no_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "LLM_MODE", "real")
    monkeypatch.setattr(config, "API_KEY", "")
    body = client.get("/api/v1/health").json()
    assert body["status"] == "degraded"
    assert body["provider"]["reachable"] is None

    monkeypatch.setattr(config, "LLM_MODE", "mock")
    assert client.get("/api/v1/health").json()["status"] == "ok"


# 15 ------------------------------------------------------------------------
def test_stage_event_accepts_canceled_status() -> None:
    event = StageEventData(stage="diagnose", status="canceled", ms=12, seq=3)
    assert event.status == "canceled"
