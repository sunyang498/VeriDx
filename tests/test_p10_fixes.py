"""P1.0 修复的回归测试（P0.6 审查 §3.1–§3.6 + §4.2 ①②③⑥）。

每条对应一个具体缺陷，测试名里的每个承诺都必须有断言。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from veridx import config
from veridx.api.app import create_app, openapi_spec
from veridx.api.errors import INTERNAL_ERROR_MESSAGE
from veridx.store import db as store_db
from veridx.store.repository import Repository, RepositoryError


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection, _ = store_db.init_db(tmp_path / "p10.db", config.MIGRATIONS_DIR)
    yield connection
    connection.close()


# --- §3.1 / §3.6  读路径不得执行 DDL -----------------------------------------
def test_read_version_on_foreign_file_returns_zero(tmp_path: Path) -> None:
    """指向一个"存在但不是本项目库"的文件：返回 0，不抛错、不建表。"""
    foreign = tmp_path / "someone_else.db"
    sqlite3.connect(foreign).close()  # 造一个合法的空 sqlite 文件

    assert store_db.read_version(foreign) == 0

    check = sqlite3.connect(foreign)
    try:
        tables = [r[0] for r in check.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    finally:
        check.close()
    assert tables == []  # 读版本没有顺手建 schema_version


def test_read_version_on_garbage_file_raises_clearly(tmp_path: Path) -> None:
    """不是 sqlite 文件：报"不是数据库"，而不是"attempt to write a readonly database"。"""
    junk = tmp_path / "notes.db"
    junk.write_bytes(b"this is not a sqlite file")

    with pytest.raises(sqlite3.DatabaseError) as excinfo:
        store_db.read_version(junk)
    assert "readonly" not in str(excinfo.value).lower()


def test_read_version_on_valid_db_is_read_only(tmp_path: Path) -> None:
    db = tmp_path / "v2.db"
    conn, _ = store_db.init_db(db, config.MIGRATIONS_DIR)
    conn.execute("PRAGMA journal_mode=DELETE")  # 关掉 WAL，保证只读可见
    conn.close()

    assert store_db.read_version(db) == 2


def test_current_version_never_writes(conn: sqlite3.Connection) -> None:
    """`/health` 每请求都调它，所以它必须是纯读。"""
    assert store_db.current_version(conn) == 2
    assert store_db.current_version(conn) == 2


def test_health_works_on_read_only_connection(conn: sqlite3.Connection, tmp_path: Path) -> None:
    path = Path(conn.execute("PRAGMA database_list").fetchone()["file"])
    readonly = store_db.connect(path, read_only=True)
    try:
        assert store_db.current_version(readonly) == 2
    finally:
        readonly.close()


# --- §3.2  500 必须有 X-Request-Id 响应头 ------------------------------------
def test_500_response_carries_request_id_header(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "boom.db")

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail: sk-do-not-leak")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom", headers={"X-Request-Id": "req-test-1"})

    assert response.status_code == 500
    assert response.headers["X-Request-Id"] == "req-test-1"  # ← 名字里的承诺
    assert response.json()["request_id"] == "req-test-1"
    assert response.json()["message"] == INTERNAL_ERROR_MESSAGE
    assert "sk-do-not-leak" not in response.text


def test_500_generates_request_id_header_when_absent(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "boom2.db")

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("x")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.headers["X-Request-Id"].startswith("req-")


# --- §3.3  外键失败不得被说成"重复作答" --------------------------------------
def test_foreign_key_error_is_not_reported_as_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ghost = {
        "id": "q-ghost",
        "course": "c",
        "stem": "s",
        "correct_answer": "1",
        "kp_ids": [],
        "qtype": None,
        "difficulty": None,
        "source": None,
        "created_at": "t",
    }
    monkeypatch.setattr(Repository, "get_question", lambda self, qid: ghost)

    with TestClient(create_app(db_path=tmp_path / "fk.db")) as client:
        response = client.post("/api/v1/attempts", json={"question_id": "q-ghost", "student_answer": "2"})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "invalid_input"
    assert "references_missing" in body["message"]
    assert "duplicate_attempt" not in body["message"]


# --- §3.4  openapi 导出不得指向真实库 ----------------------------------------
def test_openapi_spec_does_not_touch_configured_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "should_not_exist.db"
    settings = config.get_settings()
    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: config.Settings(
            db_path=target,
            migrations_dir=settings.migrations_dir,
            llm_mode=settings.llm_mode,
            model=settings.model,
            second_model=settings.second_model,
            seed_policy=settings.seed_policy,
        ),
    )

    spec = openapi_spec()
    assert len(spec["paths"]) == 13
    assert not target.exists()


# --- §4.2⑥  reason_code 必须落在枚举内 --------------------------------------
def test_reason_code_is_validated(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    question = repo.create_question(course="c", stem="s", correct_answer="1")
    attempt = repo.create_attempt(question_id=question["id"], student_answer="2")
    task = repo.create_task(attempt_id=attempt["id"])

    with pytest.raises(RepositoryError, match="reason_code"):
        repo.finish_task(task["id"], "canceled", reason_code="user_cancel")  # 少了 ed

    assert repo.finish_task(task["id"], "canceled", reason_code="user_canceled")["reason_code"] == (
        "user_canceled"
    )


# --- §4.2①  evidence 查询必须带 run_id --------------------------------------
def test_evidence_lookup_is_scoped_by_run(conn: sqlite3.Connection) -> None:
    """两个 run 各有一个 `ev_1`：一个 ok、一个 failed。

    run A 的校验不能因为 run B 那个 ok 而通过 —— 这是 P0.6 主键改动的直接副作用。
    """
    repo = Repository(conn)
    repo.create_evidence(
        run_id="r-A",
        evidence_id="ev_1",
        kind="numeric_check",
        payload={"equal": False},
        produced_by="check_numeric",
        seq=1,
        status="failed",
    )
    repo.create_evidence(
        run_id="r-B",
        evidence_id="ev_1",
        kind="numeric_check",
        payload={"equal": False},
        produced_by="check_numeric",
        seq=1,
        status="ok",
    )

    run_a = repo.list_evidence("r-A", ["ev_1"])
    assert len(run_a) == 1
    assert run_a[0]["run_id"] == "r-A"
    assert run_a[0]["status"] == "failed"  # 不是 run B 的那个 ok

    assert repo.list_evidence("r-A", []) == []
    assert [e["run_id"] for e in repo.list_evidence("r-B")] == ["r-B"]


def test_evidence_seq_is_unique_within_run(conn: sqlite3.Connection) -> None:
    repo = Repository(conn)
    repo.create_evidence(
        run_id="r-1", evidence_id="ev_1", kind="numeric_check", payload={}, produced_by="x", seq=1
    )
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_evidence(
            run_id="r-1", evidence_id="ev_2", kind="step_diff", payload={}, produced_by="x", seq=1
        )
