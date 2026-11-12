"""G0 门禁：空跑 e2e 桩。

覆盖 POST /attempts → 202 {task_id} → GET /tasks/{id} = queued 的完整动作，
外加幂等命中、取消幂等、统一错误码、以及"未实现端点明确 501 而不是假数据"。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from veridx import __version__, config
from veridx.api.app import create_app

QUESTION = {
    "course": "calculus",
    "stem": "计算 12*3 的值",
    "correct_answer": "36",
    "qtype": "numeric",
    "kp_ids": ["kp_int_by_parts"],
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """端到端 fixture：**强制 mock 模式**，绝不真打网络（scenario provider 跑得通全链路）。"""
    monkeypatch.setattr(config, "LLM_MODE", "mock")
    app = create_app(db_path=tmp_path / "e2e.db")
    with TestClient(app) as test_client:
        yield test_client


def _wait_terminal(client: TestClient, task_id: str, *, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/tasks/{task_id}").json()
        if body["terminal"]:
            return body
        time.sleep(0.05)
    raise AssertionError(f"任务未在 {timeout}s 内进入终态：{body}")


def _create_attempt(client: TestClient, student_answer: str = "e-1") -> dict:
    resp = client.post("/api/v1/attempts", json={"question": QUESTION, "student_answer": student_answer})
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_stub_path_end_to_end(client: TestClient) -> None:
    attempt = _create_attempt(client)
    assert attempt["attempt_id"].startswith("a-")
    assert attempt["question_id"].startswith("q-")
    assert len(attempt["fingerprint"]) == 16

    resp = client.post(
        f"/api/v1/attempts/{attempt['attempt_id']}/diagnose", headers={"Idempotency-Key": "idem-e2e"}
    )
    assert resp.status_code == 202, resp.text
    created = resp.json()
    assert created["status"] == "queued"
    assert created["idempotent_hit"] is False
    assert created["estimated_ms"] > 0
    task_id = created["task_id"]

    polled = client.get(f"/api/v1/tasks/{task_id}")
    assert polled.status_code == 200
    assert polled.json()["status"] in {"queued", "running", "diagnosed"}
    assert polled.json()["attempt_id"] == attempt["attempt_id"]

    done = _wait_terminal(client, task_id)
    assert done["terminal"] in {"diagnosed", "escalated", "uncertain"}
    assert done["reason_code"] is None if done["terminal"] == "diagnosed" else done["reason_code"]


def test_diagnose_produces_a_real_diagnosis(client: TestClient) -> None:
    attempt = _create_attempt(client, student_answer="35")
    task_id = client.post(f"/api/v1/attempts/{attempt['attempt_id']}/diagnose").json()["task_id"]
    done = _wait_terminal(client, task_id)
    assert done["diagnosis_id"], f"没有产出诊断：{done}"

    diagnosis = client.get(f"/api/v1/diagnoses/{done['diagnosis_id']}").json()
    assert diagnosis["terminal"] == done["terminal"]
    assert diagnosis["claims"], "诊断必须带 claims"
    assert diagnosis["evidence"], "诊断必须带证据"
    for claim in diagnosis["claims"]:
        assert claim["evidence_ids"], "每条断言都必须引用证据"
        assert claim["seq"] > 0
    assert diagnosis["steps"], "讲解步骤应存在（scenario provider 会给出 3 步）"
    assert all(step["targets_error"] for step in diagnosis["steps"])


def test_diagnose_is_idempotent(client: TestClient) -> None:
    attempt = _create_attempt(client)
    first = client.post(
        f"/api/v1/attempts/{attempt['attempt_id']}/diagnose", headers={"Idempotency-Key": "same-key"}
    )
    second = client.post(
        f"/api/v1/attempts/{attempt['attempt_id']}/diagnose", headers={"Idempotency-Key": "same-key"}
    )
    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["idempotent_hit"] is True
    assert second.json()["task_id"] == first.json()["task_id"]


def test_cancel_is_idempotent(client: TestClient) -> None:
    attempt = _create_attempt(client)
    task_id = client.post(f"/api/v1/attempts/{attempt['attempt_id']}/diagnose").json()["task_id"]

    canceled = client.post(f"/api/v1/tasks/{task_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] in {"canceled", "diagnosed", "escalated", "uncertain"}

    again = client.post(f"/api/v1/tasks/{task_id}/cancel")
    assert again.status_code == 200
    assert again.json()["already_terminal"] is True

    task = client.get(f"/api/v1/tasks/{task_id}").json()
    assert task["terminal"] is not None
    assert task["reason_code"] is not None if task["terminal"] != "diagnosed" else True


def test_health_reports_schema_version(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["version"] == __version__
    assert body["schema_version"] == 3
    assert body["provider"]["reachable"] is None  # 健康检查不做网络探测


def test_unknown_task_returns_404_envelope(client: TestClient) -> None:
    resp = client.get("/api/v1/tasks/t-does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_invalid_input_returns_400_envelope(client: TestClient) -> None:
    resp = client.post("/api/v1/attempts", json={"student_answer": ""})
    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_input"


def test_attempt_requires_exactly_one_question_source(client: TestClient) -> None:
    both = client.post(
        "/api/v1/attempts", json={"question": QUESTION, "question_id": "q-1", "student_answer": "1"}
    )
    assert both.status_code == 400
    assert both.json()["code"] == "invalid_input"


def test_unimplemented_endpoints_do_not_fake_results(client: TestClient) -> None:
    stubs = [
        ("post", "/api/v1/attempts/import", {"format": "jsonl", "content": "{}"}),
        ("get", "/api/v1/runs/r-1/replay", None),
        ("get", "/api/v1/kp/weak", None),
        ("post", "/api/v1/eval/runs", {"arm": "D", "split": "heldout", "k": 1}),
        ("get", "/api/v1/eval/runs/e-1/report", None),
    ]
    for method, path, body in stubs:
        resp = getattr(client, method)(path) if body is None else getattr(client, method)(path, json=body)
        assert resp.status_code == 501, f"{method} {path} -> {resp.status_code}"
        assert resp.json()["code"] == "not_implemented"
        assert "将在 P" in resp.json()["message"]


def test_p21_endpoints_are_real_now(client: TestClient) -> None:
    """P2.1 已落地：伪任务 / 伪诊断必须报 404，而不是 501 或假数据。"""
    assert client.get("/api/v1/tasks/t-1/stream").status_code == 404
    assert client.get("/api/v1/diagnoses/d-1").status_code == 404
