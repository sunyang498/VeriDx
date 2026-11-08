"""契约验收（P0.3 / P0.4 DoD）。

守的是三件事：
1. 13 个端点、6 类 SSE 事件**一个都不能少**（少了前端就要猜）
2. 文档 §8.4 的示例 JSON 能被模型**真正反序列化**（不是"看起来像"）
3. `TraceEvent`（内部）与 `TraceEventOut`（传输）**字段完全一致**（防止两套契约漂移）
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from pydantic import ValidationError

from veridx.api import schemas as S
from veridx.api.app import create_app
from veridx.cli import cmd_contract_export
from veridx.guard.claim import Claim, Evidence, EvidenceKind, confidence_cap
from veridx.taxonomy import MACRO_F1_CLASSES, ErrorType
from veridx.trace.event import NodeStatus, ReasonCode, TraceEvent

EXPECTED_ENDPOINTS: dict[str, set[str]] = {
    "/api/v1/questions": {"post"},
    "/api/v1/attempts": {"post"},
    "/api/v1/attempts/import": {"post"},
    "/api/v1/attempts/{attempt_id}/diagnose": {"post"},
    "/api/v1/tasks/{task_id}": {"get"},
    "/api/v1/tasks/{task_id}/stream": {"get"},
    "/api/v1/tasks/{task_id}/cancel": {"post"},
    "/api/v1/diagnoses/{diagnosis_id}": {"get"},
    "/api/v1/runs/{run_id}/replay": {"get"},
    "/api/v1/kp/weak": {"get"},
    "/api/v1/eval/runs": {"post"},
    "/api/v1/eval/runs/{eval_run_id}/report": {"get"},
    "/api/v1/health": {"get"},
}


@pytest.fixture(scope="module")
def spec() -> dict:
    return create_app(db_path=":memory:").openapi()


def test_all_13_endpoints_exist(spec: dict) -> None:
    assert len(EXPECTED_ENDPOINTS) == 13
    for path, methods in EXPECTED_ENDPOINTS.items():
        assert path in spec["paths"], f"缺少端点 {path}"
        for method in methods:
            assert method in spec["paths"][path], f"{path} 缺少方法 {method}"


def test_every_endpoint_has_response_schema(spec: dict) -> None:
    missing = []
    for path, methods in EXPECTED_ENDPOINTS.items():
        for method in methods:
            operation = spec["paths"][path][method]
            responses = operation.get("responses", {})
            if not any("content" in r for r in responses.values()):
                missing.append(f"{method.upper()} {path}")
    assert missing == []


def test_stream_endpoint_declares_sse_media_type(spec: dict) -> None:
    operation = spec["paths"]["/api/v1/tasks/{task_id}/stream"]["get"]
    assert "text/event-stream" in operation["responses"]["200"]["content"]


def test_openapi_artifact_is_in_sync() -> None:
    assert cmd_contract_export(out=None, check=True) == 0, "doc/openapi.json 与代码不一致，请重新导出"


def test_sse_six_events_roundtrip() -> None:
    assert {name.value for name in S.SSEEventName} == {"stage", "evidence", "claim", "token", "guard", "done"}

    samples = [
        ("stage", '{"stage":"blind_solve","status":"ok","ms":1840,"seq":2}'),
        (
            "evidence",
            '{"id":"ev_2","kind":"numeric_check","seq":3,"status":"ok",'
            '"payload":{"equal":false,"model_value":1.0,"student_value":1.718}}',
        ),
        ("claim", '{"field":"error_type","value":"程序性错误","evidence_ids":["ev_2"]}'),
        ("token", '{"text":"第 1 步：先求不定积分"}'),
        ("guard", '{"schema":"pass","evidence_binding":"pass","retries":1}'),
        (
            "done",
            '{"terminal":"diagnosed","reason_code":null,"run_id":"r-8f2c",'
            '"latency_ms":9420,"tokens":{"in":3120,"out":2000}}',
        ),
    ]
    for name, data in samples:
        frame = S.SSEFrame.parse(name, data)
        text = frame.to_text()
        assert text.startswith(f"event: {name}\n")
        assert text.endswith("\n\n")
        reparsed = S.SSEFrame.parse(name, text.split("data: ", 1)[1].strip())
        assert reparsed.event == frame.event


def test_done_event_requires_terminal_and_tokens() -> None:
    with pytest.raises(ValidationError):
        S.SSEFrame.parse("done", '{"terminal":"diagnosed","run_id":"r-1","latency_ms":1}')


def test_claim_event_rejects_empty_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        S.ClaimEventData(field="error_type", value="概念性错误", evidence_ids=[])


def test_diagnosis_example_from_doc_deserializes() -> None:
    """§8.4 示例 + §7.1 要求的审计字段 seq/produced_by（见 CHANGELOG C-09）。"""
    payload = {
        "id": "d-77a1",
        "attempt_id": "a-91b2",
        "run_id": "r-8f2c",
        "terminal": "diagnosed",
        "reason_code": None,
        "error_type": "程序性错误",
        "error_subtype": "分部积分符号错",
        "first_divergent_step": 2,
        "locator": "program",
        "root_cause": "分部积分时把 ∫u v' 写成 ∫u' v",
        "hypotheses": [],
        "kp_ids": ["kp_int_by_parts"],
        "confidence": 0.82,
        "needs_review": False,
        "claims": [
            {"field": "error_type", "value": "程序性错误", "evidence_ids": ["ev_2", "ev_3"], "seq": 4},
            {"field": "first_divergent_step", "value": "2", "evidence_ids": ["ev_3"], "seq": 5},
        ],
        "evidence": [
            {
                "id": "ev_2",
                "kind": "numeric_check",
                "status": "ok",
                "payload": {"model_value": 1.0, "student_value": 1.718, "equal": False},
                "produced_by": "check_numeric",
                "seq": 2,
            },
            {
                "id": "ev_3",
                "kind": "step_diff",
                "status": "ok",
                "payload": {"first_divergence": 2, "model_step": "(x-1)e^x", "student_step": "(x+1)e^x"},
                "produced_by": "step_diff",
                "seq": 3,
            },
        ],
        "steps": [
            {
                "seq": 1,
                "title": "分部积分",
                "content": "∫x e^x dx = (x-1)e^x + C",
                "key_point": "选 u=x",
                "targets_error": "符号错",
                "verifiable": True,
            }
        ],
        "similar_questions": [],
        "kps": [],
        "arm": "D",
        "prompt_version_id": "pv_diagnose_v7",
        "model": "glm-4.6",
        "temperature": 0.2,
        "seed": 7,
        "evidence_binding": "pass",
        "guard_retries": 1,
        "cascade_level": 0,
        "verify_agrees": True,
        "suspected_bad_answer_key": "none",
        "input_tokens": 3120,
        "output_tokens": 2000,
        "latency_ms": 9420,
        "created_at": "2026-11-08T15:22:31Z",
    }
    diagnosis = S.DiagnosisOut.model_validate(payload)
    assert diagnosis.error_type is ErrorType.PROCEDURAL
    assert diagnosis.claims[0].evidence_ids == ["ev_2", "ev_3"]
    assert diagnosis.model_dump(mode="json")["error_type"] == "程序性错误"


def test_trace_event_out_matches_internal_contract() -> None:
    internal_fields = {f.name for f in dataclasses.fields(TraceEvent)}
    assert set(S.TraceEventOut.model_fields) == internal_fields

    event = TraceEvent(
        run_id="r-1", node="diagnose", seq=2, start_ms=10, end_ms=20, status="ok", parent_seq=None
    )
    mirror = S.TraceEventOut.from_event(event)
    assert mirror.model_dump() == event.to_dict()


def test_trace_event_invariants() -> None:
    with pytest.raises(ValueError, match="seq"):
        TraceEvent(run_id="r-1", node="parse", seq=0, start_ms=0, status="ok")
    with pytest.raises(ValueError, match="parent_seq"):
        TraceEvent(run_id="r-1", node="parse", seq=2, parent_seq=2, start_ms=0, status="ok")
    with pytest.raises(ValueError, match="end_ms"):
        TraceEvent(run_id="r-1", node="parse", seq=1, start_ms=20, end_ms=10, status="ok")


def test_trace_status_and_reason_codes_are_stable() -> None:
    assert {s.value for s in NodeStatus} == {"ok", "retry", "fail", "skipped", "canceled"}
    assert "call_budget_exceeded" in {r.value for r in ReasonCode}
    assert "user_canceled" in {r.value for r in ReasonCode}


def test_evidence_and_claim_guards() -> None:
    with pytest.raises(ValidationError):
        Claim(field="root_cause", value="x", evidence_ids=[], seq=3)
    with pytest.raises(ValidationError):
        Evidence(id="ev-1", kind=EvidenceKind.STUDENT_TEXT, produced_by="parse", seq=1)
    with pytest.raises(ValidationError):
        Evidence(id="ev_1", kind=EvidenceKind.STUDENT_TEXT, produced_by="parse", seq=0)


def test_condition_check_caps_confidence() -> None:
    evidence = [
        Evidence(id="ev_1", kind=EvidenceKind.CONDITION_CHECK, produced_by="parse", seq=1),
        Evidence(id="ev_2", kind=EvidenceKind.STUDENT_TEXT, produced_by="parse", seq=2),
    ]
    assert confidence_cap(evidence) == 0.5
    assert confidence_cap([evidence[1]]) == 1.0


def test_terminals_are_mutually_exclusive_and_complete() -> None:
    assert S.BUSINESS_TERMINALS.isdisjoint(S.ENGINEERING_TERMINALS)
    assert S.BUSINESS_TERMINALS | S.ENGINEERING_TERMINALS == set(S.Terminal)
    assert len(S.Terminal) == 6


def test_macro_f1_uses_four_classes_only() -> None:
    assert len(MACRO_F1_CLASSES) == 4
    assert ErrorType.OTHER not in MACRO_F1_CLASSES


def test_openapi_is_valid_json_on_disk() -> None:
    from veridx import config

    spec = json.loads((config.DOC_DIR / "openapi.json").read_text(encoding="utf-8"))
    assert spec["info"]["title"] == "VeriDx"
    assert len(spec["paths"]) == 13
