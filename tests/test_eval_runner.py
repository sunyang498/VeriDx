"""实施单 §3 的验收：`veridx eval` 能跑、能落盘、能复用、能归因，而且**分母为 0 时不撒谎**。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veridx import cli, config
from veridx.eval.report import BINDINGS, TERMINALS, summarize, to_markdown
from veridx.eval.runner import ItemResult, dataset_hash, load_suite


def _item(**overrides: object) -> ItemResult:
    base: dict[str, object] = {
        "id": "q1",
        "k_index": 0,
        "run_id": "r-1",
        "terminal": "diagnosed",
        "reason_code": None,
        "binding": "pass",
        "accepted": 2,
        "rejected": 0,
        "steps": 3,
        "steps_usable": True,
        "tokens_in": 100,
        "tokens_out": 40,
        "llm_calls": 3,
        "latency_ms": 500,
        "collision_prob": 0.0,
        "failure_attribution": None,
        "notes": [],
    }
    base.update(overrides)
    return ItemResult(**base)  # type: ignore[arg-type]


# --- 数据集 ---------------------------------------------------------------
def test_load_suite_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text(
        '{"id":"q1","stem":"s","correct":"1","student":"2"}\n'
        "\n"
        '{"id":"q2","stem":"s2","correct":"3","student":"4","student_steps":["a","b"]}\n',
        encoding="utf-8",
    )
    cases = load_suite(path)
    assert [case.id for case in cases] == ["q1", "q2"]
    assert cases[1].student_steps == ["a", "b"]


def test_load_suite_rejects_bad_lines_instead_of_skipping(tmp_path: Path) -> None:
    """静默跳过坏行会让分母悄悄变小 —— 那正是"指标变好看、结论变假"。"""
    path = tmp_path / "s.jsonl"
    path.write_text('{"id":"q1","stem":"s","correct":"1","student":"2"}\n{不是 JSON\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        load_suite(path)


def test_dataset_hash_is_content_addressed(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"id":"q1","stem":"s","correct":"1","student":"2"}\n', encoding="utf-8")
    b.write_bytes(a.read_bytes())
    assert dataset_hash(a) == dataset_hash(b)
    assert len(dataset_hash(a)) == 16


# --- 汇总口径 -------------------------------------------------------------
def test_empty_dataset_summary_is_null_not_zero() -> None:
    """空数据集：分母 0 → `null`。输出 0 会被平均进总分，把"没测"变成"很差"。"""
    summary = summarize([])
    assert summary["explain_usable"] == {"value": None, "denominator": 0}
    assert summary["latency"]["mean"] is None and summary["latency"]["p95"] is None
    assert summary["tokens"]["in_mean"] is None and summary["tokens"]["denominator"] == 0
    assert summary["questions"]["completed"] == 0
    for name in TERMINALS:
        assert summary["terminals"][name] == 0
    assert summary["terminals"]["denominator"] == 0


def test_summary_binding_tiers_are_three_not_two() -> None:
    """反向验证的靶子：`partial` 必须单独成档，不能并进 pass。"""
    items = [
        _item(id="a", binding="pass"),
        _item(id="b", binding="partial", accepted=1, rejected=1),
        _item(id="c", binding="fail_after_retry", terminal="escalated", accepted=0, rejected=1),
    ]
    summary = summarize(items)
    assert summary["binding"]["pass"] == 1
    assert summary["binding"]["partial"] == 1
    assert summary["binding"]["fail_after_retry"] == 1
    assert summary["binding"]["denominator"] == 3
    assert set(BINDINGS) == {"pass", "partial", "fail_after_retry", "skipped"}


def test_explain_usable_denominator_is_diagnosed() -> None:
    items = [
        _item(id="a", steps_usable=True),
        _item(id="b", steps_usable=False),
        _item(id="c", terminal="uncertain", steps=0, steps_usable=False),
    ]
    summary = summarize(items)
    assert summary["explain_usable"] == {"value": 1, "denominator": 2}


def test_failure_attribution_groups_by_exception_type() -> None:
    items = [
        _item(
            id="a", terminal="failed", reason_code="internal_error", failure_attribution="ValueError: 泄露"
        ),
        _item(
            id="b", terminal="failed", reason_code="internal_error", failure_attribution="ValueError: 泄露"
        ),
        _item(
            id="c", terminal="failed", reason_code="internal_error", failure_attribution="RuntimeError: 炸了"
        ),
        _item(id="d"),
    ]
    summary = summarize(items)
    assert summary["failure_attribution"]["counts"] == {"ValueError": 2, "RuntimeError": 1}
    assert summary["failure_attribution"]["denominator"] == 3


def test_token_mean_uses_call_count_as_denominator() -> None:
    items = [_item(tokens_in=300, tokens_out=90, llm_calls=3)]
    summary = summarize(items)
    assert summary["tokens"]["in_total"] == 300
    assert summary["tokens"]["in_mean"] == pytest.approx(100.0)
    assert summary["tokens"]["denominator"] == 3


def test_markdown_carries_denominators() -> None:
    report = {
        "created_at": "2026-11-14T00:00:00+08:00",
        "suite": "datasets/x.jsonl",
        "dataset_hash": "abc123",
        "model": "deepseek-flash",
        "mode": "mock",
        "prompt_version": "pv_diagnose_v2",
        "git_commit": "abc-dirty",
        "arms": ["A"],
        "k": 1,
        "limit": None,
        "items": [_item().to_dict(), _item(id="b", binding="partial", accepted=1, rejected=1).to_dict()],
    }
    report["summary"] = summarize([_item(), _item(id="b", binding="partial", accepted=1, rejected=1)])
    text = to_markdown(report)
    assert "1/2 完成" in text or "2 / 2" in text
    assert "`partial` | 1 | 2" in text
    assert "分母" in text


# --- 端到端（真跑批，mock 模式）-------------------------------------------
@pytest.fixture()
def mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_MODE", "mock")


def test_eval_command_runs_and_writes_json_and_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mock_mode: None
) -> None:
    suite = tmp_path / "s.jsonl"
    suite.write_text(
        '{"id":"q001","stem":"计算 1/2 + 1/3，结果写成最简分数。","correct":"5/6","student":"5/7"}\n'
        '{"id":"q002","stem":"求 2^10 的值。","correct":"1024","student":"20"}\n'
        '{"id":"q003","stem":"求 3^2 的值。","correct":"9","student":"6"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    code = cli.main(
        [
            "eval",
            "--suite",
            str(suite),
            "--mock",
            "--limit",
            "2",
            "--out",
            str(out),
            "--db",
            str(tmp_path / "eval.db"),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    # 结果文件头：dataset_hash / model / prompt_version / git commit / arms / k
    for key in ("dataset_hash", "model", "prompt_version", "git_commit", "arms", "k"):
        assert payload[key], key
    assert len(payload["items"]) == 2
    assert payload["summary"]["terminals"]["denominator"] == 2
    assert out.with_suffix(".md").exists()

    item = payload["items"][0]
    assert item["run_id"].startswith("r-")
    assert item["tokens_out"] > 0, "token 必须分开记（实施单 §2）"
    assert item["latency_ms"] is None or item["latency_ms"] > 0
    assert "collision_prob" in item


def test_eval_reuses_finished_items_on_rerun(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mock_mode: None
) -> None:
    """中断后重跑**不重复烧额度**：已完成 (id,k) 直接复用。"""
    suite = tmp_path / "s.jsonl"
    suite.write_text(
        '{"id":"q001","stem":"计算 1/2 + 1/3，结果写成最简分数。","correct":"5/6","student":"5/7"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    args = [
        "eval",
        "--suite",
        str(suite),
        "--mock",
        "--out",
        str(out),
        "--db",
        str(tmp_path / "eval.db"),
    ]
    assert cli.main(args) == 0
    first = json.loads(out.read_text(encoding="utf-8"))["items"][0]["run_id"]

    capsys.readouterr()
    assert cli.main(args) == 0
    printed = capsys.readouterr().out
    second = json.loads(out.read_text(encoding="utf-8"))["items"][0]["run_id"]
    assert "复用" in printed
    assert second == first, "复用应当保留原来的 run_id（不重跑）"


def test_eval_empty_dataset_returns_zero_and_says_no_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mock_mode: None
) -> None:
    suite = tmp_path / "empty.jsonl"
    suite.write_text("", encoding="utf-8")
    out = tmp_path / "report.json"
    code = cli.main(
        ["eval", "--suite", str(suite), "--mock", "--out", str(out), "--db", str(tmp_path / "e.db")]
    )
    assert code == 0
    summary = json.loads(out.read_text(encoding="utf-8"))["summary"]
    assert summary["questions"]["value"] == 0
    assert summary["explain_usable"]["value"] is None
    assert "无数据" in capsys.readouterr().out


def test_eval_unknown_arm_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mock_mode: None
) -> None:
    suite = tmp_path / "s.jsonl"
    suite.write_text('{"id":"q1","stem":"s","correct":"1","student":"2"}\n', encoding="utf-8")
    code = cli.main(
        [
            "eval",
            "--suite",
            str(suite),
            "--arms",
            "B",
            "--mock",
            "--out",
            str(tmp_path / "r.json"),
            "--db",
            str(tmp_path / "e.db"),
        ]
    )
    assert code == 2
    assert "暂不支持" in capsys.readouterr().out


def test_eval_failed_item_gets_an_attribution(tmp_path: Path, mock_mode: None) -> None:
    """有 `failed` 样本时，`failure_attribution` 里必须有内容（TD4 会伪装成 internal_error）。"""
    from veridx.eval import runner as eval_runner

    case = eval_runner.EvalCase(id="q-td4", stem="计算 2+3 的值", correct="5", student="7")
    items = []

    async def _go() -> None:
        items.extend(
            await eval_runner.run_suite([case], db_path=tmp_path / "e.db", mode="mock", explain=True, k=1)
        )

    # 固定 nonce 让它必然撞上（TD4 的特征表现），从而必然 failed
    import asyncio

    from veridx.prompts import blind_solve as blind_prompt

    original = blind_prompt.new_nonce
    blind_prompt.new_nonce = lambda: "0123456789abcdef"  # type: ignore[assignment]
    try:
        asyncio.run(_go())
    finally:
        blind_prompt.new_nonce = original  # type: ignore[assignment]

    item = items[0]
    assert item.terminal == "failed"
    assert item.reason_code == "internal_error"
    assert item.failure_attribution and "ValueError" in item.failure_attribution
    assert item.binding == "skipped", "没走到 guard 就不许假装 pass"
