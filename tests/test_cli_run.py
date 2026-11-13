"""A-2：`veridx run` / `veridx replay` 两个壳的验收（离线、确定性）。

这两条命令的价值只有一句：**一条命令就能把一次 run 跑完并留下可回放的 trace**。
所以这里断言的是"跑完 + 落盘 + 打得开"，不是"诊断质量"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veridx import cli, cli_ops


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_run_case_from_dataset_produces_terminal_and_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """G1 第一条验收：`veridx run --case q007 --mock`。"""
    db = tmp_path / "cli.db"
    code, out = _run(capsys, "run", "--case", "q007", "--mock", "--db", str(db))

    assert code == 0
    assert "terminal:" in out and "run_id:" in out
    assert "evidence:" in out
    assert "ev_1" in out, f"必须打印非空证据表：{out}"
    assert "veridx replay" in out


def test_run_writes_json_when_asked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_path = tmp_path / "result.json"
    code, _ = _run(
        capsys,
        "run",
        "--case",
        "q001",
        "--mock",
        "--db",
        str(tmp_path / "cli.db"),
        "--out",
        str(out_path),
    )
    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["terminal"] in {"diagnosed", "escalated", "uncertain"}
    assert payload["run_id"].startswith("r-")
    assert payload["evidence"], "落盘的 JSON 里必须有证据"


def test_run_unknown_case_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(capsys, "run", "--case", "no-such-case", "--mock", "--db", str(tmp_path / "cli.db"))
    assert code == 1
    assert "找不到题" in out


def test_run_persists_failed_instead_of_swallowing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """provider 炸了：Runner 会把它落成 `failed/internal_error`，命令照常打印并给 run_id。"""

    class Boom:
        model = "boom"

        def describe(self) -> dict:
            return {"provider": "boom", "model": "boom", "mode": "mock"}

        async def chat(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(cli_ops, "_build_pair", lambda mode, blind_answer=None: (Boom(), None))
    db = tmp_path / "cli.db"
    code, out = _run(capsys, "run", "--case", "q007", "--mock", "--db", str(db))

    assert code == 0, f"命令本身跑完了（终态是 failed，不是命令坏了）：{out}"
    assert "terminal:   failed" in out
    assert "internal_error" in out
    assert "run_id:" in out, "failed 也必须给 run_id —— 否则没法回放"
    conn = __import__("sqlite3").connect(db)
    rows = conn.execute("SELECT node, status, reason_code FROM trace_events").fetchall()
    conn.close()
    assert rows, "异常路径也必须落盘 trace"
    assert any(row[2] == "internal_error" for row in rows)


def test_run_reports_failed_when_the_runner_itself_blows_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """连 Runner 都挂了（不是模型错）：命令**必须**打印 failed + 落一条 submit/fail 事件。

    这是"异常路径不许静默"的那条路径 —— 界面上只留一句死话，事后无从排查。
    """

    class ExplodingRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, spec: object) -> object:
            raise RuntimeError("runner exploded before any node ran")

    monkeypatch.setattr(cli_ops, "Runner", ExplodingRunner)
    db = tmp_path / "cli.db"
    code, out = _run(capsys, "run", "--case", "q007", "--mock", "--db", str(db))

    assert code == 1
    assert "terminal:   failed" in out
    assert "internal_error" in out
    assert "runner exploded" in out
    conn = __import__("sqlite3").connect(db)
    rows = conn.execute("SELECT node, status, reason_code FROM trace_events").fetchall()
    conn.close()
    assert rows, "连 Runner 挂了也必须留下 trace"
    assert any(row[0] == "submit" and row[2] == "internal_error" for row in rows)


def test_replay_prints_timeline_and_single_step(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "cli.db")
    _code, out = _run(capsys, "run", "--case", "q001", "--mock", "--db", db)
    run_id = next(line.split(":", 1)[1].strip() for line in out.splitlines() if line.startswith("run_id:"))

    code, timeline = _run(capsys, "replay", run_id, "--db", db)
    assert code == 0
    assert "parse" in timeline and "guard" in timeline

    code, single = _run(capsys, "replay", run_id, "--db", db, "--at", "1")
    assert code == 0
    assert "seq=1" in single and "input/detail" in single


def test_replay_diff_reports_facts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "cli.db")
    _c, first = _run(capsys, "run", "--case", "q001", "--mock", "--db", db)
    _c, second = _run(capsys, "run", "--case", "q001", "--mock", "--db", db)
    a = next(line.split(":", 1)[1].strip() for line in first.splitlines() if line.startswith("run_id:"))
    b = next(line.split(":", 1)[1].strip() for line in second.splitlines() if line.startswith("run_id:"))

    code, out = _run(capsys, "replay", a, "--db", db, "--diff", b)
    assert code == 0
    assert "events:" in out and "final:" in out


def test_replay_unknown_run_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "cli.db")
    _run(capsys, "run", "--case", "q001", "--mock", "--db", db)
    code, out = _run(capsys, "replay", "r-nope", "--db", db)
    assert code == 1
    assert "找不到 run" in out
