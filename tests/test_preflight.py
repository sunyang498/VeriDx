"""C 部分验收（方案 §5 / §9.1 第 4 条）：跑批**之前**就知道坏在哪。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veridx import cli
from veridx.eval.preflight import BLOCKING_RATIO, run_preflight


def _write(path: Path, rows: list[dict | str]) -> Path:
    lines = [row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_good_suite_has_no_blocker(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "good.jsonl",
        [
            {"id": "q1", "stem": "计算 1/2+1/3", "correct": "5/6", "student": "5/7"},
            {"id": "q2", "stem": "求 2^10", "correct": "1024", "student": "20"},
        ],
    )
    report = run_preflight(path)
    assert report.rows == 2
    assert report.blocking is False
    assert report.line_errors == []
    assert report.parse["parse_ok"] == 4, "两侧都要预检（分母 = 题数 × 2）"


def test_structurally_broken_rows_are_reported_line_by_line(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.jsonl",
        [
            {"id": "q1", "stem": "题", "correct": "1", "student": "2"},
            "{这不是 JSON",
            {"id": "q3", "stem": "题"},
            '{"id": "q4", "stem": "题", "correct": "1", "student": 2}',
        ],
    )
    report = run_preflight(path)
    assert report.rows == 1
    assert report.blocking is True
    joined = " ".join(report.line_errors)
    assert "第 2 行" in joined
    assert "缺字段" in joined
    assert "必须是字符串" in joined


def test_implicit_multiplication_is_not_an_unparseable_gap(tmp_path: Path) -> None:
    """审查 §3.2 的盲区：`3\\pi` / `2x` 曾被 preflight 记成 `parse_ok`，于是"没问题"。

    现在它们要么被判成 `parser_normalized`（修好了），要么被判成 `need_llm`/`unparseable`（会阻断）。
    """
    path = _write(
        tmp_path / "implicit.jsonl",
        [
            {"id": "q1", "stem": "求 3π 的值", "correct": r"3\pi", "student": "2x"},
            {"id": "q2", "stem": "求值", "correct": r"\frac{1}{2}x", "student": r"\sqrt{2}x"},
        ],
    )
    report = run_preflight(path)
    assert report.parse["parse_ok"] == 0, f"隐式乘法不该再被记成 parse_ok：{report.parse}"
    assert report.parse["parser_normalized"] == 4, report.parse
    assert report.blocking is False


def test_unparseable_ratio_above_threshold_blocks(tmp_path: Path) -> None:
    rows = [{"id": f"q{i}", "stem": "题", "correct": "1", "student": "2"} for i in range(10)]
    rows[0]["correct"] = r"\weird{x}"  # 残留 TeX → 需要模型
    rows[1]["student"] = r"\text{cm}"  # 排版盒子非数值
    path = _write(tmp_path / "risky.jsonl", rows)
    report = run_preflight(path)
    assert report.parse["need_llm"] >= 1
    # 2/20 = 10% 恰好不超过阈值 → 不阻断；再加两条就阻断
    assert report.blocking is False
    rows[2]["correct"] = r"\weird{x}"
    rows[3]["correct"] = r"\weird{x}"
    rows[4]["correct"] = r"\weird{x}"
    path = _write(tmp_path / "risky2.jsonl", rows)
    report2 = run_preflight(path)
    assert report2.blocking is True
    assert any("不可解析" in blocker for blocker in report2.blockers)


def test_td4_risk_is_summed_not_just_counted(tmp_path: Path) -> None:
    """TD4 的代价是**概率**：3 条 `7` 的期望失败数 ≈ 1.93，而不是"3 条受影响"。"""
    path = _write(
        tmp_path / "td4.jsonl",
        [{"id": f"q{i}", "stem": "计算 2+3 的值", "correct": "5", "student": "7"} for i in range(3)],
    )
    report = run_preflight(path)
    assert report.td4["affected_items"] == 3
    assert report.td4["expected_failures"] == pytest.approx(1.93, abs=0.02)


def test_duplicates_and_slices(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "dup.jsonl",
        [
            {
                "id": "q1",
                "stem": "同一道题",
                "correct": "1",
                "student": "2",
                "student_steps": ["a"],
                "error_type": "计算表达错误",
            },
            {"id": "q2", "stem": "同一道题 ", "correct": "3", "student": "4"},
            {"id": "q3", "stem": "Calculate x", "correct": "5", "student": "6"},
        ],
    )
    report = run_preflight(path)
    assert report.duplicates == 1, "去空白后同 hash 记一次重复"
    assert report.slices["has_step_truth"] == 1
    assert report.slices["has_error_label"] == 1
    assert report.slices["zh"] == 2 and report.slices["en"] == 1


def test_cli_check_writes_preflight_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(
        tmp_path / "bad.jsonl",
        [
            {"id": "q1", "stem": "题", "correct": "1", "student": "2"},
            "{坏行",
        ],
    )
    out = tmp_path / "pre.json"
    code = cli.main(["dataset", "check", "--suite", str(path), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 1, "有阻断项必须退出码非 0（CI 才能拦住坏数据）"
    assert "阻断" in printed
    payload = json.loads(out.read_text(encoding="utf-8"))
    for key in ("suite", "dataset_hash", "rows", "blocking", "parse", "normalize_risk", "td4", "slices"):
        assert key in payload, key
    assert payload["blocking"] is True


def test_cli_check_on_clean_suite_exits_zero(tmp_path: Path, capsys) -> None:
    path = _write(tmp_path / "ok.jsonl", [{"id": "q1", "stem": "题", "correct": "1", "student": "2"}])
    code = cli.main(["dataset", "check", "--suite", str(path), "--out", str(tmp_path / "p.json")])
    assert code == 0
    assert "通过" in capsys.readouterr().out


def test_blocking_ratio_is_a_named_constant() -> None:
    """阈值必须是命名常量：藏在字面量里会让"为什么这条被阻断"没法回答。"""
    assert 0 < BLOCKING_RATIO < 1
