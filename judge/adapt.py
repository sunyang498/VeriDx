"""MathDial → 评测 suite + 待标注 CSV（judge 的输入域那一半）。

**只做切字段，全程不调模型**：正确答案绝不能经过提取器（ADR-021）。

口径（照实施单 2026-11-15 §1，不自创）：
* `stem`    = `question`，**原样**
* `correct` = `ground_truth` 的**最后一行**（丢掉前面整篇推理）
* `student` = `student_incorrect_solution` 的**最后一行**
* 超长（>256）**跳过并计数**，绝不截断（截断就是改值）

分层抽样（TD4 的失败率强依赖答案长度，取前 100 条会偏）：
`len==1 : 19` / `len==2 : 37` / `len>=3 : 44`，固定种子 42。

用法：
    python judge/adapt.py --limit 100
    python judge/adapt.py --limit 100 --break-correct first   # 只用于反向验证
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "datasets" / "robustness" / "_raw" / "mathdial_test.jsonl"
OUT_DIR = ROOT / "judge" / "data"

#: 与 `config.SANDBOX_MAX_EXPR_LEN` 同值；这里不 import veridx（judge 是外部进程）
MAX_ANSWER_CHARS = 256
SEED = 42
#: 三层各抽多少（合计 100）
STRATA_PLAN = {"1": 19, "2": 37, "3+": 44}


def last_line(text: object) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def first_line(text: object) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[0] if lines else ""


def stratum_of(answer: str) -> str:
    if len(answer) == 1:
        return "1"
    if len(answer) == 2:
        return "2"
    return "3+"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(rows: list[dict], *, limit: int, break_correct: str | None) -> tuple[list[dict], dict]:
    buckets: dict[str, list[dict]] = {"1": [], "2": [], "3+": []}
    skipped_too_long = 0
    dropped: list[str] = []
    # MathDial 的 `qid` **不唯一**：同一道题挂了多个学生作答（599 行 → 394 个 qid）。
    # 实施单要求「id 必须唯一」，格式是 `md-{qid}` —— 冲突时按**规则**办：
    # 首次用 `md-{qid}`，重复的追加 `-2` / `-3`，既不丢数据也不撞号。
    seen_qid: dict[str, int] = {}

    for row in rows:
        qid = str(row.get("qid") or "").strip()
        student = last_line(row.get("student_incorrect_solution"))
        correct = (
            first_line(row["ground_truth"]) if break_correct == "first" else last_line(row["ground_truth"])
        )
        stem = str(row.get("question") or "").strip()
        if not qid or not stem or not student or not correct:
            dropped.append(qid or "<no-qid>")
            continue
        if len(student) > MAX_ANSWER_CHARS or len(correct) > MAX_ANSWER_CHARS:
            skipped_too_long += 1
            continue
        seen_qid[qid] = seen_qid.get(qid, 0) + 1
        ordinal = seen_qid[qid]
        buckets[stratum_of(student)].append(
            {
                "id": f"md-{qid}" if ordinal == 1 else f"md-{qid}-{ordinal}",
                "stem": stem,  # 原样，不清理
                "correct": correct,
                "student": student,
                "_qid": qid,
                "_confusion": str(row.get("teacher_described_confusion") or "").strip(),
            }
        )

    # 分层抽：某层不够就按 1 → 2 → 3+ 的顺序补（manifest 里如实记实际数）
    rng = random.Random(SEED)
    picked: dict[str, list[dict]] = {}
    shortage = 0
    for name, want in STRATA_PLAN.items():
        pool = buckets[name]
        take = min(want, len(pool))
        shortage += want - take
        picked[name] = rng.sample(pool, take)

    if shortage:
        for name in ("1", "2", "3+"):
            if shortage <= 0:
                break
            taken_ids = {item["id"] for item in picked[name]}
            extra_pool = [item for item in buckets[name] if item["id"] not in taken_ids]
            take = min(shortage, len(extra_pool))
            if take:
                picked[name].extend(rng.sample(extra_pool, take))
                shortage -= take

    selected = [item for name in ("1", "2", "3+") for item in picked[name]][:limit]
    return selected, {
        "total_rows": len(rows),
        "unique_qids": len(seen_qid),
        "skipped_too_long": skipped_too_long,
        "dropped_rows": dropped[:50],
        "seed": SEED,
        "strata": {name: len(picked[name]) for name in STRATA_PLAN},
        "strata_pool": {name: len(pool) for name, pool in buckets.items()},
        "strata_shortage": shortage,
    }


def write_suite(selected: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for item in selected:
            row = {key: item[key] for key in ("id", "stem", "correct", "student")}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_labels(selected: list[dict], out: Path) -> None:
    """给人工勾选：`gold_class` 由负责人填，`judge_class`/`agree` 留给 run.py。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "stem", "teacher_confusion", "gold_class", "judge_class", "agree"])
        for item in selected:
            writer.writerow([item["id"], item["stem"], item["_confusion"], "", "", ""])


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial → suite + 待标注 CSV")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default=None, help="suite 路径（默认 judge/data/mathdial_<N>.jsonl）")
    parser.add_argument(
        "--break-correct",
        choices=["first"],
        default=None,
        help="**只用于反向验证**：把 correct 取成第一行（解题过程首句），必须让 dataset check 报错",
    )
    parser.add_argument("--labels-out", default=None)
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"源数据不存在：{SOURCE}（先跑 datasets/robustness/fetch.py）", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected, stats = build(rows, limit=args.limit, break_correct=args.break_correct)

    suffix = "-broken" if args.break_correct else ""
    suite_path = Path(args.out) if args.out else OUT_DIR / f"mathdial_{args.limit}{suffix}.jsonl"
    if not suite_path.is_absolute():
        suite_path = ROOT / suite_path
    labels_path = (
        Path(args.labels_out) if args.labels_out else ROOT / "judge" / f"labels_{args.limit}{suffix}_todo.csv"
    )
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path

    write_suite(selected, suite_path)
    write_labels(selected, labels_path)

    if args.break_correct is None:
        manifest = {
            "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha256_of(SOURCE),
            **stats,
            "rows_written": len(selected),
            "dataset_hash": None,  # 由 `veridx dataset check` 产出后回填
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        manifest_path = suite_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"manifest -> {manifest_path}")

    print(f"suite   -> {suite_path}（{len(selected)} 行）")
    print(f"labels  -> {labels_path}")
    print(
        f"分层     : {stats['strata']}（池 {stats['strata_pool']}，"
        f"超长跳过 {stats['skipped_too_long']}，短缺 {stats['strata_shortage']}）"
    )
    rel = str(suite_path.relative_to(ROOT)).replace("\\", "/")
    print(f"下一步：veridx dataset check --suite {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
