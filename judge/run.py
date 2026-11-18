"""judge 主脚本：读 eval 报告 → 取 agent 的错因主张 → 调一次 Kimi → 写判分结果。

**它不参与诊断、不 import veridx、不进 `src/`。** 只读三样东西：
报告 JSON、报告头里 `db` 指向的 sqlite（只读，取 `error_type`/`root_cause`）、人工标注 CSV。

用法：
    python judge/run.py --selftest
    python judge/run.py --report eval/reports/x.json [--labels judge/labels_100.csv] [--out j/r.json]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from prompts import CLASSES, SYSTEM, build_user, prompt_hash

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "judge" / "reports"
SCHEMA = "judge-v1"
#: 少于这么多条不打准确率（"n=1 的准确率"是最容易骗人的东西）
MIN_N_FOR_ACCURACY = 30
#: κ 门槛：低于它标 `uncalibrated`，只作内部参考
KAPPA_THRESHOLD = 0.6

Classifier = Callable[[dict[str, str]], str | None]


class JudgeUnavailable(RuntimeError):
    """这条没判成——**责任在我们或链路，不在模型的判断力**。

    两种来源：① 模型连续不可用（429 / 5xx / 超时）；② 输出被 `max_tokens` 截断。
    **必须与 `unparsed` 分开**：把限流或预算不足记成 `unparsed`，等于把"我没测到"
    伪装成"模型不合格"。
    """


# ---------------------------------------------------------------------------
# 凭据：自己读 .env 的 6 行，不 import veridx
# ---------------------------------------------------------------------------
def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, _, value = text.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _pick(env: dict[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return value
    return ""


def resolve_judge(env: dict[str, str]) -> dict[str, str]:
    """`JUDGE_*` → `SECOND_*` → 遗留别名 `*_KIMI`（方案 v1.1 §4）。"""
    return {
        "model": _pick(env, "JUDGE_MODEL", "SECOND_MODEL", "MODEL_KIMI"),
        "base_url": _pick(env, "JUDGE_BASE_URL", "SECOND_BASE_URL", "BASE_KIMI_URL"),
        "api_key": _pick(env, "JUDGE_API_KEY", "SECOND_API_KEY", "API_KIMI_KEY"),
    }


# ---------------------------------------------------------------------------
# 一次模型调用
# ---------------------------------------------------------------------------
def _parse_json(text: str) -> dict[str, Any] | None:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if "```" in body[3:] else body[3:]
        body = body.removeprefix("json").strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(body[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_class(body: dict[str, Any]) -> str | None:
    """响应体 → 类目名 或 None（模型答了但不在 6 类里）。

    `finish_reason == "length"` 是**我们的预算没给够**，不是模型答歪了 → 抛
    `JudgeUnavailable`，落到 `error` 桶，绝不计入 `unparsed`。
    """
    choice = (body.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if choice.get("finish_reason") == "length":
        used = ((body.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
        raise JudgeUnavailable(f"输出被 max_tokens 截断（reasoning_tokens={used}），调大预算")
    parsed = _parse_json(content)
    if parsed is None:
        return None
    name = str(parsed.get("class") or "").strip()
    return name if name in CLASSES else None


def make_classifier(settings: dict[str, str], *, timeout: float = 90.0, attempts: int = 4) -> Classifier:
    """返回 `classifier(row) -> 类目 or None`。

    `None` = 模型确实输出了 6 类之外的东西；连续不可用则抛 `JudgeUnavailable`。
    """
    import httpx

    if not settings["model"] or not settings["api_key"] or not settings["base_url"]:
        raise SystemExit("判分模型的凭据不全：需要 JUDGE_MODEL/SECOND_MODEL/MODEL_KIMI 三者之一")

    url = settings["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings['api_key']}", "Content-Type": "application/json"}

    def classify(row: dict[str, str]) -> str | None:
        payload: dict[str, Any] = {
            "model": settings["model"],
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": build_user(
                        stem=row["stem"],
                        student=row["student"],
                        teacher=row.get("teacher", ""),
                        claim=row["claim"],
                    ),
                },
            ],
            # kimi-k3.1-pro 只接受 temperature=1；别的值会 400（下面会去掉重发一次）
            "temperature": 1,
            "response_format": {"type": "json_object"},
            # 推理模型先烧 reasoning_tokens 再吐 content；实测 kimi 单条能烧到 1497，
            # 给少了 content 直接是空串（会被记成 error，不会污染 unparsed）
            "max_tokens": 4000,
        }
        last = "unknown"
        for attempt in range(attempts):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}"
                time.sleep(2 + 3 * attempt)
                continue
            if response.status_code == 400 and "temperature" in response.text.lower():
                payload.pop("temperature", None)
                continue
            if response.status_code >= 500 or response.status_code == 429:
                # 429 是"引擎过载"，不是"我们请求写错了" —— 退避重试，耗尽后**抛**
                last = f"http_{response.status_code}"
                time.sleep(2 + 3 * attempt)
                continue
            if response.status_code >= 400:
                raise SystemExit(f"判分模型返回 {response.status_code}：{response.text[:200]}")
            body = response.json()
            return _extract_class(body)
        raise JudgeUnavailable(f"判分模型连续 {attempts} 次不可用（{last}）")

    return classify


# ---------------------------------------------------------------------------
# 取数：报告 + sqlite + 标注
# ---------------------------------------------------------------------------
def read_claim(conn: sqlite3.Connection, run_id: str) -> str | None:
    """agent 的归因主张 = `error_type` + `root_cause`（**只有结论，没有推理过程**）。"""
    row = conn.execute("SELECT error_type, root_cause FROM diagnoses WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    error_type, root_cause = (row[0] or "").strip(), (row[1] or "").strip()
    if not error_type and not root_cause:
        return None
    return f"错误大类：{error_type}；根因：{root_cause or '（未给出）'}"


def load_suite(path: Path) -> dict[str, dict[str, str]]:
    suite: dict[str, dict[str, str]] = {}
    if not path.exists():
        return suite
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        suite[str(row["id"])] = {
            "stem": str(row.get("stem") or ""),
            "student": str(row.get("student") or ""),
        }
    return suite


def load_labels(path: Path | None) -> dict[str, str]:
    """`id → gold_class`。空值不算标注；不在 6 类里的值直接报错（不静默忽略）。"""
    if path is None or not path.exists():
        return {}
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gold = (row.get("gold_class") or "").strip()
            if not gold:
                continue
            if gold not in CLASSES:
                raise SystemExit(f"{path.name} 里 `{row.get('id')}` 的 gold_class 非法：{gold}")
            labels[str(row["id"])] = gold
    return labels


def load_teacher(path: Path) -> dict[str, str]:
    """老师的原话在待标注 CSV 的 `teacher_confusion` 列（adapt.py 导出的）。"""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(row["id"]): str(row.get("teacher_confusion") or "")
            for row in csv.DictReader(handle)
            if row.get("id")
        }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """judge 与人工在**同一批行**上的一致性。退化情形（期望一致率=1）返回 None。"""
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(1 for left, right in pairs if left == right) / total
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for left, right in pairs:
        left_counts[left] = left_counts.get(left, 0) + 1
        right_counts[right] = right_counts.get(right, 0) + 1
    expected = sum((left_counts[k] / total) * (right_counts.get(k, 0) / total) for k in left_counts)
    if expected >= 1:
        return None
    return round((observed - expected) / (1 - expected), 4)


def summarize(items: list[dict[str, Any]], *, diagnosed_total: int) -> dict[str, Any]:
    counted = [item for item in items if item["verdict"] in {"correct", "wrong"}]
    abstain = [item for item in items if item["verdict"] == "abstain"]
    unparsed = [item for item in items if item["verdict"] == "unparsed"]
    unlabeled = [item for item in items if item["verdict"] == "unlabeled"]
    errored = [item for item in items if item["verdict"] == "error"]

    correct = sum(1 for item in counted if item["verdict"] == "correct")
    accuracy = round(correct / len(counted), 4) if len(counted) >= MIN_N_FOR_ACCURACY else None

    pairs = [(item["judge"], item["gold"]) for item in counted]
    kappa = cohen_kappa(pairs) if len(pairs) >= MIN_N_FOR_ACCURACY else None
    calibrated = kappa is not None and kappa >= KAPPA_THRESHOLD

    confusion: dict[str, dict[str, int]] = {}
    for item in counted:
        confusion.setdefault(item["gold"], {})
        confusion[item["gold"]][item["judge"]] = confusion[item["gold"]].get(item["judge"], 0) + 1

    return {
        "n_items": len(items),
        "n_diagnosed": diagnosed_total,
        "n_scored": len(counted),
        "n_abstain": len(abstain),
        "n_unparsed": len(unparsed),
        "n_unlabeled": len(unlabeled),
        "n_error": len(errored),
        "accuracy_strict": accuracy,
        "calibration": {
            "kappa": kappa,
            "n": len(pairs),
            "status": "calibrated" if calibrated else "uncalibrated",
        },
        # 分母为什么这么小，必须能直接读出来
        "coverage_note": (
            f"分母 = agent 走到 diagnosed 且已人工标注的条数（{len(counted)}/{len(items)}）；"
            f"abstain {len(abstain)} 条不计入，未标注 {len(unlabeled)} 条不计入，"
            f"未判成 {len(errored)} 条不计入。"
            + ("" if accuracy is not None else f" n < {MIN_N_FOR_ACCURACY}，准确率不出数（null）。")
        ),
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def classify_verdict(*, terminal: str | None, has_claim: bool, judged: str | None, gold: str | None) -> str:
    """判分口径的唯一实现（`run_judge` 与 `--selftest` 共用，避免测试测副本）。"""
    # 没走到 diagnosed = agent 明确没下结论 → abstain，**不算错**
    if terminal != "diagnosed" or not has_claim:
        return "abstain"
    if judged is None:
        return "unparsed"
    if gold is None:
        return "unlabeled"
    return "correct" if judged == gold else "wrong"


def run_judge(
    *,
    report_path: Path,
    labels_path: Path | None,
    out_path: Path | None,
    classify: Classifier,
    suite_path: Path,
    labels_csv: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = report.get("items") or []
    if limit is not None:
        items = items[:limit]

    db_path = Path(str(report.get("db") or ""))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    if not db_path.exists():
        raise SystemExit(f"报告头里的 db 不存在：{db_path}")

    suite = load_suite(suite_path)
    labels = load_labels(labels_path)
    teacher = load_teacher(labels_csv)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out_items: list[dict[str, Any]] = []
    try:
        for item in items:
            item_id = str(item["id"])
            base = suite.get(item_id, {})
            record: dict[str, Any] = {
                "id": item_id,
                "k_index": item.get("k_index", 0),
                "run_id": item.get("run_id"),
                "terminal": item.get("terminal"),
                "gold": labels.get(item_id),
                "judge": None,
            }
            # ① 没走到 diagnosed = agent 明确没下结论 → abstain，**不算错**
            claim = read_claim(conn, str(item.get("run_id"))) if item.get("terminal") == "diagnosed" else None
            record["claim"] = claim
            judged: str | None = None
            if claim is not None:
                try:
                    judged = classify(
                        {
                            "stem": base.get("stem", ""),
                            "student": base.get("student", ""),
                            "teacher": teacher.get(item_id, ""),
                            "claim": claim,
                        }
                    )
                except JudgeUnavailable as exc:
                    # 没判成 ≠ 判错。分开计数，绝不混进 unparsed。
                    record["verdict"] = "error"
                    record["note"] = str(exc)
                    out_items.append(record)
                    print(f"  {item_id}: error（{exc}）")
                    continue
            verdict = classify_verdict(
                terminal=item.get("terminal"),
                has_claim=claim is not None,
                judged=judged,
                gold=record["gold"],
            )
            record["verdict"] = verdict
            record["judge"] = judged if verdict in {"correct", "wrong", "unlabeled"} else None
            if verdict == "abstain":
                record["note"] = (
                    "no_diagnosis_row"
                    if item.get("terminal") == "diagnosed"
                    else f"terminal={item.get('terminal')}"
                )
            elif verdict == "unparsed":
                record["note"] = "模型输出不在 6 个类目内"
            out_items.append(record)
            print(f"  {item_id}: {record['verdict']}（judge={record['judge']} gold={record['gold']}）")
    finally:
        conn.close()

    diagnosed_total = sum(1 for item in items if item.get("terminal") == "diagnosed")
    summary = summarize(out_items, diagnosed_total=diagnosed_total)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "judge_model": getattr(classify, "model", None) or "unknown",
        "prompt_hash": prompt_hash(),
        "source_report": str(report_path.relative_to(ROOT)).replace("\\", "/")
        if report_path.is_relative_to(ROOT)
        else str(report_path),
        "dataset_hash": report.get("dataset_hash"),
        "min_n_for_accuracy": MIN_N_FOR_ACCURACY,
        "t2": summary,
        "items": out_items,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {out_path}")
    return payload


# ---------------------------------------------------------------------------
# 反向验证（方案 §8 的三条断言）
# ---------------------------------------------------------------------------
def selftest() -> int:
    """用**注入的分类器**测判分逻辑本身，不联网。任何一条不过就退出码非 0。"""
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    # 一个"永远把主张里的类目原样判出来"的分类器：模拟满分类模型，
    # 这样测到的是**判分逻辑**（比对/弃权/门槛），不是模型能力。
    def echo_classifier(row: dict[str, str]) -> str | None:
        for name in CLASSES:
            if name in row["claim"]:
                return name
        return None

    def build(claim: str, gold: str | None, terminal: str = "diagnosed") -> dict[str, Any]:
        """走**生产代码那一个** `classify_verdict`，不再抄一份判定逻辑。"""
        judged = echo_classifier({"claim": claim, "stem": "", "student": "", "teacher": ""})
        return {
            "id": "x",
            "verdict": classify_verdict(terminal=terminal, has_claim=True, judged=judged, gold=gold),
            "judge": judged,
            "gold": gold,
        }

    # ① 归因与金标准一致 → 判对（基线）
    check("① 一致 → correct", build("错误大类：运算错误", "运算错误")["verdict"] == "correct")

    # ② 语义近似但不同类 → 必须判错（"运算错误" ↔ "概念混淆"）
    near = build("错误大类：运算错误", "概念混淆")
    check("② 近似但不同类 → wrong", near["verdict"] == "wrong", str(near))

    # ③ 随机类目 → 必须判错
    random_like = build("错误大类：单位与量纲", "运算错误")
    check("③ 随机类目 → wrong", random_like["verdict"] == "wrong", str(random_like))

    # ④ agent 明确拒答 → abstain，**不算错**
    abstained = build("", "运算错误", terminal="escalated")
    check("④ escalated → abstain", abstained["verdict"] == "abstain", str(abstained))

    # ⑤ n < 30 时准确率必须为 null（不许出"n=1 的准确率"）
    few = summarize([build("错误大类：运算错误", "运算错误")] * 5, diagnosed_total=5)
    check("⑤ n<30 → accuracy_strict is null", few["accuracy_strict"] is None, str(few["accuracy_strict"]))

    # ⑥ 分母正常时准确率算得对，且 abstain 不进分母
    many = [build("错误大类：运算错误", "运算错误")] * 30 + [build("", "运算错误", terminal="escalated")]
    ok_summary = summarize(many, diagnosed_total=30)
    check("⑥ n>=30 → accuracy 正确", ok_summary["accuracy_strict"] == 1.0, str(ok_summary["accuracy_strict"]))
    check("⑥ abstain 不进分母", ok_summary["n_scored"] == 30 and ok_summary["n_abstain"] == 1)

    # ⑦ 模型吐了 6 类之外的词 → unparsed，不许悄悄兜底成"无法归类"
    check("⑦ 类目之外 → unparsed", build("错误大类：粗心", "运算错误")["verdict"] == "unparsed")

    # ⑧ 模型不可用（429/超时）或预算截断 → error，**绝不**记成 unparsed
    errored = summarize(
        [{"id": "x", "verdict": "error", "judge": None, "gold": "运算错误"}], diagnosed_total=1
    )
    check("⑧ 不可用 → error 单独计数", errored["n_error"] == 1 and errored["n_unparsed"] == 0)
    check("⑧ error 不进分母", errored["n_scored"] == 0 and errored["accuracy_strict"] is None)

    # ⑨ 截断（reasoning 吃光 max_tokens）≠ 模型答歪
    truncated = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
    try:
        _extract_class(truncated)
    except JudgeUnavailable:
        check("⑨ 截断 → JudgeUnavailable（不是 unparsed）", True)
    else:
        check("⑨ 截断 → JudgeUnavailable（不是 unparsed）", False)
    empty_stop = {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
    check("⑨ 模型停住但空答 → unparsed", _extract_class(empty_stop) is None)

    # ⑩ 还没人工标注 ≠ 判错，也 ≠ 进分母
    unlabeled = build("错误大类：运算错误", None)
    check("⑩ 未标注 → unlabeled", unlabeled["verdict"] == "unlabeled")
    unlabeled_summary = summarize([unlabeled], diagnosed_total=1)
    check(
        "⑩ 未标注不进分母",
        unlabeled_summary["n_scored"] == 0
        and unlabeled_summary["n_unlabeled"] == 1
        and unlabeled_summary["accuracy_strict"] is None,
    )

    # ⑪ κ 低于门槛 → uncalibrated（"看着有数"不代表"能对外报"）
    low = summarize(
        [build(f"错误大类：{CLASSES[i % 6]}", "运算错误") for i in range(MIN_N_FOR_ACCURACY)],
        diagnosed_total=MIN_N_FOR_ACCURACY,
    )
    check("⑪ κ<0.6 → uncalibrated", low["calibration"]["status"] == "uncalibrated")

    if failures:
        print(f"\n{len(failures)} 条断言失败：{failures}", file=sys.stderr)
        return 1
    print("\nselftest 全绿（判分器恒判对是这套东西最可能的失败形态）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="VeriDx 外部判分器（T2 错因归类）")
    parser.add_argument("--report", help="eval/reports/*.json")
    parser.add_argument("--labels", default=None, help="人工标注 CSV（含 gold_class 列）")
    parser.add_argument("--labels-csv", default=None, help="adapt.py 导出的待标注 CSV（取老师原话用）")
    parser.add_argument("--suite", default=None, help="suite（取题干与作答用；默认由报告头推断）")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.report:
        parser.print_help()
        return 2

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    if not report_path.exists():
        print(f"报告不存在：{report_path}", file=sys.stderr)
        return 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    suite_path = Path(args.suite) if args.suite else ROOT / str(report.get("suite") or "")
    if not suite_path.is_absolute():
        suite_path = ROOT / suite_path
    labels_csv = (
        Path(args.labels_csv)
        if args.labels_csv
        else ROOT / "judge" / f"labels_{report.get('limit') or report.get('cases') or 100}_todo.csv"
    )
    if not labels_csv.is_absolute():
        labels_csv = ROOT / labels_csv

    out_path = Path(args.out) if args.out else REPORTS_DIR / (report_path.stem + ".judge.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    settings = resolve_judge(_load_env(ROOT / ".env"))
    print(f"判分模型：{settings['model'] or '(未配置)'}  报告：{report_path.name}")
    classify = make_classifier(settings)
    classify.model = settings["model"]  # type: ignore[attr-defined]
    payload = run_judge(
        report_path=report_path,
        labels_path=Path(args.labels) if args.labels else None,
        out_path=out_path,
        classify=classify,
        suite_path=suite_path,
        labels_csv=labels_csv,
        limit=args.limit,
    )
    t2 = payload["t2"]
    print(
        f"\nitems {t2['n_items']}  diagnosed {t2['n_diagnosed']}  scored {t2['n_scored']}  "
        f"abstain {t2['n_abstain']}  unparsed {t2['n_unparsed']}  "
        f"unlabeled {t2['n_unlabeled']}  error {t2['n_error']}"
    )
    print(
        f"accuracy_strict = {t2['accuracy_strict']}   κ = {t2['calibration']['kappa']}"
        f"（{t2['calibration']['status']}）"
    )
    print(t2["coverage_note"])
    return 0


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
