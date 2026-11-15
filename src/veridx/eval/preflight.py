"""评测前检查（方案 §5）：**跑批之前**就告诉你"这 N 条里有几条是坏的、为什么坏"。

为什么它最值钱：它把"跑完之后总结失败"变成"**跑之前就知道会有多少条失败**"。
没有它，你会用一晚上跑出一张无法解释的表。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veridx.flow.blind_solve import collision_prob
from veridx.normalize.parser import NORMALIZER_VERSION, to_expr
from veridx.normalize.static import static_parse_ok

__all__ = ["PREFLIGHT_VERSION", "PreflightReport", "row_errors", "run_preflight"]

PREFLIGHT_VERSION = "preflight-v1"

#: C-6：不可解析 + 无法验证的比例超过这个数就判"阻断"
BLOCKING_RATIO = 0.10

_REQUIRED = ("id", "stem", "correct", "student")


def row_errors(raw: Any, lineno: int) -> list[str]:
    """C-1：逐行校验，**返回全部问题**（不中断、不跳过）。"""
    if not isinstance(raw, dict):
        return [f"第 {lineno} 行不是 JSON 对象"]
    errors = [f"第 {lineno} 行缺字段 `{key}`" for key in _REQUIRED if not raw.get(key)]
    for key in ("id", "stem", "correct", "student"):
        value = raw.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"第 {lineno} 行 `{key}` 必须是字符串")
    steps = raw.get("student_steps")
    if steps is not None and not isinstance(steps, list):
        errors.append(f"第 {lineno} 行 `student_steps` 必须是数组或 null")
    return errors


def _classify(value: str) -> str:
    """C-2：一格里只有一种结论，四个格子互斥且穷尽。

    **必须验到"沙箱能吃"这一层**（审查 §3.3）：只看 `to_expr().ok` 会把
    "解析器自己说成功"当成"沙箱能吃"，于是 `3\\pi` / `2x` 被记成 `parse_ok`，
    preflight 就会对着会掉出分母的数据说"没问题"。
    `to_expr` 现在已内置 `static_parse_ok`，这里再核一遍是**契约断言**：
    它一旦失效，preflight 必须当场把这类数据记成 `unparseable`，而不是继续报平安。
    """
    parsed = to_expr(value)
    if parsed.ok:
        if not static_parse_ok(parsed.text):  # 防御性断言：to_expr 的闸失效了
            return "unparseable"
        return "parser_normalized" if parsed.changed else "parse_ok"
    if not parsed.text or "\\" in parsed.text:
        return "unparseable"
    return "need_llm"


@dataclass
class PreflightReport:
    suite: str
    dataset_hash: str
    rows: int
    blocking: bool
    parse: dict[str, int]
    normalize_risk: dict[str, Any]
    td4: dict[str, Any]
    slices: dict[str, int]
    duplicates: int
    blockers: list[str] = field(default_factory=list)
    line_errors: list[str] = field(default_factory=list)
    normalizer_version: str = NORMALIZER_VERSION
    version: str = PREFLIGHT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "dataset_hash": self.dataset_hash,
            "rows": self.rows,
            "blocking": self.blocking,
            "parse": self.parse,
            "normalize_risk": self.normalize_risk,
            "td4": self.td4,
            "slices": self.slices,
            "duplicates": self.duplicates,
            "blockers": self.blockers,
            "line_errors": self.line_errors,
            "normalizer_version": self.normalizer_version,
            "version": self.version,
        }


def run_preflight(path: Path) -> PreflightReport:
    """跑一遍全部检查。**不抛异常**：坏数据要变成报告内容，而不是炸掉。"""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    rows = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("//")]

    parse_counts: Counter[str] = Counter(
        {"parse_ok": 0, "parser_normalized": 0, "need_llm": 0, "unparseable": 0}
    )
    line_errors: list[str] = []
    stems: list[str] = []
    collisions: list[dict[str, Any]] = []
    slices: Counter[str] = Counter()
    good_rows = 0

    for lineno, line in enumerate(rows, start=1):
        try:
            raw = json.loads(line)
        except ValueError as exc:
            line_errors.append(f"第 {lineno} 行不是合法 JSON：{exc}")
            continue
        errors = row_errors(raw, lineno)
        if errors:
            line_errors.extend(errors)
            continue
        good_rows += 1
        # 两侧都要预检：归一化是**相对的**，只归一化一侧会把误差直接变成"判错"
        parse_counts[_classify(str(raw["correct"]))] += 1
        parse_counts[_classify(str(raw["student"]))] += 1
        prob = collision_prob(str(raw["student"]), str(raw["stem"]))
        if prob > 0:
            collisions.append({"id": str(raw["id"]), "prob": round(prob, 6)})
        stems.append(" ".join(str(raw["stem"]).split()))
        slices[_lang_of(raw)] += 1
        if raw.get("student_steps"):
            slices["has_step_truth"] += 1
        if raw.get("error_type"):
            slices["has_error_label"] += 1
        if raw.get("source"):
            slices[f"source:{raw['source']}"] += 1

    pairs = good_rows * 2
    total_unusable = parse_counts["unparseable"]
    ratio = (total_unusable / pairs) if pairs else 0.0
    duplicates = len(stems) - len({hashlib.sha256(s.encode("utf-8")).hexdigest()[:16] for s in stems})

    blockers: list[str] = []
    if line_errors:
        blockers.append(f"结构性错误 {len(line_errors)} 条（逐行见 line_errors）")
    if pairs and ratio > BLOCKING_RATIO:
        blockers.append(
            f"不可解析 {total_unusable}/{pairs} = {ratio:.1%} > {BLOCKING_RATIO:.0%}："
            "先把归一化补齐，否则这些题会直接掉出评测分母"
        )

    return PreflightReport(
        suite=str(path),
        dataset_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        rows=good_rows,
        blocking=bool(blockers),
        parse=dict(parse_counts),
        normalize_risk={
            # 期望冲突数：模型与解析器都可能说错，这里只报"需要模型介入"的量
            "need_llm": parse_counts["need_llm"] + parse_counts["unparseable"],
            "unparseable_ratio": round(ratio, 4),
        },
        td4={
            "affected_items": len(collisions),
            "expected_failures": round(sum(item["prob"] for item in collisions), 4),
            "items": collisions[:20],
        },
        slices=dict(slices),
        duplicates=duplicates,
        blockers=blockers,
        line_errors=line_errors,
    )


def _lang_of(raw: dict[str, Any]) -> str:
    """极简判定：只为了分片统计（不是语言识别器）。"""
    text = str(raw.get("stem") or "")
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    has_latin = any("a" <= ch.lower() <= "z" for ch in text)
    if has_cjk and has_latin:
        return "mixed"
    return "zh" if has_cjk else "en"
