"""把逐题明细汇成 6 个数字（实施单 §3）。

三条硬纪律（写在这里，防止以后被"优化"掉）：

1. **分母为 0 时输出 `null`，不许输出 0** —— 输出 0 会被平均进总分，把"没测"变成"很差"；
2. **每个比例旁边必须带分母**；
3. **`failed` 必须逐条归因**（原因见 `eval.runner._failure_attribution`）。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from veridx.guard.evidence import BINDING_FAIL, BINDING_PARTIAL, BINDING_PASS

__all__ = ["TERMINALS", "build_report", "summarize", "to_markdown"]

TERMINALS: tuple[str, ...] = (
    "diagnosed",
    "escalated",
    "uncertain",
    "needs_clarification",
    "failed",
    "canceled",
)

BINDINGS: tuple[str, ...] = (BINDING_PASS, BINDING_PARTIAL, BINDING_FAIL, "skipped")


def _ratio(value: float | int | None, denominator: int) -> dict[str, Any]:
    """一个比例 = `{value, denominator}`；分母 0 → `value: null`（**不是 0**）。"""
    if denominator <= 0:
        return {"value": None, "denominator": 0}
    return {"value": value, "denominator": denominator}


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _p95(values: Sequence[float]) -> float | None:
    """nearest-rank p95：样本少时也不插值（插值会造出一个不存在的观测值）。"""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(math.ceil(0.95 * len(ordered)) - 1, 0)
    return ordered[rank]


def summarize(items: Sequence[Any]) -> dict[str, Any]:
    total = len(items)
    terminals = Counter(item.terminal for item in items)
    bindings = Counter(item.binding for item in items)

    diagnosed = terminals.get("diagnosed", 0)
    usable = sum(1 for item in items if item.terminal == "diagnosed" and item.steps_usable)

    failed = [item for item in items if item.terminal == "failed"]
    attribution = Counter((item.failure_attribution or "unknown").split(":", 1)[0].strip() for item in failed)

    # 分母 = **调用数**（不是题数）：一次 run 可能有 3 次调用，按题数平均会把成本摊薄
    calls = sum(item.llm_calls for item in items)
    latencies = [float(item.latency_ms) for item in items if item.latency_ms is not None]

    return {
        "questions": {
            "value": total,
            "denominator": total,
            # "跑成功"= 真的产出了终态（不是"诊断成功"）
            "completed": sum(1 for item in items if item.terminal),
        },
        "terminals": {
            **{name: terminals.get(name, 0) for name in TERMINALS},
            "denominator": total,
        },
        "binding": {
            **{name: bindings.get(name, 0) for name in BINDINGS},
            "denominator": total,
        },
        "explain_usable": _ratio(usable, diagnosed),
        "failure_attribution": {
            "counts": dict(attribution),
            "denominator": len(failed),
        },
        "tokens": {
            "in_total": sum(item.tokens_in for item in items),
            "out_total": sum(item.tokens_out for item in items),
            # 分母 = **调用数**（实施单 §3）：按题数平均会把 3 次调用的成本摊薄成 1/3
            "in_mean": (sum(item.tokens_in for item in items) / calls) if calls else None,
            "out_mean": (sum(item.tokens_out for item in items) / calls) if calls else None,
            "denominator": calls,
        },
        "latency": {
            "mean": _mean(latencies),
            "p95": _p95(latencies),
            "denominator": len(latencies),
        },
    }


def build_report(items: Sequence[Any], *, header: dict[str, Any]) -> dict[str, Any]:
    return {
        **header,
        "items": [item.to_dict() for item in items],
        "summary": summarize(items),
    }


def to_markdown(report: dict[str, Any]) -> str:
    """人能读的那一版：先写"跑的是什么"，再写 6 个数字，最后是逐题表。"""
    header = report
    summary = report["summary"]
    lines = [
        f"# 评测报告 {header.get('created_at', '')}",
        "",
        "| 跑的是什么 | 值 |",
        "|---|---|",
        f"| 数据集 | `{header.get('suite', '')}`（hash `{header.get('dataset_hash', '')}`）|",
        f"| 模型 | `{header.get('model', '')}`（mode `{header.get('mode', '')}`）|",
        f"| 提示词 | `{header.get('prompt_version', '')}` |",
        f"| 代码 | `{header.get('git_commit', '')}` |",
        f"| 臂 / k / 限制 | `{header.get('arms', '')}` / {header.get('k', 1)} / {header.get('limit')} |",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 | 分母 |",
        "|---|---|---|",
    ]
    q = summary["questions"]
    lines.append(f"| 题数（完成 / 总） | {q['completed']} / {q['value']} | {q['denominator']} |")
    term = summary["terminals"]
    for name in TERMINALS:
        lines.append(f"| 终态 `{name}` | {term[name]} | {term['denominator']} |")
    bind = summary["binding"]
    for name in BINDINGS:
        lines.append(f"| 证据绑定 `{name}` | {bind[name]} | {bind['denominator']} |")
    usable = summary["explain_usable"]
    usable_value = "—" if usable["value"] is None else f"{usable['value']}"
    lines.append(f"| 讲解可用 | {usable_value} | {usable['denominator']}（diagnosed）|")
    tokens = summary["tokens"]
    tin = "—" if tokens["in_mean"] is None else f"{tokens['in_mean']:.1f}"
    tout = "—" if tokens["out_mean"] is None else f"{tokens['out_mean']:.1f}"
    lines.append(
        f"| token in（mean / total） | {tin} / {tokens['in_total']} | {tokens['denominator']} 次调用 |"
    )
    lines.append(
        f"| token out（mean / total） | {tout} / {tokens['out_total']} | {tokens['denominator']} 次调用 |"
    )
    lat = summary["latency"]
    lat_mean = "—" if lat["mean"] is None else f"{lat['mean']:.0f}ms"
    lat_p95 = "—" if lat["p95"] is None else f"{lat['p95']:.0f}ms"
    lines.append(f"| latency（mean / p95） | {lat_mean} / {lat_p95} | {lat['denominator']} |")

    attribution = summary["failure_attribution"]
    lines += ["", "## 失败归因（`failed` 必须逐条说清）", ""]
    if attribution["counts"]:
        lines += ["| 异常类型 | 次数 |", "|---|---|"]
        for name, count in sorted(attribution["counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| `{name}` | {count} |")
        lines.append(f"| **合计** | {attribution['denominator']} |")
    else:
        lines.append(f"无（分母 {attribution['denominator']}）")

    lines += [
        "",
        "## 逐题明细",
        "",
        "| id | k | 终态 | reason | 绑定 | 断言 | 步数 | 可用 | token in/out | 耗时 | 误伤率 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in report["items"]:
        claims = (
            "—"
            if item["accepted"] is None
            else f"{item['accepted']}/{item['accepted'] + (item['rejected'] or 0)}"
        )
        latency = "—" if item["latency_ms"] is None else f"{item['latency_ms']}ms"
        lines.append(
            f"| `{item['id']}` | {item['k_index']} | {item['terminal']} | {item['reason_code'] or '—'} "
            f"| {item['binding']} | {claims} | {item['steps']} | {'是' if item['steps_usable'] else '否'} "
            f"| {item['tokens_in']}/{item['tokens_out']} | {latency} | {item['collision_prob']:.3f} |"
        )
    return "\n".join(lines) + "\n"
