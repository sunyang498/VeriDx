"""`veridx eval` 命令的壳：读数据集 → 跑批 → 落 JSON + Markdown。

**评测走 CLI，不进 REST 契约**（C-32 已裁定）。这里只做编排，跑批与汇总分别在
`eval/runner.py` 与 `eval/report.py` —— 那两个模块不依赖 argparse，便于单测。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from veridx import config
from veridx.eval import report as eval_report
from veridx.eval import runner as eval_runner
from veridx.prompts import blind_solve as blind_prompt
from veridx.prompts import diagnose as diagnose_prompt
from veridx.prompts import explain as explain_prompt

__all__ = ["cmd_eval"]

#: 指标口径版本：改了汇总口径就 +1，否则跨版本的数字不可比
METRICS_VERSION = "eval-v1"


def _git_commit() -> str:
    """当前 commit（带 `-dirty` 标记）。拿不到就写 `unknown`，不猜。"""
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode != 0:
            return "unknown"
        commit = head.stdout.strip()
        status = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return f"{commit}-dirty" if status.stdout.strip() else commit
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 环境异常
        return "unknown"


def _reuse_items(out_path: Path, header: dict[str, Any]) -> list[eval_runner.ItemResult]:
    """同一份数据集 + 同一套坐标 → 复用已完成的结果（不重复烧额度）。"""
    if not out_path.exists():
        return []
    try:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(previous, dict) or previous.get("items") is None:
        return []
    keys = ("dataset_hash", "model", "prompt_version", "arms", "k")
    if any(previous.get(key) != header.get(key) for key in keys):
        return []
    items: list[eval_runner.ItemResult] = []
    for raw in previous["items"]:
        try:
            items.append(
                eval_runner.ItemResult(
                    id=str(raw["id"]),
                    k_index=int(raw["k_index"]),
                    run_id=str(raw["run_id"]),
                    terminal=str(raw["terminal"]),
                    reason_code=raw.get("reason_code"),
                    binding=str(raw.get("binding") or "skipped"),
                    accepted=raw.get("accepted"),
                    rejected=raw.get("rejected"),
                    steps=int(raw.get("steps") or 0),
                    steps_usable=bool(raw.get("steps_usable")),
                    tokens_in=int(raw.get("tokens_in") or 0),
                    tokens_out=int(raw.get("tokens_out") or 0),
                    llm_calls=int(raw.get("llm_calls") or 0),
                    latency_ms=raw.get("latency_ms"),
                    collision_prob=float(raw.get("collision_prob") or 0.0),
                    failure_attribution=raw.get("failure_attribution"),
                    notes=list(raw.get("notes") or []),
                )
            )
        except (KeyError, TypeError, ValueError):
            return []  # 旧报告格式不认识 → 整体不复用，宁可重跑
    return items


def cmd_eval(
    *,
    suite: str,
    arms: str,
    k: int,
    out: str | None,
    limit: int | None,
    db_path: str | None,
    mock: bool,
    real: bool,
    explain: bool,
    reuse: bool,
) -> int:
    suite_path = Path(suite)
    if not suite_path.is_absolute():
        suite_path = config.PROJECT_ROOT / suite_path
    try:
        cases = eval_runner.load_suite(suite_path)
    except (OSError, ValueError) as exc:
        print(f"数据集读取失败：{exc}")
        return 2

    mode = "mock" if mock else ("real" if real else None)
    digest = eval_runner.dataset_hash(suite_path)
    now = datetime.now(UTC).astimezone()
    out_path = Path(out) if out else (config.EVAL_REPORTS_DIR / f"{now:%Y-%m-%d-%H%M%S}.json")
    if not out_path.is_absolute():
        out_path = config.PROJECT_ROOT / out_path
    path = Path(db_path) if db_path else eval_runner.default_db_path(digest, k)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path

    arm_list = [arm.strip() for arm in arms.split(",") if arm.strip()]
    from veridx.llm import factory as llm_factory

    header: dict[str, Any] = {
        "index": "eval-1",
        "metrics_version": METRICS_VERSION,
        "created_at": now.isoformat(timespec="seconds"),
        "suite": str(suite_path.relative_to(config.PROJECT_ROOT))
        if suite_path.is_relative_to(config.PROJECT_ROOT)
        else str(suite_path),
        "dataset_hash": digest,
        "cases": len(cases),
        "arms": arm_list,
        "k": k,
        "limit": limit,
        "mode": llm_factory.effective_mode("primary") if mode is None else mode,
        "model": config.MODEL or "mock",
        "prompt_version": diagnose_prompt.VERSION,
        "prompt_versions": {
            "blind_solve": blind_prompt.VERSION,
            "diagnose": diagnose_prompt.VERSION,
            "explain": explain_prompt.VERSION,
        },
        "git_commit": _git_commit(),
        "db": str(path.relative_to(config.PROJECT_ROOT))
        if path.is_relative_to(config.PROJECT_ROOT)
        else str(path),
    }

    cached = _reuse_items(out_path, header) if reuse else []
    try:
        items = asyncio.run(
            eval_runner.run_suite(
                cases,
                db_path=path,
                arms=arm_list,
                k=k,
                mode=mode,
                explain=explain,
                limit=limit,
                reuse=cached,
                on_progress=print,
            )
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"跑批失败：{exc}")
        return 2

    payload = eval_report.build_report(items, header=header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(eval_report.to_markdown(payload), encoding="utf-8")

    summary = payload["summary"]
    print("")
    print(f"题数：{summary['questions']['completed']}/{summary['questions']['value']} 完成")
    print("终态：" + "  ".join(f"{name}={summary['terminals'][name]}" for name in eval_report.TERMINALS))
    print("绑定：" + "  ".join(f"{name}={summary['binding'][name]}" for name in eval_report.BINDINGS))
    usable = summary["explain_usable"]
    print("讲解可用：" + ("—" if usable["value"] is None else f"{usable['value']}/{usable['denominator']}"))
    print(f"失败归因：{summary['failure_attribution']['counts'] or '无'}")
    print(f"-> {out_path}")
    print(f"-> {md_path}")
    if not items:
        print("（数据集为空：输出「无数据」，这是正常结束，不是错误）")
    return 0
