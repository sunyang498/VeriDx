"""`veridx` 命令行入口。

命令按阶段增量挂载（一张卡一个 commit）：
  P0.1  --version / info
  P0.2  db init / db version
  P0.3  serve / contract export

纪律：**只读命令不得产生写副作用**（`db version` 不能顺手建库）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from veridx import __version__, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veridx", description="VeriDx - verification-first diagnosis agent")
    parser.add_argument("--version", action="version", version=f"veridx {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info", help="打印当前配置摘要（不含密钥）")

    db = sub.add_parser("db", help="数据库迁移")
    db_sub = db.add_subparsers(dest="db_command")
    db_init = db_sub.add_parser("init", help="建库并应用全部迁移（幂等）")
    db_init.add_argument("--db", default=None, help="数据库路径（默认取 VERIDX_DB_PATH）")
    db_version = db_sub.add_parser("version", help="打印当前 schema 版本（只读，不建库）")
    db_version.add_argument("--db", default=None, help="数据库路径（默认取 VERIDX_DB_PATH）")

    serve = sub.add_parser("serve", help="启动 API 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="开发模式热重载")

    contract = sub.add_parser("contract", help="契约产物")
    contract_sub = contract.add_subparsers(dest="contract_command")
    export = contract_sub.add_parser("export", help="导出 openapi.json")
    export.add_argument("--out", default=None, help="输出路径（默认 doc/openapi.json）")
    export.add_argument("--check", action="store_true", help="只校验导出文件是否与代码一致")

    run = sub.add_parser("run", help="跑一题并打印终态 / 证据 / trace 位置")
    run.add_argument("--case", default=None, help="题目 id（先查 datasets/，再查数据库）")
    run.add_argument("--db", default=None, help="数据库路径（默认取 VERIDX_DB_PATH）")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="强制脚本 provider（离线、确定性）")
    mode.add_argument("--real", action="store_true", help="强制真模型（必须有 key）")
    run.add_argument("--student", default=None, help="覆盖学生答案（默认取样本/最近一次作答）")
    run.add_argument("--out", default=None, help="把结果写成 JSON")
    run.add_argument("--no-explain", action="store_true", help="跳过讲解节点（省一次调用）")

    replay = sub.add_parser("replay", help="打印某次 run 的 trace 时间线")
    replay.add_argument("run_id", help="run_id（`veridx run` 会打印）")
    replay.add_argument("--db", default=None, help="数据库路径（默认取 VERIDX_DB_PATH）")
    replay.add_argument("--at", type=int, default=None, help="只看第 N 条事件的完整输入")
    replay.add_argument("--diff", default=None, help="与另一个 run 对比节点序列")

    eval_parser = sub.add_parser("eval", help="批量跑题 + 自动汇总（CLI-only，C-32）")
    eval_parser.add_argument("--suite", required=True, help="数据集（JSONL，一行一题）")
    eval_parser.add_argument("--arms", default="A", help="实验臂，逗号分隔（当前只有 A）")
    eval_parser.add_argument("--k", type=int, default=1, help="每题重复次数")
    eval_parser.add_argument("--out", default=None, help="结果 JSON（默认 eval/reports/<时间>.json）")
    eval_parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（自测用）")
    eval_parser.add_argument("--db", default=None, help="数据库路径（默认 eval/tmp/<hash>-k<k>.db）")
    eval_mode = eval_parser.add_mutually_exclusive_group()
    eval_mode.add_argument("--mock", action="store_true", help="强制脚本 provider")
    eval_mode.add_argument("--real", action="store_true", help="强制真模型")
    eval_parser.add_argument("--no-explain", action="store_true", help="跳过讲解节点（省一次调用）")
    eval_parser.add_argument("--no-reuse", action="store_true", help="不复用已有结果，强制重跑")
    return parser


def cmd_info() -> int:
    payload = dict(config.get_settings().as_dict())
    payload["version"] = __version__
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_db_init(db_path: str | None) -> int:
    from veridx.store import db as store_db

    path = db_path or str(config.get_settings().db_path)
    conn, applied = store_db.init_db(path)
    try:
        print(json.dumps({"db": path, "applied": applied, "version": store_db.current_version(conn)}))
    finally:
        conn.close()
    return 0


def cmd_db_version(db_path: str | None) -> int:
    from veridx.store import db as store_db

    path = db_path or str(config.get_settings().db_path)
    print(json.dumps({"version": store_db.read_version(path), "db": path}))
    return 0


def cmd_serve(host: str, port: int, reload: bool) -> int:
    import uvicorn

    uvicorn.run("veridx.api.app:create_app", factory=True, host=host, port=port, reload=reload)
    return 0


def _openapi_text() -> str:
    from veridx.api.app import openapi_spec

    return json.dumps(openapi_spec(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def cmd_contract_export(out: str | None, check: bool) -> int:
    path = config.DOC_DIR / (out or "openapi.json")
    text = _openapi_text()
    if check:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if existing != text:
            print(f"openapi 契约与代码不一致：{path}", file=sys.stderr)
            return 1
        print(f"openapi 契约一致：{path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"已导出：{path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        return cmd_info()
    if args.command == "db":
        if args.db_command == "init":
            return cmd_db_init(args.db)
        if args.db_command == "version":
            return cmd_db_version(args.db)
    if args.command == "serve":
        return cmd_serve(args.host, args.port, args.reload)
    if args.command == "contract" and args.contract_command == "export":
        return cmd_contract_export(args.out, args.check)
    if args.command == "run":
        from veridx import cli_ops

        return cli_ops.cmd_run(
            case_id=args.case,
            db_path=args.db,
            mock=args.mock,
            real=args.real,
            student=args.student,
            out=args.out,
            explain=not args.no_explain,
        )
    if args.command == "replay":
        from veridx import cli_ops

        return cli_ops.cmd_replay(run_id=args.run_id, db_path=args.db, at=args.at, diff=args.diff)
    if args.command == "eval":
        from veridx import cli_eval

        return cli_eval.cmd_eval(
            suite=args.suite,
            arms=args.arms,
            k=args.k,
            out=args.out,
            limit=args.limit,
            db_path=args.db,
            mock=args.mock,
            real=args.real,
            explain=not args.no_explain,
            reuse=not args.no_reuse,
        )
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
