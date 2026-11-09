"""SQLite 连接、PRAGMA 与顺序迁移（不引 Alembic：单机单开发者，收益低于成本）。

迁移的四重保护（P0.6 补强，逐条对应一次真实踩坑）：

1. **checksum**——文件被改即报错（"版本号没变但内容被手改"是实验最常见的污染源）
2. **事务**——DDL 包在 `BEGIN IMMEDIATE` 里，中途失败 `ROLLBACK`，不留半套表
3. **外键自检**——`PRAGMA foreign_key_check` 必须在 COMMIT 之前跑，它是唯一的自检点
4. **备份**——有待执行迁移时先落一份 `.bak`（单文件 SQLite，成本极低）

关于 `PRAGMA foreign_keys` 的两个坑（实测过，写在这里防止被"优化"掉）：

* 它在**事务内是 no-op**（事务里设 OFF，读回来仍是 1）；
* 表重建必须用官方 12-step 顺序（建新表 → 拷贝 → `DROP` 旧表 → `RENAME` 新表）。
  **不能"先把旧表改名"**——那会让子表的 `REFERENCES` 被改写成指向旧表名，
  而错误会拖到几十行测试之后才爆，报错地点与病因完全无关。
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from veridx import config
from veridx.clock import now_utc

SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
  version    INTEGER PRIMARY KEY,
  name       TEXT NOT NULL,
  checksum   TEXT NOT NULL,
  applied_at TEXT NOT NULL
)
"""

_TABLE_EXISTS = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"--[^\n]*")


class MigrationError(RuntimeError):
    """迁移计划与磁盘状态不一致（checksum 变更 / 文件缺失 / 版本号重复 / 外键悬空）。"""


class DatabaseNotFound(RuntimeError):
    """只读模式打开一个不存在的库。"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path
    checksum: str
    sql: str


def _migration_version(path: Path) -> int:
    prefix = path.name.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError as exc:
        raise MigrationError(f"迁移文件名必须以版本号开头：{path.name}") from exc


def load_migrations(migrations_dir: Path) -> list[Migration]:
    migrations: list[Migration] = []
    seen: dict[int, Path] = {}
    for path in sorted(Path(migrations_dir).glob("*.sql")):
        version = _migration_version(path)
        if version in seen:
            raise MigrationError(f"迁移版本重复：{seen[version].name} / {path.name}")
        seen[version] = path
        sql = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=version,
                name=path.stem,
                path=path,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16],
                sql=sql,
            )
        )
    migrations.sort(key=lambda m: m.version)
    return migrations


def split_statements(sql: str) -> list[str]:
    """把脚本切成可逐条 `execute` 的语句。

    不能用 `sql.split(';')`：字符串字面量与注释里都可能出现分号（`/* ; */`、
    `SELECT ';'`），切坏了会静默少执行一半语句。这里用标准库的
    `sqlite3.complete_statement` 状态机，它认得引号与注释。
    """
    statements: list[str] = []
    buffer: list[str] = []
    for line in sql.splitlines(keepends=True):
        buffer.append(line)
        chunk = "".join(buffer)
        if sqlite3.complete_statement(chunk):
            text = chunk.strip()
            if text:
                statements.append(text)
            buffer.clear()

    tail = "".join(buffer)
    if _has_executable_content(tail):
        raise MigrationError(f"迁移脚本末尾存在未结束的语句：{tail.strip()[:80]!r}")
    return statements


def _has_executable_content(text: str) -> bool:
    stripped = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))
    return bool(stripped.strip())


def _connect_target(db_path: str | Path, *, read_only: bool) -> tuple[str, bool]:
    path = Path(db_path)
    if str(path) == ":memory:":
        return ":memory:", False
    if read_only:
        if not path.exists():
            raise DatabaseNotFound(f"数据库不存在：{path}")
        # mode=ro 让"只读路径"物理上不可能写库，比"约定不要写"强
        return f"file:{path.as_posix()}?mode=ro", True
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path), False


def connect(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    target, uri = _connect_target(db_path, read_only=read_only)
    # check_same_thread=False：FastAPI 的同步依赖与同步端点跑在线程池里，不保证同一线程；
    # 连接是"每请求私有"的，从不跨请求共享，禁掉线程指纹检查是安全的。
    conn = sqlite3.connect(target, uri=uri, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if not read_only:
        conn.execute(f"PRAGMA journal_mode={config.DB_JOURNAL_MODE}")
    conn.execute(f"PRAGMA busy_timeout={config.DB_BUSY_TIMEOUT_MS}")
    if not read_only:
        conn.execute(f"PRAGMA synchronous={config.DB_SYNCHRONOUS}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(_TABLE_EXISTS, (name,)).fetchone() is not None


def applied_migrations(conn: sqlite3.Connection) -> dict[int, str]:
    """读取已应用的迁移。

    **读路径不执行任何 DDL**：早先这里直接 `CREATE TABLE IF NOT EXISTS`，
    在只读连接上等于"试图写库"（`attempt to write a readonly database`），
    于是 `db version` 指向一个陌生文件时给出的报错与病因完全无关（P0.6 审查 §3.1/§3.6）。
    建表只在写路径（`apply_migrations` 开头的 `ensure_schema_version_table`）发生。
    """
    if not _table_exists(conn, "schema_version"):
        return {}
    rows = conn.execute("SELECT version, checksum FROM schema_version").fetchall()
    return {int(r["version"]): str(r["checksum"]) for r in rows}


def ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA_VERSION_TABLE)


def current_version(conn: sqlite3.Connection) -> int:
    applied = applied_migrations(conn)
    return max(applied) if applied else 0


def _db_file_path(conn: sqlite3.Connection) -> Path | None:
    for row in conn.execute("PRAGMA database_list"):
        if row["name"] == "main" and row["file"]:
            return Path(str(row["file"]))
    return None


def _backup_database(conn: sqlite3.Connection, target_version: int) -> Path | None:
    path = _db_file_path(conn)
    if path is None or not path.exists():
        return None
    with contextlib.suppress(sqlite3.Error):  # 非 WAL 模式或只读库：备份照做
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    backup = path.with_suffix(path.suffix + f".pre{target_version}.bak")
    shutil.copy2(path, backup)
    return backup


def _apply_one(conn: sqlite3.Connection, migration: Migration) -> None:
    statements = split_statements(migration.sql)
    conn.execute("BEGIN IMMEDIATE")
    try:
        for statement in statements:
            conn.execute(statement)
        conn.execute(
            "INSERT INTO schema_version (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
            (migration.version, migration.name, migration.checksum, now_utc()),
        )
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise MigrationError(
                f"迁移 {migration.name} 后外键校验失败（{len(violations)} 处），已回滚："
                f"{[dict(v) for v in violations[:3]]}"
            )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def apply_migrations(conn: sqlite3.Connection, migrations_dir: str | Path) -> list[int]:
    """应用未执行的迁移，返回本次新应用的版本号列表。"""
    migrations_dir = Path(migrations_dir)
    ensure_schema_version_table(conn)
    applied = applied_migrations(conn)
    planned = load_migrations(migrations_dir)
    planned_versions = {m.version for m in planned}

    missing = sorted(set(applied) - planned_versions)
    if missing:
        raise MigrationError(f"数据库已应用的迁移在磁盘上缺失：{missing}")

    for migration in planned:
        if migration.version in applied and applied[migration.version] != migration.checksum:
            raise MigrationError(
                f"迁移 {migration.name} 已被修改（checksum {applied[migration.version]} -> "
                f"{migration.checksum}）。已冻结的迁移不可回改，请新增 00X_*.sql。"
            )

    pending = [m for m in planned if m.version not in applied]
    if not pending:
        return []

    if applied:  # 空库没什么可丢的，不必留 .bak
        _backup_database(conn, pending[0].version)

    # PRAGMA 必须在事务之外设置，且异常路径也要恢复（事务回滚不会撤销 PRAGMA）
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        for migration in pending:
            _apply_one(conn, migration)
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
    return [m.version for m in pending]


def init_db(
    db_path: str | Path, migrations_dir: str | Path | None = None
) -> tuple[sqlite3.Connection, list[int]]:
    conn = connect(db_path)
    applied = apply_migrations(conn, migrations_dir or config.MIGRATIONS_DIR)
    return conn, applied


def read_version(db_path: str | Path) -> int:
    """只读地取 schema 版本；库不存在时返回 0，且**不创建任何文件**。"""
    path = Path(db_path)
    if str(path) != ":memory:" and not path.exists():
        return 0
    conn = connect(path, read_only=True)
    try:
        return current_version(conn)
    finally:
        conn.close()


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(r["name"]) for r in rows]
