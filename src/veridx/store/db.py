"""SQLite 连接、PRAGMA 与顺序迁移（不引 Alembic：单机单开发者，收益低于成本）。

迁移语义（设计文档 §7.5 / TODO §0.7）：
* `migrations/00X_*.sql` 按版本号顺序执行一次，执行记录写入 `schema_version`
* 已执行过的迁移**校验 checksum**：文件被改动即报错（防止"版本号没变但内容被改"）
* 重复 `db init` 幂等：已应用的版本直接跳过
"""

from __future__ import annotations

import hashlib
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


class MigrationError(RuntimeError):
    """迁移计划与磁盘状态不一致（checksum 变更 / 文件缺失 / 版本号重复）。"""


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


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA journal_mode={config.DB_JOURNAL_MODE}")
    conn.execute(f"PRAGMA busy_timeout={config.DB_BUSY_TIMEOUT_MS}")
    conn.execute(f"PRAGMA synchronous={config.DB_SYNCHRONOUS}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def applied_migrations(conn: sqlite3.Connection) -> dict[int, str]:
    conn.execute(SCHEMA_VERSION_TABLE)
    rows = conn.execute("SELECT version, checksum FROM schema_version").fetchall()
    return {int(r["version"]): str(r["checksum"]) for r in rows}


def current_version(conn: sqlite3.Connection) -> int:
    applied = applied_migrations(conn)
    return max(applied) if applied else 0


def apply_migrations(conn: sqlite3.Connection, migrations_dir: str | Path) -> list[int]:
    """应用未执行的迁移，返回本次新应用的版本号列表。"""
    migrations_dir = Path(migrations_dir)
    applied = applied_migrations(conn)
    planned = load_migrations(migrations_dir)
    planned_versions = {m.version for m in planned}

    missing = sorted(set(applied) - planned_versions)
    if missing:
        raise MigrationError(f"数据库已应用的迁移在磁盘上缺失：{missing}")

    newly_applied: list[int] = []
    for migration in planned:
        if migration.version in applied:
            if applied[migration.version] != migration.checksum:
                raise MigrationError(
                    f"迁移 {migration.name} 已被修改（checksum {applied[migration.version]} -> "
                    f"{migration.checksum}）。已冻结的迁移不可回改，请新增 00X_*.sql。"
                )
            continue
        conn.executescript(migration.sql)
        conn.execute(
            "INSERT INTO schema_version (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
            (migration.version, migration.name, migration.checksum, now_utc()),
        )
        newly_applied.append(migration.version)
    return newly_applied


def init_db(
    db_path: str | Path, migrations_dir: str | Path | None = None
) -> tuple[sqlite3.Connection, list[int]]:
    conn = connect(db_path)
    applied = apply_migrations(conn, migrations_dir or config.MIGRATIONS_DIR)
    return conn, applied


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(r["name"]) for r in rows]
