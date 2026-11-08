"""VeriDx 全局配置：路径、运行时设置、预算常量。

设计约束（设计文档 §0.7 / TODO P0.5）：**所有预算类常量集中在此文件**，
其它模块不允许出现散落的魔法数——因为任何一个数字被写死两处，
实验结论就无法归因。

⋆ 标 ⟨T⟩ 的值是文档占位，必须由首批实测（`veridx eval --limit 5`）反推后回填。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DOC_DIR = PROJECT_ROOT / "doc"
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"
DATASETS_DIR = PROJECT_ROOT / "datasets"
EVAL_REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
RUNTIME_DIR = PROJECT_ROOT / "runtime"

load_dotenv(PROJECT_ROOT / ".env")


def _env_path(name: str) -> Path:
    raw = os.getenv(name, "runtime/veridx.db")
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---- API 契约相关（§8）----
ESTIMATED_LATENCY_MS = 9000  # 202 响应里的 estimated_ms，实测后按 P95 回填

# ---- SQLite（NFR-9：评测与在线服务不互相锁死）----
DB_JOURNAL_MODE = "WAL"
DB_BUSY_TIMEOUT_MS = 5000
DB_SYNCHRONOUS = "NORMAL"

# ---- 模型 Provider（主 / 第二模型，见 .env.example）----
LLM_MODE = os.getenv("LLM_MODE", "auto")  # auto | real | mock
MODEL = os.getenv("MODEL", "glm-4.6")
BASE_URL = os.getenv("BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
API_KEY = os.getenv("API_KEY", "").strip()

SECOND_MODEL = os.getenv("SECOND_MODEL", "").strip()
SECOND_BASE_URL = os.getenv("SECOND_BASE_URL", "").strip()
SECOND_API_KEY = os.getenv("SECOND_API_KEY", "").strip()

_PLACEHOLDERS = {"", "your-api-key", "填写你的key", "sk-xxx"}


def has_key(key: str) -> bool:
    return key.strip() not in _PLACEHOLDERS


def has_second_model() -> bool:
    """第二模型是否可用：分支 B 闸 2 与 judge 都依赖它（设计文档 §3.1.5 / §10.5）。"""
    return bool(SECOND_MODEL) and has_key(SECOND_API_KEY)


@dataclass(frozen=True)
class Settings:
    db_path: Path
    migrations_dir: Path
    llm_mode: str
    model: str
    second_model: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "db_path": str(self.db_path),
            "migrations_dir": str(self.migrations_dir),
            "llm_mode": self.llm_mode,
            "model": self.model,
            "second_model": self.second_model,
        }


def get_settings() -> Settings:
    return Settings(
        db_path=_env_path("VERIDX_DB_PATH"),
        migrations_dir=MIGRATIONS_DIR,
        llm_mode=LLM_MODE,
        model=MODEL,
        second_model=SECOND_MODEL if has_second_model() else None,
    )
