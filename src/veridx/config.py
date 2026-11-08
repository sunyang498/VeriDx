"""VeriDx 全局配置：路径、运行时设置、预算常量。

设计约束（设计文档 §0.7 / TODO P0.5）：**所有预算类常量集中在此文件**，
其它模块不允许出现散落的魔法数——因为任何一个数字被写死两处，
实验结论就无法归因。

⋆ 标 ⟨T⟩ 的值是文档占位，必须由首批实测（`veridx eval --limit 5`）反推后回填。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
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

# ---- 预算与调用上限（NFR-1 / NFR-2 / §2.3.2 / §5.3.3）----
#: 单题 LLM 调用上限；本地 sympy 计算（check_numeric / step_diff）不计次数、不计 token
MAX_LLM_CALLS = 7
#: §2.3.2 的节点预算；四者之和必须等于 MAX_LLM_CALLS（由 tests/test_config.py 守护）
NODE_MAX_CALLS: dict[str, int] = {"parse": 1, "blind_solve": 2, "diagnose": 3, "explain": 1}
LOCAL_TOOL_MAX_CALLS = 3
#: 单次调用的上下文窗口（NFR-2 的第一个维度）
PER_CALL_CTX_TOKENS = 8000
#: ⟨T⟩ 单题累计 token（NFR-2 的第二个维度），首批实测后回填
PER_ITEM_TOKENS = 40_000
WALL_CLOCK_SECONDS = 180
#: T1 用例口径：注入 300s 挂死题时，整批仍须在该上限内出报告
BATCH_WALL_CLOCK_SECONDS = 200
#: NFR-1：SSE 首 token 的感知阈值
FIRST_TOKEN_SLA_SECONDS = 3

# ---- 表达式沙箱（§4.4 四层防御）----
SANDBOX_TIMEOUT_SECONDS = 3.0
SANDBOX_TERMINATE_GRACE_SECONDS = 1.0
#: 口径冻结：256（TODO 里写的 4096 与 §4.8.1「300 字符样本必须 rejected」冲突）
SANDBOX_MAX_EXPR_LEN = 256
SANDBOX_MAX_AST_NODES = 200
SANDBOX_MAX_EXPONENT = 200
SANDBOX_MAX_NESTED_POW = 1
SANDBOX_MAX_INT_LITERAL = 10**9
SANDBOX_LRU_SIZE = 128

# ---- 守卫 / 级联 / 置信度（§3.2 / §5.5）----
GUARD_MAX_RETRIES = 2
EVIDENCE_MAX_RETRIES = 2
CASCADE_MAX_LEVEL = 1
#: 进入 diagnosed 的置信度下限（§2.6）
CONFIDENCE_THETA = 0.60
#: 低于此值时输出 Top-3 并列假设（FR-2.4）
CONFIDENCE_TOP3_BELOW = 0.60
#: 低于此值升级一档模型（§5.5）
CASCADE_UPGRADE_BELOW = 0.70
#: 高于此值时"该省的省"，不升级（§5.5）
CONFIDENCE_HIGH = 0.85

# ---- 判定容差（§4.8.2：1/3 vs 0.333333 必须判等）----
REL_TOL = 1e-6

# ---- 温度与 seed 策略（不变量 I5：seed 策略必须显式，不能只写"固定 seed"）----
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_SEED = 0


class SeedPolicy(StrEnum):
    """`fixed` = 复现/CI；`per_run_recorded` = pass^k 的独立试验；`unset` = 探测 provider 是否真支持 seed。"""

    FIXED = "fixed"
    PER_RUN_RECORDED = "per_run_recorded"
    UNSET = "unset"


def resolve_seed(
    policy: SeedPolicy, *, base_seed: int = DEFAULT_SEED, run_index: int | None = None
) -> int | None:
    if policy is SeedPolicy.UNSET:
        return None
    if policy is SeedPolicy.FIXED:
        return base_seed
    if run_index is None:
        raise ValueError("seed 策略 per_run_recorded 必须提供 run_index，否则 k 次试验不独立")
    return base_seed + run_index


def _parse_seed_policy(raw: str) -> SeedPolicy:
    try:
        return SeedPolicy(raw.strip())
    except ValueError as exc:
        allowed = ", ".join(p.value for p in SeedPolicy)
        raise ValueError(f"VERIDX_SEED_POLICY 只能是 [{allowed}]，收到 {raw!r}") from exc


SEED_POLICY = _parse_seed_policy(os.getenv("VERIDX_SEED_POLICY", SeedPolicy.FIXED.value))

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
    seed_policy: SeedPolicy

    def as_dict(self) -> dict[str, str | None]:
        return {
            "db_path": str(self.db_path),
            "migrations_dir": str(self.migrations_dir),
            "llm_mode": self.llm_mode,
            "model": self.model,
            "second_model": self.second_model,
            "seed_policy": self.seed_policy.value,
        }


def get_settings() -> Settings:
    return Settings(
        db_path=_env_path("VERIDX_DB_PATH"),
        migrations_dir=MIGRATIONS_DIR,
        llm_mode=LLM_MODE,
        model=MODEL,
        second_model=SECOND_MODEL if has_second_model() else None,
        seed_policy=SEED_POLICY,
    )
