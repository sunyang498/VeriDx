"""配置验收（P0.5 DoD）：预算常量集中、调用数算得平、seed 三态可校验、魔法数不散落。"""

from __future__ import annotations

import re

import pytest

from veridx import config


def test_llm_call_budget_adds_up() -> None:
    """§2.3.2 最容易算不平的一处：1 + 2 + 3 + 1 = 7。"""
    assert sum(config.NODE_MAX_CALLS.values()) == config.MAX_LLM_CALLS == 7
    assert config.LOCAL_TOOL_MAX_CALLS == 3  # 纯本地，不计入 7 次


def test_two_token_dimensions_are_independent() -> None:
    """NFR-2：单次上下文 8k 与单题累计 ⟨T⟩ 是两个独立上限。"""
    assert config.PER_CALL_CTX_TOKENS == 8000
    assert config.PER_ITEM_TOKENS > config.PER_CALL_CTX_TOKENS
    assert config.PER_ITEM_TOKENS >= config.PER_CALL_CTX_TOKENS * 3  # ≈7 次调用


def test_timeouts_ordered() -> None:
    assert config.BATCH_WALL_CLOCK_SECONDS > config.WALL_CLOCK_SECONDS
    assert config.FIRST_TOKEN_SLA_SECONDS < config.WALL_CLOCK_SECONDS


def test_sandbox_limits_match_frozen_decision() -> None:
    assert config.SANDBOX_MAX_EXPR_LEN == 256  # 口径 v1：不是 TODO 里的 4096
    assert config.SANDBOX_MAX_NESTED_POW == 1  # 拒 a**b**c
    assert config.SANDBOX_TIMEOUT_SECONDS == 3.0
    assert config.SANDBOX_TERMINATE_GRACE_SECONDS == 1.0
    assert config.SANDBOX_MAX_INT_LITERAL == 10**9


def test_guard_and_confidence_ladder_ordered() -> None:
    assert config.GUARD_MAX_RETRIES == 2
    assert config.EVIDENCE_MAX_RETRIES == 2
    assert config.CASCADE_MAX_LEVEL == 1
    assert config.CONFIDENCE_TOP3_BELOW <= config.CONFIDENCE_THETA < config.CASCADE_UPGRADE_BELOW
    assert config.CASCADE_UPGRADE_BELOW < config.CONFIDENCE_HIGH <= 1.0


def test_float_tolerance_is_frozen() -> None:
    assert config.REL_TOL == 1e-6


def test_seed_policy_has_exactly_three_modes() -> None:
    assert {p.value for p in config.SeedPolicy} == {"fixed", "per_run_recorded", "unset"}
    assert isinstance(config.SEED_POLICY, config.SeedPolicy)
    assert config.get_settings().seed_policy is config.SEED_POLICY


def test_resolve_seed_semantics() -> None:
    assert config.resolve_seed(config.SeedPolicy.UNSET) is None
    assert config.resolve_seed(config.SeedPolicy.FIXED, base_seed=7) == 7
    assert config.resolve_seed(config.SeedPolicy.PER_RUN_RECORDED, base_seed=7, run_index=3) == 10
    with pytest.raises(ValueError, match="run_index"):
        config.resolve_seed(config.SeedPolicy.PER_RUN_RECORDED)


def test_invalid_seed_policy_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="VERIDX_SEED_POLICY"):
        config._parse_seed_policy("fixed-seed")


FORBIDDEN_BUDGET_LITERALS = ("8000", "40000", "180", "5000", "1e-6")


def test_budget_numbers_are_not_duplicated_outside_config() -> None:
    """P0.5 验收：预算常量只能出现在 config.py，别处出现即视为硬编码。"""
    offenders: list[str] = []
    for path in sorted(config.PACKAGE_ROOT.rglob("*.py")):
        if path.name == "config.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            code = line.split("#", 1)[0]
            if "port" in code.lower():  # HTTP 默认端口 8000 与 token 预算无关
                continue
            for literal in FORBIDDEN_BUDGET_LITERALS:
                if re.search(rf"(?<![\w.]){re.escape(literal)}(?![\w.])", code):
                    offenders.append(f"{path.relative_to(config.PACKAGE_ROOT)}:{lineno}: {literal}")
    assert offenders == [], f"预算常量散落在 config.py 之外：{offenders}"


def test_secrets_are_never_exposed_by_settings() -> None:
    dumped = config.get_settings().as_dict()
    lowered = " ".join(f"{k}={v}" for k, v in dumped.items()).lower()
    assert "api_key" not in lowered
    assert config.has_key("") is False
    assert config.has_key("  ") is False
    assert config.has_key("sk-real-looking-key") is True
