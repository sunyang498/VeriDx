"""幂等指纹验收（§2.7）：改 prompt_version 必变 / fresh 模式 k 次不同 / fresh 必须给 run_index。"""

from __future__ import annotations

import pytest

from veridx.sched.fingerprint import CachePolicy, fingerprint

BASE = {"question": "求 ∫_0^1 x·e^x dx", "correct": "1", "student_answer": "e-1"}


def test_same_input_same_fingerprint() -> None:
    assert fingerprint(**BASE) == fingerprint(**BASE)
    assert len(fingerprint(**BASE)) == 16


def test_prompt_version_change_breaks_cache() -> None:
    old = fingerprint(**BASE, prompt_version="pv_v1")
    new = fingerprint(**BASE, prompt_version="pv_v2")
    assert old != new


def test_reuse_ignores_run_index() -> None:
    assert fingerprint(**BASE, run_index=1) == fingerprint(**BASE, run_index=2)


def test_fresh_mode_makes_runs_independent() -> None:
    prints = {
        fingerprint(**BASE, cache_policy=CachePolicy.FRESH, run_index=i, model="glm-4.6") for i in range(3)
    }
    assert len(prints) == 3


def test_fresh_requires_run_index() -> None:
    with pytest.raises(ValueError, match="run_index"):
        fingerprint(**BASE, cache_policy=CachePolicy.FRESH)
