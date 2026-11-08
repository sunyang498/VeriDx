"""幂等指纹（设计文档 §2.7）。

反直觉但必须写进论文的一点：**pass^k 与缓存天然冲突**。
* `cache_policy="reuse"`：同指纹复用结果（省钱、断点续跑）
* `cache_policy="fresh"`：把 `run_index` 计入指纹，制造**独立试验**（否则 pass^k 测的是缓存命中率）
"""

from __future__ import annotations

import hashlib
from enum import StrEnum


class CachePolicy(StrEnum):
    REUSE = "reuse"
    FRESH = "fresh"


def fingerprint(
    *,
    question: str,
    correct: str,
    student_answer: str,
    prompt_version: str = "",
    model: str = "",
    paradigm: str = "",
    cache_policy: CachePolicy = CachePolicy.REUSE,
    run_index: int | None = None,
) -> str:
    if cache_policy is CachePolicy.FRESH and run_index is None:
        raise ValueError("cache_policy=fresh 必须提供 run_index，否则 k 次运行会命中同一指纹")
    parts = [
        question,
        correct,
        student_answer,
        prompt_version,
        model,
        paradigm,
        str(run_index) if cache_policy is CachePolicy.FRESH else "-",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
