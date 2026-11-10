"""脱敏：**密钥三不落**（不落盘 / 不落 trace / 不落异常消息）。

`httpx` 的异常消息可能带完整 URL（含 `?key=...`），`Authorization` 头也可能被某些日志
中间件拼进字符串。所以所有要落盘的自由文本都先过 `scrub()`。

设计要点：

* **值必须被替换成 `***`**，而不是只把键名留下（`api_key=SECRET` → `api_key=***`）；
* `scrub()` **幂等**：抹过的文本再抹一次不变；
* `contains_secret()` 判的是"**还有没有未抹掉的值**"，所以 `api_key=***` 不算命中 ——
  否则"已脱敏"永远报阳性，测试就没意义了。
"""

from __future__ import annotations

import re

#: 一个"像密钥的值"：≥6 位字母数字/符号，且**不以 `*` 开头**（避免匹配已脱敏的 `***`）
_VALUE = r"(?!\*)[A-Za-z0-9._\-]{6,}"

_KEY_NAMES = r"(?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|secret|password)"

#: `api_key=…` / `api_key: …` / `"api_key": "…"` 三种形态都要覆盖（键名后可能跟一个闭合引号）
_KEY_SEP = r"\b" + _KEY_NAMES + r"\"?\s*[=:]\s*[\"']?"

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"(?i)\b(bearer)\s+{_VALUE}"), r"\1 ***"),
    (re.compile(rf"(?i)\bsk-{_VALUE}"), "sk-***"),
    (re.compile(rf"(?i)({_KEY_SEP}){_VALUE}"), r"\1***"),
    (re.compile(rf"(?i)([?&](?:key|api_key|access_token)=){_VALUE}"), r"\1***"),
    (re.compile(rf"(?i)(authorization\"?\s*[:=]\s*\"?){_VALUE}"), r"\1***"),
)

#: 检测是否**残留**未脱敏的值
_SECRET_HINT = re.compile(
    rf"(?i)(sk-{_VALUE}"
    rf"|bearer\s+{_VALUE}"
    rf"|{_KEY_SEP}{_VALUE}"
    rf"|[?&](?:key|api_key|access_token)={_VALUE})"
)


def scrub(text: str) -> str:
    """抹掉疑似密钥，返回**新字符串**。"""
    if not isinstance(text, str):
        return text
    cleaned = text
    for pattern, replacement in _PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def contains_secret(text: str) -> bool:
    """是否**残留**疑似密钥（已脱敏的 `***` 不算）。"""
    if not isinstance(text, str):
        return False
    return _SECRET_HINT.search(text) is not None


def safe_exception(exc: BaseException) -> str:
    """异常 → 可落盘的一句话（类名 + 脱敏后的消息，截断 200 字）。"""
    return scrub(f"{type(exc).__name__}: {exc}")[:200]


__all__ = ["contains_secret", "safe_exception", "scrub"]
