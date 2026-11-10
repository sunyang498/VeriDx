"""P1.3 验收：容错解析 F1–F12 + 四条不变量（P1–P4）。

不变量比样本重要：

```
P1  tolerant_parse 对任意输入永不抛
P2  合法 JSON 的 round-trip：parse(dumps(x)) == x（含中文引号值、含 } 在字符串里）
P3  宽容解析不得改变任何字符串字段的值（F6 的守门人）
P4  同一输入两次解析结果相同（无随机性，才能回放）
```
"""

from __future__ import annotations

import json

import pytest

from veridx.guard.samples import NEVER_RAISE_INPUTS, SAMPLES
from veridx.guard.schema import tolerant_parse


def _label(sample) -> str:  # noqa: ANN001
    return f"{sample.kind}"


# --- 分类表覆盖 -------------------------------------------------------------
def test_every_family_has_at_least_one_sample() -> None:
    families = {sample.kind.split("_")[0] for sample in SAMPLES}
    assert families == {f"F{i}" for i in range(1, 13)}, f"缺样本的分类：{families}"


@pytest.mark.parametrize("sample", SAMPLES, ids=_label)
def test_sample_matches_expected_outcome(sample) -> None:  # noqa: ANN001
    outcome = tolerant_parse(sample.raw)
    assert outcome.ok is sample.parse_ok, f"{sample.kind}（{sample.why}）-> {outcome.errors}"
    assert outcome.ambiguous is sample.ambiguous, sample.why
    if sample.expect_transform:
        assert sample.expect_transform in outcome.transformations, (
            f"{sample.kind} 期望走 {sample.expect_transform}，实际 {outcome.transformations}"
        )
    if sample.parse_ok:
        assert isinstance(outcome.value, (dict, list))


# --- P1 永不抛 --------------------------------------------------------------
@pytest.mark.parametrize("bad", NEVER_RAISE_INPUTS, ids=lambda v: repr(v)[:20])
def test_p1_tolerant_parse_never_raises(bad: object) -> None:
    outcome = tolerant_parse(bad)
    assert isinstance(outcome.errors, list)


# --- P2 round-trip ----------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"a": 1},
        {"root_cause": "他说“好”"},
        {"root_cause": "集合 {x|x>0}"},
        {"nested": {"k": ["v", "含 } 括号"]}},
        {"unicode": "积分 ∫_0^1 x·e^x dx"},
    ],
)
def test_p2_roundtrip_preserves_payload(payload: dict) -> None:
    outcome = tolerant_parse(json.dumps(payload, ensure_ascii=False))
    assert outcome.ok
    assert outcome.value == payload


def test_p2_roundtrip_with_ascii_escapes() -> None:
    payload = {"a": "line\nbreak", "b": 'quote"inside'}
    assert tolerant_parse(json.dumps(payload)).value == payload


# --- P3 不改值（F6 守门人）--------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"root_cause": "他说“这步符号写反了”"}', "他说“这步符号写反了”"),
        ('{"a": "前“后”再“一次"}', "前“后”再“一次"),
        ('{"a": "集合 {x|x>0} 与 } 混排"}', "集合 {x|x>0} 与 } 混排"),
        ('{"a": "含,逗号和}括号,"}', "含,逗号和}括号,"),
    ],
)
def test_p3_string_values_are_byte_identical(raw: str, expected: str) -> None:
    outcome = tolerant_parse(raw)
    assert outcome.ok, outcome.errors
    assert outcome.value["root_cause" if "root_cause" in outcome.value else "a"] == expected


def test_p3_structural_quotes_still_normalized() -> None:
    outcome = tolerant_parse("{“error_type”: “程序性错误”}")
    assert outcome.ok
    assert outcome.value == {"error_type": "程序性错误"}


# --- P4 确定性 --------------------------------------------------------------
@pytest.mark.parametrize("sample", SAMPLES[:8], ids=_label)
def test_p4_parsing_is_deterministic(sample) -> None:  # noqa: ANN001
    first = tolerant_parse(sample.raw)
    second = tolerant_parse(sample.raw)
    assert first.value == second.value
    assert first.transformations == second.transformations
    assert first.source_sha == second.source_sha


# --- 具体分类行为 -----------------------------------------------------------
def test_f2_extracts_outermost_object() -> None:
    outcome = tolerant_parse('好的，这是结果：{"a": 1} 希望有帮助')
    assert outcome.value == {"a": 1}
    assert "extract_outermost" in outcome.transformations


def test_f7_truncated_is_diagnosed_not_guessed() -> None:
    outcome = tolerant_parse('{"a": 1, "b":')
    assert not outcome.ok
    assert outcome.errors == ["truncated"], "截断必须报 truncated，不能猜补"


def test_f10_takes_first_and_marks_ambiguous() -> None:
    outcome = tolerant_parse('{"a": 1}\n{"b": 2}')
    assert outcome.value == {"a": 1}
    assert outcome.ambiguous is True


def test_f12_too_long_is_rejected_without_parsing() -> None:
    outcome = tolerant_parse('{"a": "' + "x" * 9000 + '"}')
    assert not outcome.ok
    assert outcome.errors[0].startswith("too_long")


def test_json_array_is_accepted() -> None:
    assert tolerant_parse("[1, 2, 3]").value == [1, 2, 3]


def test_scalar_is_not_an_object() -> None:
    outcome = tolerant_parse("42")
    assert not outcome.ok
