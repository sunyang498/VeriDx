"""提取器的输出 schema 与进入主模型的唯一形态（方案 §2.1 / §2.2）。

两条铁律（每个字段的设计都是它的推论）：

1. **提取器的产出只是"数据"，不是"判据"** —— 所以 `ExtractDraft` 里
   **物理上没有** `answer_key` / `symbols` / `qtype` 三个字段，多给一个就 schema 拒；
2. **提取器只能"选区"，不能"改写"** —— 所以每个字段都强制带 `span`，由 `verify_span` 机械核对。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from veridx import config

__all__ = [
    "CanonicalProblem",
    "ExtractDraft",
    "FieldSource",
    "UntrustedJudgedInput",
]


class FieldSource(StrEnum):
    DATASET = "dataset"  # 数据集标注（可信）
    USER_FORM = "user_form"  # 表单独立字段（可信）
    EXTRACTED = "extracted"  # 隔离提取器（**不可信**，必须带 span）
    DEFAULT = "default"  # 系统默认策略（如 symbols 为空）


class ExtractDraft(BaseModel):
    """**严格、有界、逐字可定位**。`extra="forbid"`：模型多给一个字段就整条拒。

    ⚠️ v1.1（方案 §2.2）：**不要模型给偏移**。实测让千问数字符的准确率只有 2/5，
    而让它"输出原文片段"是它的强项。偏移由服务端 `locate()` 算，落 trace 供审计 ——
    安全性不依赖"输出版式"，只依赖"能不能在原文里逐字定位到"。
    """

    model_config = ConfigDict(extra="forbid")

    stem: str = Field(min_length=1, max_length=config.STEM_MAX_CHARS)
    student_answer: str | None = Field(default=None, max_length=config.SANDBOX_MAX_EXPR_LEN)
    detected_lang: Literal["zh", "en", "mixed"]
    #: 只允许枚举：自由文本会被原样搬到下游，那等于开了一条新的注入通道
    layout_hint: Literal["single", "with_solution", "with_answer_line", "unknown"] = "unknown"


class UntrustedJudgedInput(ValueError):
    """判据字段来自不可信来源 —— **代码 bug**，正常不该出现（方案 §6.3）。"""

    reason_code = "judged_input_from_untrusted"

    def __init__(self, field: str) -> None:
        super().__init__(f"判据字段 `{field}` 不能来自提取器（这是注入通道）")
        self.field = field


class CanonicalProblem(BaseModel):
    """进入主模型的**唯一**输入形态（方案 §2.1）。"""

    model_config = ConfigDict(extra="forbid")

    stem: str
    stem_source: FieldSource

    student_answer: str | None = None
    student_answer_source: FieldSource = FieldSource.DEFAULT

    #: ② 判据输入：**绝不允许 EXTRACTED**
    answer_key: str | None = None
    answer_key_source: FieldSource = FieldSource.DEFAULT
    symbols: tuple[str, ...] = ()
    symbols_source: FieldSource = FieldSource.DEFAULT
    qtype: str = "numeric"
    qtype_source: FieldSource = FieldSource.DEFAULT

    input_mode: Literal["structured", "freetext"] = "structured"
    detected_lang: Literal["zh", "en", "mixed"] = "zh"

    #: v1.1：这些偏移**不是模型给的**，是服务端 `locate()` 算出来的（审计用）
    stem_span: tuple[int, int] | None = None
    student_answer_span: tuple[int, int] | None = None

    @model_validator(mode="after")
    def _enforce_provenance(self) -> CanonicalProblem:
        """校验规则（方案 §2.1 的表）——**写成测试，不只是写在文档里**。"""
        judged = {
            "answer_key": (self.answer_key, self.answer_key_source),
            "symbols": (self.symbols, self.symbols_source),
            "qtype": (self.qtype, self.qtype_source),
        }
        for name, (value, source) in judged.items():
            if source is FieldSource.EXTRACTED:
                raise UntrustedJudgedInput(name)
            if source is FieldSource.DEFAULT and value not in (None, (), "", "numeric"):
                # 有值却说是"默认策略" = 出处丢了（`answer_key` 尤其危险）
                raise ValueError(f"`{name}` 有值但 source=DEFAULT：出处不明")
        # v1.1：EXTRACTED 字段**不要求**带 span（模型不报偏移）；
        # 逐字可定位由 `extract.extractor` 在装配时强制，落 trace 供审计。
        return self

    @property
    def key_mode(self) -> Literal["present", "absent"]:
        """方案 §2.1：`answer_key is None` **合法** —— 进入纯盲解模式。"""
        return "present" if self.answer_key else "absent"

    def to_spec_fields(self) -> dict[str, object]:
        """给 `RunSpec` 的平铺字段（保持 runner 的构造签名不变）。"""
        return {
            "stem": self.stem,
            "student_answer": self.student_answer or "",
            "answer_key": self.answer_key or "",
            "input_mode": self.input_mode,
            "stem_source": self.stem_source.value,
            "student_answer_source": self.student_answer_source.value,
            "answer_key_source": self.answer_key_source.value,
        }
