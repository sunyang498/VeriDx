"""错因分类体系（设计文档 §9.3）——让"错因"可统计的前提。

三条约束：
1. 枚举内 + "其他"兜底：既能统计，又不逼模型硬分类
2. **只用大类算 M2**，子类仅作参考（子类标注一致性差）
3. 每个大类绑定一条不同的教学建议 → 分类才有产品价值
"""

from __future__ import annotations

from enum import StrEnum


class ErrorType(StrEnum):
    CONCEPTUAL = "概念性错误"
    PROCEDURAL = "程序性错误"
    READING = "审题性错误"
    COMPUTATION = "计算表达错误"
    OTHER = "其他"


#: MR2 集专用：学生其实做对，不参与 Macro-F1
NO_ERROR = "无错"

#: M2（Macro-F1）的类别集合：4 类等权平均，"其他"与"无错"不进分母
MACRO_F1_CLASSES: tuple[ErrorType, ...] = (
    ErrorType.CONCEPTUAL,
    ErrorType.PROCEDURAL,
    ErrorType.READING,
    ErrorType.COMPUTATION,
)

TEACHING_ADVICE: dict[ErrorType, str] = {
    ErrorType.CONCEPTUAL: "回看定义与定理条件",
    ErrorType.PROCEDURAL: "练同类步骤拆分",
    ErrorType.READING: "先圈条件再动笔",
    ErrorType.COMPUTATION: "练验算习惯",
    ErrorType.OTHER: "人工复核",
}
