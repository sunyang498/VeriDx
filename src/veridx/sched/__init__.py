"""调度层：预算闸门 / 超时 / 取消 / 幂等指纹 / 自适应并发。

不理解"错因"是什么（设计文档 §2.2）。
"""

from __future__ import annotations
