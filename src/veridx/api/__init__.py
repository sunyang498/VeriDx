"""接入层：REST 路由 + SSE 建连 + 入参校验 + 幂等键。

不该做：不拼 prompt、不做业务判断（设计文档 §2.2）。
"""

from __future__ import annotations
