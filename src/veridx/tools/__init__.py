"""工具层：blind_solve / check_numeric / step_diff / kb_lookup / escalate。

不调用 LLM 做判断（避免递归与自证，设计文档 §2.2）。
"""

from __future__ import annotations
