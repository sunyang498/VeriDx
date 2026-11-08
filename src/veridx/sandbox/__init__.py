"""表达式沙箱：AST 白名单 → 名字白名单 → 静态资源上限 → 子进程超时。

三分类 ok / unsupported / rejected（设计文档 §4.6）。
"""

from __future__ import annotations
