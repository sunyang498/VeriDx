# judge/ —— 评测判分器（T2：把 agent 的错因主张归到 6 类，再与人工标注比）

**内部工具，不 import `veridx`、不进 `src/`。** 只读三样东西：

| 输入 | 从哪来 |
|---|---|
| 报告 JSON | `veridx eval --out eval/reports/xxx.json` |
| 错因主张 | 报告头 `db` 指向的 sqlite：`SELECT error_type, root_cause FROM diagnoses WHERE run_id=?`（**只读打开**） |
| 人工标注 | `labels_<n>_todo.csv` 的 `gold_class` 列 |

为什么单独一个进程：`correct` 在 veridx 里是不可信输入（ADR-021 禁作判据），在 judge 里却是标准答案；
`teacher_described_confusion` 永远不进 agent，是 judge 的金标准。视角不同，所以分开放。

## 三条命令

```powershell
# 1) MathDial → suite + 待标注 CSV（分层 19/37/44，seed 42）
python judge/adapt.py --limit 100

# 2) 检查 suite 能不能被加载、TD4 风险多大
.venv\Scripts\veridx.exe dataset check --suite judge/data/mathdial_100.jsonl

# 3) 判分（--selftest 不用模型，先跑它）
python judge/run.py --selftest
python judge/run.py --report eval/reports/smoke-mathdial-10.json
```

判分模型取 `JUDGE_*` → `SECOND_*` → 遗留别名 `*_KIMI`，都从 `.env` 读，**不读环境变量**。
想临时换模型交叉验证：往 `.env` 加 `JUDGE_MODEL` / `JUDGE_BASE_URL` / `JUDGE_API_KEY`，别改代码。

## 6 个类目（`prompts.py::CLASSES`）

概念混淆 · 运算错误 · 漏步或多步 · 读题或条件误用 · 单位与量纲 · 无法归类

## verdict 七种，别混

| verdict | 含义 | 进分母？ |
|---|---|---|
| `correct` | 类目 == `gold_class` | 是 |
| `wrong` | 类目 != `gold_class` | 是 |
| `abstain` | agent 没走到 `diagnosed`（或没有 diagnosis 行）——它明确说"我不判"，不算错 | 否 |
| `unlabeled` | 判出来了，但这行还没人工标注 | 否 |
| `unparsed` | 模型**答了**，但不在 6 类里 | 否（单独计数，不兜底成"无法归类"） |
| `error` | **没判成**：429 / 5xx / 超时重试耗尽 / 被 `max_tokens` 截断 | 否 |
| — | `n_scored < 30` → `accuracy_strict` 必须是 `null`，不出数 | — |

`unparsed` 与 `error` 分开是刻意的：把限流或预算不足记成 `unparsed`，等于把"我没测到"伪装成"模型不合格"。
`selftest` 有 3 条断言专门守这条线。

## 没实现（故意的，方案里没有就别加）

- **partial credit**（上位类算半分）：方案 §6 没给类目层级，自己造层级属于添加功能。
- 多模型投票、置信度、二次复核。
- 跨 run 聚合、排行榜。

## 反向验证（改坏了会变红）

```powershell
python judge/adapt.py --limit 100 --break-correct first   # correct 列取第一行 → 题干不再含答案
.venv\Scripts\veridx.exe dataset check --suite judge/data/_rv_broken.jsonl   # 必须 exit 1
```

自检逻辑本身的反向验证（7 组注入，跑完自动还原）：`python runtime/rv_judge.py`（会话产物，不入库）
