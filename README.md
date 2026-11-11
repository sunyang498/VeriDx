# VeriDx

**Verification-first diagnosis for student mistakes — every claim carries evidence.**

An LLM agent that diagnoses *why* a student's answer is wrong. Every diagnostic
claim must be backed by executable, program-generated evidence; when no such
evidence can be produced, the system escalates instead of guessing.

## Why this is not "just an API wrapper"

| Concern | How it is handled |
|---|---|
| The model invents a plausible cause | **Claim–evidence binding**: each claim must cite ≥1 `status=ok` program fact, produced *before* the claim (no post-hoc justification), otherwise the guard refuses to persist it |
| The model is anchored by the student's answer | **Blind solve**: the solver never sees the student's answer; the comparison is done by the program, and only a program judgment counts |
| The model's output is code, not just text | **Expression sandbox**: AST allow-list → name allow-list → static limits → subprocess timeout, three-way verdict `ok / unsupported / rejected` |
| A single success rate is misleading | **pass^k** (k consecutive successes) as the headline reliability metric, not pass@1 |
| "Wrong" is the dangerous direction | **False-accusation rate** reported as a first-class metric |
| Prompt edits silently regress | **Prompt regression CI** on a frozen subset, with thresholds |

## Status

**P1.7 (gate G0.6).** The agent core is real end-to-end: blind solve → program
verification → claim–evidence binding → step explanation. `POST /diagnose` runs
the pipeline in the background, streams progress over SSE, and persists a
diagnosis whose every claim cites program-generated evidence. `step_diff` is
still a placeholder (P1.8), so non-numeric questions escalate instead of guessing.

## Demo (Web UI)

```bash
./.venv/Scripts/pip install -e ".[dev]"
./.venv/Scripts/veridx serve --port 8000     # http://127.0.0.1:8000
```

The page is served by FastAPI from `web/dist` (same origin, no CORS). Rebuild it
after editing `web/src`:

```bash
cd web && npm ci && npm run build
```

Without any API key the pipeline still runs end-to-end in `LLM_MODE=mock` (a
scenario provider that answers by prompt type) — useful for offline demos; every
quality metric, however, must be measured on a real model.

## Layout

```
src/veridx/
├── api/        REST routers + SSE protocol + error envelope
├── flow/       controlled 5-node pipeline and terminal states
├── guard/      schema checks + claim–evidence binding (veto power)
├── sandbox/    expression evaluation: AST allow-list + process timeout
├── sched/      budget, timeout, cancellation, idempotency fingerprint
├── tools/      blind_solve / check_numeric / step_diff / kb_lookup / escalate
├── trace/      trace contract and replay
├── store/      SQLite (WAL) + sequential migrations
├── eval/       runner / metrics / report / compare
└── prompts/    versioned prompt templates
```

## Quick start

```bash
python -m venv .venv && ./.venv/Scripts/pip install -e ".[dev]"   # Windows
cp .env.example .env        # fill in an OpenAI-compatible provider

veridx db init              # create schema (idempotent)
veridx serve --port 8000    # http://127.0.0.1:8000/docs
ruff check . && pytest
```

## Design

The design document (Chinese) lives at
[`doc/VeriDx-需求与设计文档-v3.1.md`](doc/VeriDx-需求与设计文档-v3.1.md).
It is the single source of truth for the invariants, the 22 metrics, and the
experiment protocol.
