-- ============================================================================
-- VeriDx schema v1  (contract frozen at gate G0)
-- 依据：doc/VeriDx-需求与设计文档-v3.1.md §7.1 / §7.2
-- 约定：
--   * 时间戳一律 UTC ISO8601 文本（形如 2026-11-08T15:22:31Z）
--   * JSON 字段以 TEXT 存储，读写方负责 (de)serialize
--   * 迁移只增不改：新增结构写 00X_*.sql，不回改本文件
-- P0 冻结时的补齐（逐条记在 doc/CHANGELOG-contract.md）：
--   C-01 tasks 表：§8.1 有 /tasks/{id} 端点但 §7.1 无表，202+task_id 无处持久化
--   C-02 diagnoses.needs_review：§3.3.6 要求 condition_check 诊断带 needs_review
--   C-03 claims.seq：E4「禁止倒因为果」需要 claim 的产生序号
--   C-04 eval_runs.seed_policy / split：I5 与报告头（§10.4.5）要求
--   C-05 eval_items.run_index / seed：pass^k 需要逐 run 独立试验坐标（§10.4.1）
-- ============================================================================

CREATE TABLE questions (
  id             TEXT PRIMARY KEY,
  course         TEXT NOT NULL,
  stem           TEXT NOT NULL,
  correct_answer TEXT NOT NULL,
  qtype          TEXT,
  difficulty     TEXT,
  kp_ids         TEXT NOT NULL DEFAULT '[]',
  source         TEXT,
  created_at     TEXT NOT NULL
);

CREATE TABLE attempts (
  id             TEXT PRIMARY KEY,
  question_id    TEXT NOT NULL REFERENCES questions(id),
  student_answer TEXT NOT NULL,
  fingerprint    TEXT,
  created_at     TEXT NOT NULL
);

CREATE INDEX idx_attempts_question ON attempts(question_id);
CREATE INDEX idx_attempts_fingerprint ON attempts(fingerprint);

CREATE TABLE tasks (
  id              TEXT PRIMARY KEY,
  attempt_id      TEXT NOT NULL REFERENCES attempts(id),
  run_id          TEXT,
  status          TEXT NOT NULL,
  terminal        TEXT,
  reason_code     TEXT,
  idempotency_key TEXT,
  estimated_ms    INTEGER,
  tokens_in       INTEGER NOT NULL DEFAULT 0,
  tokens_out      INTEGER NOT NULL DEFAULT 0,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);

CREATE UNIQUE INDEX uq_tasks_idempotency ON tasks(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX idx_tasks_attempt ON tasks(attempt_id);
CREATE INDEX idx_tasks_status ON tasks(status);

CREATE TABLE diagnoses (
  id                       TEXT PRIMARY KEY,
  attempt_id               TEXT NOT NULL REFERENCES attempts(id),
  run_id                   TEXT NOT NULL,
  error_type               TEXT NOT NULL,
  error_subtype            TEXT,
  first_divergent_step     INTEGER,
  locator                  TEXT,
  root_cause               TEXT,
  hypotheses               TEXT,
  kp_ids                   TEXT,
  confidence               REAL,
  needs_review             INTEGER NOT NULL DEFAULT 0,
  terminal                 TEXT NOT NULL,
  reason_code              TEXT,
  arm                      TEXT NOT NULL,
  prompt_version_id        TEXT NOT NULL,
  model                    TEXT NOT NULL,
  temperature              REAL NOT NULL,
  seed                     INTEGER,
  evidence_binding         TEXT NOT NULL,
  guard_retries            INTEGER NOT NULL DEFAULT 0,
  cascade_level            INTEGER NOT NULL DEFAULT 0,
  verify_agrees            INTEGER,
  suspected_bad_answer_key TEXT NOT NULL DEFAULT 'none',
  input_tokens             INTEGER,
  output_tokens            INTEGER,
  latency_ms               INTEGER,
  created_at               TEXT NOT NULL
);

CREATE INDEX idx_diag_attempt ON diagnoses(attempt_id);
CREATE INDEX idx_diag_run ON diagnoses(run_id);
CREATE INDEX idx_diag_type ON diagnoses(error_type);

CREATE TABLE evidence (
  id          TEXT PRIMARY KEY,
  run_id      TEXT NOT NULL,
  item_id     TEXT,
  kind        TEXT NOT NULL,
  payload     TEXT NOT NULL DEFAULT '{}',
  status      TEXT NOT NULL,
  produced_by TEXT NOT NULL,
  seq         INTEGER NOT NULL,
  created_at  TEXT NOT NULL
);

CREATE UNIQUE INDEX uq_evidence_run_seq ON evidence(run_id, seq);
CREATE INDEX idx_evidence_run ON evidence(run_id);

CREATE TABLE claims (
  id            TEXT PRIMARY KEY,
  diagnosis_id  TEXT NOT NULL REFERENCES diagnoses(id),
  field         TEXT NOT NULL,
  value         TEXT NOT NULL,
  evidence_ids  TEXT NOT NULL DEFAULT '[]',
  seq           INTEGER NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE INDEX idx_claims_diag ON claims(diagnosis_id);

CREATE TABLE explanation_steps (
  id            TEXT PRIMARY KEY,
  diagnosis_id  TEXT NOT NULL REFERENCES diagnoses(id),
  seq           INTEGER NOT NULL,
  title         TEXT NOT NULL,
  content       TEXT NOT NULL,
  key_point     TEXT,
  targets_error TEXT,
  verifiable    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_steps_diag ON explanation_steps(diagnosis_id, seq);

CREATE TABLE similar_questions (
  id           TEXT PRIMARY KEY,
  diagnosis_id TEXT NOT NULL REFERENCES diagnoses(id),
  stem         TEXT NOT NULL,
  answer       TEXT,
  solution     TEXT,
  check_level  TEXT,
  check_passed INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_similar_diag ON similar_questions(diagnosis_id);

CREATE TABLE kp (
  id        TEXT PRIMARY KEY,
  name      TEXT NOT NULL,
  parent_id TEXT REFERENCES kp(id)
);

CREATE TABLE kp_links (
  diagnosis_id TEXT NOT NULL REFERENCES diagnoses(id),
  kp_id        TEXT NOT NULL REFERENCES kp(id),
  PRIMARY KEY (diagnosis_id, kp_id)
);

CREATE TABLE prompt_versions (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  arm          TEXT NOT NULL,
  version      TEXT NOT NULL,
  content      TEXT NOT NULL,
  few_shot     TEXT NOT NULL DEFAULT '[]',
  is_active    INTEGER NOT NULL DEFAULT 0,
  hash         TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  UNIQUE (name, version)
);

CREATE TABLE eval_runs (
  id                TEXT PRIMARY KEY,
  arm               TEXT NOT NULL,
  split             TEXT NOT NULL DEFAULT 'heldout',
  dataset_hash      TEXT NOT NULL,
  prompt_version_id TEXT NOT NULL,
  model             TEXT NOT NULL,
  temperature       REAL NOT NULL,
  seed              INTEGER,
  seed_policy       TEXT NOT NULL DEFAULT 'fixed',
  k                 INTEGER NOT NULL DEFAULT 1,
  started_at        TEXT,
  finished_at       TEXT,
  total_tokens      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE eval_items (
  id         TEXT PRIMARY KEY,
  run_id     TEXT NOT NULL REFERENCES eval_runs(id),
  attempt_id TEXT NOT NULL REFERENCES attempts(id),
  run_index  INTEGER NOT NULL DEFAULT 0,
  seed       INTEGER,
  predicted  TEXT,
  gold       TEXT,
  metrics    TEXT,
  terminal   TEXT,
  latency_ms INTEGER,
  tokens     INTEGER
);

CREATE INDEX idx_eval_items_run ON eval_items(run_id);
CREATE UNIQUE INDEX uq_eval_items_run_attempt_index ON eval_items(run_id, attempt_id, run_index);

CREATE TABLE trace_events (
  id          TEXT PRIMARY KEY,
  run_id      TEXT NOT NULL,
  item_id     TEXT,
  node        TEXT NOT NULL,
  seq         INTEGER NOT NULL,
  parent_seq  INTEGER,
  start_ms    INTEGER NOT NULL,
  end_ms      INTEGER,
  tokens_in   INTEGER NOT NULL DEFAULT 0,
  tokens_out  INTEGER NOT NULL DEFAULT 0,
  status      TEXT NOT NULL,
  reason_code TEXT,
  detail      TEXT NOT NULL DEFAULT '{}',
  UNIQUE (run_id, seq)
);

CREATE INDEX idx_trace_run ON trace_events(run_id, seq);
CREATE INDEX idx_trace_item ON trace_events(item_id);

CREATE TABLE sandbox_rejects (
  id          TEXT PRIMARY KEY,
  run_id      TEXT,
  verdict     TEXT NOT NULL,
  reason      TEXT NOT NULL,
  expr_digest TEXT NOT NULL,
  expr_len    INTEGER NOT NULL,
  created_at  TEXT NOT NULL
);

CREATE INDEX idx_sandbox_run ON sandbox_rejects(run_id);

CREATE TABLE redteam_cases (
  id             TEXT PRIMARY KEY,
  category       TEXT NOT NULL,
  stem           TEXT NOT NULL,
  student_answer TEXT NOT NULL,
  target         TEXT,
  expect         TEXT
);

CREATE TABLE gold_labels (
  id                TEXT PRIMARY KEY,
  attempt_id        TEXT NOT NULL REFERENCES attempts(id),
  error_type_major  TEXT NOT NULL,
  key_steps         TEXT NOT NULL DEFAULT '[]',
  first_wrong_step  INTEGER,
  labeler           TEXT NOT NULL,
  labeled_at        TEXT NOT NULL,
  round             INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_gold_attempt ON gold_labels(attempt_id);
