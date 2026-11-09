-- ============================================================================
-- VeriDx schema v2 —— P0.6 补丁（G0.6）
-- 依据：doc/review/P0-review.md #1/#6/#7 + P0.6 审批意见 §3.2/§4.C
-- 纪律：001_init.sql 一字不改；本文件只做增量与「官方 12-step 表重建」。
--
-- 表重建统一用下面的顺序（**不能先改名旧表**，否则子表 REFERENCES 会被改写）：
--   建新表（临时名） → INSERT SELECT 拷贝 → DROP 旧表 → RENAME 新表
-- 外键开关由 store/db.py 在事务之外关掉，事务内跑 PRAGMA foreign_key_check。
-- ============================================================================

-- ---------------------------------------------------------------------------
-- C-11  evidence 主键：全局 id -> (run_id, id)
-- 原因：ev_N 的语义是「run 内编号」，全局唯一从来不是它的属性。
-- 实测：第二个 run 插 ev_1 会 UNIQUE constraint failed。
-- ---------------------------------------------------------------------------
CREATE TABLE evidence_new (
  id          TEXT NOT NULL,
  run_id      TEXT NOT NULL,
  item_id     TEXT,
  kind        TEXT NOT NULL,
  payload     TEXT NOT NULL DEFAULT '{}',
  status      TEXT NOT NULL,
  produced_by TEXT NOT NULL,
  seq         INTEGER NOT NULL,
  created_at  TEXT NOT NULL,
  PRIMARY KEY (run_id, id)
);

INSERT INTO evidence_new (id, run_id, item_id, kind, payload, status, produced_by, seq, created_at)
SELECT id, run_id, item_id, kind, payload, status, produced_by, seq, created_at FROM evidence;

DROP TABLE evidence;

ALTER TABLE evidence_new RENAME TO evidence;

CREATE UNIQUE INDEX uq_evidence_run_seq ON evidence(run_id, seq);
CREATE INDEX idx_evidence_run ON evidence(run_id);

-- ---------------------------------------------------------------------------
-- C-16  gold_labels 补两列（可空）：M18 的分母来源
-- 不用 DEFAULT 0：未标注必须显式暴露为 NULL，否则会静默缩小 M18 分母。
-- ---------------------------------------------------------------------------
ALTER TABLE gold_labels ADD COLUMN student_is_correct INTEGER;
ALTER TABLE gold_labels ADD COLUMN answer_key_correct INTEGER;

-- ---------------------------------------------------------------------------
-- C-14/C-15  attempts：fingerprint -> answer_digest（只描述内容体，不含实验坐标）
-- C-17       UNIQUE(question_id, student_answer)
-- 注：这条唯一约束建立在「单用户假设」上；将来加 user_id 时必须改成
--     (user_id, question_id, student_answer)，否则会误伤多用户场景。
-- ---------------------------------------------------------------------------
CREATE TABLE attempts_new (
  id             TEXT PRIMARY KEY,
  question_id    TEXT NOT NULL REFERENCES questions(id),
  student_answer TEXT NOT NULL,
  answer_digest  TEXT,
  created_at     TEXT NOT NULL
);

INSERT INTO attempts_new (id, question_id, student_answer, answer_digest, created_at)
SELECT id, question_id, student_answer, fingerprint, created_at FROM attempts;

DROP TABLE attempts;

ALTER TABLE attempts_new RENAME TO attempts;

CREATE INDEX idx_attempts_question ON attempts(question_id);
CREATE UNIQUE INDEX uq_attempts_question_answer ON attempts(question_id, student_answer);

-- ---------------------------------------------------------------------------
-- C-17  tasks：status 与 terminal 不得互相矛盾
-- 保守版（不改已冻结的 TaskStatus 枚举）：两个状态必须同时成立或同时为空。
-- ---------------------------------------------------------------------------
CREATE TABLE tasks_new (
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
  updated_at      TEXT NOT NULL,
  CONSTRAINT ck_tasks_status CHECK (
    status IN ('queued', 'running', 'diagnosed', 'escalated',
               'needs_clarification', 'uncertain', 'canceled', 'failed')
  ),
  CONSTRAINT ck_tasks_terminal CHECK (
    (status IN ('queued', 'running') AND terminal IS NULL)
    OR (status NOT IN ('queued', 'running') AND terminal = status)
  )
);

INSERT INTO tasks_new (id, attempt_id, run_id, status, terminal, reason_code,
                       idempotency_key, estimated_ms, tokens_in, tokens_out,
                       created_at, updated_at)
SELECT id, attempt_id, run_id, status, terminal, reason_code,
       idempotency_key, estimated_ms, tokens_in, tokens_out,
       created_at, updated_at
FROM tasks;

DROP TABLE tasks;

ALTER TABLE tasks_new RENAME TO tasks;

-- 幂等键保持「全局唯一」（P0.6 审批 §3.1：放宽到 (attempt_id, key) 会让串号合法化）
CREATE UNIQUE INDEX uq_tasks_idempotency ON tasks(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX idx_tasks_attempt ON tasks(attempt_id);
CREATE INDEX idx_tasks_status ON tasks(status);
