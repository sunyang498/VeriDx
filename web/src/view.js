// 界面上"说的每一句话"都由这里生成 —— 纯函数、无 DOM、无 Vue，
// 目的是让 `node scripts/check_view.mjs` 能直接跑断言（B-1/B-2/B-3 的可测性）。

/** 三档口径（C-33）：部分通过必须可见，不能被"通过"两个字吞掉。 */
export const BINDING_LABELS = {
  pass: '证据绑定通过',
  partial: '部分通过',
  fail_after_retry: '证据绑定未通过 → 转人工',
  skipped: '证据绑定未执行',
}

function ratio(d) {
  const accepted = typeof d.accepted === 'number' ? d.accepted : null
  const total = typeof d.total === 'number' ? d.total : null
  return accepted === null || total === null ? '' : `${accepted}/${total} 条断言有证据`
}

/**
 * guard 事件的展示口径：`schema` / `evidence_binding` / 重试次数是**三个独立判据**。
 * 拿不到的值一律显示 `—`，绝不拿别的字段顶替。
 */
export function guardView(d) {
  const binding = d && d.evidence_binding
  const known = Object.prototype.hasOwnProperty.call(BINDING_LABELS, binding)
  const schema =
    d && d.schema === 'pass'
      ? 'schema 通过'
      : d && d.schema === 'retry'
        ? 'schema 第 2 次才通过'
        : d && d.schema === 'fail'
          ? 'schema 未通过'
          : 'schema —'
  const retries =
    d && typeof d.retries === 'number' && d.retries > 0 ? `重试 ${d.retries} 次` : ''
  return {
    level: known ? (binding === 'fail_after_retry' ? 'fail' : binding) : 'unknown',
    label: known ? BINDING_LABELS[binding] : '证据绑定 —',
    detail: [ratio(d), schema, retries].filter(Boolean).join(' · '),
  }
}

/** 终态的一句话。`failed` 必须给出 run_id —— 那才是能回放的那把钥匙。 */
export function terminalMessage(terminal, reasonCode, runId) {
  const tail = runId ? `（run_id ${runId}）` : '（本次未拿到 run_id）'
  if (terminal === 'failed') {
    return `系统内部错误，已记录可回放${tail}`
  }
  if (terminal === 'diagnosed') {
    return ''
  }
  return reasonCode ? `原因：${reasonCode}${tail}` : `未完成${tail}`
}

/** 没有耗时数据就不显示 —— `0.0s` 会被读成"秒失败 = 参数错"。 */
export function durationText(ms) {
  return typeof ms === 'number' && ms > 0 ? `· ${(ms / 1000).toFixed(1)}s` : ''
}

// 开发备注不许出现在产品界面上（B-3）。这两个清单是**唯一**的出口过滤。
const DEV_KEYS = new Set(['note', 'notes', 'dev_note', 'todo'])
const DEV_VALUES = new Set(['not_implemented'])

function sanitize(value) {
  if (Array.isArray(value)) return value.map(sanitize)
  if (!value || typeof value !== 'object') return DEV_VALUES.has(value) ? '—' : value
  const out = {}
  for (const [key, item] of Object.entries(value)) {
    if (DEV_KEYS.has(key)) continue
    if (typeof item === 'string' && DEV_VALUES.has(item)) continue
    out[key] = sanitize(item)
  }
  return out
}

function fmt(value) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

/** 按 `kind` 的白名单取业务字段；原始数据（已剔除开发备注）折进折叠区。 */
export function evidenceView(e) {
  const p = (e && e.payload) || {}
  const rows = []
  const push = (label, value) => rows.push({ label, value: fmt(value) })

  if (e.kind === 'numeric_check') {
    if ('expr' in p) push('程序核对的式子', p.expr)
    if ('student_answer' in p) push('学生作答', p.student_answer)
    if ('equal' in p) push('判等', p.equal)
    if (p.equal === false) push('结论', '学生答案与本式不等')
    if (p.comparable === false) push('为什么判不了', p.blocked_reason)
  } else if (e.kind === 'step_diff') {
    push('首个分歧步', p.first_divergence === null ? '未能定位（程序侧未实现逐步对齐）' : p.first_divergence)
    if ('model_steps' in p) push('模型步数', p.model_steps)
    if ('student_steps' in p) push('学生步数', p.student_steps)
  } else if (e.kind === 'kb_lookup') {
    push('查询知识点', p.requested)
    push('命中', p.found)
    push('缺失', p.missing)
  } else if (e.kind === 'condition_check') {
    push('条件', p.conditions ?? p.condition)
    push('结果', p.holds ?? p.result)
  } else if (e.kind === 'student_text') {
    push('学生作答原文', p.text ?? e.id)
  }

  if (rows.length === 0) {
    for (const [key, value] of Object.entries(sanitize(p))) {
      if (['string', 'number', 'boolean'].includes(typeof value)) push(key, value)
    }
  }
  return {
    title: `${e.id} · ${e.kind} · ${e.status}`,
    fields: rows,
    raw: JSON.stringify(sanitize(p), null, 2),
  }
}
