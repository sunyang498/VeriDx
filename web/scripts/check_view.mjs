// B-1 / B-2 / B-3 的界面口径断言（不需要浏览器，`node scripts/check_view.mjs` 直接跑）。
// 反向验证：把 view.js 里对应的分支删掉/改回旧的"一个布尔"版本，这里必须变红。
import { BINDING_LABELS, durationText, evidenceView, guardView, terminalMessage } from '../src/view.js'

let failed = 0
function check(name, ok, extra) {
  if (ok) {
    console.log(`ok   ${name}`)
    return
  }
  failed += 1
  console.log(`FAIL ${name}${extra ? `  ${extra}` : ''}`)
}

// ---- B-1：三档口径 -------------------------------------------------------
const partial = guardView({ evidence_binding: 'partial', accepted: 1, total: 2, schema: 'pass', retries: 0 })
check('B-1 partial 显示为"部分通过"', partial.label === BINDING_LABELS.partial, partial.label)
check('B-1 partial 带分子分母', partial.detail.includes('1/2 条断言有证据'), partial.detail)

const pass = guardView({ evidence_binding: 'pass', accepted: 2, total: 2, schema: 'pass', retries: null })
check('B-1 pass 带分子分母', pass.detail.includes('2/2 条断言有证据'), pass.detail)
check('B-1 retries=null 不显示"重试 0 次"', !pass.detail.includes('重试'), pass.detail)

const retried = guardView({ evidence_binding: 'pass', accepted: 2, total: 2, schema: 'retry', retries: 2 })
check(
  'B-1 schema 重试独立可见',
  retried.detail.includes('第 2 次才通过') && retried.detail.includes('重试 2 次'),
  retried.detail,
)

const unknown = guardView({ evidence_binding: 'pass', accepted: null, total: null, schema: null })
check(
  'B-1 缺值显示 —，不顶替',
  unknown.detail.includes('schema —') && !unknown.detail.includes('条断言'),
  unknown.detail,
)

// ---- B-2：failed 必须可排查 ---------------------------------------------
const failedMsg = terminalMessage('failed', 'internal_error', 'r-abc123')
check('B-2 failed 给 run_id', failedMsg.includes('r-abc123'), failedMsg)
check('B-2 failed 说人话', failedMsg.includes('内部错误'), failedMsg)
check('B-2 无耗时/0 不显示秒', durationText(null) === '' && durationText(0) === '' && durationText(1500).includes('1.5s'))
check('B-2 diagnosed 不重复说证据绑定', terminalMessage('diagnosed', null, 'r-1') === '')

// ---- B-3：白名单渲染 ----------------------------------------------------
const stepDiff = evidenceView({
  id: 'ev_2',
  kind: 'step_diff',
  status: 'skipped',
  payload: {
    status: 'not_implemented',
    model_steps: 2,
    student_steps: 1,
    first_divergence: null,
    locator: 'model',
    note: 'P1.8 实现 sympy 逐步对齐；在此之前 first_divergent_step 只能由模型判断',
  },
})
const dump = JSON.stringify(stepDiff)
check('B-3 界面文本里没有 not_implemented', !dump.includes('not_implemented'), dump)
check('B-3 界面文本里没有 P1.8', !dump.includes('P1.8'), dump)
check(
  'B-3 仍保留业务字段',
  stepDiff.fields.some((f) => f.label === '模型步数' && f.value === '2'),
  JSON.stringify(stepDiff.fields),
)

const numeric = evidenceView({
  id: 'ev_1',
  kind: 'numeric_check',
  status: 'ok',
  payload: { expr: '5/6', student_answer: '5/7', equal: false, comparable: true },
})
check(
  'B-3 numeric_check 显示判等',
  numeric.fields.some((f) => f.label === '判等' && f.value === '否'),
  JSON.stringify(numeric.fields),
)

if (failed) {
  console.log(`\n${failed} 条界面口径断言失败`)
  process.exit(1)
}
console.log('\n界面口径全部通过')
