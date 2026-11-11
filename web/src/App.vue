<script setup>
import { computed, onMounted, ref } from 'vue'
import { createAttempt, getDiagnosis, getHealth, getTask, startDiagnose, streamTask } from './api'

const PRESETS = [
  { label: '分数加法', stem: '计算 1/2 + 1/3，结果写成最简分数。', correct: '5/6', student: '5/7' },
  { label: '指数', stem: '求 2^10 的值。', correct: '1024', student: '20' },
  { label: '对数', stem: '求 log_2(64) 的值。', correct: '6', student: '8' },
  { label: '定积分（会转人工）', stem: '求 ∫_0^1 x^2 dx 的值。', correct: '1/3', student: '1/4' },
]

const LABELS = {
  diagnosed: '已诊断',
  escalated: '转人工复核',
  needs_clarification: '需要补充信息',
  uncertain: '未完成',
  canceled: '已取消',
  failed: '失败',
}

const stem = ref(PRESETS[0].stem)
const correct = ref(PRESETS[0].correct)
const student = ref(PRESETS[0].student)
const running = ref(false)
const error = ref('')
const health = ref('连接中…')
const ok = ref(false)
const stages = ref([])
const evidence = ref([])
const guard = ref('')
const result = ref(null)
const diagnosis = ref(null)

const badge = computed(() =>
  result.value ? LABELS[result.value.terminal] || result.value.terminal : '',
)

function preset(index) {
  stem.value = PRESETS[index].stem
  correct.value = PRESETS[index].correct
  student.value = PRESETS[index].student
  reset()
}

function reset() {
  error.value = ''
  stages.value = []
  evidence.value = []
  guard.value = ''
  result.value = null
  diagnosis.value = null
}

function onFrame(frame) {
  const d = frame.data
  if (frame.event === 'stage' && d.stage !== 'heartbeat') {
    const i = stages.value.findIndex((s) => s.name === d.stage)
    const item = { name: d.stage, status: d.status }
    if (i >= 0) stages.value[i] = item
    else stages.value.push(item)
  } else if (frame.event === 'evidence') {
    evidence.value.push(d)
  } else if (frame.event === 'guard') {
    guard.value = d.schema === 'pass' ? '证据绑定通过' : '证据绑定未通过 → 转人工'
  } else if (frame.event === 'done') {
    result.value = { terminal: d.terminal, reason_code: d.reason_code, ms: d.latency_ms }
  }
}

async function run() {
  reset()
  running.value = true
  try {
    const attempt = await createAttempt({
      question: {
        course: '数学',
        stem: stem.value,
        correct_answer: correct.value,
        qtype: 'numeric',
      },
      student_answer: student.value,
    })
    const task = await startDiagnose(attempt.attempt_id)

    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('等待超时（90s）')), 90000)
      streamTask(task.task_id, onFrame, () => {
        clearTimeout(timer)
        resolve()
      })
    })

    const final = await getTask(task.task_id)
    result.value = {
      terminal: final.terminal,
      reason_code: final.reason_code,
      ms: result.value ? result.value.ms : null,
    }
    if (final.diagnosis_id) diagnosis.value = await getDiagnosis(final.diagnosis_id)
  } catch (e) {
    error.value = e.message
  } finally {
    running.value = false
  }
}

onMounted(async () => {
  try {
    const body = await getHealth()
    ok.value = body.status === 'ok'
    health.value = `${body.provider.model} · ${body.provider.effective_mode}`
  } catch {
    health.value = '后端未启动'
  }
})
</script>

<template>
  <div class="app">
    <header>
      <h1>VeriDx 证诊</h1>
      <span class="sub">模型只解释「为什么错」，对错由程序裁决</span>
      <span class="health"><i :class="ok ? 'dot ok' : 'dot'"></i>{{ health }}</span>
    </header>

    <div class="grid">
      <section class="card">
        <h2>录入错题</h2>
        <label>题干</label>
        <textarea v-model="stem" rows="3" :disabled="running"></textarea>
        <label>标准答案</label>
        <input v-model="correct" :disabled="running" />
        <label>学生答案</label>
        <input v-model="student" :disabled="running" />

        <div class="presets">
          <button v-for="(p, i) in PRESETS" :key="p.label" @click="preset(i)">{{ p.label }}</button>
        </div>

        <button class="primary" :disabled="running" @click="run">
          {{ running ? '诊断中…' : '开始诊断' }}
        </button>
        <p v-if="error" class="error">{{ error }}</p>
      </section>

      <section class="card">
        <h2>诊断结果</h2>

        <div v-if="stages.length" class="stages">
          <span v-for="s in stages" :key="s.name" class="stage" :class="s.status">{{ s.name }}</span>
        </div>

        <div v-if="result" class="terminal" :class="result.terminal">
          <b>{{ badge }}</b>
          <span v-if="result.reason_code">reason: {{ result.reason_code }}</span>
          <span v-else-if="result.terminal === 'diagnosed'">证据绑定通过</span>
          <span v-if="result.ms">· {{ (result.ms / 1000).toFixed(1) }}s</span>
        </div>

        <p v-if="guard" class="muted">{{ guard }}</p>

        <div v-if="diagnosis">
          <dl class="kv">
            <dt>错因</dt>
            <dd>
              {{ diagnosis.error_type }}
              <template v-if="diagnosis.error_subtype"> / {{ diagnosis.error_subtype }}</template>
            </dd>
            <dt>根因</dt>
            <dd>{{ diagnosis.root_cause || '—' }}</dd>
            <dt>置信度</dt>
            <dd>
              {{ diagnosis.confidence }}
              <span v-if="diagnosis.needs_review" class="tag">需人工复核</span>
            </dd>
            <dt>执行坐标</dt>
            <dd>{{ diagnosis.model }} · {{ diagnosis.arm }} 臂 · {{ diagnosis.prompt_version_id }}</dd>
          </dl>

          <h3>断言 → 证据</h3>
          <div v-for="c in diagnosis.claims" :key="c.field" class="claim">
            <div class="row"><b>{{ c.field }}</b><span>seq {{ c.seq }}</span></div>
            <div>{{ Array.isArray(c.value) ? c.value.join(' / ') : c.value }}</div>
            <div class="chips"><span v-for="id in c.evidence_ids" :key="id">{{ id }}</span></div>
          </div>

          <h3>分步讲解</h3>
          <div v-for="s in diagnosis.steps" :key="s.seq" class="claim">
            <div class="row">
              <b>{{ s.seq }}. {{ s.title }}</b>
              <span class="tag">{{ s.verifiable ? '可验证' : '教学性表述' }}</span>
            </div>
            <div class="muted">针对：{{ s.targets_error }}</div>
            <div>{{ s.content }}</div>
          </div>

          <h3>程序证据</h3>
          <div v-for="e in diagnosis.evidence" :key="e.id" class="claim">
            <div class="row">
              <b>{{ e.id }}</b>
              <span>{{ e.kind }} · {{ e.status }} · by {{ e.produced_by }}</span>
            </div>
            <pre>{{ JSON.stringify(e.payload, null, 2) }}</pre>
          </div>
        </div>

        <!-- 没产出诊断时也要展示：程序证据已经产生，只是结论立不住（能力边界） -->
        <div v-else-if="result && evidence.length">
          <h3>本次收集到的程序证据</h3>
          <div v-for="e in evidence" :key="e.id" class="claim">
            <div class="row">
              <b>{{ e.id }}</b>
              <span>{{ e.kind }} · {{ e.status }}</span>
            </div>
            <pre>{{ JSON.stringify(e.payload, null, 2) }}</pre>
          </div>
        </div>

        <p v-else-if="!running && !result" class="muted center">
          左侧填一道错题并点「开始诊断」，过程与证据会实时出现在这里。
        </p>
        <p v-else-if="running && evidence.length" class="muted">
          已收到 {{ evidence.length }} 条程序证据，等待模型断言…
        </p>
      </section>
    </div>
  </div>
</template>
