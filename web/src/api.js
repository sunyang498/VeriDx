// 演示用的最小 API 封装：4 个 REST + 1 个 SSE
const BASE = '/api/v1'

async function call(path, options) {
  const response = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  const text = await response.text()
  const body = text ? JSON.parse(text) : null
  if (!response.ok) throw new Error(body?.message || body?.detail || `HTTP ${response.status}`)
  return body
}

export const createAttempt = (payload) =>
  call('/attempts', { method: 'POST', body: JSON.stringify(payload) })

export const startDiagnose = (attemptId) =>
  call(`/attempts/${attemptId}/diagnose`, { method: 'POST', body: '{}' })

export const getTask = (taskId) => call(`/tasks/${taskId}`)
export const getDiagnosis = (id) => call(`/diagnoses/${id}`)
export const getHealth = () => call('/health')

/** 订阅 SSE；`done` 到达后关闭。返回取消函数。 */
export function streamTask(taskId, onFrame, onClose) {
  const source = new EventSource(`${BASE}/tasks/${taskId}/stream`)
  const names = ['stage', 'evidence', 'claim', 'guard', 'done']

  names.forEach((name) => {
    source.addEventListener(name, (event) => {
      onFrame({ event: name, data: JSON.parse(event.data) })
      if (name === 'done') {
        source.close()
        onClose()
      }
    })
  })

  return () => source.close()
}
