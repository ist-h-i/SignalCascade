import { useEffect, useState } from 'react'
import { normalizeDashboardData } from './dashboard/data-contract'
import DashboardPage from './dashboard/DashboardPage'
import type { DashboardData } from './dashboard/types'
import type { DashboardDataInput } from './dashboard/data-contract'

function App() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function loadDashboard() {
      try {
        const response = await fetch('/dashboard-data.json')
        if (!response.ok) {
          throw new Error(`dashboard-data.json の取得に失敗しました: ${response.status}`)
        }

        const payload = normalizeDashboardData((await response.json()) as DashboardDataInput)
        if (!cancelled) {
          setData(payload)
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'ダッシュボードデータを読み込めませんでした。')
        }
      }
    }

    void loadDashboard()

    return () => {
      cancelled = true
    }
  }, [])

  if (error) {
    return (
      <main className="dashboard-status-shell">
        <section className="dashboard-status">
          <p className="dashboard-eyebrow">SignalCascade / Frontend</p>
          <h1>読み込みエラー</h1>
          <p>{error}</p>
          <p className="dashboard-status__hint">`Frontend` で `npm run sync:data` を実行してから再読み込みしてください。</p>
        </section>
      </main>
    )
  }

  if (!data) {
    return (
      <main className="dashboard-status-shell">
        <section className="dashboard-status">
          <p className="dashboard-eyebrow">SignalCascade / Frontend</p>
          <h1>読み込み中</h1>
          <p>ダッシュボードデータを読み込んでいます。</p>
        </section>
      </main>
    )
  }

  return <DashboardPage data={data} />
}

export default App
