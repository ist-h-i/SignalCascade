import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  PolarAngleAxis,
  RadialBar,
  RadialBarChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  useXAxisScale,
  useYAxisScale,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

type HorizonRow = {
  horizon: number
  hours: number
  predictedClose: number
  lowerClose: number
  upperClose: number
  uncertainty: number
  expectedReturnPct: number
  actualClose: number | null
  actualReturnPct: number | null
}

type ChartRow = {
  ts: number
  label: string
  phase: 'history' | 'forecast'
  close: number | null
  high: number | null
  low: number | null
  forecastBase: number | null
  forecastLower: number | null
  forecastUpper: number | null
  forecastSpread: number | null
  actualClose: number | null
  actualRangeStart: number | null
  actualRangeEnd: number | null
}

type MicroCandleRow = {
  ts: number
  open: number
  high: number
  low: number
  close: number
}

type MetricPoint = {
  epoch: number
  trainTotal: number
  validationTotal: number
  trainReturn: number
  validationReturn: number
  trainOverlay: number
  validationOverlay: number
}

type DashboardData = {
  generatedAt: string
  instrument: string
  provenance: {
    rawRows: number
    sourcePath: string
    start: string
    end: string
  }
  run: {
    anchorTime: string
    anchorClose: number
    selectedHorizon: number
    selectedHours: number
    position: number
    overlayAction: string
    trainSamples: number
    validationSamples: number
    sampleCount: number
    bestValidationLoss: number
    bestEpoch: number
    convergenceGain: number
    generalizationGap: number
    sourceRows: number
    tuningSessionId?: string | null
    bestParams?: {
      epochs: number
      batchSize: number
      learningRate: number
      hiddenDim: number
      dropout: number
      weightDecay: number
    } | null
  }
  chart: {
    rows: ChartRow[]
    microRows: MicroCandleRow[]
    yDomain: [number, number]
  }
  horizons: HorizonRow[]
  metrics: {
    history: MetricPoint[]
    validation: {
      returnRmse: number | null
      returnMae: number | null
      directionalAccuracy: number | null
      uncertaintyCalibrationError: number | null
      coverageAt1Sigma: number | null
      overlayAccuracy: number | null
      overlayMacroF1: number | null
      valuePerSignal: number | null
      downsidePerSignal: number | null
      valueCaptureRatio: number | null
      utilityScore: number | null
    } | null
  }
  narrative: {
    title: string
    summary: string
    bullets: string[]
  }
}

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
        const payload = (await response.json()) as DashboardData
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
      <main className="shell">
        <section className="status-view">
          <p className="eyebrow">SignalCascade / Frontend</p>
          <h1>読み込みエラー</h1>
          <p>{error}</p>
          <p className="status-view__hint">`frontend` で `npm run sync:data` を実行してから再読み込みしてください。</p>
        </section>
      </main>
    )
  }

  if (!data) {
    return (
      <main className="shell">
        <section className="status-view">
          <p className="eyebrow">SignalCascade / Frontend</p>
          <h1>読み込み中</h1>
          <p>データを読み込んでいます。</p>
        </section>
      </main>
    )
  }

  const selectedForecast = data.horizons.find((row) => row.horizon === data.run.selectedHorizon) ?? data.horizons[0]
  const bestEpochMetric = data.metrics.history.find((row) => row.epoch === data.run.bestEpoch) ?? data.metrics.history[0]
  const latestMetric = data.metrics.history[data.metrics.history.length - 1]
  const chartYDomain = getVisiblePriceDomain(data.chart.rows)
  const anchorTs = new Date(data.run.anchorTime).getTime()
  const forecastRows = data.chart.rows.filter((row) => row.phase === 'forecast')
  const availableHorizons = data.horizons.filter(
    (row) => row.actualClose !== null && row.actualReturnPct !== null,
  )
  const directionalAccuracy =
    data.metrics.validation?.directionalAccuracy ??
    (availableHorizons.length > 0
      ? availableHorizons.filter((row) => Math.sign(row.expectedReturnPct) === Math.sign(row.actualReturnPct ?? 0)).length /
        availableHorizons.length
      : null)
  const rangeCoverage =
    data.metrics.validation?.coverageAt1Sigma ??
    (availableHorizons.length > 0
      ? availableHorizons.filter(
          (row) => (row.actualClose ?? Number.NaN) >= row.lowerClose && (row.actualClose ?? Number.NaN) <= row.upperClose,
        ).length / availableHorizons.length
      : null)
  const averagePriceErrorPct =
    data.metrics.validation?.returnMae ??
    (availableHorizons.length > 0
      ? availableHorizons.reduce((sum, row) => {
          const actualClose = row.actualClose ?? row.predictedClose
          return sum + Math.abs((actualClose - row.predictedClose) / actualClose)
        }, 0) / availableHorizons.length
      : null)
  const utilityScore = data.metrics.validation?.utilityScore ?? null
  const valueCaptureRatio = data.metrics.validation?.valueCaptureRatio ?? null
  const valuePerSignal = data.metrics.validation?.valuePerSignal ?? null
  const overlayMacroF1 = data.metrics.validation?.overlayMacroF1 ?? null
  const overlayAccuracy = data.metrics.validation?.overlayAccuracy ?? null
  const downsidePerSignal = data.metrics.validation?.downsidePerSignal ?? null
  const uncertaintyCalibrationError = data.metrics.validation?.uncertaintyCalibrationError ?? null
  const normalizedUncertainty = getNormalizedUncertainty(selectedForecast.uncertainty, data.horizons)
  const normalizedGap = clamp01(1 - Math.min(Math.abs(data.run.generalizationGap), 1))
  const normalizedError = averagePriceErrorPct === null ? 0.5 : clamp01(1 - Math.min(averagePriceErrorPct / 0.05, 1))
  const trustScore = Math.round(
    (
      normalizedUncertainty * 0.28 +
      normalizedGap * 0.24 +
      (directionalAccuracy ?? 0.5) * 0.28 +
      normalizedError * 0.2
    ) * 100,
  )
  const confidenceTone = getConfidenceTone(trustScore)
  const confidenceLabel = getConfidenceLabel(trustScore)
  const comparisonChartData = availableHorizons.map((row) => ({
    label: `${row.hours}h`,
    hours: row.hours,
    expectedReturnPct: row.expectedReturnPct,
    actualReturnPct: row.actualReturnPct ?? 0,
  }))
  const reuseNarrative = buildReuseNarrative({
    trustScore,
    directionalAccuracy,
    rangeCoverage,
    averagePriceErrorPct,
    generalizationGap: data.run.generalizationGap,
  })
  const epochDrift = latestMetric.validationTotal - bestEpochMetric.validationTotal
  const predictionBandWidthPct = ((selectedForecast.upperClose - selectedForecast.lowerClose) / data.run.anchorClose) * 100
  const selectedUncertaintyScore = getUncertaintyScore(selectedForecast.uncertainty, data.horizons)
  const selectedUncertaintyLabel = getUncertaintyLabel(selectedUncertaintyScore)
  const positionGauge = [
    {
      name: 'position',
      value: Math.abs(data.run.position) * 100,
      fill: data.run.position >= 0 ? '#daa23a' : '#7f8cff',
    },
  ]

  return (
    <main className="shell">
      <motion.header
        className="topbar"
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: 'easeOut' }}
      >
        <div className="topbar__title">
          <p className="eyebrow">SignalCascade / Gold</p>
          <h1>XAU/USD</h1>
          <span>{formatTimestamp(data.run.anchorTime)}</span>
        </div>

        <div className="topbar__stats">
          <MetricTile label="基準価格" value={formatPrice(data.run.anchorClose)} />
          <MetricTile label={`${selectedForecast.hours}時間先の中心予測`} value={formatPrice(selectedForecast.predictedClose)} />
          <MetricTile label="学習価値スコア" value={formatNullableScore(utilityScore)} accent />
          <MetricTile label="価値回収率" value={formatNullablePercent(valueCaptureRatio)} />
          <MetricTile label="信頼度" value={`${trustScore} / 100`} accent={confidenceTone !== 'confidence-pill--caution'} />
          <MetricTile label="アクション" value={translateOverlay(data.run.overlayAction)} accent />
        </div>
      </motion.header>

      <motion.section
        className="workspace"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.05, ease: 'easeOut' }}
      >
        <section className="surface surface--chart">
          <div className="surface__head">
            <div className="legend">
              <span><i className="legend__dot legend__dot--history" />過去</span>
              <span><i className="legend__dot legend__dot--micro" />30分足</span>
              <span><i className="legend__dot legend__dot--forecast" />予測</span>
              <span><i className="legend__dot legend__dot--actual" />4時間実績</span>
            </div>
          </div>

          <div className="chart-summary">
            <MiniStat label="選択時間軸" value={`${selectedForecast.hours}時間先`} />
            <MiniStat label="予測幅" value={`${predictionBandWidthPct.toFixed(2)}%`} />
            <MiniStat label="予想変化率" value={formatSignedPercent(selectedForecast.expectedReturnPct)} />
            <MiniStat label="不確実性" value={`${selectedUncertaintyScore} / 100 ${selectedUncertaintyLabel}`} />
          </div>

          <div className="chart-frame">
            <ResponsiveContainer width="100%" height={430}>
              <ComposedChart
                data={data.chart.rows}
                baseValue={chartYDomain[0]}
                margin={{ top: 18, right: 8, bottom: 8, left: 0 }}
                >
                <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.08)" />
                <XAxis
                  dataKey="ts"
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={formatAxisDate}
                  stroke="rgba(255,255,255,0.45)"
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  domain={chartYDomain}
                  allowDataOverflow
                  tickFormatter={formatCompactPrice}
                  stroke="rgba(255,255,255,0.45)"
                  tick={{ fontSize: 11 }}
                  width={68}
                />
                <Tooltip content={<ChartTooltip />} />
                <ReferenceArea
                  x1={anchorTs}
                  x2={forecastRows[forecastRows.length - 1]?.ts ?? anchorTs}
                  fill="rgba(255,202,112,0.06)"
                  ifOverflow="extendDomain"
                />
                <ReferenceLine
                  x={anchorTs}
                  stroke="rgba(255,214,122,0.9)"
                  strokeDasharray="4 6"
                  label={{ value: 'Now', position: 'top', fill: '#ffca70', fontSize: 11 }}
                />
                <MicroCandleLayer rows={data.chart.microRows} />
                <Area type="monotone" dataKey="forecastLower" stackId="forecastBand" stroke="transparent" fill="transparent" connectNulls isAnimationActive />
                <Area type="monotone" dataKey="forecastSpread" stackId="forecastBand" stroke="transparent" fill="url(#forecastBand)" connectNulls isAnimationActive />
                <Line type="monotone" dataKey="close" stroke="#f3ede3" strokeWidth={2.2} dot={false} connectNulls isAnimationActive={false} />
                <Line type="monotone" dataKey="forecastBase" stroke="#ffca70" strokeWidth={3} dot={false} connectNulls />
                <Line type="linear" dataKey="actualClose" stroke="#9db0ff" strokeWidth={2} dot={false} connectNulls strokeDasharray="6 5" />
                <defs>
                  <linearGradient id="forecastBand" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(255,204,114,0.42)" />
                    <stop offset="100%" stopColor="rgba(255,204,114,0.02)" />
                  </linearGradient>
                </defs>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </section>

        <aside className="workspace__rail">
          <section className="surface surface--decision">
            <div className="signal-head">
              <div>
                <p className="eyebrow">Now Decision</p>
                <h2>判断</h2>
              </div>
              <div className={`confidence-pill ${confidenceTone}`}>{confidenceLabel}</div>
            </div>

            <div className="decision-hero">
              <div className="decision-hero__copy">
                <span>アクション</span>
                <strong>{translateOverlay(data.run.overlayAction)}</strong>
                <p>{describeExpectedMove(selectedForecast.expectedReturnPct, selectedForecast.hours)}</p>
              </div>
              <div className="gauge-chart">
                <ResponsiveContainer width="100%" height={156}>
                <RadialBarChart data={positionGauge} innerRadius="72%" outerRadius="100%" startAngle={210} endAngle={-30} barSize={16}>
                  <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                  <RadialBar background dataKey="value" cornerRadius={999} />
                </RadialBarChart>
                </ResponsiveContainer>
                <div className="gauge-value">
                  <span>強さ</span>
                  <strong>{data.run.position.toFixed(2)}</strong>
                  <em>{selectedForecast.hours}時間先</em>
                </div>
              </div>
            </div>

            <div className="confidence-meter" aria-label="予測信頼度">
              <div className="confidence-meter__track">
                <div className={`confidence-meter__fill ${confidenceTone}`} style={{ width: `${trustScore}%` }} />
              </div>
              <div className="confidence-meter__meta">
                <span>信頼度</span>
                <strong>{trustScore} / 100</strong>
              </div>
            </div>

            <div className="signal-grid">
              <MiniStat label="予測値" value={formatPrice(selectedForecast.predictedClose)} />
              <MiniStat label="予測幅" value={`${formatPrice(selectedForecast.lowerClose)} - ${formatPrice(selectedForecast.upperClose)}`} />
              <MiniStat label="変化率" value={formatSignedPercent(selectedForecast.expectedReturnPct)} />
              <MiniStat label="採用エポック" value={`${data.run.bestEpoch}回`} />
              <MiniStat label="1シグナル価値" value={formatNullableSignedDecimal(valuePerSignal, 3)} />
              <MiniStat label="学習データ" value={`${formatInteger(data.run.sourceRows)} rows`} />
            </div>

            <div className="decision-note">
              <strong>評価</strong>
              <p>{reuseNarrative}</p>
            </div>

            <div className="insight-list">
              {data.narrative.bullets.slice(0, 3).map((bullet) => (
                <div className="insight-list__item" key={bullet}>
                  <i />
                  <span>{bullet}</span>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </motion.section>

      <motion.section
        className="analysis-grid"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1, ease: 'easeOut' }}
      >
        <div className="analysis-grid__stack">
          <section className="surface">
            <div className="surface__head">
              <div>
                <p className="surface__eyebrow">Horizons</p>
                <h2>時間軸の見通し</h2>
              </div>
              <p className="surface__hint">各時間軸の中心予測とレンジ幅を同じスケールで見比べます。</p>
            </div>
            <div className="horizon-list">
              {data.horizons.map((row) => (
                <article className={`horizon-item${row.horizon === data.run.selectedHorizon ? ' is-selected' : ''}`} key={row.horizon}>
                  <div className="horizon-item__meta">
                    <strong>{row.hours}時間先</strong>
                    <span>中心予測</span>
                  </div>
                  <div className="horizon-item__content">
                    <div className="horizon-item__summary">
                      <span className={`horizon-item__delta ${row.expectedReturnPct >= 0 ? 'is-up' : 'is-down'}`}>{formatSignedPercent(row.expectedReturnPct)}</span>
                      <p>{describeExpectedMove(row.expectedReturnPct, row.hours)}</p>
                    </div>
                    <div className="horizon-item__range">
                      <div
                        className="horizon-item__band"
                        style={{
                          left: `${normalizeRange(row.lowerClose, chartYDomain)}%`,
                          width: `${normalizeRange(row.upperClose, chartYDomain) - normalizeRange(row.lowerClose, chartYDomain)}%`,
                        }}
                      />
                      <div
                        className="horizon-item__point"
                        style={{ left: `${normalizeRange(row.predictedClose, chartYDomain)}%` }}
                      />
                    </div>
                    <div className="horizon-item__facts">
                      <span>不確実性 {getUncertaintyScore(row.uncertainty, data.horizons)} / 100 {getUncertaintyLabel(getUncertaintyScore(row.uncertainty, data.horizons))}</span>
                      <span>予測幅 {(((row.upperClose - row.lowerClose) / data.run.anchorClose) * 100).toFixed(2)}%</span>
                      {getHorizonActualLabel(row) ? <span>{getHorizonActualLabel(row)}</span> : null}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </div>

        <div className="analysis-grid__stack">
          <section className="surface surface--metrics">
            <div className="surface__head">
              <div>
                <p className="surface__eyebrow">Epoch</p>
                <h2>学習の安定性</h2>
              </div>
              <p className="surface__hint">採用したエポックと最新の状態を比べます。</p>
            </div>
            <div className="epoch-grid">
              <article className="epoch-card epoch-card--accent">
                <span>採用</span>
                <strong>{data.run.bestEpoch}回</strong>
                <p>検証損失 {bestEpochMetric.validationTotal.toFixed(3)}</p>
              </article>
              <article className="epoch-card">
                <span>最新</span>
                <strong>{latestMetric.epoch}回</strong>
                <p>差分 {epochDrift >= 0 ? '+' : ''}{epochDrift.toFixed(3)}</p>
              </article>
            </div>
            <div className="quality-strip quality-strip--compact">
              <MiniStat label="改善幅" value={data.run.convergenceGain.toFixed(3)} />
              <MiniStat label="リターン損失" value={bestEpochMetric.validationReturn.toFixed(3)} />
              <MiniStat label="判定損失" value={latestMetric.validationOverlay.toFixed(3)} />
              <MiniStat label="検証件数" value={formatInteger(data.run.validationSamples)} />
              <MiniStat label="回収率" value={formatNullablePercent(valueCaptureRatio)} />
              <MiniStat label="Overlay F1" value={formatNullableScore(overlayMacroF1)} />
            </div>
            <div className="series-guide" aria-label="学習指標の凡例">
              <LegendChip label="訓練" tone="series-chip--ink" />
              <LegendChip label="検証" tone="series-chip--gold" />
              <LegendChip label="リターン" tone="series-chip--violet" />
            </div>
            <div className="metrics-chart">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={data.metrics.history}>
                  <CartesianGrid vertical={false} stroke="rgba(15,23,42,0.08)" />
                  <XAxis dataKey="epoch" tickFormatter={formatEpochLabel} tick={{ fontSize: 11 }} stroke="rgba(15,23,42,0.4)" />
                  <YAxis tickFormatter={formatLossLabel} tick={{ fontSize: 11 }} stroke="rgba(15,23,42,0.4)" width={60} />
                  <Tooltip content={<MetricTooltip />} />
                  <ReferenceLine x={data.run.bestEpoch} stroke="rgba(211,154,56,0.45)" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="trainTotal" stroke="#212121" strokeWidth={2.1} dot={false} />
                  <Line type="monotone" dataKey="validationTotal" stroke="#d39a38" strokeWidth={2.4} dot={false} />
                  <Line type="monotone" dataKey="validationReturn" stroke="#7b89ff" strokeWidth={1.7} dot={false} strokeDasharray="4 4" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {comparisonChartData.length > 0 ? (
            <section className="surface surface--confidence">
              <div className="surface__head">
                <div>
                  <p className="surface__eyebrow">Model Check</p>
                  <h2>予測の安定性</h2>
                </div>
                <p className="surface__hint">実績がある時間軸で予測を見直します。</p>
              </div>
              <div className="quality-strip">
                <MiniStat label="方向一致" value={formatNullablePercent(directionalAccuracy)} />
                <MiniStat label="レンジ内" value={formatNullablePercent(rangeCoverage)} />
                <MiniStat label="価格誤差" value={formatNullableSignedDecimal(averagePriceErrorPct, 3)} />
                <MiniStat label="汎化ギャップ" value={data.run.generalizationGap.toFixed(3)} />
                <MiniStat label="Overlay 正答" value={formatNullablePercent(overlayAccuracy)} />
                <MiniStat label="不確実性ずれ" value={formatNullableSignedDecimal(uncertaintyCalibrationError, 3)} />
                <MiniStat label="下振れ価値" value={formatNullableSignedDecimal(downsidePerSignal, 3)} />
              </div>
              <div className="decision-note decision-note--soft">
                <strong>要点</strong>
                <p>{reuseNarrative}</p>
              </div>
              <div className="series-guide" aria-label="予測と実績の凡例">
                <LegendChip label="予測" tone="series-chip--gold" />
                <LegendChip label="実績" tone="series-chip--violet" />
              </div>
              <div className="comparison-chart">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={comparisonChartData}>
                    <CartesianGrid vertical={false} stroke="rgba(15,23,42,0.08)" />
                    <XAxis dataKey="hours" tickFormatter={formatHourLabel} tick={{ fontSize: 11 }} stroke="rgba(15,23,42,0.4)" />
                    <YAxis tickFormatter={(value) => `${value.toFixed(0)}%`} tick={{ fontSize: 11 }} stroke="rgba(15,23,42,0.4)" />
                    <Tooltip content={<ReturnTooltip />} />
                    <Bar dataKey="expectedReturnPct" fill="#d39a38" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="actualReturnPct" fill="#93a5ff" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          ) : null}
        </div>
      </motion.section>
    </main>
  )
}

function MicroCandleLayer({ rows }: { rows: MicroCandleRow[] }) {
  const xScale = useXAxisScale()
  const yScale = useYAxisScale()

  if (!xScale || !yScale || rows.length === 0) {
    return null
  }

  const gaps = rows
    .slice(0, 12)
    .map((row, index) => {
      if (index === 0) {
        return 0
      }

      const prev = rows[index - 1]
      const currentX = xScale(row.ts)
      const previousX = xScale(prev.ts)

      return typeof currentX === 'number' && typeof previousX === 'number' ? Math.abs(currentX - previousX) : 0
    })
    .filter((gap) => gap > 0)

  const inferredGap = gaps.length > 0 ? Math.min(...gaps) : 0
  const candleWidth = Math.max(1, Math.min(3.2, inferredGap * 0.42))
  const wickOpacity = 0.22
  const bodyOpacity = 0.12

  return (
    <g aria-hidden="true" pointerEvents="none">
      {rows.map((row) => {
        const x = xScale(row.ts)
        const highY = yScale(row.high)
        const lowY = yScale(row.low)
        const openY = yScale(row.open)
        const closeY = yScale(row.close)

        if (
          typeof x !== 'number' ||
          typeof highY !== 'number' ||
          typeof lowY !== 'number' ||
          typeof openY !== 'number' ||
          typeof closeY !== 'number'
        ) {
          return null
        }

        const bodyTop = Math.min(openY, closeY)
        const bodyBottom = Math.max(openY, closeY)
        const bodyHeight = Math.max(bodyBottom - bodyTop, 1)
        const bullish = row.close >= row.open

        return (
          <g key={row.ts}>
            <line
              x1={x}
              y1={highY}
              x2={x}
              y2={lowY}
              stroke={bullish ? 'rgba(157,176,255,0.55)' : 'rgba(243,237,227,0.45)'}
              strokeWidth={1}
              strokeOpacity={wickOpacity}
              strokeLinecap="round"
            />
            <rect
              x={x - candleWidth / 2}
              y={bodyTop}
              width={candleWidth}
              height={bodyHeight}
              fill={bullish ? 'rgba(157,176,255,0.38)' : 'rgba(243,237,227,0.22)'}
              fillOpacity={bodyOpacity}
              stroke={bullish ? 'rgba(157,176,255,0.5)' : 'rgba(243,237,227,0.32)'}
              strokeOpacity={0.18}
            />
          </g>
        )
      })}
    </g>
  )
}

function MetricTile({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`metric-tile${accent ? ' metric-tile--accent' : ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="mini-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function LegendChip({ label, tone }: { label: string; tone: string }) {
  return (
    <span className={`series-chip ${tone}`}>
      <i />
      {label}
    </span>
  )
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: Array<{ payload?: ChartRow }>
  label?: number
}) {
  if (!active || !payload?.length || typeof label !== 'number') {
    return null
  }

  const row = payload[0]?.payload
  if (!row) {
    return null
  }

  return (
    <div className="tooltip tooltip--dark">
      <strong>{formatAxisDate(label)}</strong>
      {row.close !== null ? <span>終値 {formatPrice(row.close)}</span> : null}
      {row.forecastBase !== null ? <span>予測値 {formatPrice(row.forecastBase)}</span> : null}
      {row.forecastLower !== null && row.forecastUpper !== null ? <span>予測幅 {formatPrice(row.forecastLower)} - {formatPrice(row.forecastUpper)}</span> : null}
      {row.actualClose !== null ? <span>4時間足終値 {formatPrice(row.actualClose)}</span> : null}
      {row.actualRangeStart !== null && row.actualRangeEnd !== null ? <span>対象区間 {formatActualWindow(row.actualRangeStart, row.actualRangeEnd)}</span> : null}
    </div>
  )
}

function MetricTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: Array<{ value?: number; name?: string }>
  label?: number
}) {
  if (!active || !payload?.length) {
    return null
  }

  return (
    <div className="tooltip">
      <strong>エポック {label}</strong>
      {payload.map((item) => (
        <span key={item.name}>
          {translateMetricName(item.name)} {item.value?.toFixed(3)}
        </span>
      ))}
    </div>
  )
}

function ReturnTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: Array<{ value?: number; name?: string }>
  label?: string
}) {
  if (!active || !payload?.length) {
    return null
  }

  return (
    <div className="tooltip">
      <strong>{label}</strong>
      {payload.map((item) => (
        <span key={item.name}>
          {item.name === 'expectedReturnPct' ? '予測' : '実績'} {item.value?.toFixed(2)}%
        </span>
      ))}
    </div>
  )
}

function formatTimestamp(value: string): string {
  return new Intl.DateTimeFormat('ja-JP', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Asia/Tokyo',
  }).format(new Date(value))
}

function formatAxisDate(value: number): string {
  return new Intl.DateTimeFormat('ja-JP', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Asia/Tokyo',
  }).format(new Date(value))
}

function formatActualWindow(start: number, end: number): string {
  const startText = new Intl.DateTimeFormat('ja-JP', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Asia/Tokyo',
  }).format(new Date(start))
  const endText = new Intl.DateTimeFormat('ja-JP', {
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Asia/Tokyo',
  }).format(new Date(end))

  return `${startText} - ${endText}`
}

function formatPrice(value: number): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)
}

function formatCompactPrice(value: number): string {
  return `${new Intl.NumberFormat('ja-JP', { maximumFractionDigits: 0 }).format(value)}`
}

function formatSignedPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatNullablePercent(value: number | null): string {
  if (value === null) {
    return '-'
  }

  return `${(value * 100).toFixed(1)}%`
}

function formatNullableScore(value: number | null): string {
  if (value === null) {
    return '-'
  }

  return value.toFixed(3)
}

function formatNullableSignedDecimal(value: number | null, digits = 2): string {
  if (value === null) {
    return '-'
  }

  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

function formatEpochLabel(value: number): string {
  return `${value}回`
}

function formatLossLabel(value: number): string {
  return value.toFixed(1)
}

function formatHourLabel(value: number): string {
  return `${value}h`
}

function normalizeRange(value: number, [min, max]: [number, number]): number {
  return ((value - min) / (max - min)) * 100
}

function getVisiblePriceDomain(rows: ChartRow[]): [number, number] {
  const values = rows
    .flatMap((row) => [row.close, row.forecastBase, row.forecastLower, row.forecastUpper, row.actualClose])
    .filter((value): value is number => value !== null)

  if (!values.length) {
    return [0, 1]
  }

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min
  const pad = range === 0 ? Math.max(min * 0.0025, 1) : Math.max(range * 0.035, max * 0.0025, 1)

  return [Math.floor(min - pad), Math.ceil(max + pad)]
}

function translateOverlay(value: string): string {
  switch (value) {
    case 'hold':
      return '維持'
    case 'reduce':
      return '縮小'
    case 'full_exit':
      return '全解消'
    case 'hard_exit':
      return '即時解消'
    default:
      return value
  }
}

function translateMetricName(value?: string): string {
  switch (value) {
    case 'trainTotal':
      return '訓練'
    case 'validationTotal':
      return '検証'
    case 'trainReturn':
      return '訓練リターン'
    case 'validationReturn':
      return '検証リターン'
    case 'trainOverlay':
      return '訓練判定'
    case 'validationOverlay':
      return '検証判定'
    default:
      return value ?? ''
  }
}

function describeExpectedMove(expectedReturnPct: number, hours: number): string {
  const abs = Math.abs(expectedReturnPct)

  if (abs < 0.15) {
    return `${hours}時間先は横ばいです。`
  }

  if (expectedReturnPct > 0) {
    return `${hours}時間先は上向きです。`
  }

  return `${hours}時間先は下向きです。`
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value))
}

function getNormalizedUncertainty(value: number, horizons: HorizonRow[]): number {
  const maxUncertainty = Math.max(...horizons.map((row) => row.uncertainty))
  const minUncertainty = Math.min(...horizons.map((row) => row.uncertainty))

  if (maxUncertainty === minUncertainty) {
    return 1
  }

  return clamp01(1 - (value - minUncertainty) / (maxUncertainty - minUncertainty))
}

function getUncertaintyScore(value: number, horizons: HorizonRow[]): number {
  return Math.round((1 - getNormalizedUncertainty(value, horizons)) * 100)
}

function getUncertaintyLabel(score: number): string {
  if (score <= 33) {
    return '低め'
  }
  if (score <= 66) {
    return '標準'
  }

  return '高め'
}

function getConfidenceTone(score: number): string {
  if (score >= 72) {
    return 'confidence-pill--strong'
  }
  if (score >= 55) {
    return 'confidence-pill--watch'
  }

  return 'confidence-pill--caution'
}

function getConfidenceLabel(score: number): string {
  if (score >= 72) {
    return '高い'
  }
  if (score >= 55) {
    return '標準'
  }

  return '注意'
}

function buildReuseNarrative({
  trustScore,
  directionalAccuracy,
  rangeCoverage,
  averagePriceErrorPct,
  generalizationGap,
}: {
  trustScore: number
  directionalAccuracy: number | null
  rangeCoverage: number | null
  averagePriceErrorPct: number | null
  generalizationGap: number
}): string {
  const points = [`信頼度 ${trustScore}/100`]

  if (directionalAccuracy !== null) {
    points.push(
      directionalAccuracy >= 0.65
        ? '方向は安定しています。'
        : directionalAccuracy >= 0.5
          ? '方向はまずまずです。'
          : '方向はまだ不安定です。',
    )
  }

  if (rangeCoverage !== null) {
    points.push(rangeCoverage >= 0.6 ? '予測幅は適切です。' : '予測幅は狭めです。')
  }

  if (averagePriceErrorPct !== null) {
    points.push(averagePriceErrorPct <= 0.02 ? '価格のズレは小さめです。' : '価格のズレはやや大きめです。')
  }

  points.push(Math.abs(generalizationGap) <= 0.4 ? '学習の偏りは小さめです。' : '学習の偏りが残っています。')

  return points.join(' ')
}

function getHorizonActualLabel(row: HorizonRow): string {
  if (row.actualReturnPct === null) {
    return ''
  }

  return `実績 ${formatSignedPercent(row.actualReturnPct)}`
}

export default App
