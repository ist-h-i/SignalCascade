import { useEffect, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ChartRow, DashboardData } from './types'
import type { RecommendedAction } from './view-model'
import {
  buildDashboardViewModel,
  formatActualWindow,
  formatAxisDateJst,
  formatCompactPrice,
  formatCurrencyPrice,
  formatRelativeTick,
  formatTimestampJst,
  getFocusedPriceDomain,
} from './view-model'
import './dashboard.css'

type SeriesFocus = 'all' | 'history' | 'forecast' | 'actual'
type GlyphName =
  | 'wait'
  | 'go'
  | 'pass'
  | 'signal'
  | 'size'
  | 'trust'
  | 'economy'
  | 'lag'
  | 'direction'
  | 'range'
  | 'utility'
  | 'fresh'

const HOUR_MS = 60 * 60 * 1000

function DashboardPage({ data }: { data: DashboardData }) {
  const shouldReduceMotion = Boolean(useReducedMotion())
  const [activeHorizon, setActiveHorizon] = useState(() => data.run.selectedHorizon)
  const [pinnedSeries, setPinnedSeries] = useState<SeriesFocus>('all')
  const [previewSeries, setPreviewSeries] = useState<SeriesFocus | null>(null)

  const view = buildDashboardViewModel(data, activeHorizon)
  const activeSeries = previewSeries ?? pinnedSeries
  const viewportWidth = useViewportWidth()
  const chartHeight = getChartHeight(viewportWidth)
  const chartWindowMs = view.chartWindowHours * HOUR_MS
  const focusedRows = data.chart.rows.filter(
    (row) => row.ts >= view.anchorTs - chartWindowMs && row.ts <= view.anchorTs + chartWindowMs,
  )
  const chartRows = focusedRows.length > 0 ? focusedRows : data.chart.rows
  const chartYDomain = getFocusedPriceDomain(chartRows)
  const chartXDomain: [number, number] = [view.anchorTs - chartWindowMs, view.anchorTs + chartWindowMs]
  const chartTicks = [
    view.anchorTs - chartWindowMs,
    view.anchorTs - chartWindowMs / 2,
    view.anchorTs,
    view.anchorTs + chartWindowMs / 2,
    view.anchorTs + chartWindowMs,
  ]
  const hasActualSeries = chartRows.some((row) => row.actualClose !== null)
  const historyOpacity = activeSeries === 'all' || activeSeries === 'history' ? 0.7 : 0.16
  const forecastOpacity = activeSeries === 'all' || activeSeries === 'forecast' ? 1 : 0.18
  const actualOpacity = activeSeries === 'all' || activeSeries === 'actual' ? 0.94 : 0.18
  const biasWord = getBiasWord(view.selectedForecast.expectedReturnPct)
  const lagHours = data.provenance.freshness?.predictionLagHours ?? null
  const productionCurrentCandidate = data.governance?.productionCurrentCandidate ?? null
  const acceptedCandidate = data.governance?.acceptedCandidate ?? null
  const decisionFacts = [
    {
      icon: 'trust' as const,
      label: 'Confidence',
      value: `${view.trustScore}%`,
      detail: view.reliabilityVerdict.label,
    },
    {
      icon: 'fresh' as const,
      label: 'Freshness',
      value: formatLag(lagHours),
      detail: getFreshnessCaption(data.run.runQuality, lagHours),
    },
    {
      icon: 'range' as const,
      label: 'Prediction band',
      value: `${view.boundaryCards[2]?.detail ?? '-'} / ${view.boundaryCards[0]?.detail ?? '-'}`,
      detail: `${view.selectedForecast.hours}H ${view.uncertaintyLabel}`,
    },
    {
      icon: 'size' as const,
      label: 'Position delta',
      value: formatExposure(data.run.tradeDelta),
      detail: `保有 ${formatExposure(data.run.position)}`,
    },
  ]
  const chartLegend = [
    { key: 'history', label: '履歴', detail: 'solid line' },
    { key: 'forecast', label: '予測', detail: 'bold line' },
    { key: 'band', label: '想定帯', detail: 'dashed band' },
    { key: 'actual', label: '実績', detail: 'dashed line' },
  ]
  const railFreshnessFacts = view.freshnessFacts.slice(0, 3)
  const provenanceFacts = [
    {
      label: 'Source',
      value: data.provenance.sourceOriginPath ?? data.provenance.sourcePath ?? '-',
    },
    {
      label: 'Artifact',
      value: formatShortHash(data.provenance.artifactId),
    },
    {
      label: 'Generated',
      value: formatTimestampJst(data.generatedAt),
    },
    {
      label: 'Git',
      value: `${formatShortHash(data.provenance.gitCommitSha)}${data.provenance.gitDirty ? ' / dirty' : ''}`,
    },
    {
      label: 'Production',
      value: productionCurrentCandidate ?? '-',
    },
    {
      label: 'Accepted',
      value: acceptedCandidate ?? '-',
    },
  ]
  const narrativeSegments = (data.narrative.segments ?? []).map((segment) => ({
    ...segment,
    tone: getSegmentTone(segment.summary),
    label: getSegmentLabel(segment.summary),
  }))

  function toggleSeries(series: Exclude<SeriesFocus, 'all'>) {
    setPinnedSeries((current) => (current === series ? 'all' : series))
  }

  return (
    <main className="ops-shell" aria-labelledby="dashboard-page-title">
      <section className="signal-grid" data-lk-component="section" data-lk-section-padding="lg">
        <motion.section
          className="stage-canvas"
          data-lk-component="card"
          data-lk-card-element="surface"
          data-lk-card-variant="fill"
          data-lk-card-material="glass"
          data-lk-card-optical-correction="all"
          data-lk-slot="signal-stage"
          {...(getMotionProps(shouldReduceMotion, 0) ?? {})}
        >
            <header className="stage-topline">
              <div className="stage-topline__copy">
                <p className="ops-eyebrow">SignalCascade / {data.instrument}</p>
                <strong className="stage-kicker">{view.selectedForecast.hours}H Forecast Chart</strong>
              </div>

              <div className="series-controls stage-series" role="toolbar" aria-label="chart series">
                <SeriesToggle
                  label="履歴"
                  active={activeSeries === 'all' || activeSeries === 'history'}
                  onClick={() => toggleSeries('history')}
                  onPreview={() => setPreviewSeries('history')}
                  onPreviewClear={() => setPreviewSeries(null)}
                />
                <SeriesToggle
                  label="予測"
                  active={activeSeries === 'all' || activeSeries === 'forecast'}
                  onClick={() => toggleSeries('forecast')}
                  onPreview={() => setPreviewSeries('forecast')}
                  onPreviewClear={() => setPreviewSeries(null)}
                />
                <SeriesToggle
                  label="実績"
                  active={activeSeries === 'all' || activeSeries === 'actual'}
                  disabled={!hasActualSeries}
                  onClick={() => toggleSeries('actual')}
                  onPreview={() => setPreviewSeries('actual')}
                  onPreviewClear={() => setPreviewSeries(null)}
                />
              </div>
            </header>

            <section className="stage-chart" aria-labelledby="stage-chart-label">
              <header className="stage-chart__head">
                <div>
                  <PanelHeader id="stage-chart-label" label="Forecast Chart" />
                  <p className="chart-shell__caption">actual / forecast / band を同じ時間軸で確認する</p>
                </div>

                <ul className="chart-legend" aria-label="chart legend">
                  {chartLegend.map((item) => (
                    <li key={item.key} className={`chart-legend__item chart-legend__item--${item.key}`}>
                      <i aria-hidden="true" />
                      <span>{item.label}</span>
                      <em>{item.detail}</em>
                    </li>
                  ))}
                </ul>
              </header>

              <div className="stage-chart__canvas">
                <ResponsiveContainer width="100%" height={chartHeight}>
                  <ComposedChart data={chartRows} margin={{ top: 18, right: 10, bottom: 6, left: 0 }}>
                    <CartesianGrid vertical stroke="var(--surface-border-anchor)" horizontal={false} />
                    <XAxis
                      dataKey="ts"
                      type="number"
                      domain={chartXDomain}
                      ticks={chartTicks}
                      tickFormatter={(value) => formatRelativeTick(value, view.anchorTs)}
                      tick={{ fontSize: 11, fill: 'var(--text-axis)' }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      domain={chartYDomain}
                      tickFormatter={formatCompactPrice}
                      tick={{ fontSize: 11, fill: 'var(--text-axis-subtle)' }}
                      axisLine={false}
                      tickLine={false}
                      width={66}
                    />
                    <Tooltip content={<ForecastTooltip />} />
                    <Line
                      type="monotone"
                      dataKey="close"
                      stroke="var(--chart-history-line)"
                      strokeWidth={1.45}
                      strokeOpacity={historyOpacity}
                      dot={false}
                      connectNulls
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="forecastBase"
                      stroke="var(--chart-forecast-line)"
                      strokeWidth={2.5}
                      strokeOpacity={forecastOpacity}
                      dot={false}
                      connectNulls
                      isAnimationActive={!shouldReduceMotion}
                    />
                    <Line
                      type="monotone"
                      dataKey="forecastLower"
                      stroke="var(--chart-band-line)"
                      strokeWidth={1.15}
                      strokeDasharray="6 5"
                      strokeOpacity={forecastOpacity}
                      dot={false}
                      connectNulls
                      isAnimationActive={!shouldReduceMotion}
                    />
                    <Line
                      type="monotone"
                      dataKey="forecastUpper"
                      stroke="var(--chart-band-line)"
                      strokeWidth={1.15}
                      strokeDasharray="6 5"
                      strokeOpacity={forecastOpacity}
                      dot={false}
                      connectNulls
                      isAnimationActive={!shouldReduceMotion}
                    />
                    <Line
                      type="linear"
                      dataKey="actualClose"
                      stroke="var(--chart-actual-line)"
                      strokeDasharray="7 4"
                      strokeWidth={2}
                      strokeOpacity={actualOpacity}
                      dot={false}
                      connectNulls
                      isAnimationActive={!shouldReduceMotion}
                    />
                    <ReferenceLine
                      x={view.anchorTs}
                      stroke="var(--surface-border-anchor)"
                      strokeDasharray="3 6"
                      label={{ value: '現在', position: 'insideTop', fill: 'var(--text-axis-strong)', fontSize: 10 }}
                    />
                    <ReferenceDot
                      x={view.anchorTs}
                      y={data.run.anchorClose}
                      r={5}
                      fill="var(--chart-forecast-line)"
                      stroke="var(--chart-forecast-soft)"
                      strokeWidth={2}
                      ifOverflow="extendDomain"
                      label={{ value: 'Live', position: 'right', fill: 'var(--chart-forecast-soft)', fontSize: 10 }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </section>

            <footer className="stage-foot">
              <div className="stage-foot__head">
                <div>
                  <PanelHeader id="stage-horizon-label" label="Horizon" />
                  <p className="stage-foot__copy">時間軸を替えると、予測帯と判断 rail が更新される</p>
                </div>
              </div>

              <nav className="horizon-switch" aria-label="Horizon">
                {view.horizonOptions.map((option) => {
                  const isActive = option.horizon === activeHorizon

                  return (
                    <button
                      key={option.horizon}
                      type="button"
                      className={`horizon-switch__button${isActive ? ' is-active' : ''}`}
                      data-lk-component="button"
                      data-lk-button-variant={isActive ? 'fill' : 'outline'}
                      data-lk-button-size="sm"
                      aria-pressed={isActive}
                      aria-label={`${option.hours}H ${option.expectedReturnLabel} ${option.stateLabel}`}
                      onClick={() => setActiveHorizon(option.horizon)}
                    >
                      <StateLayer forcedState={isActive ? 'active' : undefined} />
                      <span className="lk-button__stack" data-lk-button-content-wrap>
                        <span data-lk-button-child>{option.hours}H</span>
                        <strong data-lk-button-child>{option.expectedReturnLabel}</strong>
                      </span>
                    </button>
                  )
                })}
              </nav>
            </footer>
        </motion.section>

        <div className="signal-side">
          <motion.aside
            className={`decision-dossier decision-dossier--${view.decisionTone}`}
            data-lk-component="card"
            data-lk-card-element="surface"
            data-lk-card-variant="fill"
            data-lk-card-material="flat"
            data-lk-card-optical-correction="y"
            data-lk-slot="decision-rail"
            {...(getMotionProps(shouldReduceMotion, 0.08) ?? {})}
          >
            <section className="rail-action" aria-labelledby="dashboard-page-title">
              <div className={`stage-mark stage-mark--${view.decisionTone}`} aria-hidden="true">
                <Glyph name={getActionGlyph(view.recommendedAction)} />
              </div>
              <div className="rail-action__copy">
                <PanelHeader id="decision-label" label="Decision Rail" />
                <h1 id="dashboard-page-title">{getActionEnglish(view.recommendedAction)}</h1>
                <p>{getHeroSummary(view.recommendedAction, view.selectedForecast.expectedReturnPct, lagHours)}</p>
              </div>
              <div className="rail-tags" aria-label="action and bias">
                <span className="decision-chip">{getActionLabel(view.recommendedAction)}</span>
                <span className={`stage-bias stage-bias--${getBiasTone(view.selectedForecast.expectedReturnPct)}`}>{biasWord}</span>
              </div>
            </section>

            <dl className="rail-facts" aria-label="decision facts">
              {decisionFacts.map((fact) => (
                <div key={fact.label} className="rail-fact">
                  <dt>
                    <span className="rail-fact__icon" aria-hidden="true">
                      <Glyph name={fact.icon} />
                    </span>
                    {fact.label}
                  </dt>
                  <dd>{fact.value}</dd>
                  <p>{fact.detail}</p>
                </div>
              ))}
            </dl>

            <section className="rail-section" aria-labelledby="why-label">
              <PanelHeader id="why-label" label="Why now / Why not now" />
              <div className="reason-lines">
                {view.reasons.map((reason) => (
                  <article
                    key={reason.title}
                    className={`reason-line reason-line--${reason.tone}`}
                    data-lk-component="card"
                    data-lk-card-element="surface"
                    data-lk-card-variant="transparent"
                    data-lk-card-material="flat"
                    data-lk-slot={`reason-${reason.title}`}
                  >
                    <strong>{reason.title}</strong>
                    <span>{reason.body}</span>
                  </article>
                ))}
              </div>
            </section>

            <section className="rail-section rail-section--freshness" aria-labelledby="freshness-label">
              <PanelHeader id="freshness-label" label="Freshness" />
              <div className="fact-list">
                {railFreshnessFacts.map((fact) => (
                  <article
                    key={fact.label}
                    className={`fact-row${fact.tone ? ` fact-row--${fact.tone}` : ''}`}
                    data-lk-component="card"
                    data-lk-card-element="surface"
                    data-lk-card-variant="transparent"
                    data-lk-card-material="flat"
                    data-lk-slot={`fact-${fact.label}`}
                  >
                    <span>{fact.label}</span>
                    <strong>{fact.value}</strong>
                  </article>
                ))}
              </div>
            </section>

            <section className="rail-section rail-section--ladder" aria-labelledby="action-ladder-label">
              <PanelHeader id="action-ladder-label" label="Action Ladder" />
              <div className="choice-strip" aria-label="Go Wait Pass">
                {view.actionCards.map((card) => (
                  <article
                    key={card.key}
                    className={`choice-row${card.isCurrent ? ' is-current' : ''}${card.key === 'follow' ? ' is-follow' : card.key === 'hold' ? ' is-watch' : ' is-pass'}`}
                    data-lk-component="card"
                    data-lk-card-element="surface"
                    data-lk-card-variant="transparent"
                    data-lk-card-material="flat"
                    data-lk-slot={`choice-${card.key}`}
                  >
                    <div className="choice-row__head">
                      <strong>{card.title}</strong>
                      <span>{card.summary}</span>
                    </div>
                    <p>{card.detail}</p>
                  </article>
                ))}
              </div>
            </section>

            <a className="primary-cta" data-lk-component="button" data-lk-button-variant="fill" href="#validation-summary">
              <StateLayer />
              <span data-lk-button-content-wrap>検証詳細を見る</span>
            </a>
          </motion.aside>
        </div>

        <motion.section
          id="validation-summary"
          className="support-panel support-panel--evidence"
          data-lk-component="card"
          data-lk-card-element="surface"
          data-lk-card-variant="fill"
          data-lk-card-material="flat"
          data-lk-slot="evidence-panel"
          {...(getMotionProps(shouldReduceMotion, 0.22) ?? {})}
        >
          <header className="panel-head">
            <div>
              <PanelHeader id="evidence-label" label="Validation Summary" />
              <h2>検証サマリー</h2>
              <p className="panel-copy">収益、校正、構造の偏りをここでまとめて確認する</p>
            </div>
            <dl className="panel-kpis panel-kpis--evidence" aria-label="evidence highlights">
              <div className="panel-kpi">
                <dt>信頼</dt>
                <dd>{view.trustScore}%</dd>
              </div>
              <div className="panel-kpi">
                <dt>収益</dt>
                <dd>{view.economicScore}%</dd>
              </div>
            </dl>
          </header>

          <div className="proof-grid">
            <div className="proof-metrics">
              {view.detailMetrics.map((metric) => (
                <article key={metric.label} className="proof-metric">
                  <span>{metric.label}</span>
                  <strong>{metric.value}</strong>
                  <p>{metric.hint}</p>
                </article>
              ))}
            </div>

            <div className="proof-sidecar">
              <section className="proof-health" aria-labelledby="proof-health-label">
                <PanelHeader id="proof-health-label" label="構造の状態" />
                <div className="health-list">
                  {view.systemHealthRows.map((row) => (
                    <div key={row.label} className="health-list__row">
                      <span className="health-list__label">{row.label}</span>
                      <span className={`health-pill health-pill--${row.tone}`}>
                        <i />
                        {row.status}
                      </span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="proof-distribution" aria-labelledby="proof-distribution-label">
                <PanelHeader id="proof-distribution-label" label="時間軸の偏り" />
                <div className="distribution-list">
                  {view.horizonMixRows.map((row) => (
                    <article key={row.label} className={`distribution-row distribution-row--${row.tone}`}>
                      <div className="distribution-row__head">
                        <strong>{row.label}</strong>
                        <span>{row.shareLabel}</span>
                      </div>
                      <div className="distribution-row__track" aria-hidden="true">
                        <i style={{ width: `${row.share}%` }} />
                      </div>
                      <p>{row.detail}</p>
                    </article>
                  ))}
                </div>
              </section>
            </div>
          </div>
        </motion.section>

        <motion.section
          className="support-panel support-panel--provenance"
          data-lk-component="card"
          data-lk-card-element="surface"
          data-lk-card-variant="fill"
          data-lk-card-material="flat"
          data-lk-slot="provenance-panel"
          {...(getMotionProps(shouldReduceMotion, 0.26) ?? {})}
        >
          <header className="panel-head">
            <div>
              <PanelHeader id="provenance-label" label="Provenance" />
              <h2>出所と候補</h2>
              <p className="panel-copy">first view から外した運用メタ情報をここで確認する</p>
            </div>
          </header>

          <dl className="provenance-list" aria-label="provenance details">
            {provenanceFacts.map((fact) => (
              <div key={fact.label} className="provenance-row">
                <dt>{fact.label}</dt>
                <dd>{fact.value}</dd>
              </div>
            ))}
          </dl>
        </motion.section>

        <motion.section
          className="support-panel support-panel--scenario"
          data-lk-component="card"
          data-lk-card-element="surface"
          data-lk-card-variant="fill"
          data-lk-card-material="flat"
          data-lk-slot="scenario-panel"
          {...(getMotionProps(shouldReduceMotion, 0.16) ?? {})}
        >
          <header className="panel-head">
            <div>
              <PanelHeader id="scenario-label" label="Similar-case Replay" />
              <h2>類似ケースの再生</h2>
              <p className="panel-copy">{view.selectedForecast.hours}H までの判断変化を短いステップで追う</p>
            </div>
            <dl className="panel-kpis" aria-label="scenario highlights">
              <div className="panel-kpi">
                <dt>方向</dt>
                <dd>{biasWord}</dd>
              </div>
              <div className="panel-kpi">
                <dt>期待値</dt>
                <dd>{view.selectedExpectedReturnLabel.replace(/^Exp\.\s*/, '')}</dd>
              </div>
              <div className="panel-kpi">
                <dt>中心</dt>
                <dd>{view.selectedForecastPriceLabel}</dd>
              </div>
            </dl>
          </header>

          <ol className="segment-list scenario-ribbon">
            {narrativeSegments.map((segment) => (
              <li key={`${segment.fromStep}-${segment.toStep}`} className="segment-item">
                <span className={`segment-chip segment-chip--${segment.tone}`}>{segment.label}</span>
                <strong>{segment.title}</strong>
                <p>{segment.summary}</p>
              </li>
            ))}
          </ol>
        </motion.section>
      </section>
    </main>
  )
}

function PanelHeader({ id, label }: { id?: string; label: string }) {
  return (
    <p id={id} className="panel-label">
      {label}
    </p>
  )
}

function SeriesToggle({
  label,
  active,
  disabled = false,
  onClick,
  onPreview,
  onPreviewClear,
}: {
  label: string
  active: boolean
  disabled?: boolean
  onClick: () => void
  onPreview: () => void
  onPreviewClear: () => void
}) {
  return (
    <button
      type="button"
      className={`series-toggle${active ? ' is-active' : ''}`}
      data-lk-component="button"
      data-lk-button-variant={active ? 'fill' : 'text'}
      data-lk-button-size="sm"
      aria-pressed={active}
      disabled={disabled}
      onClick={onClick}
      onMouseEnter={onPreview}
      onMouseLeave={onPreviewClear}
      onFocus={onPreview}
      onBlur={onPreviewClear}
    >
      <StateLayer forcedState={active ? 'active' : undefined} />
      <span data-lk-button-content-wrap>
        <span data-lk-button-child>{label}</span>
      </span>
    </button>
  )
}

function StateLayer({ forcedState }: { forcedState?: 'hover' | 'active' | 'focus' }) {
  return (
    <span
      aria-hidden="true"
      className="lk-state-layer"
      data-lk-component="state-layer"
      {...(forcedState ? { 'data-lk-forced-state': forcedState } : {})}
    />
  )
}

function ForecastTooltip({
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
    <div className="ops-tooltip">
      <strong>{formatAxisDateJst(label)}</strong>
      {row.close !== null ? <span>履歴 {formatCurrencyPrice(row.close)}</span> : null}
      {row.forecastBase !== null ? <span>予測 {formatCurrencyPrice(row.forecastBase)}</span> : null}
      {row.forecastLower !== null && row.forecastUpper !== null ? (
        <span>
          レンジ {formatCurrencyPrice(row.forecastLower)} - {formatCurrencyPrice(row.forecastUpper)}
        </span>
      ) : null}
      {row.actualClose !== null ? <span>実績 {formatCurrencyPrice(row.actualClose)}</span> : null}
      {row.actualRangeStart !== null && row.actualRangeEnd !== null ? (
        <span>実績窓 {formatActualWindow(row.actualRangeStart, row.actualRangeEnd)}</span>
      ) : null}
    </div>
  )
}

function Glyph({ name }: { name: GlyphName }) {
  switch (name) {
    case 'wait':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2 12s3.7-6 10-6 10 6 10 6-3.7 6-10 6S2 12 2 12Z" />
          <circle cx="12" cy="12" r="2.7" />
        </svg>
      )
    case 'go':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 17 17 7" />
          <path d="M9 7h8v8" />
          <path d="M4 12a8 8 0 1 0 8-8" />
        </svg>
      )
    case 'pass':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="m12 3 7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3Z" />
          <path d="m9 9 6 6" />
          <path d="m15 9-6 6" />
        </svg>
      )
    case 'signal':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 16h4l2-8 4 12 2-6h4" />
        </svg>
      )
    case 'size':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M5 19V9" />
          <path d="M12 19V5" />
          <path d="M19 19v-7" />
        </svg>
      )
    case 'trust':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 14a8 8 0 1 1 16 0" />
          <path d="M12 14 16 10" />
          <circle cx="12" cy="14" r="1.2" fill="currentColor" stroke="none" />
        </svg>
      )
    case 'economy':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="m4 17 5-5 3 3 8-8" />
          <path d="M16 7h4v4" />
        </svg>
      )
    case 'lag':
    case 'fresh':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="8" />
          <path d="M12 8v5l3 2" />
        </svg>
      )
    case 'direction':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="m5 15 4-4 3 3 7-7" />
          <path d="M15 7h4v4" />
        </svg>
      )
    case 'range':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 5H5v14h2" />
          <path d="M17 5h2v14h-2" />
          <path d="M9 8h6" />
          <path d="M9 16h6" />
        </svg>
      )
    case 'utility':
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="m12 3 2.8 5.7L21 9.6l-4.5 4.4 1.1 6.2L12 17.3 6.4 20.2 7.5 14 3 9.6l6.2-.9L12 3Z" />
        </svg>
      )
  }
}

function getActionGlyph(action: RecommendedAction): GlyphName {
  if (action === 'Active') {
    return 'go'
  }
  if (action === 'Reduce') {
    return 'pass'
  }

  return 'wait'
}

function getActionLabel(action: RecommendedAction) {
  if (action === 'Active') {
    return '強気'
  }
  if (action === 'Reduce') {
    return '見送り'
  }

  return '様子見'
}

function getActionEnglish(action: RecommendedAction) {
  if (action === 'Active') {
    return 'Go'
  }
  if (action === 'Reduce') {
    return 'Pass'
  }

  return 'Wait'
}

function getHeroSummary(action: RecommendedAction, expectedReturnPct: number, lagHours: number | null) {
  if (action === 'Reduce' && lagHours !== null && lagHours >= 48) {
    return '更新が古い。今回は見送り。'
  }
  if (action === 'Reduce') {
    return '根拠が弱い。今回は見送り。'
  }
  if (action === 'Active') {
    return `${getBiasNarrative(expectedReturnPct)}入る条件が揃っている。`
  }

  return `${getBiasNarrative(expectedReturnPct)}エントリーはまだ早い。`
}

function getBiasWord(expectedReturnPct: number) {
  if (expectedReturnPct > 0.2) {
    return 'Bullish'
  }
  if (expectedReturnPct < -0.2) {
    return 'Bearish'
  }

  return 'Neutral'
}

function getBiasNarrative(expectedReturnPct: number) {
  if (expectedReturnPct > 0.2) {
    return '上方向だが、'
  }
  if (expectedReturnPct < -0.2) {
    return '下方向だが、'
  }

  return '方向感はあるが、'
}

function getBiasTone(expectedReturnPct: number) {
  if (expectedReturnPct > 0.2) {
    return 'bull'
  }
  if (expectedReturnPct < -0.2) {
    return 'bear'
  }

  return 'flat'
}

function getSegmentLabel(summary: string) {
  if (summary.includes('下')) {
    return 'Bear'
  }
  if (summary.includes('上')) {
    return 'Bull'
  }

  return 'Flat'
}

function getSegmentTone(summary: string) {
  if (summary.includes('下')) {
    return 'bear'
  }
  if (summary.includes('上')) {
    return 'bull'
  }

  return 'flat'
}

function formatExposure(value: number) {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(0)}%`
}

function formatLag(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return '-'
  }

  return `${value.toFixed(1)}h`
}

function getFreshnessCaption(runQuality: string, lagHours: number | null | undefined) {
  if (lagHours === null || lagHours === undefined) {
    return runQuality === 'fresh' ? 'fresh' : runQuality
  }
  if (lagHours >= 48) {
    return 'stale'
  }
  if (lagHours >= 24) {
    return 'watch'
  }

  return 'fresh'
}

function formatShortHash(value: string | null | undefined) {
  if (!value) {
    return '-'
  }

  return value.length > 12 ? `${value.slice(0, 12)}` : value
}

function useViewportWidth() {
  const [viewportWidth, setViewportWidth] = useState(() =>
    typeof window === 'undefined' ? 1440 : window.innerWidth,
  )

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const onResize = () => setViewportWidth(window.innerWidth)

    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  return viewportWidth
}

function getChartHeight(viewportWidth: number) {
  if (viewportWidth <= 480) {
    return 248
  }
  if (viewportWidth <= 768) {
    return 292
  }
  if (viewportWidth <= 1120) {
    return 332
  }

  return 392
}

function getMotionProps(disabled: boolean, delay = 0) {
  if (disabled) {
    return undefined
  }

  return {
    initial: { opacity: 1, y: 12 },
    animate: { opacity: 1, y: 0 },
    transition: {
      duration: 0.42,
      delay,
      ease: [0.22, 1, 0.36, 1] as const,
    },
  }
}

export default DashboardPage
