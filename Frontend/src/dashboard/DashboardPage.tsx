import { useState } from 'react'
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

const HOUR_MS = 60 * 60 * 1000

function DashboardPage({ data }: { data: DashboardData }) {
  const shouldReduceMotion = useReducedMotion()
  const [activeHorizon, setActiveHorizon] = useState(() => data.run.selectedHorizon)
  const [pinnedSeries, setPinnedSeries] = useState<SeriesFocus>('all')
  const [previewSeries, setPreviewSeries] = useState<SeriesFocus | null>(null)

  const view = buildDashboardViewModel(data, activeHorizon)
  const activeSeries = previewSeries ?? pinnedSeries
  const motionProps = shouldReduceMotion
    ? undefined
    : {
        initial: { opacity: 0, y: 14 },
        animate: { opacity: 1, y: 0 },
        transition: { duration: 0.28, ease: [0.22, 1, 0.36, 1] as const },
      }

  const chartWindowMs = view.chartWindowHours * HOUR_MS
  const focusedRows = data.chart.rows.filter(
    (row) => row.ts >= view.anchorTs - chartWindowMs && row.ts <= view.anchorTs + chartWindowMs,
  )
  const chartRows = focusedRows.length > 0 ? focusedRows : data.chart.rows
  const chartSeriesRows = chartRows.map((row) => ({
    ...row,
    forecastInnerUpper:
      row.forecastInnerLower !== null && row.forecastInnerSpread !== null
        ? row.forecastInnerLower + row.forecastInnerSpread
        : null,
  }))
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
  const historyOpacity = activeSeries === 'all' || activeSeries === 'history' ? 0.78 : 0.16
  const forecastOpacity = activeSeries === 'all' || activeSeries === 'forecast' ? 1 : 0.18
  const actualOpacity = activeSeries === 'all' || activeSeries === 'actual' ? 0.92 : 0.18

  function toggleSeries(series: Exclude<SeriesFocus, 'all'>) {
    setPinnedSeries((current) => (current === series ? 'all' : series))
  }

  return (
    <main className="ops-shell">
      <section className="ops-layout">
        <section className="ops-workspace">
          <motion.section className="panel panel--forecast" {...(motionProps ?? {})}>
            <div className="forecast-head">
              <div className="forecast-head__copy">
                <div className="forecast-meta">
                  <span>SignalCascade</span>
                  <span>{data.instrument}</span>
                  <span>{view.selectedForecast.hours}H</span>
                  <span>{formatTimestampJst(data.generatedAt)}</span>
                </div>
                <PanelHeader label="Market Canvas" />
                <div className="forecast-summary">
                  <strong>{view.selectedForecastPriceLabel}</strong>
                  <span>{view.selectedExpectedReturnLabel}</span>
                </div>
                <p className="forecast-range">{view.selectedRangeLabel}</p>
              </div>

              <div className="forecast-head__controls">
                <nav className="horizon-switch horizon-switch--inline" aria-label="Horizon">
                  {view.horizonOptions.map((option) => {
                    const isActive = option.horizon === activeHorizon

                    return (
                      <button
                        key={option.horizon}
                        type="button"
                        className={`horizon-switch__button${isActive ? ' is-active' : ''}`}
                        aria-pressed={isActive}
                        aria-label={`${option.hours}H ${option.expectedReturnLabel} ${option.stateLabel}`}
                        onClick={() => setActiveHorizon(option.horizon)}
                      >
                        <span>{option.hours}H</span>
                        <strong>{option.expectedReturnLabel}</strong>
                      </button>
                    )
                  })}
                </nav>

                <div className="series-controls" role="toolbar" aria-label="chart series">
                  <SeriesToggle
                    label="History"
                    active={activeSeries === 'all' || activeSeries === 'history'}
                    onClick={() => toggleSeries('history')}
                    onPreview={() => setPreviewSeries('history')}
                    onPreviewClear={() => setPreviewSeries(null)}
                  />
                  <SeriesToggle
                    label="Forecast"
                    active={activeSeries === 'all' || activeSeries === 'forecast'}
                    onClick={() => toggleSeries('forecast')}
                    onPreview={() => setPreviewSeries('forecast')}
                    onPreviewClear={() => setPreviewSeries(null)}
                  />
                  <SeriesToggle
                    label="Actual"
                    active={activeSeries === 'all' || activeSeries === 'actual'}
                    disabled={!hasActualSeries}
                    onClick={() => toggleSeries('actual')}
                    onPreview={() => setPreviewSeries('actual')}
                    onPreviewClear={() => setPreviewSeries(null)}
                  />
                </div>
              </div>
            </div>

            <div className="forecast-stage">
              <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={chartSeriesRows} margin={{ top: 12, right: 8, bottom: 8, left: 0 }}>
                  <CartesianGrid vertical stroke="var(--surface-border-soft)" horizontal={false} />
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
                    stroke="var(--text-history)"
                    strokeWidth={1.6}
                    strokeOpacity={historyOpacity}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecastBase"
                    stroke="var(--accent-cyan)"
                    strokeWidth={2.45}
                    strokeOpacity={forecastOpacity}
                    dot={false}
                    connectNulls
                    isAnimationActive={!shouldReduceMotion}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecastInnerLower"
                    stroke="var(--accent-cyan-line)"
                    strokeWidth={1.3}
                    strokeOpacity={forecastOpacity}
                    dot={false}
                    connectNulls
                    isAnimationActive={!shouldReduceMotion}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecastInnerUpper"
                    stroke="var(--accent-cyan-line)"
                    strokeWidth={1.3}
                    strokeOpacity={forecastOpacity}
                    dot={false}
                    connectNulls
                    isAnimationActive={!shouldReduceMotion}
                  />
                  <Line
                    type="linear"
                    dataKey="actualClose"
                    stroke="var(--accent-sand)"
                    strokeDasharray="8 5"
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
                    label={{ value: 'Now', position: 'insideTop', fill: 'var(--text-axis-strong)', fontSize: 10 }}
                  />
                  <ReferenceDot
                    x={view.anchorTs}
                    y={data.run.anchorClose}
                    r={5}
                    fill="var(--accent-cyan)"
                    stroke="var(--accent-cyan-soft)"
                    strokeWidth={2}
                    ifOverflow="extendDomain"
                    label={{ value: 'Live', position: 'right', fill: 'var(--accent-cyan-soft)', fontSize: 10 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="boundary-grid boundary-grid--embedded">
              {view.boundaryCards.map((card) => (
                <article key={card.label} className={`boundary-card boundary-card--${card.tone}`}>
                  <span>{card.label}</span>
                  <strong>{card.value}</strong>
                  <p>{card.detail}</p>
                </article>
              ))}
            </div>
          </motion.section>

          <motion.section id="detail-anchor" className="panel panel--evidence" {...(motionProps ?? {})}>
            <div className="panel-head panel-head--detail">
              <div>
                <PanelHeader label="Evidence" />
              </div>
              <p className="detail-chip">Why / Validation</p>
            </div>

            <div className="evidence-grid">
              <section className="detail-block">
                <h2>Why</h2>
                <ul className="reason-list">
                  {view.reasons.map((reason) => (
                    <li key={reason.title} className={`reason-list__item reason-list__item--${reason.tone}`}>
                      <strong>{reason.title}</strong>
                      <p>{reason.body}</p>
                    </li>
                  ))}
                </ul>
              </section>

              <section className="detail-block">
                <h2>Validation</h2>
                <div className="mini-evidence-grid">
                  {view.detailMetrics.map((metric) => (
                    <article key={metric.label} className="mini-evidence">
                      <span>{metric.label}</span>
                      <strong>{metric.value}</strong>
                      <p>{metric.hint}</p>
                    </article>
                  ))}
                </div>
              </section>
            </div>
          </motion.section>
        </section>

        <aside className="ops-rail">
          <motion.section className="panel panel--action" {...(motionProps ?? {})}>
            <div className="panel-head">
              <PanelHeader label="Decision Rail" />
            </div>

            <div className={`action-hero action-hero--${view.decisionTone}`}>
              <span className="action-hero__label">Next Action</span>
              <strong>{view.recommendedAction}</strong>
              <p>{view.decisionSummary}</p>
            </div>

            <div className="action-stack" aria-label="decision modes">
              {view.actionCards.map((card) => (
                <article key={card.key} className={`action-row${card.isCurrent ? ' is-current' : ''}`}>
                  <div>
                    <span className="action-row__label">{card.title}</span>
                    <strong>{card.summary}</strong>
                  </div>
                  <p>{card.detail}</p>
                </article>
              ))}
            </div>
          </motion.section>

          <motion.section className="panel panel--context" {...(motionProps ?? {})}>
            <section className="context-block">
              <PanelHeader label="Market State" />
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

            <section className="context-block">
              <PanelHeader label="Reliability" />
              <div className="meter-list">
                {view.reliabilityMeters.map((meter) => (
                  <article key={meter.label} className="meter">
                    <div className="meter__head">
                      <span>{meter.label}</span>
                      <strong>{meter.valueLabel}</strong>
                    </div>
                    <div className="meter__track" aria-hidden="true">
                      <div className={`meter__fill meter__fill--${meter.tone}`} style={{ width: `${meter.value}%` }} />
                    </div>
                    <p>{meter.caption}</p>
                  </article>
                ))}
              </div>

              <div className={`verdict verdict--${view.reliabilityVerdict.tone}`}>
                <span className="verdict__icon" aria-hidden="true" />
                <div>
                  <strong>{view.reliabilityVerdict.label}</strong>
                  <p>{view.reliabilityVerdict.summary}</p>
                </div>
              </div>
            </section>

            <section className="context-block">
              <PanelHeader label="Freshness" />
              <dl className="source-list">
                {view.freshnessFacts.map((fact) => (
                  <div key={fact.label} className={`source-list__row source-list__row--${fact.tone ?? 'watch'}`}>
                    <dt>{fact.label}</dt>
                    <dd title={fact.value}>{fact.value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          </motion.section>
        </aside>
      </section>
    </main>
  )
}

function PanelHeader({ label }: { label: string }) {
  return <p className="panel-label">{label}</p>
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
      aria-pressed={active}
      disabled={disabled}
      onClick={onClick}
      onMouseEnter={onPreview}
      onMouseLeave={onPreviewClear}
      onFocus={onPreview}
      onBlur={onPreviewClear}
    >
      {label}
    </button>
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
      {row.close !== null ? <span>History {formatCurrencyPrice(row.close)}</span> : null}
      {row.forecastBase !== null ? <span>Forecast {formatCurrencyPrice(row.forecastBase)}</span> : null}
      {row.forecastLower !== null && row.forecastUpper !== null ? (
        <span>
          Range {formatCurrencyPrice(row.forecastLower)} - {formatCurrencyPrice(row.forecastUpper)}
        </span>
      ) : null}
      {row.actualClose !== null ? <span>Actual {formatCurrencyPrice(row.actualClose)}</span> : null}
      {row.actualRangeStart !== null && row.actualRangeEnd !== null ? (
        <span>Window {formatActualWindow(row.actualRangeStart, row.actualRangeEnd)}</span>
      ) : null}
    </div>
  )
}

export default DashboardPage
