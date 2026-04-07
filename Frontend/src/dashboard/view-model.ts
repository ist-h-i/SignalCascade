import type { ChartRow, DashboardData, HorizonRow } from './types'

export type RecommendedAction = 'Active' | 'Hold' | 'Reduce'
export type DecisionTone = 'follow' | 'watch' | 'skip'
export type ConfidenceTone = 'strong' | 'watch' | 'caution'

export type HorizonOption = {
  horizon: number
  hours: number
  expectedReturnLabel: string
  stateLabel: string
}

export type ActionCard = {
  key: 'follow' | 'hold' | 'reduce'
  title: string
  summary: string
  detail: string
  isCurrent: boolean
}

export type SystemHealthRow = {
  label: string
  status: string
  tone: 'positive' | 'neutral' | 'risk'
}

export type ReliabilityMeter = {
  label: string
  value: number
  valueLabel: string
  caption: string
  tone: 'cool' | 'warm' | 'risk'
}

export type ReliabilityVerdict = {
  label: string
  summary: string
  tone: ConfidenceTone
}

export type BoundaryCard = {
  label: string
  value: string
  detail: string
  tone: 'upper' | 'pivot' | 'lower'
}

export type DetailMetric = {
  label: string
  value: string
  hint: string
}

export type DetailReason = {
  title: string
  body: string
  tone: ConfidenceTone
}

export type FreshnessFact = {
  label: string
  value: string
  tone?: ConfidenceTone
}

export type DashboardViewModel = {
  selectedForecast: HorizonRow
  horizonOptions: HorizonOption[]
  anchorTs: number
  chartWindowHours: number
  trustScore: number
  uncertaintyScore: number
  uncertaintyLabel: string
  directionalAccuracy: number | null
  rangeCoverage: number | null
  recommendedAction: RecommendedAction
  decisionTone: DecisionTone
  decisionSummary: string
  selectedForecastPriceLabel: string
  selectedExpectedReturnLabel: string
  selectedRangeLabel: string
  actionCards: ActionCard[]
  systemHealthRows: SystemHealthRow[]
  reliabilityMeters: ReliabilityMeter[]
  reliabilityVerdict: ReliabilityVerdict
  boundaryCards: BoundaryCard[]
  reasons: DetailReason[]
  freshnessFacts: FreshnessFact[]
  detailMetrics: DetailMetric[]
}

export function buildDashboardViewModel(data: DashboardData, activeHorizon: number): DashboardViewModel {
  const selectedForecast = data.horizons.find((row) => row.horizon === activeHorizon) ?? data.horizons[0]
  const validation = data.metrics.validation
  const availableHorizons = data.horizons.filter(
    (row) => row.actualClose !== null && row.actualReturnPct !== null,
  )
  const directionalAccuracy =
    validation?.directionalAccuracy ??
    (availableHorizons.length > 0
      ? availableHorizons.filter((row) => Math.sign(row.expectedReturnPct) === Math.sign(row.actualReturnPct ?? 0)).length /
        availableHorizons.length
      : null)
  const rangeCoverage =
    validation?.noTradeBandHitRate !== null && validation?.noTradeBandHitRate !== undefined
      ? 1 - validation.noTradeBandHitRate
      : availableHorizons.length > 0
        ? availableHorizons.filter(
            (row) => (row.actualClose ?? Number.NaN) >= row.lowerClose && (row.actualClose ?? Number.NaN) <= row.upperClose,
          ).length / availableHorizons.length
        : null
  const averagePriceErrorPct =
    validation?.muCalibration ??
    (availableHorizons.length > 0
      ? availableHorizons.reduce((sum, row) => {
          const actualClose = row.actualClose ?? row.predictedClose
          return sum + Math.abs((actualClose - row.predictedClose) / actualClose)
        }, 0) / availableHorizons.length
      : null)
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
  const uncertaintyScore = getUncertaintyScore(selectedForecast.uncertainty, data.horizons)
  const uncertaintyLabel = getUncertaintyLabel(uncertaintyScore)
  const freshness = data.provenance.freshness
  const utilityScore = validation?.utilityScore ?? null
  const recommendedAction = deriveRecommendedAction({
    policyStatus: data.run.policyStatus,
    predictionLagHours: freshness?.predictionLagHours ?? null,
    trustScore,
    directionalAccuracy,
    utilityScore,
  })
  const decisionTone = getDecisionTone(recommendedAction)
  const freshnessScore = deriveFreshnessScore({
    runQuality: data.run.runQuality,
    freshness,
  })

  return {
    selectedForecast,
    horizonOptions: data.horizons.map((row) => ({
      horizon: row.horizon,
      hours: row.hours,
      expectedReturnLabel: formatSignedPercent(row.expectedReturnPct),
      stateLabel: compactDirectionLabel(row.expectedReturnPct),
    })),
    anchorTs: data.chart.rows.find((row) => row.phase === 'forecast')?.ts ?? new Date(data.run.anchorTime).getTime(),
    chartWindowHours: selectedForecast.hours,
    trustScore,
    uncertaintyScore,
    uncertaintyLabel,
    directionalAccuracy,
    rangeCoverage,
    recommendedAction,
    decisionTone,
    decisionSummary: buildDecisionSummary({
      recommendedAction,
      hours: selectedForecast.hours,
      freshness,
    }),
    selectedForecastPriceLabel: formatCurrencyPrice(selectedForecast.predictedClose),
    selectedExpectedReturnLabel: `Exp. ${formatSignedPercent(selectedForecast.expectedReturnPct)}`,
    selectedRangeLabel: `${formatCurrencyPrice(selectedForecast.lowerClose)} - ${formatCurrencyPrice(selectedForecast.upperClose)}`,
    actionCards: buildActionCards({
      recommendedAction,
      trustScore,
      uncertaintyScore,
      freshness,
    }),
    systemHealthRows: buildSystemHealthRows({
      horizons: data.horizons,
      selectedHorizon: selectedForecast.horizon,
    }),
    reliabilityMeters: [
      {
        label: '確度',
        value: trustScore,
        valueLabel: `${trustScore}%`,
        caption: '方向一致 + レンジ',
        tone: trustScore >= 72 ? 'cool' : trustScore >= 55 ? 'warm' : 'risk',
      },
      {
        label: '鮮度',
        value: freshnessScore,
        valueLabel: `${freshnessScore}%`,
        caption: '更新状態 + ラグ',
        tone: freshnessScore >= 72 ? 'cool' : freshnessScore >= 55 ? 'warm' : 'risk',
      },
      {
        label: '不確実性',
        value: uncertaintyScore,
        valueLabel: `${uncertaintyScore}%`,
        caption: `${selectedForecast.hours}H のブレ`,
        tone: uncertaintyScore <= 33 ? 'cool' : uncertaintyScore <= 66 ? 'warm' : 'risk',
      },
    ],
    reliabilityVerdict: buildReliabilityVerdict({
      recommendedAction,
      freshnessScore,
      trustScore,
    }),
    boundaryCards: [
      {
        label: '上限',
        value: formatCurrencyPrice(selectedForecast.upperClose),
        detail: formatSignedPercent(((selectedForecast.upperClose - data.run.anchorClose) / data.run.anchorClose) * 100),
        tone: 'upper',
      },
      {
        label: '中心',
        value: formatCurrencyPrice(selectedForecast.predictedClose),
        detail: `${selectedForecast.hours}H 中心値`,
        tone: 'pivot',
      },
      {
        label: '下限',
        value: formatCurrencyPrice(selectedForecast.lowerClose),
        detail: formatSignedPercent(((selectedForecast.lowerClose - data.run.anchorClose) / data.run.anchorClose) * 100),
        tone: 'lower',
      },
    ],
    reasons: buildReasons({
      runQuality: data.run.runQuality,
      freshness,
      directionalAccuracy,
      rangeCoverage,
      utilityScore,
      uncertaintyScore,
    }),
    freshnessFacts: [
      {
        label: 'Anchor',
        value: formatTimestampJst(data.run.anchorTime),
      },
      {
        label: '予測ラグ',
        value: formatNullableHours(freshness?.predictionLagHours),
        tone:
          freshness?.predictionLagHours !== null && freshness?.predictionLagHours !== undefined && freshness.predictionLagHours >= 48
            ? 'caution'
            : 'strong',
      },
      {
        label: '更新ラグ',
        value: formatNullableHours(freshness?.artifactLagHours),
        tone:
          freshness?.artifactLagHours !== null && freshness?.artifactLagHours !== undefined && freshness.artifactLagHours >= 24
            ? 'caution'
            : 'watch',
      },
      {
        label: 'データ元',
        value: data.provenance.sourceOriginPath ?? data.provenance.sourcePath ?? '-',
      },
    ],
    detailMetrics: [
      {
        label: '方向一致',
        value: formatNullablePercent(directionalAccuracy),
        hint: 'direction',
      },
      {
        label: 'レンジ捕捉',
        value: formatNullablePercent(rangeCoverage),
        hint: 'coverage',
      },
      {
        label: 'Utility',
        value: formatNullableScore(validation?.utilityScore),
        hint: 'policy value',
      },
      {
        label: '汎化差',
        value: formatSignedDecimal(data.run.generalizationGap, 3),
        hint: 'train vs val',
      },
    ],
  }
}

export function formatTimestampJst(value: string): string {
  return new Intl.DateTimeFormat('ja-JP', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Asia/Tokyo',
  }).format(new Date(value))
}

export function formatAxisDateJst(value: number): string {
  return new Intl.DateTimeFormat('ja-JP', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Asia/Tokyo',
  }).format(new Date(value))
}

export function formatRelativeTick(value: number, anchorTs: number): string {
  const diffHours = Math.round((value - anchorTs) / (1000 * 60 * 60))

  if (diffHours === 0) {
    return '現在'
  }

  return `${diffHours > 0 ? '+' : '-'}${Math.abs(diffHours)}H`
}

export function formatActualWindow(start: number, end: number): string {
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

export function formatCurrencyPrice(value: number, digits = 2): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value)
}

export function formatCompactPrice(value: number): string {
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0,
  }).format(value)
}

export function formatSignedPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

export function formatNullablePercent(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return '-'
  }

  return `${(value * 100).toFixed(1)}%`
}

export function formatNullableScore(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return '-'
  }

  return value.toFixed(3)
}

export function formatSignedDecimal(value: number, digits = 2): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`
}

export function getFocusedPriceDomain(rows: ChartRow[]): [number, number] {
  const values = rows
    .flatMap((row) => {
      const acc: number[] = []
      if (row.close !== null) acc.push(row.close)
      if (row.forecastBase !== null) acc.push(row.forecastBase)
      if (row.forecastLower !== null) acc.push(row.forecastLower)
      if (row.forecastUpper !== null) acc.push(row.forecastUpper)
      if (row.forecastInnerLower !== null) acc.push(row.forecastInnerLower)
      if (row.forecastInnerLower !== null && row.forecastInnerSpread !== null) {
        acc.push(row.forecastInnerLower + row.forecastInnerSpread)
      }
      if (row.actualClose !== null) acc.push(row.actualClose)
      return acc
    })
    .filter((value) => Number.isFinite(value))

  if (values.length === 0) {
    return getVisiblePriceDomain(rows)
  }

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = Math.max(max - min, 10)
  const pad = Math.max(range * 0.1, 6)

  return [Math.floor(min - pad), Math.ceil(max + pad)]
}

function getVisiblePriceDomain(rows: ChartRow[]): [number, number] {
  const values = rows
    .flatMap((row) => [row.close, row.forecastBase, row.forecastLower, row.forecastUpper, row.actualClose])
    .filter((value): value is number => value !== null)

  if (values.length === 0) {
    return [0, 1]
  }

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min
  const pad = range === 0 ? Math.max(min * 0.0025, 1) : Math.max(range * 0.05, max * 0.0025, 1)

  return [Math.floor(min - pad), Math.ceil(max + pad)]
}

function deriveRecommendedAction({
  policyStatus,
  predictionLagHours,
  trustScore,
  directionalAccuracy,
  utilityScore,
}: {
  policyStatus: string
  predictionLagHours: number | null
  trustScore: number
  directionalAccuracy: number | null
  utilityScore: number | null
}): RecommendedAction {
  if (policyStatus === 'no-trade' || (predictionLagHours !== null && predictionLagHours >= 48)) {
    return 'Reduce'
  }

  if (trustScore >= 72 && (directionalAccuracy ?? 0) >= 0.7 && (utilityScore ?? 0) >= 0.6) {
    return 'Active'
  }

  if (trustScore >= 55) {
    return 'Hold'
  }

  return 'Reduce'
}

function buildDecisionSummary({
  recommendedAction,
  hours,
  freshness,
}: {
  recommendedAction: RecommendedAction
  hours: number
  freshness: DashboardData['provenance']['freshness']
}): string {
  if (recommendedAction === 'Reduce') {
    if ((freshness?.predictionLagHours ?? 0) >= 48) {
      return '更新が古いため、今回は見送り。'
    }

    return `${hours}H シグナルだが、今回は見送り。`
  }

  if (recommendedAction === 'Active') {
    return `${hours}H はエントリー候補。`
  }

  return `${hours}H はタイミング待ち。`
}

function buildActionCards({
  recommendedAction,
  trustScore,
  uncertaintyScore,
  freshness,
}: {
  recommendedAction: RecommendedAction
  trustScore: number
  uncertaintyScore: number
  freshness: DashboardData['provenance']['freshness']
}) {
  return [
    {
      key: 'follow' as const,
      title: 'Go',
      summary: '入る',
      detail: `確度 ${trustScore}%`,
      isCurrent: recommendedAction === 'Active',
    },
    {
      key: 'hold' as const,
      title: 'Wait',
      summary: '様子見',
      detail: `不確実 ${uncertaintyScore}%`,
      isCurrent: recommendedAction === 'Hold',
    },
    {
      key: 'reduce' as const,
      title: 'Pass',
      summary: '見送り',
      detail: (freshness?.predictionLagHours ?? 0) >= 48 ? '鮮度不足' : '根拠不足',
      isCurrent: recommendedAction === 'Reduce',
    },
  ]
}

function buildSystemHealthRows({
  horizons,
  selectedHorizon,
}: {
  horizons: HorizonRow[]
  selectedHorizon: number
}) {
  const short = horizons[0]
  const current = horizons.find((row) => row.horizon === selectedHorizon) ?? horizons[0]
  const long = horizons[horizons.length - 1] ?? current
  const rows = [short, current, long].filter(
    (row, index, list) => list.findIndex((candidate) => candidate.horizon === row.horizon) === index,
  )

  return rows.slice(0, 3).map((row) => {
    const uncertaintyScore = getUncertaintyScore(row.uncertainty, horizons)
    const state = classifyHealth(row.expectedReturnPct, uncertaintyScore)

    return {
      label: `${row.hours}H`,
      status: state.label,
      tone: state.tone,
    }
  })
}

function buildReliabilityVerdict({
  recommendedAction,
  freshnessScore,
  trustScore,
}: {
  recommendedAction: RecommendedAction
  freshnessScore: number
  trustScore: number
}): ReliabilityVerdict {
  if (recommendedAction === 'Reduce') {
    return {
      label: 'Pass',
      summary: '鮮度か根拠が不足している。',
      tone: 'caution',
    }
  }

  if (recommendedAction === 'Active' && freshnessScore >= 70 && trustScore >= 72) {
    return {
      label: 'Go',
      summary: '入る条件が揃っている。',
      tone: 'strong',
    }
  }

  return {
    label: 'Wait',
    summary: '方向性はあるが、入るにはまだ早い。',
    tone: 'watch',
  }
}

function buildReasons({
  runQuality,
  freshness,
  directionalAccuracy,
  rangeCoverage,
  utilityScore,
  uncertaintyScore,
}: {
  runQuality: string
  freshness: DashboardData['provenance']['freshness']
  directionalAccuracy: number | null
  rangeCoverage: number | null
  utilityScore: number | null
  uncertaintyScore: number
}): DetailReason[] {
  const reasons: DetailReason[] = []

  if (runQuality !== 'fresh' || (freshness?.predictionLagHours ?? 0) >= 48) {
    reasons.push({
      title: '鮮度に注意',
      body: `予測ラグ ${formatNullableHours(freshness?.predictionLagHours)}`,
      tone: 'caution',
    })
  }

  if (directionalAccuracy !== null) {
    reasons.push({
      title: directionalAccuracy >= 0.7 ? '方向一致が強い' : '方向一致は中位',
      body: `一致率 ${formatNullablePercent(directionalAccuracy)}`,
      tone: directionalAccuracy >= 0.7 ? 'strong' : 'watch',
    })
  }

  if (rangeCoverage !== null) {
    reasons.push({
      title: rangeCoverage >= 0.7 ? 'レンジ捕捉は十分' : 'レンジ捕捉は限定的',
      body: `帯域 ${formatNullablePercent(rangeCoverage)}`,
      tone: rangeCoverage >= 0.7 ? 'strong' : 'watch',
    })
  }

  if (utilityScore !== null) {
    reasons.push({
      title: utilityScore >= 0.6 ? 'Utility 良好' : 'Utility は弱め',
      body: `score ${utilityScore.toFixed(3)}`,
      tone: utilityScore >= 0.6 ? 'strong' : 'watch',
    })
  }

  reasons.push({
    title: uncertaintyScore <= 33 ? '不確実性は低め' : uncertaintyScore <= 66 ? '不確実性は標準' : '不確実性は高め',
    body: `${uncertaintyScore}%`,
    tone: uncertaintyScore <= 33 ? 'strong' : uncertaintyScore <= 66 ? 'watch' : 'caution',
  })

  return reasons.slice(0, 3)
}

function deriveFreshnessScore({
  runQuality,
  freshness,
}: {
  runQuality: string
  freshness: DashboardData['provenance']['freshness']
}): number {
  const base = runQuality === 'fresh' ? 88 : runQuality === 'degraded' ? 64 : 44
  const penalty =
    (freshness?.predictionLagHours ?? 0) >= 72
      ? 16
      : (freshness?.predictionLagHours ?? 0) >= 48
        ? 10
        : (freshness?.artifactLagHours ?? 0) >= 24
          ? 4
          : 0

  return clampRange(base - penalty, 12, 96)
}

function classifyHealth(expectedReturnPct: number, uncertaintyScore: number) {
  if (Math.abs(expectedReturnPct) < 0.4) {
    return { label: '中立', tone: 'neutral' as const }
  }
  if (expectedReturnPct > 0 && uncertaintyScore <= 45) {
    return { label: '上向き', tone: 'positive' as const }
  }
  if (expectedReturnPct > 0) {
    return { label: 'やや上向き', tone: 'positive' as const }
  }
  return { label: '警戒', tone: 'risk' as const }
}

function compactDirectionLabel(expectedReturnPct: number) {
  if (Math.abs(expectedReturnPct) < 0.2) {
    return '横ばい'
  }

  return expectedReturnPct > 0 ? '上向き' : '下向き'
}

function getDecisionTone(value: RecommendedAction): DecisionTone {
  switch (value) {
    case 'Active':
      return 'follow'
    case 'Hold':
      return 'watch'
    default:
      return 'skip'
  }
}

function getNormalizedUncertainty(value: number, horizons: HorizonRow[]) {
  const maxUncertainty = Math.max(...horizons.map((row) => row.uncertainty))
  const minUncertainty = Math.min(...horizons.map((row) => row.uncertainty))

  if (maxUncertainty === minUncertainty) {
    return 1
  }

  return clamp01(1 - (value - minUncertainty) / (maxUncertainty - minUncertainty))
}

function getUncertaintyScore(value: number, horizons: HorizonRow[]) {
  return Math.round((1 - getNormalizedUncertainty(value, horizons)) * 100)
}

function getUncertaintyLabel(score: number) {
  if (score <= 33) {
    return '低め'
  }
  if (score <= 66) {
    return '標準'
  }

  return '高め'
}

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value))
}

function clampRange(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function formatNullableHours(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return '-'
  }

  return `${value.toFixed(1)}h`
}
