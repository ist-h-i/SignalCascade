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

export type HorizonMixRow = {
  label: string
  share: number
  shareLabel: string
  detail: string
  tone: 'positive' | 'neutral' | 'risk'
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
  economicScore: number
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
  horizonMixRows: HorizonMixRow[]
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
  const blocked = data.metrics.blocked ?? null
  const live = data.metrics.live ?? null
  const structure = data.metrics.structure ?? null
  const horizonDiagnostics = data.metrics.horizonDiagnostics ?? []
  const availableHorizons = data.horizons.filter(
    (row) => row.actualClose !== null && row.actualReturnPct !== null,
  )
  const directionalAccuracy =
    live?.directionalAccuracy ??
    blocked?.directionalAccuracyMean ??
    validation?.directionalAccuracy ??
    (availableHorizons.length > 0
      ? availableHorizons.filter((row) => Math.sign(row.expectedReturnPct) === Math.sign(row.actualReturnPct ?? 0)).length /
        availableHorizons.length
      : null)
  const rangeCoverage =
    live?.interval1SigmaCoverage ??
    blocked?.interval1SigmaCoverageMean ??
    validation?.interval1SigmaCoverage ??
    (availableHorizons.length > 0
      ? availableHorizons.filter(
          (row) => (row.actualClose ?? Number.NaN) >= row.lowerClose && (row.actualClose ?? Number.NaN) <= row.upperClose,
        ).length / availableHorizons.length
      : null)
  const averagePriceErrorPct =
    live?.meanAbsoluteErrorPct ??
    validation?.muCalibration ??
    (availableHorizons.length > 0
      ? availableHorizons.reduce((sum, row) => {
          const actualClose = row.actualClose ?? row.predictedClose
          return sum + Math.abs((actualClose - row.predictedClose) / actualClose)
        }, 0) / availableHorizons.length
      : null)
  const probabilisticCalibrationScore =
    live?.probabilisticCalibrationScore ??
    blocked?.probabilisticCalibrationScoreMean ??
    validation?.probabilisticCalibrationScore ??
    null
  const runtimeAlignmentScore = structure?.runtimePolicyAlignmentScore ?? 1
  const shapeConcentration = structure?.dominantShapeClassShare ?? null
  const normalizedGap = clamp01(1 - Math.min(Math.abs(data.run.generalizationGap), 1))
  const normalizedError = averagePriceErrorPct === null ? 0.5 : clamp01(1 - Math.min(averagePriceErrorPct / 0.05, 1))
  const normalizedCoverage = rangeCoverage === null ? 0.5 : clamp01(rangeCoverage)
  const normalizedProbabilistic = probabilisticCalibrationScore === null ? 0.5 : clamp01(probabilisticCalibrationScore)
  const normalizedRuntimeAlignment = clamp01(runtimeAlignmentScore)
  const trustScore = Math.round(
    (
      normalizedCoverage * 0.24 +
      (directionalAccuracy ?? 0.5) * 0.24 +
      normalizedProbabilistic * 0.18 +
      normalizedRuntimeAlignment * 0.12 +
      normalizedError * 0.12 +
      normalizedGap * 0.10
    ) * 100,
  )
  const blockedObjective = blocked?.objectiveLogWealthMinusLambdaCvarMean ?? null
  const normalizedBlockedObjective =
    blockedObjective === null ? 0.35 : clamp01((blockedObjective + 0.001) / 0.002)
  const normalizedPnl =
    validation?.realizedPnlPerAnchor === null || validation?.realizedPnlPerAnchor === undefined
      ? 0.4
      : clamp01((validation.realizedPnlPerAnchor + 0.001) / 0.002)
  const normalizedDrawdown =
    validation?.maxDrawdown === null || validation?.maxDrawdown === undefined
      ? 0.5
      : clamp01(1 - Math.min(validation.maxDrawdown / 0.15, 1))
  const normalizedUtility =
    validation?.utilityScore === null || validation?.utilityScore === undefined
      ? 0.5
      : clamp01(validation.utilityScore)
  const economicScore = Math.round(
    (
      normalizedBlockedObjective * 0.36 +
      normalizedUtility * 0.26 +
      normalizedPnl * 0.18 +
      normalizedDrawdown * 0.20
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
    economicScore,
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
    economicScore,
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
      economicScore,
      uncertaintyScore,
      freshness,
    }),
    systemHealthRows: buildSystemHealthRows({
      structure,
      selectedHorizon: selectedForecast.horizon,
    }),
    horizonMixRows: buildHorizonMixRows(horizonDiagnostics),
    reliabilityMeters: [
      {
        label: '信頼',
        value: trustScore,
        valueLabel: `${trustScore}%`,
        caption: '方向一致・1σ・runtime',
        tone: trustScore >= 72 ? 'cool' : trustScore >= 55 ? 'warm' : 'risk',
      },
      {
        label: '収益',
        value: economicScore,
        valueLabel: `${economicScore}%`,
        caption: blockedObjective === null ? '収益評価は未取得' : '収益評価と方策価値',
        tone: economicScore >= 62 ? 'cool' : economicScore >= 46 ? 'warm' : 'risk',
      },
      {
        label: '鮮度',
        value: freshnessScore,
        valueLabel: `${freshnessScore}%`,
        caption: '更新状態とラグ',
        tone: freshnessScore >= 72 ? 'cool' : freshnessScore >= 55 ? 'warm' : 'risk',
      },
      {
        label: '不確実性',
        value: uncertaintyScore,
        valueLabel: `${uncertaintyScore}%`,
        caption: `${selectedForecast.hours}H の振れ幅`,
        tone: uncertaintyScore <= 33 ? 'cool' : uncertaintyScore <= 66 ? 'warm' : 'risk',
      },
    ],
    reliabilityVerdict: buildReliabilityVerdict({
      recommendedAction,
      freshnessScore,
      trustScore,
      economicScore,
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
      economicScore,
      runtimeAlignmentScore,
      shapeConcentration,
    }),
    freshnessFacts: [
      {
        label: '基準時刻',
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
    ],
    detailMetrics: [
      {
        label: '方向一致',
        value: formatNullablePercent(directionalAccuracy),
        hint: '方向の一致',
      },
      {
        label: '1σ捕捉',
        value: formatNullablePercent(rangeCoverage),
        hint: '実績が1σ内',
      },
      {
        label: '確率校正',
        value: formatNullablePercent(probabilisticCalibrationScore),
        hint: 'カバレッジと PIT',
      },
      {
        label: '収益評価',
        value: formatNullableSignedMetric(blockedObjective, 5),
        hint: 'wealth - λ*CVaR',
      },
      {
        label: 'Runtime整合',
        value: formatNullablePercent(runtimeAlignmentScore),
        hint: 'runtime 設定との整合',
      },
      {
        label: 'Shape偏り',
        value: formatNullablePercent(shapeConcentration),
        hint: '最頻クラスの集中',
      },
      {
        label: '方策価値',
        value: formatNullableScore(validation?.utilityScore),
        hint: '方策の価値',
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

export function formatNullableSignedMetric(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined) {
    return '-'
  }

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
  economicScore,
  directionalAccuracy,
  utilityScore,
}: {
  policyStatus: string
  predictionLagHours: number | null
  trustScore: number
  economicScore: number
  directionalAccuracy: number | null
  utilityScore: number | null
}): RecommendedAction {
  if (policyStatus === 'no-trade' || (predictionLagHours !== null && predictionLagHours >= 48)) {
    return 'Reduce'
  }

  if (trustScore >= 70 && economicScore >= 58 && (directionalAccuracy ?? 0) >= 0.55 && (utilityScore ?? 0) >= 0.55) {
    return 'Active'
  }

  if (trustScore >= 55 && economicScore >= 42) {
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
  economicScore,
  uncertaintyScore,
  freshness,
}: {
  recommendedAction: RecommendedAction
  trustScore: number
  economicScore: number
  uncertaintyScore: number
  freshness: DashboardData['provenance']['freshness']
}) {
  return [
    {
      key: 'follow' as const,
      title: 'Go',
      summary: '入る',
      detail: `信頼 ${trustScore}% / 収益 ${economicScore}%`,
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
  structure,
  selectedHorizon,
}: {
  structure: DashboardData['metrics']['structure']
  selectedHorizon: number
}): SystemHealthRow[] {
  const runtimeAlignmentScore = structure?.runtimePolicyAlignmentScore ?? null
  const dominantShapeClassShare = structure?.dominantShapeClassShare ?? null
  const dominantHorizon = structure?.dominantHorizon ?? selectedHorizon
  const dominantHorizonShare = structure?.dominantHorizonShare ?? null

  return [
    {
      label: 'Runtime',
      status: runtimeAlignmentScore === null ? '-' : `一致 ${formatNullablePercent(runtimeAlignmentScore)}`,
      tone:
        runtimeAlignmentScore === null
          ? 'neutral'
          : runtimeAlignmentScore >= 0.8
            ? 'positive'
            : runtimeAlignmentScore >= 0.6
              ? 'neutral'
              : 'risk' as const,
    },
    {
      label: 'Shape',
      status:
        dominantShapeClassShare === null
          ? '-'
          : `上位 ${formatNullablePercent(dominantShapeClassShare)}`,
      tone:
        dominantShapeClassShare === null
          ? 'neutral'
          : dominantShapeClassShare <= 0.55
            ? 'positive'
            : dominantShapeClassShare <= 0.75
              ? 'neutral'
              : 'risk' as const,
    },
    {
      label: 'Policy',
      status:
        dominantHorizonShare === null
          ? `${dominantHorizon * 4}H`
          : `${dominantHorizon * 4}H ${formatNullablePercent(dominantHorizonShare)}`,
      tone:
        dominantHorizonShare === null
          ? 'neutral'
          : dominantHorizonShare <= 0.55
            ? 'positive'
            : dominantHorizonShare <= 0.75
              ? 'neutral'
              : 'risk' as const,
    },
  ]
}

function buildHorizonMixRows(horizonDiagnostics: DashboardData['metrics']['horizonDiagnostics']): HorizonMixRow[] {
  return [...(horizonDiagnostics ?? [])]
    .sort((left, right) => left.horizon - right.horizon)
    .map((row) => {
      const share = row.policyHorizonShare ?? row.selectionRate ?? 0
      const coverage = row.interval1SigmaCoverage
      const tone =
        share > 0.75
          ? 'risk'
          : coverage !== null && coverage >= 0.65
            ? 'positive'
            : share > 0.3
              ? 'neutral'
              : 'neutral'

      return {
        label: `${row.hours}H`,
        share: Math.round(clamp01(share) * 100),
        shareLabel: formatNullablePercent(share),
        detail:
          coverage === null
            ? '選択率のみ'
            : `1σ ${formatNullablePercent(coverage)} / 方向 ${formatNullablePercent(row.directionalAccuracy)}`,
        tone,
      }
    })
}

function buildReliabilityVerdict({
  recommendedAction,
  freshnessScore,
  trustScore,
  economicScore,
}: {
  recommendedAction: RecommendedAction
  freshnessScore: number
  trustScore: number
  economicScore: number
}): ReliabilityVerdict {
  if (recommendedAction === 'Reduce') {
    return {
      label: '見送り',
      summary: '鮮度・信頼・収益のどれかが不足している。',
      tone: 'caution',
    }
  }

  if (recommendedAction === 'Active' && freshnessScore >= 70 && trustScore >= 70 && economicScore >= 58) {
    return {
      label: '強気',
      summary: '信頼と収益の両方が基準を超えている。',
      tone: 'strong',
    }
  }

  return {
    label: '様子見',
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
  economicScore,
  runtimeAlignmentScore,
  shapeConcentration,
}: {
  runQuality: string
  freshness: DashboardData['provenance']['freshness']
  directionalAccuracy: number | null
  rangeCoverage: number | null
  utilityScore: number | null
  uncertaintyScore: number
  economicScore: number
  runtimeAlignmentScore: number | null
  shapeConcentration: number | null
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
      title: directionalAccuracy >= 0.6 ? '方向一致が維持' : '方向一致は弱め',
      body: `一致率 ${formatNullablePercent(directionalAccuracy)}`,
      tone: directionalAccuracy >= 0.6 ? 'strong' : 'watch',
    })
  }

  if (rangeCoverage !== null) {
    reasons.push({
      title: rangeCoverage >= 0.6 ? '1σ帯域は機能' : '1σ帯域は弱い',
      body: `捕捉率 ${formatNullablePercent(rangeCoverage)}`,
      tone: rangeCoverage >= 0.6 ? 'strong' : 'watch',
    })
  }

  if (runtimeAlignmentScore !== null && runtimeAlignmentScore < 0.8) {
    reasons.push({
      title: runtimeAlignmentScore >= 0.6 ? 'Runtime設定にズレ' : 'Runtime設定が大きくズレ',
      body: `整合 ${formatNullablePercent(runtimeAlignmentScore)}`,
      tone: runtimeAlignmentScore >= 0.6 ? 'watch' : 'caution',
    })
  }

  if (shapeConcentration !== null && shapeConcentration > 0.75) {
    reasons.push({
      title: 'shape が偏っている',
      body: `上位比率 ${formatNullablePercent(shapeConcentration)}`,
      tone: 'caution',
    })
  }

  if (utilityScore !== null && economicScore < 50) {
    reasons.push({
      title: '収益面はまだ弱い',
      body: `方策価値 ${utilityScore.toFixed(3)} / 収益 ${economicScore}%`,
      tone: 'watch',
    })
  }

  if (reasons.length < 3) {
    reasons.push({
      title: uncertaintyScore <= 33 ? '不確実性は低め' : uncertaintyScore <= 66 ? '不確実性は標準' : '不確実性は高め',
      body: `${uncertaintyScore}%`,
      tone: uncertaintyScore <= 33 ? 'strong' : uncertaintyScore <= 66 ? 'watch' : 'caution',
    })
  }

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
