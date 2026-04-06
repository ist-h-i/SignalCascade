export type HorizonRow = {
  horizon: number
  hours: number
  muT: number | null
  sigmaT: number
  sigmaTSq: number | null
  predictedClose: number
  lowerClose: number
  upperClose: number
  uncertainty: number
  expectedReturnPct: number
  actualClose: number | null
  actualReturnPct: number | null
}

export type ChartRow = {
  ts: number
  label: string
  phase: 'history' | 'forecast'
  close: number | null
  high: number | null
  low: number | null
  forecastBase: number | null
  forecastLower: number | null
  forecastInnerLower: number | null
  forecastInnerSpread: number | null
  forecastUpper: number | null
  forecastSpread: number | null
  forecastOuterLowerBase: number | null
  forecastOuterLowerSpread: number | null
  forecastOuterUpperBase: number | null
  forecastOuterUpperSpread: number | null
  actualClose: number | null
  actualRangeStart: number | null
  actualRangeEnd: number | null
}

export type MicroCandleRow = {
  ts: number
  open: number
  high: number
  low: number
  close: number
}

export type MetricPoint = {
  epoch: number
  trainTotal: number
  validationTotal: number
  trainReturn: number
  validationReturn: number
  trainProfit: number
  validationProfit: number
  trainLogWealth: number | null
  validationLogWealth: number | null
}

export type ValidationMetrics = {
  averageLogWealth: number | null
  realizedPnlPerAnchor: number | null
  cvarTailLoss: number | null
  noTradeBandHitRate: number | null
  shapeGateUsage: number | null
  expertEntropy: number | null
  muCalibration: number | null
  sigmaCalibration: number | null
  directionalAccuracy: number | null
  projectValueScore: number | null
  utilityScore: number | null
  exactSmoothHorizonAgreement: number | null
  exactSmoothNoTradeAgreement: number | null
  exactSmoothPositionMae: number | null
  exactSmoothUtilityRegret: number | null
  logWealthClampHitRate: number | null
  stateResetMode: string | null
  costMultiplier: number | null
  gammaMultiplier: number | null
  minPolicySigma: number | null
  turnover: number | null
  maxDrawdown: number | null
}

export type NarrativeSegment = {
  title: string
  summary: string
  fromStep: number
  toStep: number
}

export type DashboardData = {
  schemaVersion: number
  generatedAt: string
  instrument: string
  artifacts: {
    metricsSchemaVersion: number | null
    predictionSchemaVersion: number | null
    forecastSchemaVersion: number | null
    sourceSchemaVersion?: number | null
  }
  provenance: {
    rawRows: number
    artifactKind?: string | null
    artifactId?: string | null
    parentArtifactId?: string | null
    dataSnapshotSha256?: string | null
    configOrigin?: string | null
    sourcePath: string | null
    sourceOriginPath?: string | null
    gitCommitSha?: string | null
    gitDirty?: boolean | null
    start: string
    end: string
    manifestGeneratedAt?: string | null
    diagnosticsGeneratedAt?: string | null
    forecastGeneratedAt?: string | null
    predictionAnchorTime?: string | null
    freshness?: {
      dashboardGeneratedAt: string
      manifestGeneratedAt: string | null
      diagnosticsGeneratedAt: string | null
      forecastGeneratedAt: string | null
      predictionAnchorTime: string | null
      artifactLagHours: number | null
      forecastAgeHours: number | null
      diagnosticsAgeHours: number | null
      predictionLagHours: number | null
    }
  }
  run: {
    anchorTime: string
    anchorClose: number
    anchorCloseRaw?: number
    effectivePriceScale: number
    priceScale?: number
    selectedHorizon: number
    executedHorizon: number | null
    selectedHours: number
    previousPosition: number
    position: number
    tradeDelta: number
    noTradeBandHit: boolean
    gT: number
    selectedPolicyUtility: number
    overlayAction: string
    policyStatus: string
    stateResetMode: string | null
    costMultiplier: number | null
    gammaMultiplier: number | null
    minPolicySigma: number | null
    interruptedTuning: boolean
    runQuality: string
    trainSamples: number
    validationSamples: number
    sampleCount: number
    effectiveSampleCount: number | null
    purgedSamples: number | null
    bestValidationLoss: number
    bestEpoch: number
    convergenceGain: number
    generalizationGap: number
    sourceRows: number
    modelDirectory?: string | null
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
    validation: ValidationMetrics | null
  }
  narrative: {
    title: string
    summary: string
    bullets: string[]
    segments?: NarrativeSegment[]
  }
}
