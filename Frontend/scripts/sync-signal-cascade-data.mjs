import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = path.resolve(frontendRoot, '..')
const pyTorchRoot = path.resolve(repoRoot, 'PyTorch')
const artifactRoot = path.resolve(repoRoot, 'PyTorch/artifacts/gold_xauusd_m30')
const currentRunDir = path.join(artifactRoot, 'current')
const predictionPath = path.join(currentRunDir, 'prediction.json')
const metricsPath = path.join(currentRunDir, 'metrics.json')
const configPath = path.join(currentRunDir, 'config.json')
const sourceMetaPath = path.join(currentRunDir, 'source.json')
const forecastSummaryPath = path.join(currentRunDir, 'forecast_summary.json')
const manifestPath = path.join(currentRunDir, 'manifest.json')
const validationSummaryPath = path.join(currentRunDir, 'validation_summary.json')
const policySummaryPath = path.join(currentRunDir, 'policy_summary.csv')
const horizonDiagPath = path.join(currentRunDir, 'horizon_diag.csv')
const validationRowsPath = path.join(currentRunDir, 'validation_rows.csv')
const outputPath = path.resolve(frontendRoot, 'public/dashboard-data.json')
const liveDataDir = path.join(artifactRoot, 'live')
const liveCsvPath = path.join(liveDataDir, 'xauusd_m30_latest.csv')
const pyTorchPython = path.resolve(pyTorchRoot, '.venv/bin/python')
const defaultLookbackDays = 360
const defaultDownloadAttempts = 3
const defaultDownloadRetryDelayMs = 5_000

const targetDateJst = resolveTargetDateJst(process.env.SIGNAL_CASCADE_TARGET_DATE)
const configuredLookbackDays = resolveLookbackDays()
const configuredDownloadAttempts = resolvePositiveIntegerEnv('SIGNAL_CASCADE_DOWNLOAD_ATTEMPTS', defaultDownloadAttempts)
const configuredDownloadRetryDelayMs = resolveNonNegativeIntegerEnv(
  'SIGNAL_CASCADE_DOWNLOAD_RETRY_MS',
  defaultDownloadRetryDelayMs,
)
const initialCsvPath = resolveCsvPath(targetDateJst)
ensureCurrentRun(initialCsvPath)
const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'))
const config = JSON.parse(fs.readFileSync(configPath, 'utf8'))
const sourceMeta = fs.existsSync(sourceMetaPath)
  ? JSON.parse(fs.readFileSync(sourceMetaPath, 'utf8'))
  : null
const prediction = JSON.parse(fs.readFileSync(predictionPath, 'utf8'))
const forecastSummary = fs.existsSync(forecastSummaryPath)
  ? JSON.parse(fs.readFileSync(forecastSummaryPath, 'utf8'))
  : null
const manifest = fs.existsSync(manifestPath)
  ? JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  : null
const validationSummary = loadRequiredCurrentDiagnosticsSummary()
const validationRowsTable = loadRequiredDiagnosticsCsv(validationRowsPath)
const horizonDiagTable = loadRequiredDiagnosticsCsv(horizonDiagPath)
const diagnosticsGeneratedAt = validationSummary.generated_at_utc
const manifestGeneratedAt = resolveManifestGeneratedAt(manifest)
const anchorTime = normalizeDate(prediction.anchor_time)
const {
  csvPath,
  csvRows,
  bars4h,
  anchorIndex,
} = resolveDashboardCsvContext({
  initialCsvPath,
  metrics,
  prediction,
})

const anchorBar = bars4h[anchorIndex]
const priceScale = resolvePositiveNumber(
  prediction.effective_price_scale ??
    prediction.price_scale ??
    forecastSummary?.effective_price_scale ??
    forecastSummary?.price_scale ??
    sourceMeta?.effective_price_scale ??
    sourceMeta?.price_scale,
  1,
)
const anchorCloseDisplay = resolvePositiveNumber(
  prediction.current_close_display ??
    forecastSummary?.anchor_close_display,
  anchorBar.close / priceScale,
)
const meanByHorizon = prediction.mu_t ?? prediction.expected_log_returns ?? {}
const sigmaByHorizon = prediction.sigma_t ?? prediction.uncertainties ?? {}
const sigmaSqByHorizon = prediction.sigma_t_sq ?? {}
const predictedCloseByHorizonRaw =
  prediction.median_predicted_close_by_horizon ??
  prediction.median_predicted_closes ??
  prediction.predicted_closes ??
  {}
const predictedCloseByHorizon =
  prediction.median_predicted_close_display_by_horizon ??
  prediction.median_predicted_closes_display ??
  scalePriceMap(predictedCloseByHorizonRaw, priceScale)
const selectedPolicyUtility = prediction.selected_policy_utility ?? prediction.policy_score ?? 0
const gT = prediction.g_t ?? prediction.tradeability_gate ?? 0
const horizons = Object.keys(meanByHorizon)
  .map((value) => Number(value))
  .filter((value) => Number.isFinite(value))
  .sort((left, right) => left - right)
const horizonRows = buildHorizonRows({
  horizons,
  bars4h,
  anchorIndex,
  anchorClose: anchorCloseDisplay,
  meanByHorizon,
  sigmaByHorizon,
  sigmaSqByHorizon,
  predictedCloseByHorizon,
  priceScale,
})
const validationRowDiagnostics = buildValidationRowDiagnostics(validationRowsTable)
const horizonDiagnostics = buildHorizonDiagnostics({
  horizonRows,
  horizonDiagTable,
  validationRowDiagnostics,
  validationSummary,
})
const chartRows = buildChartRows({
  rawRows: csvRows,
  anchorTime,
  anchorClose: anchorCloseDisplay,
  anchorBar,
  horizonRows,
  bars4h,
  anchorIndex,
  priceScale,
})
const selectedHorizon =
  prediction.executed_horizon ??
  prediction.policy_horizon ??
  prediction.accepted_horizon ??
  prediction.proposed_horizon ??
  horizons[0]
const history = metrics.history.map((row) => ({
  epoch: row.epoch,
  trainTotal: row.train_total,
  validationTotal: row.validation_total,
  trainReturn: row.train_return,
  validationReturn: row.validation_return,
  trainProfit: row.train_profit ?? row.train_total,
  validationProfit: row.validation_profit ?? row.validation_total,
  trainLogWealth: row.train_log_wealth ?? null,
  validationLogWealth: row.validation_log_wealth ?? null,
}))
const bestEpochRow = history.reduce((best, row) => (row.validationTotal < best.validationTotal ? row : best))
const convergenceGain = history[0].validationTotal - bestEpochRow.validationTotal
const generalizationGap = bestEpochRow.validationTotal - bestEpochRow.trainTotal
const validationMetrics = metrics.validation_metrics ?? null
const artifactProvenance = resolveArtifactProvenance(sourceMeta)
const governance = resolveCurrentGovernance(sourceMeta)
const primaryStateResetMode =
  validationMetrics?.state_reset_mode ??
  validationSummary?.primary_state_reset_mode ??
  null
const blockedMetrics = buildBlockedMetrics({
  validationSummary,
  primaryStateResetMode,
  cvarWeight: config.cvar_weight ?? validationSummary?.checkpoint_audit?.cvar_weight ?? 0.2,
})
const structureMetrics = buildStructureMetrics({
  validationSummary,
  config,
})
const operatingPoint = {
  stateResetMode: primaryStateResetMode,
  costMultiplier: validationMetrics?.cost_multiplier ?? 1.0,
  gammaMultiplier: validationMetrics?.gamma_multiplier ?? 1.0,
  minPolicySigma: validationMetrics?.min_policy_sigma ?? config.min_policy_sigma ?? null,
}
const freshness = buildFreshness({
  dashboardGeneratedAt: new Date().toISOString(),
  manifestGeneratedAt,
  diagnosticsGeneratedAt,
  forecastGeneratedAt: forecastSummary?.generated_at_utc ?? null,
  predictionAnchorTime: prediction.anchor_time ?? null,
})
const interruptedTuning = Boolean(manifest?.interrupted_tuning)
const runQuality = deriveRunQuality({
  freshness,
  interruptedTuning,
})
const liveMetrics = buildLiveMetrics({
  horizonRows,
  selectedHorizon,
})
const overlayAction = prediction.no_trade_band_hit ? 'hold' : 'reduce'
const payload = {
  schemaVersion: 6,
  generatedAt: freshness.dashboardGeneratedAt,
  instrument: '金 / XAUUSD',
  artifacts: {
    metricsSchemaVersion: metrics.schema_version ?? null,
    predictionSchemaVersion: prediction.schema_version ?? null,
    forecastSchemaVersion: forecastSummary?.schema_version ?? null,
    sourceSchemaVersion: artifactProvenance.artifactSchemaVersion,
  },
  provenance: {
    rawRows: csvRows.length,
    artifactKind: artifactProvenance.artifactKind,
    artifactId: artifactProvenance.artifactId,
    parentArtifactId: artifactProvenance.parentArtifactId,
    dataSnapshotSha256: artifactProvenance.dataSnapshotSha256,
    configOrigin: artifactProvenance.configOrigin,
    sourcePath: artifactProvenance.sourcePath,
    sourceOriginPath: artifactProvenance.sourceOriginPath,
    gitCommitSha: artifactProvenance.gitCommitSha,
    gitDirty: artifactProvenance.gitDirty,
    start: toIso(csvRows[0].ts),
    end: toIso(csvRows[csvRows.length - 1].ts),
    manifestGeneratedAt,
    diagnosticsGeneratedAt,
    forecastGeneratedAt: forecastSummary?.generated_at_utc ?? null,
    predictionAnchorTime: prediction.anchor_time ?? null,
    freshness,
  },
  governance,
  run: {
    anchorTime: toIso(anchorBar.ts),
    anchorClose: anchorCloseDisplay,
    anchorCloseRaw: prediction.current_close_raw ?? forecastSummary?.anchor_close_raw ?? anchorBar.close,
    effectivePriceScale: priceScale,
    priceScale,
    selectedHorizon,
    executedHorizon: prediction.executed_horizon ?? null,
    selectedHours: selectedHorizon * 4,
    previousPosition: prediction.q_t_prev ?? prediction.previous_position ?? 0,
    position: prediction.position,
    tradeDelta: prediction.q_t_trade_delta ?? prediction.trade_delta ?? 0,
    noTradeBandHit: prediction.no_trade_band_hit === true,
    gT,
    selectedPolicyUtility,
    overlayAction,
    policyStatus: prediction.no_trade_band_hit ? 'no-trade' : 'active',
    stateResetMode: operatingPoint.stateResetMode,
    costMultiplier: operatingPoint.costMultiplier,
    gammaMultiplier: operatingPoint.gammaMultiplier,
    minPolicySigma: operatingPoint.minPolicySigma,
    interruptedTuning,
    runQuality,
    trainSamples: metrics.train_samples,
    validationSamples: metrics.validation_samples,
    sampleCount: metrics.sample_count,
    effectiveSampleCount: metrics.effective_sample_count ?? null,
    purgedSamples: metrics.purged_samples ?? null,
    bestValidationLoss: metrics.best_validation_loss,
    bestEpoch: bestEpochRow.epoch,
    convergenceGain,
    generalizationGap,
    sourceRows: metrics.source_rows_used ?? csvRows.length,
    modelDirectory: toRepoRelativePath(currentRunDir),
    tuningSessionId: manifest?.session_id ?? null,
    bestParams: manifest?.best_candidate
      ? {
          epochs: manifest.best_candidate.epochs,
          batchSize: manifest.best_candidate.batch_size,
          learningRate: manifest.best_candidate.learning_rate,
          hiddenDim: manifest.best_candidate.hidden_dim,
          dropout: manifest.best_candidate.dropout,
          weightDecay: manifest.best_candidate.weight_decay,
        }
      : null,
  },
  chart: {
    rows: chartRows.rows,
    microRows: chartRows.microRows,
    yDomain: chartRows.yDomain,
  },
  horizons: horizonRows,
  metrics: {
    history,
    validation: validationMetrics
      ? {
          averageLogWealth: validationMetrics.average_log_wealth ?? null,
          realizedPnlPerAnchor: validationMetrics.realized_pnl_per_anchor ?? null,
          cvarTailLoss: validationMetrics.cvar_tail_loss ?? null,
          noTradeBandHitRate: validationMetrics.no_trade_band_hit_rate ?? null,
          shapeGateUsage: validationMetrics.shape_gate_usage ?? null,
          expertEntropy: validationMetrics.expert_entropy ?? null,
          muCalibration: validationMetrics.mu_calibration ?? null,
          sigmaCalibration: validationMetrics.sigma_calibration ?? null,
          directionalAccuracy: validationMetrics.directional_accuracy ?? null,
          projectValueScore: validationMetrics.project_value_score ?? null,
          exactSmoothHorizonAgreement: validationMetrics.exact_smooth_horizon_agreement ?? null,
          exactSmoothNoTradeAgreement: validationMetrics.exact_smooth_no_trade_agreement ?? null,
          exactSmoothPositionMae: validationMetrics.exact_smooth_position_mae ?? null,
          exactSmoothUtilityRegret: validationMetrics.exact_smooth_utility_regret ?? null,
          logWealthClampHitRate: validationMetrics.log_wealth_clamp_hit_rate ?? null,
          stateResetMode: validationMetrics.state_reset_mode ?? null,
          costMultiplier: validationMetrics.cost_multiplier ?? null,
          gammaMultiplier: validationMetrics.gamma_multiplier ?? null,
          minPolicySigma: validationMetrics.min_policy_sigma ?? null,
          turnover: validationMetrics.turnover ?? null,
          maxDrawdown: validationMetrics.max_drawdown ?? null,
          utilityScore: validationMetrics.utility_score ?? null,
          interval1SigmaCoverage: validationMetrics.interval_1sigma_coverage ?? null,
          interval2SigmaCoverage: validationMetrics.interval_2sigma_coverage ?? null,
          pitMean: validationMetrics.pit_mean ?? null,
          pitVariance: validationMetrics.pit_variance ?? null,
          normalizedAbsError: validationMetrics.normalized_abs_error ?? null,
          gaussianNll: validationMetrics.gaussian_nll ?? null,
          probabilisticCalibrationScore: validationMetrics.probabilistic_calibration_score ?? null,
        }
      : null,
    blocked: blockedMetrics,
    live: liveMetrics,
    structure: structureMetrics,
    horizonDiagnostics,
  },
  narrative: buildNarrative({ prediction, horizonRows }),
}

assertDashboardLineage({
  payload,
  sourceMeta,
  manifest,
  prediction,
  forecastSummary,
  diagnosticsGeneratedAt,
})

fs.mkdirSync(path.dirname(outputPath), { recursive: true })
fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2))
console.log(`Wrote ${outputPath}`)

function assertDashboardLineage({ payload, sourceMeta, manifest, prediction, forecastSummary, diagnosticsGeneratedAt }) {
  const expectedArtifactId =
    typeof sourceMeta?.artifact_id === 'string' && sourceMeta.artifact_id.length > 0
      ? sourceMeta.artifact_id
      : null

  if (expectedArtifactId === null) {
    throw new Error(`Current artifact is missing source.json artifact_id: ${sourceMetaPath}`)
  }

  if (payload.provenance.artifactId !== expectedArtifactId) {
    throw new Error(
      `dashboard provenance mismatch: dashboard artifactId=${payload.provenance.artifactId ?? 'null'} current artifact_id=${expectedArtifactId}`,
    )
  }

  const expectedSessionId =
    typeof manifest?.session_id === 'string' && manifest.session_id.length > 0
      ? manifest.session_id
      : null
  if (expectedSessionId !== null && payload.run.tuningSessionId !== expectedSessionId) {
    throw new Error(
      `dashboard session mismatch: dashboard tuningSessionId=${payload.run.tuningSessionId ?? 'null'} current session_id=${expectedSessionId}`,
    )
  }

  const expectedAnchorClose = resolveFiniteNumber(
    prediction.current_close_display ??
      forecastSummary?.anchor_close_display,
  )
  if (expectedAnchorClose !== null && !numbersMatch(payload.run.anchorClose, expectedAnchorClose)) {
    throw new Error(
      `dashboard anchorClose mismatch: dashboard anchorClose=${payload.run.anchorClose} current close display=${expectedAnchorClose}`,
    )
  }

  if (payload.provenance.diagnosticsGeneratedAt !== diagnosticsGeneratedAt) {
    throw new Error(
      `dashboard diagnostics mismatch: dashboard diagnosticsGeneratedAt=${payload.provenance.diagnosticsGeneratedAt ?? 'null'} current diagnosticsGeneratedAt=${diagnosticsGeneratedAt}`,
    )
  }

  const expectedPriceScale = resolveFiniteNumber(
    prediction.effective_price_scale ??
      prediction.price_scale ??
      forecastSummary?.effective_price_scale ??
      forecastSummary?.price_scale ??
      manifest?.effective_price_scale ??
      sourceMeta?.effective_price_scale ??
      sourceMeta?.price_scale,
  )
  if (expectedPriceScale !== null && !numbersMatch(payload.run.effectivePriceScale, expectedPriceScale)) {
    throw new Error(
      `dashboard effectivePriceScale mismatch: dashboard effectivePriceScale=${payload.run.effectivePriceScale} current effective_price_scale=${expectedPriceScale}`,
    )
  }
  if (
    payload.run.priceScale !== undefined &&
    !numbersMatch(payload.run.priceScale, payload.run.effectivePriceScale)
  ) {
    throw new Error(
      `dashboard priceScale alias mismatch: dashboard priceScale=${payload.run.priceScale} effectivePriceScale=${payload.run.effectivePriceScale}`,
    )
  }
}

function resolvePositiveNumber(value, fallback) {
  const numeric = Number(value)
  return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback
}

function resolveManifestGeneratedAt(manifest) {
  return toNonEmptyString(manifest?.generated_at_utc) ?? toNonEmptyString(manifest?.generated_at)
}

function resolveFiniteNumber(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function numbersMatch(left, right) {
  return Math.abs(Number(left) - Number(right)) <= 1e-9
}

function scalePriceMap(valueByKey, priceScale) {
  return Object.fromEntries(
    Object.entries(valueByKey ?? {}).map(([key, value]) => [key, Number(value) / priceScale]),
  )
}

function loadRequiredCurrentDiagnosticsSummary() {
  const missingFiles = [
    ['validation_summary.json', validationSummaryPath],
    ['policy_summary.csv', policySummaryPath],
    ['horizon_diag.csv', horizonDiagPath],
    ['validation_rows.csv', validationRowsPath],
  ].filter(([, filePath]) => !fs.existsSync(filePath))

  if (missingFiles.length > 0) {
    throw new Error(
      `diagnostics unpublished for current artifact: missing ${missingFiles.map(([name]) => name).join(', ')} under ${currentRunDir}`,
    )
  }

  const validationSummary = JSON.parse(fs.readFileSync(validationSummaryPath, 'utf8'))
  if (
    typeof validationSummary.generated_at_utc !== 'string' ||
    validationSummary.generated_at_utc.trim().length === 0
  ) {
    throw new Error(
      `diagnostics unpublished for current artifact: ${validationSummaryPath} is missing generated_at_utc`,
    )
  }

  return validationSummary
}

function loadRequiredDiagnosticsCsv(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`diagnostics CSV is missing: ${filePath}`)
  }
  return parseStructuredCsv(fs.readFileSync(filePath, 'utf8'))
}

function resolveCsvPath(targetDateJst) {
  if (process.env.SIGNAL_CASCADE_CSV_PATH) {
    return path.resolve(process.env.SIGNAL_CASCADE_CSV_PATH)
  }

  if (process.env.SIGNAL_CASCADE_DISABLE_TRAINING === '1') {
    return resolveStoredCsvPath()
  }

  if (process.env.SIGNAL_CASCADE_DISABLE_LIVE_SYNC === '1') {
    return resolveStoredCsvPath()
  }

  return refreshLatestCsv(targetDateJst)
}

function ensureCurrentRun(csvPath) {
  if (process.env.SIGNAL_CASCADE_DISABLE_TRAINING === '1') {
    if (!fs.existsSync(metricsPath) || !fs.existsSync(predictionPath)) {
      throw new Error(`Current run is missing under ${currentRunDir}`)
    }
    return
  }

  if (!fs.existsSync(pyTorchPython)) {
    throw new Error(`Python runtime was not found: ${pyTorchPython}`)
  }

  const train = spawnSync(
    pyTorchPython,
    [
      '-c',
      'from signal_cascade_pytorch.interfaces.cli import main; main()',
      'tune-latest',
      '--artifact-root',
      artifactRoot,
      '--csv',
      csvPath,
      '--csv-lookback-days',
      String(configuredLookbackDays),
    ],
    {
      cwd: pyTorchRoot,
      encoding: 'utf8',
      env: {
        ...process.env,
        PYTHONPATH: buildPythonPath(path.resolve(pyTorchRoot, 'src')),
      },
    },
  )

  if (train.status !== 0) {
    throw new Error(`Failed to train the latest model: ${train.stderr || train.stdout}`)
  }

  if (train.stdout.trim()) {
    console.log(train.stdout.trim())
  }
}

function resolveStoredCsvPath() {
  if (fs.existsSync(metricsPath)) {
    const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'))
    if (metrics?.source?.path) {
      return path.resolve(metrics.source.path)
    }
  }

  if (fs.existsSync(sourceMetaPath)) {
    const sourceMeta = JSON.parse(fs.readFileSync(sourceMetaPath, 'utf8'))
    if (sourceMeta?.path) {
      return path.resolve(sourceMeta.path)
    }
  }

  if (fs.existsSync(liveCsvPath)) {
    return liveCsvPath
  }

  throw new Error('No stored CSV source was found for the current SignalCascade run.')
}

function resolveDashboardCsvContext({ initialCsvPath, metrics, prediction }) {
  const anchorTime = normalizeDate(prediction.anchor_time)
  const initialContext = buildDashboardCsvContext(initialCsvPath, metrics)
  const initialAnchorIndex = initialContext.bars4h.findIndex((row) => row.ts === anchorTime.getTime())

  if (initialAnchorIndex >= 0) {
    return {
      ...initialContext,
      anchorIndex: initialAnchorIndex,
    }
  }

  const initialResolvedPath = path.resolve(initialCsvPath)
  const liveResolvedPath = path.resolve(liveCsvPath)
  if (initialResolvedPath !== liveResolvedPath && fs.existsSync(liveResolvedPath)) {
    const liveContext = buildDashboardCsvContext(liveResolvedPath, metrics)
    const liveAnchorIndex = liveContext.bars4h.findIndex((row) => row.ts === anchorTime.getTime())
    if (liveAnchorIndex >= 0) {
      return {
        ...liveContext,
        anchorIndex: liveAnchorIndex,
      }
    }
  }

  throw new Error(
    `Anchor time ${prediction.anchor_time} was not found in the 4h resampled series.`,
  )
}

function buildDashboardCsvContext(csvPath, metrics) {
  const allCsvRows = parseCsv(fs.readFileSync(csvPath, 'utf8'))
  const requiredSourceRows = resolveRequiredSourceRows(metrics, allCsvRows.length)
  const csvRows = allCsvRows.slice(-requiredSourceRows)
  const bars4h = resampleTo4h(csvRows)

  return {
    csvPath,
    csvRows,
    bars4h,
  }
}

function resolveArtifactProvenance(sourceMeta) {
  const artifactSchemaVersion = Number.isInteger(sourceMeta?.artifact_schema_version)
    ? sourceMeta.artifact_schema_version
    : null
  const isVersionedArtifact = artifactSchemaVersion !== null && artifactSchemaVersion >= 2
  const git = sourceMeta?.git ?? {}

  return {
    artifactSchemaVersion,
    artifactKind: isVersionedArtifact ? sourceMeta?.artifact_kind ?? null : null,
    artifactId: isVersionedArtifact ? sourceMeta?.artifact_id ?? null : null,
    parentArtifactId: isVersionedArtifact ? sourceMeta?.parent_artifact_id ?? null : null,
    dataSnapshotSha256: isVersionedArtifact ? sourceMeta?.data_snapshot_sha256 ?? null : null,
    configOrigin: isVersionedArtifact ? sourceMeta?.config_origin ?? null : null,
    sourcePath: isVersionedArtifact ? toRepoRelativePath(sourceMeta?.path) : null,
    sourceOriginPath: isVersionedArtifact ? toRepoRelativePath(sourceMeta?.source_origin_path) : null,
    gitCommitSha: isVersionedArtifact ? git.git_commit_sha ?? git.commit_sha ?? git.head ?? null : null,
    gitDirty: isVersionedArtifact ? git.git_dirty ?? git.dirty ?? null : null,
  }
}

function resolveCurrentGovernance(sourceMeta) {
  const governance = sourceMeta?.current_selection_governance ?? {}
  const productionCurrent = governance?.production_current ?? {}
  const acceptedCandidate = governance?.accepted_candidate ?? {}

  return {
    selectionMode: typeof governance?.selection_mode === 'string' ? governance.selection_mode : null,
    overrideReason: typeof governance?.override_reason === 'string' ? governance.override_reason : null,
    selectionStatus: typeof governance?.selection_status === 'string' ? governance.selection_status : null,
    productionCurrentCandidate:
      typeof productionCurrent?.candidate === 'string' ? productionCurrent.candidate : null,
    acceptedCandidate: typeof acceptedCandidate?.candidate === 'string' ? acceptedCandidate.candidate : null,
  }
}

function toRepoRelativePath(value) {
  if (!value || typeof value !== 'string') {
    return null
  }

  const resolved = path.resolve(value)
  const relative = path.relative(repoRoot, resolved)
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    return value
  }
  return relative.split(path.sep).join('/')
}

function refreshLatestCsv(targetDateJst) {
  if (!fs.existsSync(pyTorchPython)) {
    throw new Error(`Python runtime was not found: ${pyTorchPython}`)
  }

  fs.mkdirSync(liveDataDir, { recursive: true })
  const npmCacheDir = path.join(os.tmpdir(), 'signal-cascade-npm-cache')
  fs.mkdirSync(npmCacheDir, { recursive: true })

  const rawDownloadDir = fs.mkdtempSync(path.join(os.tmpdir(), 'signal-cascade-xauusd-'))
  const rawBaseName = 'xauusd_m30_dukascopy'
  const rawCsvPath = path.join(rawDownloadDir, `${rawBaseName}.csv`)
  const fromDate = shiftDate(targetDateJst, -configuredLookbackDays)
  const toDateExclusive = shiftDate(targetDateJst, 1)
  const downloadArgs = [
    '--yes',
    'dukascopy-node',
    '-i',
    'xauusd',
    '-from',
    fromDate,
    '-to',
    toDateExclusive,
    '-t',
    'm30',
    '-f',
    'csv',
    '-v',
    '-vu',
    'units',
    '-dir',
    rawDownloadDir,
    '-fn',
    rawBaseName,
  ]
  const downloadOptions = {
    cwd: repoRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      npm_config_cache: npmCacheDir,
    },
  }
  const download = runDukascopyDownload(downloadArgs, downloadOptions)

  if (download.status !== 0) {
    throw new Error(`Failed to download XAUUSD market data: ${download.stderr || download.stdout}`)
  }

  normalizeDukascopyCsv(rawCsvPath, liveCsvPath, endOfJstDayUtcMs(targetDateJst))

  if (download.stdout.trim().length > 0) {
    console.log(download.stdout.trim())
  }
  return liveCsvPath
}

function runDukascopyDownload(args, options) {
  let lastFailure = null
  for (let attempt = 1; attempt <= configuredDownloadAttempts; attempt += 1) {
    const result = spawnSync('npx', args, options)
    if (result.status === 0) {
      return result
    }

    lastFailure = result
    if (attempt >= configuredDownloadAttempts) {
      break
    }

    const failureOutput = (result.stderr || result.stdout || '').trim()
    if (failureOutput.length > 0) {
      console.warn(
        `Dukascopy download attempt ${attempt}/${configuredDownloadAttempts} failed; retrying in ${configuredDownloadRetryDelayMs}ms.\n${failureOutput}`,
      )
    } else {
      console.warn(
        `Dukascopy download attempt ${attempt}/${configuredDownloadAttempts} failed; retrying in ${configuredDownloadRetryDelayMs}ms.`,
      )
    }
    sleepMs(configuredDownloadRetryDelayMs)
  }

  return lastFailure ?? {
    status: 1,
    stdout: '',
    stderr: 'Dukascopy download did not run.',
  }
}

function sleepMs(durationMs) {
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return
  }
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, durationMs)
}

function toNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function buildFreshness({
  dashboardGeneratedAt,
  manifestGeneratedAt,
  diagnosticsGeneratedAt,
  forecastGeneratedAt,
  predictionAnchorTime,
}) {
  const timestamps = {
    dashboardGeneratedAt,
    manifestGeneratedAt,
    diagnosticsGeneratedAt,
    forecastGeneratedAt,
    predictionAnchorTime,
  }
  const parsed = Object.values(timestamps)
    .map((value) => parseIsoDate(value))
    .filter((value) => value !== null)
  const minTs = parsed.length > 0 ? Math.min(...parsed.map((value) => value.getTime())) : null
  const maxTs = parsed.length > 0 ? Math.max(...parsed.map((value) => value.getTime())) : null

  return {
    ...timestamps,
    artifactLagHours:
      minTs === null || maxTs === null ? null : Number(((maxTs - minTs) / (1000 * 60 * 60)).toFixed(2)),
    forecastAgeHours:
      timestamps.forecastGeneratedAt && timestamps.dashboardGeneratedAt
        ? computeHoursBetween(timestamps.forecastGeneratedAt, timestamps.dashboardGeneratedAt)
        : null,
    diagnosticsAgeHours:
      timestamps.diagnosticsGeneratedAt && timestamps.dashboardGeneratedAt
        ? computeHoursBetween(timestamps.diagnosticsGeneratedAt, timestamps.dashboardGeneratedAt)
        : null,
    predictionLagHours:
      timestamps.predictionAnchorTime && timestamps.dashboardGeneratedAt
        ? computeHoursBetween(timestamps.predictionAnchorTime, timestamps.dashboardGeneratedAt)
        : null,
  }
}

function deriveRunQuality({ freshness, interruptedTuning }) {
  if (Boolean(interruptedTuning)) {
    return 'degraded'
  }

  if ((freshness.artifactLagHours ?? 0) >= 24 || (freshness.predictionLagHours ?? 0) >= 48) {
    return 'stale'
  }

  return 'fresh'
}

function parseIsoDate(value) {
  if (!value) {
    return null
  }

  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? null : parsed
}

function computeHoursBetween(left, right) {
  const leftDate = parseIsoDate(left)
  const rightDate = parseIsoDate(right)

  if (!leftDate || !rightDate) {
    return null
  }

  return Number((Math.abs(rightDate.getTime() - leftDate.getTime()) / (1000 * 60 * 60)).toFixed(2))
}

function normalizeDukascopyCsv(inputPath, outputPath, cutoffTs) {
  const lines = fs
    .readFileSync(inputPath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)

  const rows = lines
    .slice(1)
    .map((line) => {
      const [timestamp, open, high, low, close, volume] = line.split(',')
      return {
        ts: Number(timestamp),
        open: Number(open),
        high: Number(high),
        low: Number(low),
        close: Number(close),
        volume: Number(volume ?? 0),
      }
    })
    .filter((row) => Number.isFinite(row.ts) && row.ts <= cutoffTs)
    .sort((left, right) => left.ts - right.ts)

  if (rows.length < 512) {
    throw new Error(`Live market data is too short after cutoff: ${rows.length} rows.`)
  }

  const content = [
    'timestamp,open,high,low,close,volume',
    ...rows.map((row) => (
      `${new Date(row.ts).toISOString().replace('.000Z', 'Z')},${row.open},${row.high},${row.low},${row.close},${row.volume}`
    )),
  ].join('\n')

  fs.writeFileSync(outputPath, `${content}\n`)
}

function parseStructuredCsv(content) {
  const lines = content
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
  if (lines.length === 0) {
    return []
  }

  const headers = lines[0].split(',')
  return lines.slice(1).map((line) => {
    const cells = line.split(',')
    return Object.fromEntries(
      headers.map((header, index) => [header, cells[index] ?? '']),
    )
  })
}

function parseCsv(content) {
  const lines = content
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
  return lines.slice(1).map((line) => {
    const [timestamp, open, high, low, close, volume] = line.split(',')
    return {
      ts: parseTimestamp(timestamp),
      open: Number(open),
      high: Number(high),
      low: Number(low),
      close: Number(close),
      volume: Number(volume),
    }
  })
}

function buildValidationRowDiagnostics(rows) {
  const byHorizon = new Map()

  for (const row of rows) {
    const horizon = resolveFiniteNumber(row.horizon)
    const actual = resolveFiniteNumber(row.y_raw)
    const predicted = resolveFiniteNumber(row.mu_t)
    const sigma = resolveFiniteNumber(row.sigma_t)
    const positionIfChosen = resolveFiniteNumber(row.position_if_chosen)
    const policyUtility = resolveFiniteNumber(row.policy_utility)
    const selectionFlag = resolveFiniteNumber(row.selected) ?? resolveFiniteNumber(row.policy_horizon_selected)

    if (horizon === null || actual === null || predicted === null) {
      continue
    }

    const horizonKey = Math.trunc(horizon)
    const sigmaSafe = Math.max(Math.abs(sigma ?? 0), 1e-6)
    const absError = Math.abs(actual - predicted)
    const sigmaAbsError = Math.abs(absError - sigmaSafe)
    const zScore = (actual - predicted) / sigmaSafe
    const pit = gaussianPit(zScore)
    const gaussianNll = 0.5 * Math.log(2 * Math.PI * sigmaSafe * sigmaSafe) + (0.5 * zScore * zScore)
    const bucket = byHorizon.get(horizonKey) ?? {
      sampleCount: 0,
      selectionCount: 0,
      absErrorSum: 0,
      sigmaAbsErrorSum: 0,
      directionCorrectCount: 0,
      interval1SigmaHits: 0,
      interval2SigmaHits: 0,
      pitSum: 0,
      pitSqSum: 0,
      normalizedAbsErrorSum: 0,
      gaussianNllSum: 0,
      meanPositionSum: 0,
      meanPolicyUtilitySum: 0,
    }

    bucket.sampleCount += 1
    bucket.selectionCount += selectionFlag === 1 ? 1 : 0
    bucket.absErrorSum += absError
    bucket.sigmaAbsErrorSum += sigmaAbsError
    bucket.directionCorrectCount += Math.sign(actual) === Math.sign(predicted) ? 1 : 0
    bucket.interval1SigmaHits += absError <= sigmaSafe ? 1 : 0
    bucket.interval2SigmaHits += absError <= (2 * sigmaSafe) ? 1 : 0
    bucket.pitSum += pit
    bucket.pitSqSum += pit * pit
    bucket.normalizedAbsErrorSum += absError / sigmaSafe
    bucket.gaussianNllSum += gaussianNll
    bucket.meanPositionSum += positionIfChosen ?? 0
    bucket.meanPolicyUtilitySum += policyUtility ?? 0
    byHorizon.set(horizonKey, bucket)
  }

  return new Map(
    [...byHorizon.entries()].map(([horizon, bucket]) => {
      const sampleCount = bucket.sampleCount
      const pitMean = sampleCount > 0 ? bucket.pitSum / sampleCount : null
      const pitVariance =
        sampleCount > 0 && pitMean !== null ? (bucket.pitSqSum / sampleCount) - (pitMean * pitMean) : null
      const interval1SigmaCoverage = sampleCount > 0 ? bucket.interval1SigmaHits / sampleCount : null
      const interval2SigmaCoverage = sampleCount > 0 ? bucket.interval2SigmaHits / sampleCount : null

      return [
        horizon,
        {
          horizon,
          sampleCount,
          selectionRate: sampleCount > 0 ? bucket.selectionCount / sampleCount : null,
          meanPolicyUtility: sampleCount > 0 ? bucket.meanPolicyUtilitySum / sampleCount : null,
          meanPosition: sampleCount > 0 ? bucket.meanPositionSum / sampleCount : null,
          muCalibration: sampleCount > 0 ? bucket.absErrorSum / sampleCount : null,
          sigmaCalibration: sampleCount > 0 ? bucket.sigmaAbsErrorSum / sampleCount : null,
          directionalAccuracy: sampleCount > 0 ? bucket.directionCorrectCount / sampleCount : null,
          interval1SigmaCoverage,
          interval2SigmaCoverage,
          pitMean,
          pitVariance,
          normalizedAbsError: sampleCount > 0 ? bucket.normalizedAbsErrorSum / sampleCount : null,
          gaussianNll: sampleCount > 0 ? bucket.gaussianNllSum / sampleCount : null,
          probabilisticCalibrationScore:
            interval1SigmaCoverage === null || interval2SigmaCoverage === null || pitMean === null || pitVariance === null
              ? null
              : probabilisticCalibrationScore({
                  interval1SigmaCoverage,
                  interval2SigmaCoverage,
                  pitMean,
                  pitVariance,
                }),
        },
      ]
    }),
  )
}

function buildHorizonDiagnostics({
  horizonRows,
  horizonDiagTable,
  validationRowDiagnostics,
  validationSummary,
}) {
  const policyDistribution = validationSummary?.validation?.policy_horizon_distribution ?? {}
  const horizons = new Set(
    [
      ...horizonRows.map((row) => row.horizon),
      ...horizonDiagTable.map((row) => Number(row.horizon)),
      ...Object.keys(policyDistribution).map((value) => Number(value)),
      ...validationRowDiagnostics.keys(),
    ].filter((value) => Number.isFinite(value)),
  )

  return [...horizons]
    .sort((left, right) => left - right)
    .map((horizon) => {
      const csvRow = horizonDiagTable.find((row) => Number(row.horizon) === horizon) ?? null
      const derived = validationRowDiagnostics.get(horizon) ?? null
      return {
        horizon,
        hours: horizon * 4,
        sampleCount:
          resolveFiniteNumber(csvRow?.sample_count) ??
          derived?.sampleCount ??
          null,
        policyHorizonShare:
          resolveFiniteNumber(policyDistribution[String(horizon)]) ??
          resolveFiniteNumber(csvRow?.policy_horizon_share) ??
          derived?.selectionRate ??
          null,
        selectionRate:
          resolveFiniteNumber(csvRow?.selection_rate) ??
          derived?.selectionRate ??
          null,
        meanPolicyUtility:
          resolveFiniteNumber(csvRow?.mean_policy_utility) ??
          derived?.meanPolicyUtility ??
          null,
        meanPosition:
          resolveFiniteNumber(csvRow?.mean_position) ??
          derived?.meanPosition ??
          null,
        muCalibration:
          resolveFiniteNumber(csvRow?.mu_calibration) ??
          derived?.muCalibration ??
          null,
        sigmaCalibration:
          resolveFiniteNumber(csvRow?.sigma_calibration) ??
          derived?.sigmaCalibration ??
          null,
        directionalAccuracy:
          resolveFiniteNumber(csvRow?.directional_accuracy) ??
          derived?.directionalAccuracy ??
          null,
        interval1SigmaCoverage:
          resolveFiniteNumber(csvRow?.interval_1sigma_coverage) ??
          derived?.interval1SigmaCoverage ??
          null,
        interval2SigmaCoverage:
          resolveFiniteNumber(csvRow?.interval_2sigma_coverage) ??
          derived?.interval2SigmaCoverage ??
          null,
        pitMean:
          resolveFiniteNumber(csvRow?.pit_mean) ??
          derived?.pitMean ??
          null,
        pitVariance:
          resolveFiniteNumber(csvRow?.pit_variance) ??
          derived?.pitVariance ??
          null,
        normalizedAbsError:
          resolveFiniteNumber(csvRow?.normalized_abs_error) ??
          derived?.normalizedAbsError ??
          null,
        gaussianNll:
          resolveFiniteNumber(csvRow?.gaussian_nll) ??
          derived?.gaussianNll ??
          null,
        probabilisticCalibrationScore:
          resolveFiniteNumber(csvRow?.probabilistic_calibration_score) ??
          derived?.probabilisticCalibrationScore ??
          null,
      }
    })
}

function buildBlockedMetrics({ validationSummary, primaryStateResetMode, cvarWeight }) {
  const blockedEvaluation = validationSummary?.blocked_walk_forward_evaluation
  const stateResetModes = blockedEvaluation?.state_reset_modes
  if (!stateResetModes || typeof stateResetModes !== 'object') {
    return null
  }

  const resolvedMode =
    primaryStateResetMode ??
    validationSummary?.primary_state_reset_mode ??
    blockedEvaluation?.best_state_reset_mode_by_mean_log_wealth ??
    null
  const modePayload = resolvedMode ? stateResetModes[resolvedMode] : null
  if (!modePayload || typeof modePayload !== 'object') {
    return null
  }

  const averageLogWealthMean = resolveFiniteNumber(modePayload.average_log_wealth_mean)
  const cvarTailLossMean =
    resolveFiniteNumber(modePayload.cvar_tail_loss_mean) ??
    averageFinite(modePayload.folds, 'cvar_tail_loss')
  return {
    stateResetMode: resolvedMode,
    averageLogWealthMean,
    turnoverMean: resolveFiniteNumber(modePayload.turnover_mean),
    directionalAccuracyMean: resolveFiniteNumber(modePayload.directional_accuracy_mean),
    exactSmoothPositionMaeMean: resolveFiniteNumber(modePayload.exact_smooth_position_mae_mean),
    cvarTailLossMean,
    objectiveLogWealthMinusLambdaCvarMean:
      resolveFiniteNumber(modePayload.objective_log_wealth_minus_lambda_cvar_mean) ??
      resolveFiniteNumber(modePayload.blocked_objective_log_wealth_minus_lambda_cvar_mean) ??
      (
        averageLogWealthMean !== null && cvarTailLossMean !== null
          ? averageLogWealthMean - (Number(cvarWeight) * cvarTailLossMean)
          : null
      ),
    interval1SigmaCoverageMean: resolveFiniteNumber(modePayload.interval_1sigma_coverage_mean),
    interval2SigmaCoverageMean: resolveFiniteNumber(modePayload.interval_2sigma_coverage_mean),
    probabilisticCalibrationScoreMean: resolveFiniteNumber(modePayload.probabilistic_calibration_score_mean),
  }
}

function buildStructureMetrics({ validationSummary, config }) {
  const policyCalibration = validationSummary?.policy_calibration_summary ?? {}
  const selectedRow = policyCalibration.selected_row ?? null
  const appliedRuntimePolicy =
    policyCalibration.applied_runtime_policy ??
    buildRuntimePolicyPayload(config)
  const shapeTopClassShare = validationSummary?.state_vector_summary?.shape_posterior_top_class_share ?? {}
  const shapePosteriorMean = validationSummary?.state_vector_summary?.shape_posterior_mean ?? {}
  const dominantShapeEntry = Object.entries(shapeTopClassShare)
    .map(([shapeId, share]) => [shapeId, resolveFiniteNumber(share)]).filter(([, share]) => share !== null)
    .sort((left, right) => right[1] - left[1])[0] ?? [null, null]
  const policyHorizonDistribution = Object.entries(
    validationSummary?.validation?.policy_horizon_distribution ?? {},
  )
    .map(([horizon, share]) => ({
      horizon: Number(horizon),
      share: resolveFiniteNumber(share),
    }))
    .filter((entry) => Number.isFinite(entry.horizon) && entry.share !== null)
    .sort((left, right) => left.horizon - right.horizon)
  const dominantHorizon = [...policyHorizonDistribution]
    .sort((left, right) => (right.share ?? 0) - (left.share ?? 0))[0] ?? null
  const runtimePolicyAlignmentScore = computeRuntimePolicyAlignmentScore({
    selectedRow,
    appliedRuntimePolicy,
  })

  return {
    selectedRowMatchesRuntime:
      typeof policyCalibration.selected_row_matches_applied_runtime === 'boolean'
        ? policyCalibration.selected_row_matches_applied_runtime
        : runtimePolicyAlignmentScore >= 0.999,
    runtimePolicyAlignmentScore,
    selectedRowRole: toNonEmptyString(policyCalibration.selected_row_role),
    dominantShapeClass: dominantShapeEntry[0],
    dominantShapeClassShare: dominantShapeEntry[1],
    activeShapeCount: Object.values(shapePosteriorMean).filter((value) => (resolveFiniteNumber(value) ?? 0) >= 0.05).length,
    policyHorizonDistribution: policyHorizonDistribution.map((entry) => ({
      horizon: entry.horizon,
      share: entry.share,
    })),
    dominantHorizon: dominantHorizon?.horizon ?? null,
    dominantHorizonShare: dominantHorizon?.share ?? null,
  }
}

function buildLiveMetrics({ horizonRows, selectedHorizon }) {
  const available = horizonRows.filter((row) => row.actualClose !== null && row.actualReturnPct !== null)
  if (available.length === 0) {
    return {
      sampleCount: 0,
      directionalAccuracy: null,
      interval1SigmaCoverage: null,
      interval2SigmaCoverage: null,
      meanAbsoluteErrorPct: null,
      probabilisticCalibrationScore: null,
      selectedActualReturnPct: null,
      selectedRangeCaptured: null,
    }
  }

  let directionCorrectCount = 0
  let interval1SigmaHits = 0
  let interval2SigmaHits = 0
  let absErrorPctSum = 0
  let pitSum = 0
  let pitSqSum = 0

  for (const row of available) {
    const actualClose = row.actualClose
    const absError = Math.abs(actualClose - row.predictedClose)
    const sigmaSafe = Math.max(row.uncertainty, 1e-6)
    const zScore = Math.log(actualClose / row.predictedClose) / sigmaSafe
    const pit = gaussianPit(zScore)

    directionCorrectCount += Math.sign(row.expectedReturnPct) === Math.sign(row.actualReturnPct ?? 0) ? 1 : 0
    interval1SigmaHits += actualClose >= row.lowerClose && actualClose <= row.upperClose ? 1 : 0
    interval2SigmaHits += actualClose >= row.predictedClose * Math.exp(-2 * sigmaSafe) && actualClose <= row.predictedClose * Math.exp(2 * sigmaSafe) ? 1 : 0
    absErrorPctSum += actualClose === 0 ? 0 : absError / actualClose
    pitSum += pit
    pitSqSum += pit * pit
  }

  const sampleCount = available.length
  const directionalAccuracy = directionCorrectCount / sampleCount
  const interval1SigmaCoverage = interval1SigmaHits / sampleCount
  const interval2SigmaCoverage = interval2SigmaHits / sampleCount
  const pitMean = pitSum / sampleCount
  const pitVariance = (pitSqSum / sampleCount) - (pitMean * pitMean)
  const selectedRow = available.find((row) => row.horizon === selectedHorizon) ?? null

  return {
    sampleCount,
    directionalAccuracy,
    interval1SigmaCoverage,
    interval2SigmaCoverage,
    meanAbsoluteErrorPct: absErrorPctSum / sampleCount,
    probabilisticCalibrationScore: probabilisticCalibrationScore({
      interval1SigmaCoverage,
      interval2SigmaCoverage,
      pitMean,
      pitVariance,
    }),
    selectedActualReturnPct: selectedRow?.actualReturnPct ?? null,
    selectedRangeCaptured:
      selectedRow !== null
        ? selectedRow.actualClose >= selectedRow.lowerClose && selectedRow.actualClose <= selectedRow.upperClose
        : null,
  }
}

function averageFinite(rows, key) {
  if (!Array.isArray(rows)) {
    return null
  }
  const values = rows
    .map((row) => resolveFiniteNumber(row?.[key]))
    .filter((value) => value !== null)
  if (values.length === 0) {
    return null
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function gaussianPit(zScore) {
  const value = 0.5 * (1 + erf(zScore / Math.sqrt(2)))
  return clamp(value, 1e-6, 1 - 1e-6)
}

function probabilisticCalibrationScore({
  interval1SigmaCoverage,
  interval2SigmaCoverage,
  pitMean,
  pitVariance,
}) {
  const coverage1Target = 0.6826894921370859
  const coverage2Target = 0.9544997361036416
  const uniformPitVariance = 1 / 12
  const coverage1Score = clamp(1 - (Math.abs(interval1SigmaCoverage - coverage1Target) / 0.35), 0, 1)
  const coverage2Score = clamp(1 - (Math.abs(interval2SigmaCoverage - coverage2Target) / 0.35), 0, 1)
  const pitMeanScore = clamp(1 - (Math.abs(pitMean - 0.5) / 0.5), 0, 1)
  const pitVarianceScore = clamp(1 - (Math.abs(pitVariance - uniformPitVariance) / uniformPitVariance), 0, 1)
  return (
    (0.35 * coverage1Score) +
    (0.35 * coverage2Score) +
    (0.15 * pitMeanScore) +
    (0.15 * pitVarianceScore)
  )
}

function buildRuntimePolicyPayload(config) {
  return {
    state_reset_mode: config.evaluation_state_reset_mode,
    cost_multiplier: config.policy_cost_multiplier,
    gamma_multiplier: config.policy_gamma_multiplier,
    min_policy_sigma: config.min_policy_sigma,
    q_max: config.q_max,
    cvar_weight: config.cvar_weight,
  }
}

function computeRuntimePolicyAlignmentScore({ selectedRow, appliedRuntimePolicy }) {
  if (!selectedRow || !appliedRuntimePolicy) {
    return 1
  }

  const stateResetScore = Number(selectedRow.state_reset_mode === appliedRuntimePolicy.state_reset_mode)
  return clamp(
    (0.15 * stateResetScore) +
      (0.35 * ratioAlignmentScore(selectedRow.cost_multiplier, appliedRuntimePolicy.cost_multiplier, 12)) +
      (0.15 * ratioAlignmentScore(selectedRow.gamma_multiplier, appliedRuntimePolicy.gamma_multiplier, 12)) +
      (0.15 * ratioAlignmentScore(selectedRow.min_policy_sigma, appliedRuntimePolicy.min_policy_sigma, 8)) +
      (0.10 * ratioAlignmentScore(selectedRow.q_max, appliedRuntimePolicy.q_max, 3)) +
      (0.10 * ratioAlignmentScore(selectedRow.cvar_weight, appliedRuntimePolicy.cvar_weight, 16)),
    0,
    1,
  )
}

function ratioAlignmentScore(selectedValue, appliedValue, maxRatio) {
  const selectedNumeric = resolveFiniteNumber(selectedValue)
  const appliedNumeric = resolveFiniteNumber(appliedValue)
  if (
    selectedNumeric === null ||
    appliedNumeric === null ||
    selectedNumeric <= 0 ||
    appliedNumeric <= 0 ||
    maxRatio <= 1
  ) {
    return 0
  }
  const logRatio = Math.abs(Math.log(selectedNumeric / appliedNumeric))
  return clamp(1 - (logRatio / Math.log(maxRatio)), 0, 1)
}

function erf(value) {
  const sign = value < 0 ? -1 : 1
  const x = Math.abs(value)
  const a1 = 0.254829592
  const a2 = -0.284496736
  const a3 = 1.421413741
  const a4 = -1.453152027
  const a5 = 1.061405429
  const p = 0.3275911
  const t = 1 / (1 + p * x)
  const y = 1 - (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)
  return sign * y
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function parseTimestamp(value) {
  if (/^\d+$/.test(value)) {
    return Number(value)
  }

  const normalized = value.includes('T') ? value : value.replace(' ', 'T')
  const zoned = /(?:Z|[+-]\d{2}:\d{2})$/.test(normalized) ? normalized : `${normalized}Z`
  return new Date(zoned).getTime()
}

function normalizeDate(value) {
  const normalized = /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? value : `${value}Z`
  return new Date(normalized)
}

function toIso(value) {
  return new Date(value).toISOString()
}

function resolveTargetDateJst(value) {
  if (value) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
      throw new Error(`SIGNAL_CASCADE_TARGET_DATE must be YYYY-MM-DD, received: ${value}`)
    }
    return value
  }

  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Asia/Tokyo',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  })
  const parts = formatter.formatToParts(new Date())
  const year = parts.find((part) => part.type === 'year')?.value
  const month = parts.find((part) => part.type === 'month')?.value
  const day = parts.find((part) => part.type === 'day')?.value

  if (!year || !month || !day) {
    throw new Error('Could not resolve the current JST date.')
  }

  return `${year}-${month}-${day}`
}

function resolveLookbackDays() {
  const rawValue = process.env.SIGNAL_CASCADE_LOOKBACK_DAYS
  if (!rawValue) {
    return defaultLookbackDays
  }

  return resolvePositiveIntegerEnv('SIGNAL_CASCADE_LOOKBACK_DAYS', defaultLookbackDays)
}

function resolvePositiveIntegerEnv(name, defaultValue) {
  const rawValue = process.env[name]
  if (!rawValue) {
    return defaultValue
  }

  const parsed = Number.parseInt(rawValue, 10)
  if (!Number.isFinite(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer, received: ${rawValue}`)
  }

  return parsed
}

function resolveNonNegativeIntegerEnv(name, defaultValue) {
  const rawValue = process.env[name]
  if (!rawValue) {
    return defaultValue
  }

  const parsed = Number.parseInt(rawValue, 10)
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative integer, received: ${rawValue}`)
  }

  return parsed
}

function shiftDate(dateYmd, days) {
  const date = new Date(`${dateYmd}T00:00:00Z`)
  date.setUTCDate(date.getUTCDate() + days)
  return date.toISOString().slice(0, 10)
}

function resolveRequiredSourceRows(metrics, availableRows) {
  const preferred = Number(metrics.source_rows_used)
  if (Number.isFinite(preferred) && preferred > 0) {
    return Math.min(availableRows, Math.floor(preferred))
  }
  return availableRows
}

function endOfJstDayUtcMs(dateYmd) {
  const [year, month, day] = dateYmd.split('-').map(Number)
  return Date.UTC(year, month - 1, day, 14, 59, 59, 999)
}

function buildPythonPath(srcPath) {
  return process.env.PYTHONPATH ? `${srcPath}${path.delimiter}${process.env.PYTHONPATH}` : srcPath
}

function resampleTo4h(rows) {
  const buckets = []
  let bucket = []
  let currentKey = null

  for (const row of rows) {
    const end = closeBucketEnd(row.ts, 240)
    if (currentKey === null || currentKey !== end) {
      if (bucket.length) {
        buckets.push(mergeBucket(bucket, currentKey))
      }
      bucket = [row]
      currentKey = end
    } else {
      bucket.push(row)
    }
  }

  if (bucket.length && currentKey !== null) {
    buckets.push(mergeBucket(bucket, currentKey))
  }

  return buckets
}

function closeBucketEnd(ts, minutes) {
  return bucketStart(ts - 1, minutes) + minutes * 60_000
}

function bucketStart(ts, minutes) {
  const date = new Date(ts)
  const hour = date.getUTCHours()
  const minute = date.getUTCMinutes()
  const totalMinutes = hour * 60 + minute
  const floored = totalMinutes - (totalMinutes % minutes)
  const start = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate(), 0, 0, 0, 0))
  return start.getTime() + floored * 60_000
}

function mergeBucket(rows, bucketEnd) {
  return {
    ts: bucketEnd,
    firstTs: rows[0].ts,
    lastTs: rows[rows.length - 1].ts,
    sampleCount: rows.length,
    open: rows[0].open,
    high: Math.max(...rows.map((row) => row.high)),
    low: Math.min(...rows.map((row) => row.low)),
    close: rows[rows.length - 1].close,
  }
}

function buildHorizonRows({
  horizons,
  bars4h,
  anchorIndex,
  anchorClose,
  meanByHorizon,
  sigmaByHorizon,
  sigmaSqByHorizon,
  predictedCloseByHorizon,
  priceScale,
}) {
  return horizons.map((horizon) => {
    const muT = meanByHorizon[String(horizon)] ?? null
    const sigmaTSq =
      sigmaSqByHorizon[String(horizon)] ??
      (Number.isFinite(sigmaByHorizon[String(horizon)])
        ? sigmaByHorizon[String(horizon)] * sigmaByHorizon[String(horizon)]
        : null)
    const uncertainty = sigmaByHorizon[String(horizon)] ?? (sigmaTSq !== null ? Math.sqrt(sigmaTSq) : 0)
    const predictedClose =
      predictedCloseByHorizon[String(horizon)] ??
      (muT !== null ? anchorClose * Math.exp(muT) : anchorClose)
    const lowerClose = predictedClose * Math.exp(-uncertainty)
    const upperClose = predictedClose * Math.exp(uncertainty)
    const actualBar = bars4h[anchorIndex + horizon] ?? null
    const actualClose = actualBar ? actualBar.close / priceScale : null

    return {
      horizon,
      hours: horizon * 4,
      muT,
      sigmaT: uncertainty,
      sigmaTSq,
      predictedClose,
      lowerClose,
      upperClose,
      uncertainty,
      expectedReturnPct: ((predictedClose / anchorClose) - 1) * 100,
      actualClose,
      actualReturnPct: actualClose ? ((actualClose / anchorClose) - 1) * 100 : null,
    }
  })
}

function buildChartRows({ rawRows, anchorTime, anchorClose, anchorBar, horizonRows, bars4h, anchorIndex, priceScale }) {
  const maxHorizon = horizonRows.reduce((currentMax, row) => Math.max(currentMax, row.horizon), 1)
  const historyWindowBars = Math.max(64, Math.min(160, maxHorizon * 16))
  const historyRows = rawRows
    .filter((row) => row.ts <= anchorTime.getTime())
    .slice(-historyWindowBars)
    .map((row) => ({
      ts: row.ts,
      label: toIso(row.ts),
      phase: 'history',
      close: row.close / priceScale,
      high: row.high / priceScale,
      low: row.low / priceScale,
      forecastBase: null,
      forecastLower: null,
      forecastInnerLower: null,
      forecastInnerSpread: null,
      forecastUpper: null,
      forecastSpread: null,
      forecastOuterLowerBase: null,
      forecastOuterLowerSpread: null,
      forecastOuterUpperBase: null,
      forecastOuterUpperSpread: null,
      actualClose: null,
      actualRangeStart: null,
      actualRangeEnd: null,
    }))

  const historyEndTs = historyRows.at(-1)?.ts ?? anchorBar.lastTs ?? anchorBar.ts
  const interpolatedForecast = interpolateForecast({
    anchorClose,
    anchorBar,
    horizonRows,
    bars4h,
    anchorIndex,
    historyEndTs,
    priceScale,
  })
  const forecastRows = interpolatedForecast.map((row) => ({
    ts: row.ts,
    label: toIso(row.ts),
    phase: 'forecast',
    close: null,
    high: null,
    low: null,
    forecastBase: row.base,
    forecastLower: row.lower,
    forecastInnerLower: row.innerLower,
    forecastInnerSpread: row.innerUpper - row.innerLower,
    forecastUpper: row.upper,
    forecastSpread: row.upper - row.lower,
    forecastOuterLowerBase: row.lower,
    forecastOuterLowerSpread: row.innerLower - row.lower,
    forecastOuterUpperBase: row.innerUpper,
    forecastOuterUpperSpread: row.upper - row.innerUpper,
    actualClose: row.actual,
    actualRangeStart: row.actualRangeStart,
    actualRangeEnd: row.actualRangeEnd,
  }))

  const rows = [...historyRows, ...forecastRows]
  const microRows = buildMicroRows({
    rawRows,
    startTs: historyRows[0]?.ts ?? anchorBar.ts,
    endTs: forecastRows.at(-1)?.ts ?? anchorBar.ts,
    priceScale,
  })

  return {
    rows,
    microRows,
    yDomain: calculateVisiblePriceDomain(rows),
  }
}

function buildMicroRows({ rawRows, startTs, endTs, priceScale }) {
  return rawRows
    .filter((row) => row.ts >= startTs && row.ts <= endTs)
    .map((row) => ({
      ts: row.ts,
      open: row.open / priceScale,
      high: row.high / priceScale,
      low: row.low / priceScale,
      close: row.close / priceScale,
    }))
}

function calculateVisiblePriceDomain(rows) {
  const values = rows
    .flatMap((row) => [row.close, row.forecastBase, row.actualClose])
    .filter((value) => value !== null)

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min
  const pad = range === 0 ? Math.max(min * 0.003, 1) : Math.max(range * 0.08, max * 0.003, 1)

  return [Math.floor(min - pad), Math.ceil(max + pad)]
}

function interpolateForecast({ anchorClose, horizonRows, bars4h, anchorIndex, historyEndTs, priceScale }) {
  const horizonActuals = new Map(horizonRows.map((row) => [row.horizon, row.actualClose]))
  const points = [
    {
      step: 0,
      ts: historyEndTs,
      logBase: 0,
      sigma: 0,
      actual: anchorClose,
    },
    ...horizonRows.map((row) => ({
      step: row.horizon,
      ts:
        bars4h[anchorIndex + row.horizon]?.lastTs ??
        historyEndTs + row.horizon * 4 * 60 * 60 * 1000,
      logBase: Math.log(row.predictedClose / anchorClose),
      sigma: row.uncertainty,
      actual: row.actualClose,
    })),
  ]

  const rows = []

  for (let step = 0; step <= 30; step += 1) {
    const next = points.find((point) => point.step >= step)
    const prev = [...points].reverse().find((point) => point.step <= step)
    if (!next || !prev) {
      continue
    }
    const span = Math.max(next.step - prev.step, 1)
    const ratio = (step - prev.step) / span
    const logBase = prev.logBase + (next.logBase - prev.logBase) * ratio
    const sigma = prev.sigma + (next.sigma - prev.sigma) * ratio
    const halfSigma = sigma * 0.5
    const base = anchorClose * Math.exp(logBase)
    const lower = base * Math.exp(-sigma)
    const innerLower = base * Math.exp(-halfSigma)
    const innerUpper = base * Math.exp(halfSigma)
    const upper = base * Math.exp(sigma)
    const actualBar = bars4h[anchorIndex + step] ?? null
    const actualPoint = horizonActuals.get(step) ?? (actualBar ? actualBar.close / priceScale : null)

    rows.push({
      ts:
        bars4h[anchorIndex + step]?.lastTs ??
        historyEndTs + step * 4 * 60 * 60 * 1000,
      base,
      lower,
      innerLower,
      innerUpper,
      upper,
      actual: actualPoint,
      actualRangeStart: actualBar?.firstTs ?? null,
      actualRangeEnd: actualBar?.lastTs ?? null,
    })
  }

  return rows
}

function buildNarrative({ prediction, horizonRows }) {
  const selected = horizonRows.find((row) => row.horizon === (prediction.policy_horizon ?? prediction.selected_horizon)) ?? horizonRows[0]
  const selectedPolicyUtility = prediction.selected_policy_utility ?? prediction.policy_score ?? 0
  const directionalWord = prediction.no_trade_band_hit === true
    ? 'ノートレードです。'
    : prediction.position < -0.25
      ? '下向きです。'
      : prediction.position > 0.25
        ? '上向きです。'
        : '横ばいです。'
  const overlayWord = prediction.no_trade_band_hit === true
    ? '短期判断は維持です。'
    : '短期判断はポジション調整です。'
  const ordered = [...horizonRows].sort((left, right) => left.horizon - right.horizon)
  const segments = []

  for (let index = 0; index < ordered.length - 1; index += 1) {
    const current = ordered[index]
    const next = ordered[index + 1]
    const delta = next.expectedReturnPct - current.expectedReturnPct
    segments.push({
      title: `${current.hours}時間→${next.hours}時間`,
      summary: delta > 0.35 ? 'やや上向きます。' : delta < -0.35 ? 'やや下向きます。' : 'ほぼ横ばいです。',
      fromStep: current.horizon,
      toStep: next.horizon,
    })
  }

  return {
    title: directionalWord,
    summary: `${overlayWord} ${selected.hours}時間先の予測値は ${Math.round(selected.predictedClose).toLocaleString('ja-JP')} です。`,
    bullets: [
      `変化率は ${selected.expectedReturnPct.toFixed(2)}%、不確実性は ${selected.uncertainty.toFixed(4)} です。`,
      `予測幅は ${Math.round(selected.lowerClose).toLocaleString('ja-JP')} - ${Math.round(selected.upperClose).toLocaleString('ja-JP')} です。`,
      prediction.no_trade_band_hit === true
        ? `no-trade band に入りました。直前ポジションは ${Number(prediction.q_t_prev ?? prediction.previous_position ?? 0).toFixed(2)} です。`
        : `selected policy utility は ${Number(selectedPolicyUtility).toFixed(3)}、強さは ${Math.abs(prediction.position).toFixed(2)} です。`,
    ],
    segments,
  }
}

function translateOverlay(value) {
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
