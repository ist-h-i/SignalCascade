import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(scriptDir, '..')
const artifactRoot = path.resolve(frontendRoot, '..', 'PyTorch', 'artifacts', 'gold_xauusd_m30')
const currentRunDir = path.join(artifactRoot, 'current')
const dashboardDataPath = path.join(frontendRoot, 'public', 'dashboard-data.json')
const configPath = path.join(currentRunDir, 'config.json')
const analysisPath = path.join(currentRunDir, 'analysis.json')
const sourceMetaPath = path.join(currentRunDir, 'source.json')
const manifestPath = path.join(currentRunDir, 'manifest.json')
const validationSummaryPath = path.join(currentRunDir, 'validation_summary.json')

const dashboardData = readJson(dashboardDataPath)
const config = readJson(configPath)
const analysis = readJson(analysisPath)
const sourceMeta = readJson(sourceMetaPath)
const manifest = readJson(manifestPath)
const validationSummary = readJson(validationSummaryPath)

if (dashboardData.schemaVersion !== 6) {
  throw new Error(`dashboard-data schemaVersion must be 6, received ${dashboardData.schemaVersion ?? 'null'}`)
}

if (!dashboardData.metrics?.live || typeof dashboardData.metrics.live !== 'object') {
  throw new Error('dashboard-data metrics.live must be present')
}

if (!dashboardData.metrics?.structure || typeof dashboardData.metrics.structure !== 'object') {
  throw new Error('dashboard-data metrics.structure must be present')
}

if (!Array.isArray(dashboardData.metrics?.horizonDiagnostics) || dashboardData.metrics.horizonDiagnostics.length === 0) {
  throw new Error('dashboard-data metrics.horizonDiagnostics must be a non-empty array')
}

const effectivePriceScale = toPositiveNumber(dashboardData.run?.effectivePriceScale)
if (effectivePriceScale === null) {
  throw new Error('dashboard-data run.effectivePriceScale must be a positive number')
}

const diagnosticsGeneratedAt = toNonEmptyString(dashboardData.provenance?.diagnosticsGeneratedAt)
if (diagnosticsGeneratedAt === null) {
  throw new Error('dashboard-data provenance.diagnosticsGeneratedAt must be present')
}

const manifestGeneratedAt = toNonEmptyString(manifest.generated_at)
if (manifestGeneratedAt === null) {
  throw new Error(`current manifest.json is missing generated_at: ${manifestPath}`)
}

const manifestGeneratedAtUtc = toNonEmptyString(manifest.generated_at_utc)
if (manifestGeneratedAtUtc === null) {
  throw new Error(`current manifest.json is missing generated_at_utc: ${manifestPath}`)
}
if (manifestGeneratedAtUtc !== manifestGeneratedAt) {
  throw new Error(
    `current manifest.json timestamp mismatch: generated_at_utc=${manifestGeneratedAtUtc} generated_at=${manifestGeneratedAt}`,
  )
}

const sourceGeneratedAtUtc = toNonEmptyString(sourceMeta.generated_at_utc)
if (sourceGeneratedAtUtc === null) {
  throw new Error(`current source.json is missing generated_at_utc: ${sourceMetaPath}`)
}
if (sourceGeneratedAtUtc !== manifestGeneratedAtUtc) {
  throw new Error(
    `current artifact timestamp mismatch: manifest generated_at_utc=${manifestGeneratedAtUtc} source generated_at_utc=${sourceGeneratedAtUtc}`,
  )
}

const dashboardManifestGeneratedAt = toNonEmptyString(dashboardData.provenance?.manifestGeneratedAt)
if (dashboardManifestGeneratedAt === null) {
  throw new Error('dashboard-data provenance.manifestGeneratedAt must be present')
}
if (dashboardManifestGeneratedAt !== manifestGeneratedAtUtc) {
  throw new Error(
    `dashboard-data manifestGeneratedAt mismatch: dashboard=${dashboardManifestGeneratedAt} current=${manifestGeneratedAtUtc}`,
  )
}

const freshnessManifestGeneratedAt = toNonEmptyString(dashboardData.provenance?.freshness?.manifestGeneratedAt)
if (freshnessManifestGeneratedAt === null) {
  throw new Error('dashboard-data provenance.freshness.manifestGeneratedAt must be present')
}
if (freshnessManifestGeneratedAt !== manifestGeneratedAtUtc) {
  throw new Error(
    `dashboard-data freshness.manifestGeneratedAt mismatch: dashboard=${freshnessManifestGeneratedAt} current=${manifestGeneratedAtUtc}`,
  )
}

const expectedArtifactId = toNonEmptyString(sourceMeta.artifact_id)
if (expectedArtifactId === null) {
  throw new Error(`current source.json is missing artifact_id: ${sourceMetaPath}`)
}
if (dashboardData.provenance?.artifactId !== expectedArtifactId) {
  throw new Error(
    `dashboard-data artifactId mismatch: dashboard=${dashboardData.provenance?.artifactId ?? 'null'} current=${expectedArtifactId}`,
  )
}

const expectedSessionId = toNonEmptyString(manifest.session_id)
if (expectedSessionId !== null && dashboardData.run?.tuningSessionId !== expectedSessionId) {
  throw new Error(
    `dashboard-data tuningSessionId mismatch: dashboard=${dashboardData.run?.tuningSessionId ?? 'null'} current=${expectedSessionId}`,
  )
}

const expectedDiagnosticsGeneratedAt = toNonEmptyString(validationSummary.generated_at_utc)
if (expectedDiagnosticsGeneratedAt === null) {
  throw new Error(`current validation_summary.json is missing generated_at_utc: ${validationSummaryPath}`)
}
if (diagnosticsGeneratedAt !== expectedDiagnosticsGeneratedAt) {
  throw new Error(
    `dashboard-data diagnosticsGeneratedAt mismatch: dashboard=${diagnosticsGeneratedAt} current=${expectedDiagnosticsGeneratedAt}`,
  )
}

const selectionDiagnostics = validationSummary.selection_diagnostics
if (!selectionDiagnostics || typeof selectionDiagnostics !== 'object') {
  throw new Error(`current validation_summary.json is missing selection_diagnostics: ${validationSummaryPath}`)
}

const runtimeCurrent = validationSummary.runtime_current
if (!runtimeCurrent || typeof runtimeCurrent !== 'object') {
  throw new Error(`current validation_summary.json is missing runtime_current: ${validationSummaryPath}`)
}

const runtimeOperatingPoint = runtimeCurrent.operating_point
if (!runtimeOperatingPoint || typeof runtimeOperatingPoint !== 'object') {
  throw new Error(`current validation_summary.json is missing runtime_current.operating_point: ${validationSummaryPath}`)
}

const selectionValidation = selectionDiagnostics.validation
if (!selectionValidation || typeof selectionValidation !== 'object') {
  throw new Error(`current validation_summary.json is missing selection_diagnostics.validation: ${validationSummaryPath}`)
}

assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.run?.stateResetMode,
  runtimeValue:
    toNonEmptyString(runtimeOperatingPoint.state_reset_mode) ??
    toNonEmptyString(runtimeCurrent.state_reset_mode) ??
    toNonEmptyString(config.evaluation_state_reset_mode),
  fieldName: 'run.stateResetMode',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.run?.costMultiplier,
  runtimeValue: toFiniteNumber(runtimeOperatingPoint.cost_multiplier) ?? toFiniteNumber(config.policy_cost_multiplier),
  fieldName: 'run.costMultiplier',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.run?.gammaMultiplier,
  runtimeValue: toFiniteNumber(runtimeOperatingPoint.gamma_multiplier) ?? toFiniteNumber(config.policy_gamma_multiplier),
  fieldName: 'run.gammaMultiplier',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.run?.minPolicySigma,
  runtimeValue: toFiniteNumber(runtimeOperatingPoint.min_policy_sigma) ?? toFiniteNumber(config.min_policy_sigma),
  fieldName: 'run.minPolicySigma',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.metrics?.validation?.stateResetMode,
  runtimeValue: toNonEmptyString(selectionValidation.state_reset_mode),
  fieldName: 'metrics.validation.stateResetMode',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.metrics?.validation?.costMultiplier,
  runtimeValue: toFiniteNumber(selectionValidation.cost_multiplier),
  fieldName: 'metrics.validation.costMultiplier',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.metrics?.validation?.gammaMultiplier,
  runtimeValue: toFiniteNumber(selectionValidation.gamma_multiplier),
  fieldName: 'metrics.validation.gammaMultiplier',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.metrics?.validation?.minPolicySigma,
  runtimeValue: toFiniteNumber(selectionValidation.min_policy_sigma),
  fieldName: 'metrics.validation.minPolicySigma',
})

const rankingDiagnostics = analysis.forecast_quality_ranking_diagnostics
if (!rankingDiagnostics || typeof rankingDiagnostics !== 'object') {
  throw new Error(`current analysis.json is missing forecast_quality_ranking_diagnostics: ${analysisPath}`)
}

const divergenceScorecard = analysis.selection_divergence_scorecard
if (!divergenceScorecard || typeof divergenceScorecard !== 'object') {
  throw new Error(`current analysis.json is missing selection_divergence_scorecard: ${analysisPath}`)
}

if (!dashboardData.metrics?.selection || typeof dashboardData.metrics.selection !== 'object') {
  throw new Error('dashboard-data metrics.selection must be present')
}

assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.metrics?.selection?.currentTopCandidate,
  runtimeValue: toNonEmptyString(rankingDiagnostics.current_top_candidate),
  fieldName: 'metrics.selection.currentTopCandidate',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.metrics?.selection?.allHorizonVsCurrentSpearmanRankCorrelation,
  runtimeValue: toFiniteNumber(rankingDiagnostics.all_horizon_vs_current_spearman_rank_correlation),
  fieldName: 'metrics.selection.allHorizonVsCurrentSpearmanRankCorrelation',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.metrics?.selection?.divergenceScorecard?.sessionCount,
  runtimeValue: toFiniteNumber(divergenceScorecard.session_count),
  fieldName: 'metrics.selection.divergenceScorecard.sessionCount',
})
assertOptionalRunFieldMatchesRuntimeLane({
  dashboardValue: dashboardData.metrics?.selection?.divergenceScorecard?.fullCoverageSessionCount,
  runtimeValue: toFiniteNumber(divergenceScorecard.full_coverage_session_count),
  fieldName: 'metrics.selection.divergenceScorecard.fullCoverageSessionCount',
})

const expectedProductionCurrentCandidate = toNonEmptyString(sourceMeta.current_selection_governance?.production_current?.candidate)
if (
  expectedProductionCurrentCandidate !== null &&
  dashboardData.governance?.productionCurrentCandidate !== expectedProductionCurrentCandidate
) {
  throw new Error(
    `dashboard-data productionCurrentCandidate mismatch: dashboard=${dashboardData.governance?.productionCurrentCandidate ?? 'null'} current=${expectedProductionCurrentCandidate}`,
  )
}

const expectedAcceptedCandidate = toNonEmptyString(sourceMeta.current_selection_governance?.accepted_candidate?.candidate)
if (expectedAcceptedCandidate !== null && dashboardData.governance?.acceptedCandidate !== expectedAcceptedCandidate) {
  throw new Error(
    `dashboard-data acceptedCandidate mismatch: dashboard=${dashboardData.governance?.acceptedCandidate ?? 'null'} current=${expectedAcceptedCandidate}`,
  )
}
const expectedSelectionStatus = toNonEmptyString(sourceMeta.current_selection_governance?.selection_status)
if (
  expectedSelectionStatus !== null &&
  dashboardData.governance?.selectionStatus !== expectedSelectionStatus
) {
  throw new Error(
    `dashboard-data selectionStatus mismatch: dashboard=${dashboardData.governance?.selectionStatus ?? 'null'} current=${expectedSelectionStatus}`,
  )
}

const priceScaleAlias = toPositiveNumber(dashboardData.run?.priceScale)
if (priceScaleAlias !== null && !numbersMatch(priceScaleAlias, effectivePriceScale)) {
  throw new Error(
    `dashboard-data priceScale alias mismatch: priceScale=${priceScaleAlias} effectivePriceScale=${effectivePriceScale}`,
  )
}

console.log(`dashboard-data contract OK: ${dashboardDataPath}`)

function readJson(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`required file is missing: ${filePath}`)
  }

  return JSON.parse(fs.readFileSync(filePath, 'utf8'))
}

function toNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function toPositiveNumber(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) && numeric > 0 ? numeric : null
}

function toFiniteNumber(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function numbersMatch(left, right) {
  return Math.abs(left - right) <= 1e-9
}

function assertOptionalRunFieldMatchesRuntimeLane({ dashboardValue, runtimeValue, fieldName }) {
  if (typeof runtimeValue === 'string') {
    const dashboardString = toNonEmptyString(dashboardValue)
    if (dashboardString === null) {
      throw new Error(`dashboard-data ${fieldName} must be present`)
    }
    if (dashboardString !== runtimeValue) {
      throw new Error(`dashboard-data ${fieldName} mismatch: dashboard=${dashboardString} current=${runtimeValue}`)
    }
    return
  }

  if (runtimeValue === null) {
    return
  }

  const dashboardNumber = toFiniteNumber(dashboardValue)
  if (dashboardNumber === null) {
    throw new Error(`dashboard-data ${fieldName} must be a finite number`)
  }
  if (!numbersMatch(dashboardNumber, runtimeValue)) {
    throw new Error(`dashboard-data ${fieldName} mismatch: dashboard=${dashboardNumber} current=${runtimeValue}`)
  }
}
