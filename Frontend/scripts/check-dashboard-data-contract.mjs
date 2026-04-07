import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(scriptDir, '..')
const artifactRoot = path.resolve(frontendRoot, '..', 'PyTorch', 'artifacts', 'gold_xauusd_m30')
const currentRunDir = path.join(artifactRoot, 'current')
const dashboardDataPath = path.join(frontendRoot, 'public', 'dashboard-data.json')
const sourceMetaPath = path.join(currentRunDir, 'source.json')
const manifestPath = path.join(currentRunDir, 'manifest.json')
const validationSummaryPath = path.join(currentRunDir, 'validation_summary.json')

const dashboardData = readJson(dashboardDataPath)
const sourceMeta = readJson(sourceMetaPath)
const manifest = readJson(manifestPath)
const validationSummary = readJson(validationSummaryPath)

if (dashboardData.schemaVersion !== 6) {
  throw new Error(`dashboard-data schemaVersion must be 6, received ${dashboardData.schemaVersion ?? 'null'}`)
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

function numbersMatch(left, right) {
  return Math.abs(left - right) <= 1e-9
}
