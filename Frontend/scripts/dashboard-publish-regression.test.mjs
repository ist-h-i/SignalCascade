import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const scriptsDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(scriptsDir, '..')
const repoRoot = path.resolve(frontendRoot, '..')
const pyTorchRoot = path.join(repoRoot, 'PyTorch')
const pyTorchPython = path.join(pyTorchRoot, '.venv', 'bin', 'python')
const sourceArtifactRoot = path.join(pyTorchRoot, 'artifacts', 'gold_xauusd_m30')
const sourceCurrentRunDir = path.join(sourceArtifactRoot, 'current')
const sourceLiveCsvPath = path.join(sourceArtifactRoot, 'live', 'xauusd_m30_latest.csv')

test(
  'predict -> sync:data:fast publishes dashboard data from the predicted current artifact',
  { timeout: 300_000 },
  (t) => {
    assert.ok(fs.existsSync(pyTorchPython), `Python runtime was not found: ${pyTorchPython}`)
    assert.ok(fs.existsSync(sourceCurrentRunDir), `Current artifact directory is missing: ${sourceCurrentRunDir}`)
    assert.ok(fs.existsSync(sourceLiveCsvPath), `Live CSV is missing: ${sourceLiveCsvPath}`)

    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'signal-cascade-publish-'))
    t.after(() => {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    })

    const tempFrontendRoot = path.join(tempRoot, 'Frontend')
    const tempScriptsDir = path.join(tempFrontendRoot, 'scripts')
    const tempPublicDir = path.join(tempFrontendRoot, 'public')
    const tempArtifactRoot = path.join(tempRoot, 'PyTorch', 'artifacts', 'gold_xauusd_m30')
    const tempCurrentRunDir = path.join(tempArtifactRoot, 'current')
    const tempLiveDir = path.join(tempArtifactRoot, 'live')
    const tempDashboardPath = path.join(tempPublicDir, 'dashboard-data.json')
    const tempLiveCsvPath = path.join(tempLiveDir, 'xauusd_m30_latest.csv')

    fs.mkdirSync(tempScriptsDir, { recursive: true })
    fs.mkdirSync(tempPublicDir, { recursive: true })
    fs.mkdirSync(tempLiveDir, { recursive: true })
    fs.cpSync(path.join(scriptsDir, 'sync-signal-cascade-data.mjs'), path.join(tempScriptsDir, 'sync-signal-cascade-data.mjs'))
    fs.cpSync(
      path.join(scriptsDir, 'check-dashboard-data-contract.mjs'),
      path.join(tempScriptsDir, 'check-dashboard-data-contract.mjs'),
    )
    fs.cpSync(sourceCurrentRunDir, tempCurrentRunDir, { recursive: true })
    fs.copyFileSync(sourceLiveCsvPath, tempLiveCsvPath)

    const predict = spawnSync(
      pyTorchPython,
      [
        '-m',
        'signal_cascade_pytorch.interfaces.cli',
        'predict',
        '--output-dir',
        tempCurrentRunDir,
        '--csv',
        tempLiveCsvPath,
      ],
      {
        cwd: pyTorchRoot,
        encoding: 'utf8',
        env: {
          ...process.env,
          PYTHONPATH: buildPythonPath(path.join(pyTorchRoot, 'src')),
        },
      },
    )
    assert.equal(predict.status, 0, formatFailure('predict', predict))

    const sync = spawnSync(process.execPath, ['./scripts/sync-signal-cascade-data.mjs'], {
      cwd: tempFrontendRoot,
      encoding: 'utf8',
      env: {
        ...process.env,
        SIGNAL_CASCADE_DISABLE_TRAINING: '1',
        SIGNAL_CASCADE_CSV_PATH: tempLiveCsvPath,
      },
    })
    assert.equal(sync.status, 0, formatFailure('sync:data:fast', sync))

    const contract = spawnSync(process.execPath, ['./scripts/check-dashboard-data-contract.mjs'], {
      cwd: tempFrontendRoot,
      encoding: 'utf8',
      env: { ...process.env },
    })
    assert.equal(contract.status, 0, formatFailure('check:data:contract', contract))

    const dashboardData = readJson(tempDashboardPath)
    const manifest = readJson(path.join(tempCurrentRunDir, 'manifest.json'))
    const sourceMeta = readJson(path.join(tempCurrentRunDir, 'source.json'))
    const validationSummary = readJson(path.join(tempCurrentRunDir, 'validation_summary.json'))
    const forecastSummary = readJson(path.join(tempCurrentRunDir, 'forecast_summary.json'))
    const prediction = readJson(path.join(tempCurrentRunDir, 'prediction.json'))

    assert.equal(dashboardData.schemaVersion, 6)
    assert.equal(dashboardData.provenance.manifestGeneratedAt, manifest.generated_at_utc)
    assert.equal(dashboardData.provenance.manifestGeneratedAt, sourceMeta.generated_at_utc)
    assert.equal(dashboardData.provenance.diagnosticsGeneratedAt, validationSummary.generated_at_utc)
    assert.equal(dashboardData.provenance.forecastGeneratedAt, forecastSummary.generated_at_utc)
    assert.equal(dashboardData.provenance.predictionAnchorTime, prediction.anchor_time)
    assert.equal(dashboardData.run.selectedHorizon, prediction.policy_horizon)
    assert.equal(dashboardData.run.position, prediction.position)
  },
)

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'))
}

function buildPythonPath(srcPath) {
  return process.env.PYTHONPATH ? `${srcPath}${path.delimiter}${process.env.PYTHONPATH}` : srcPath
}

function formatFailure(step, result) {
  return [
    `${step} failed with code ${result.status} signal=${result.signal ?? 'null'}`,
    `stdout:\n${result.stdout ?? '<empty>'}`,
    `stderr:\n${result.stderr ?? '<empty>'}`,
    result.error ? `error:\n${result.error.stack ?? String(result.error)}` : null,
  ]
    .filter(Boolean)
    .join('\n')
}
