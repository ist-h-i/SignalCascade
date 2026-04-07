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
    assert.equal(
      dashboardData.governance?.productionCurrentCandidate,
      sourceMeta.current_selection_governance?.production_current?.candidate ?? null,
    )
    assert.equal(
      dashboardData.governance?.acceptedCandidate,
      sourceMeta.current_selection_governance?.accepted_candidate?.candidate ?? null,
    )
    assert.equal(dashboardData.run.selectedHorizon, prediction.policy_horizon)
    assert.equal(dashboardData.run.position, prediction.position)
  },
)

test(
  'sync:data passes the 360-day lookback window to tune-latest',
  { timeout: 300_000 },
  (t) => {
    assert.ok(fs.existsSync(sourceCurrentRunDir), `Current artifact directory is missing: ${sourceCurrentRunDir}`)
    assert.ok(fs.existsSync(sourceLiveCsvPath), `Live CSV is missing: ${sourceLiveCsvPath}`)

    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'signal-cascade-lookback-'))
    t.after(() => {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    })

    const tempFrontendRoot = path.join(tempRoot, 'Frontend')
    const tempScriptsDir = path.join(tempFrontendRoot, 'scripts')
    const tempPublicDir = path.join(tempFrontendRoot, 'public')
    const tempPyTorchRoot = path.join(tempRoot, 'PyTorch')
    const tempArtifactRoot = path.join(tempPyTorchRoot, 'artifacts', 'gold_xauusd_m30')
    const tempLiveDir = path.join(tempArtifactRoot, 'live')
    const tempDashboardPath = path.join(tempPublicDir, 'dashboard-data.json')
    const tempLiveCsvPath = path.join(tempLiveDir, 'xauusd_m30_latest.csv')
    const fakePythonPath = path.join(tempPyTorchRoot, '.venv', 'bin', 'python')
    const trainLogPath = path.join(tempRoot, 'train-log.json')

    fs.mkdirSync(tempScriptsDir, { recursive: true })
    fs.mkdirSync(tempPublicDir, { recursive: true })
    fs.mkdirSync(tempLiveDir, { recursive: true })
    fs.mkdirSync(path.dirname(fakePythonPath), { recursive: true })
    fs.mkdirSync(tempArtifactRoot, { recursive: true })

    fs.cpSync(path.join(scriptsDir, 'sync-signal-cascade-data.mjs'), path.join(tempScriptsDir, 'sync-signal-cascade-data.mjs'))
    fs.copyFileSync(sourceLiveCsvPath, tempLiveCsvPath)

    fs.writeFileSync(
      fakePythonPath,
      `#!/usr/bin/env node
const fs = require('node:fs')
const path = require('node:path')

const args = process.argv.slice(2)
const artifactRootIndex = args.indexOf('--artifact-root')
if (artifactRootIndex < 0 || artifactRootIndex + 1 >= args.length) {
  throw new Error('missing --artifact-root')
}
const artifactRoot = args[artifactRootIndex + 1]
const currentDir = path.join(artifactRoot, 'current')
const sourceCurrentRunDir = ${JSON.stringify(sourceCurrentRunDir)}
const logPath = process.env.SIGNAL_CASCADE_FAKE_TRAIN_LOG

if (logPath) {
  fs.writeFileSync(logPath, JSON.stringify({ args }, null, 2))
}

fs.rmSync(currentDir, { recursive: true, force: true })
fs.cpSync(sourceCurrentRunDir, currentDir, { recursive: true })
process.stdout.write('fake tune complete\\n')
`,
      { encoding: 'utf8' },
    )
    fs.chmodSync(fakePythonPath, 0o755)

    const sync = spawnSync(process.execPath, ['./scripts/sync-signal-cascade-data.mjs'], {
      cwd: tempFrontendRoot,
      encoding: 'utf8',
      env: {
        ...process.env,
        SIGNAL_CASCADE_CSV_PATH: tempLiveCsvPath,
        SIGNAL_CASCADE_DISABLE_LIVE_SYNC: '1',
        SIGNAL_CASCADE_LOOKBACK_DAYS: '360',
        SIGNAL_CASCADE_FAKE_TRAIN_LOG: trainLogPath,
      },
    })
    assert.equal(sync.status, 0, formatFailure('sync:data:lookback', sync))

    const logged = readJson(trainLogPath)
    const lookbackIndex = logged.args.indexOf('--csv-lookback-days')
    const csvIndex = logged.args.indexOf('--csv')

    assert.notEqual(lookbackIndex, -1)
    assert.equal(logged.args[lookbackIndex + 1], '360')
    assert.notEqual(csvIndex, -1)
    assert.equal(logged.args[csvIndex + 1], tempLiveCsvPath)
    assert.ok(fs.existsSync(tempDashboardPath), `Dashboard output is missing: ${tempDashboardPath}`)
  },
)

test(
  'sync:data retries the Dukascopy download after a transient fetch failure',
  { timeout: 300_000 },
  (t) => {
    assert.ok(fs.existsSync(sourceCurrentRunDir), `Current artifact directory is missing: ${sourceCurrentRunDir}`)
    assert.ok(fs.existsSync(sourceLiveCsvPath), `Live CSV is missing: ${sourceLiveCsvPath}`)

    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'signal-cascade-download-retry-'))
    t.after(() => {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    })

    const tempFrontendRoot = path.join(tempRoot, 'Frontend')
    const tempScriptsDir = path.join(tempFrontendRoot, 'scripts')
    const tempPublicDir = path.join(tempFrontendRoot, 'public')
    const tempPyTorchRoot = path.join(tempRoot, 'PyTorch')
    const tempArtifactRoot = path.join(tempPyTorchRoot, 'artifacts', 'gold_xauusd_m30')
    const tempCurrentRunDir = path.join(tempArtifactRoot, 'current')
    const tempBinDir = path.join(tempRoot, 'bin')
    const fakePythonPath = path.join(tempPyTorchRoot, '.venv', 'bin', 'python')
    const fakeNpxPath = path.join(tempBinDir, 'npx')
    const attemptFilePath = path.join(tempRoot, 'dukascopy-attempt.txt')
    const trainLogPath = path.join(tempRoot, 'train-log.json')
    const tempDashboardPath = path.join(tempPublicDir, 'dashboard-data.json')

    fs.mkdirSync(tempScriptsDir, { recursive: true })
    fs.mkdirSync(tempPublicDir, { recursive: true })
    fs.mkdirSync(tempArtifactRoot, { recursive: true })
    fs.mkdirSync(path.dirname(fakePythonPath), { recursive: true })
    fs.mkdirSync(tempBinDir, { recursive: true })

    fs.cpSync(path.join(scriptsDir, 'sync-signal-cascade-data.mjs'), path.join(tempScriptsDir, 'sync-signal-cascade-data.mjs'))

    fs.writeFileSync(
      fakePythonPath,
      `#!/usr/bin/env node
const fs = require('node:fs')
const path = require('node:path')

const args = process.argv.slice(2)
const artifactRootIndex = args.indexOf('--artifact-root')
if (artifactRootIndex < 0 || artifactRootIndex + 1 >= args.length) {
  throw new Error('missing --artifact-root')
}
const artifactRoot = args[artifactRootIndex + 1]
const currentDir = path.join(artifactRoot, 'current')
const sourceCurrentRunDir = ${JSON.stringify(sourceCurrentRunDir)}
const logPath = process.env.SIGNAL_CASCADE_FAKE_TRAIN_LOG

if (logPath) {
  fs.writeFileSync(logPath, JSON.stringify({ args }, null, 2))
}

fs.rmSync(currentDir, { recursive: true, force: true })
fs.cpSync(sourceCurrentRunDir, currentDir, { recursive: true })
process.stdout.write('fake tune complete\\n')
`,
      { encoding: 'utf8' },
    )
    fs.chmodSync(fakePythonPath, 0o755)

    fs.writeFileSync(
      fakeNpxPath,
      `#!/usr/bin/env node
const fs = require('node:fs')
const path = require('node:path')

const args = process.argv.slice(2)
const attemptFilePath = process.env.SIGNAL_CASCADE_FAKE_NPX_ATTEMPT_FILE
const sourceCsvPath = ${JSON.stringify(sourceLiveCsvPath)}
const dirIndex = args.indexOf('-dir')
const baseNameIndex = args.indexOf('-fn')
if (!attemptFilePath || dirIndex < 0 || baseNameIndex < 0) {
  throw new Error('missing fake dukascopy configuration')
}

const nextAttempt =
  Number.parseInt(fs.existsSync(attemptFilePath) ? fs.readFileSync(attemptFilePath, 'utf8') : '0', 10) + 1
fs.writeFileSync(attemptFilePath, String(nextAttempt))

if (nextAttempt === 1) {
  process.stdout.write('Something went wrong:\\n > fetch failed\\n')
  process.exit(1)
}

const rawDir = args[dirIndex + 1]
const rawBaseName = args[baseNameIndex + 1]
const targetPath = path.join(rawDir, \`\${rawBaseName}.csv\`)
fs.mkdirSync(rawDir, { recursive: true })
const normalizedLines = fs.readFileSync(sourceCsvPath, 'utf8').trim().split(/\\r?\\n/)
const dukascopyLines = [
  normalizedLines[0],
  ...normalizedLines.slice(1).map((line) => {
    const [timestamp, open, high, low, close, volume] = line.split(',')
    return [Date.parse(timestamp), open, high, low, close, volume].join(',')
  }),
]
fs.writeFileSync(targetPath, \`\${dukascopyLines.join('\\n')}\\n\`)
process.stdout.write('download ok\\n')
`,
      { encoding: 'utf8' },
    )
    fs.chmodSync(fakeNpxPath, 0o755)

    const sync = spawnSync(process.execPath, ['./scripts/sync-signal-cascade-data.mjs'], {
      cwd: tempFrontendRoot,
      encoding: 'utf8',
      env: {
        ...process.env,
        PATH: `${tempBinDir}${path.delimiter}${process.env.PATH ?? ''}`,
        SIGNAL_CASCADE_DOWNLOAD_ATTEMPTS: '2',
        SIGNAL_CASCADE_DOWNLOAD_RETRY_MS: '1',
        SIGNAL_CASCADE_FAKE_NPX_ATTEMPT_FILE: attemptFilePath,
        SIGNAL_CASCADE_FAKE_TRAIN_LOG: trainLogPath,
      },
    })
    assert.equal(sync.status, 0, formatFailure('sync:data:download-retry', sync))

    const attempts = fs.readFileSync(attemptFilePath, 'utf8')
    assert.equal(attempts, '2')
    assert.match(sync.stderr ?? '', /Dukascopy download attempt 1\/2 failed/)
    assert.ok(fs.existsSync(trainLogPath), `Train log is missing: ${trainLogPath}`)
    assert.ok(fs.existsSync(tempDashboardPath), `Dashboard output is missing: ${tempDashboardPath}`)
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
