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
const manifestPath = path.join(currentRunDir, 'manifest.json')
const outputPath = path.resolve(frontendRoot, 'public/dashboard-data.json')
const liveDataDir = path.join(artifactRoot, 'live')
const liveCsvPath = path.join(liveDataDir, 'xauusd_m30_latest.csv')
const pyTorchPython = path.resolve(pyTorchRoot, '.venv/bin/python')

const targetDateJst = resolveTargetDateJst(process.env.SIGNAL_CASCADE_TARGET_DATE)
const csvPath = resolveCsvPath(targetDateJst)
ensureCurrentRun(csvPath)
const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'))
const prediction = JSON.parse(fs.readFileSync(predictionPath, 'utf8'))
const manifest = fs.existsSync(manifestPath)
  ? JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  : null
const allCsvRows = parseCsv(fs.readFileSync(csvPath, 'utf8'))
const requiredSourceRows = resolveRequiredSourceRows(metrics, allCsvRows.length)
const csvRows = allCsvRows.slice(-requiredSourceRows)
const bars4h = resampleTo4h(csvRows)

const anchorTime = normalizeDate(prediction.anchor_time)
const anchorIndex = bars4h.findIndex((row) => row.ts === anchorTime.getTime())

if (anchorIndex < 0) {
  throw new Error(`Anchor time ${prediction.anchor_time} was not found in the 4h resampled series.`)
}

const anchorBar = bars4h[anchorIndex]
const horizons = [1, 2, 3, 6, 12, 18, 30]
const horizonRows = buildHorizonRows({ prediction, horizons, bars4h, anchorIndex, anchorClose: anchorBar.close })
const chartRows = buildChartRows({ rawRows: csvRows, anchorTime, anchorBar, horizonRows, bars4h, anchorIndex })
const history = metrics.history.map((row) => ({
  epoch: row.epoch,
  trainTotal: row.train_total,
  validationTotal: row.validation_total,
  trainReturn: row.train_return,
  validationReturn: row.validation_return,
  trainOverlay: row.train_overlay,
  validationOverlay: row.validation_overlay,
}))
const bestEpochRow = history.reduce((best, row) => (row.validationTotal < best.validationTotal ? row : best))
const convergenceGain = history[0].validationTotal - bestEpochRow.validationTotal
const generalizationGap = bestEpochRow.validationTotal - bestEpochRow.trainTotal
const validationMetrics = metrics.validation_metrics ?? null
const payload = {
  generatedAt: new Date().toISOString(),
  instrument: '金 / XAUUSD',
  provenance: {
    rawRows: csvRows.length,
    sourcePath: csvPath,
    start: toIso(csvRows[0].ts),
    end: toIso(csvRows[csvRows.length - 1].ts),
  },
  run: {
    anchorTime: toIso(anchorBar.ts),
    anchorClose: anchorBar.close,
    selectedHorizon: prediction.selected_horizon,
    selectedHours: prediction.selected_horizon * 4,
    position: prediction.position,
    overlayAction: prediction.overlay_action,
    trainSamples: metrics.train_samples,
    validationSamples: metrics.validation_samples,
    sampleCount: metrics.sample_count,
    bestValidationLoss: metrics.best_validation_loss,
    bestEpoch: bestEpochRow.epoch,
    convergenceGain,
    generalizationGap,
    sourceRows: metrics.source_rows_used ?? csvRows.length,
    modelDirectory: currentRunDir,
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
          returnRmse: validationMetrics.return_rmse ?? null,
          returnMae: validationMetrics.return_mae ?? null,
          directionalAccuracy: validationMetrics.directional_accuracy ?? null,
          uncertaintyCalibrationError: validationMetrics.uncertainty_calibration_error ?? null,
          coverageAt1Sigma: validationMetrics.coverage_at_1sigma ?? null,
          overlayAccuracy: validationMetrics.overlay_accuracy ?? null,
          overlayMacroF1: validationMetrics.overlay_macro_f1 ?? null,
          precisionFeasible: validationMetrics.precision_feasible ?? null,
          thresholdCalibrationFeasible: validationMetrics.threshold_calibration_feasible ?? null,
          selectionPrecision: validationMetrics.selection_precision ?? null,
          selectionSupport: validationMetrics.selection_support ?? null,
          coverageAtTargetPrecision: validationMetrics.coverage_at_target_precision ?? null,
          noTradeRate: validationMetrics.no_trade_rate ?? null,
          valuePerSignal: validationMetrics.value_per_signal ?? null,
          acceptedValuePerSignal: validationMetrics.accepted_value_per_signal ?? null,
          downsidePerSignal: validationMetrics.downside_per_signal ?? null,
          valueCaptureRatio: validationMetrics.value_capture_ratio ?? null,
          profitFactor: validationMetrics.profit_factor ?? null,
          signalSharpe: validationMetrics.signal_sharpe ?? null,
          signalSortino: validationMetrics.signal_sortino ?? null,
          directionBrierScore: validationMetrics.direction_brier_score ?? null,
          selectionBrierScore: validationMetrics.selection_brier_score ?? null,
          selectionCalibrationError: validationMetrics.selection_calibration_error ?? null,
          holdBrierScore: validationMetrics.hold_brier_score ?? null,
          turnover: validationMetrics.turnover ?? null,
          maxDrawdown: validationMetrics.max_drawdown ?? null,
          projectValueScore: validationMetrics.project_value_score ?? null,
          utilityScore: validationMetrics.utility_score ?? null,
        }
      : null,
  },
  narrative: buildNarrative({ prediction, horizonRows }),
}

fs.mkdirSync(path.dirname(outputPath), { recursive: true })
fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2))
console.log(`Wrote ${outputPath}`)

function resolveCsvPath(targetDateJst) {
  if (process.env.SIGNAL_CASCADE_CSV_PATH) {
    return path.resolve(process.env.SIGNAL_CASCADE_CSV_PATH)
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

  if (fs.existsSync(liveCsvPath)) {
    return liveCsvPath
  }

  throw new Error('No stored CSV source was found for the current SignalCascade run.')
}

function refreshLatestCsv(targetDateJst) {
  if (!fs.existsSync(pyTorchPython)) {
    throw new Error(`Python runtime was not found: ${pyTorchPython}`)
  }

  fs.mkdirSync(liveDataDir, { recursive: true })

  const rawDownloadDir = fs.mkdtempSync(path.join(os.tmpdir(), 'signal-cascade-xauusd-'))
  const rawBaseName = 'xauusd_m30_dukascopy'
  const rawCsvPath = path.join(rawDownloadDir, `${rawBaseName}.csv`)
  const fromDate = shiftDate(targetDateJst, -90)
  const toDateExclusive = shiftDate(targetDateJst, 1)
  const download = spawnSync(
    'npx',
    [
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
    ],
    {
      cwd: repoRoot,
      encoding: 'utf8',
    },
  )

  if (download.status !== 0) {
    throw new Error(`Failed to download XAUUSD market data: ${download.stderr || download.stdout}`)
  }

  normalizeDukascopyCsv(rawCsvPath, liveCsvPath, endOfJstDayUtcMs(targetDateJst))

  console.log(download.stdout.trim())
  return liveCsvPath
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

function buildHorizonRows({ prediction, horizons, bars4h, anchorIndex, anchorClose }) {
  return horizons.map((horizon) => {
    const predictedClose = prediction.predicted_closes[String(horizon)]
    const uncertainty = prediction.uncertainties[String(horizon)]
    const lowerClose = predictedClose * Math.exp(-uncertainty)
    const upperClose = predictedClose * Math.exp(uncertainty)
    const actualBar = bars4h[anchorIndex + horizon] ?? null
    const actualClose = actualBar ? actualBar.close : null

    return {
      horizon,
      hours: horizon * 4,
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

function buildChartRows({ rawRows, anchorTime, anchorBar, horizonRows, bars4h, anchorIndex }) {
  const historyRows = rawRows
    .filter((row) => row.ts <= anchorTime.getTime())
    .slice(-192)
    .map((row) => ({
      ts: row.ts,
      label: toIso(row.ts),
      phase: 'history',
      close: row.close,
      high: row.high,
      low: row.low,
      forecastBase: null,
      forecastLower: null,
      forecastUpper: null,
      forecastSpread: null,
      actualClose: null,
      actualRangeStart: null,
      actualRangeEnd: null,
    }))

  const interpolatedForecast = interpolateForecast({ anchorBar, horizonRows, bars4h, anchorIndex })
  const forecastRows = interpolatedForecast.map((row) => ({
    ts: row.ts,
    label: toIso(row.ts),
    phase: 'forecast',
    close: null,
    high: null,
    low: null,
    forecastBase: row.base,
    forecastLower: row.lower,
    forecastUpper: row.upper,
    forecastSpread: row.upper - row.lower,
    actualClose: row.actual,
    actualRangeStart: row.actualRangeStart,
    actualRangeEnd: row.actualRangeEnd,
  }))

  const rows = [...historyRows, ...forecastRows]
  const microRows = buildMicroRows({
    rawRows,
    startTs: historyRows[0]?.ts ?? anchorBar.ts,
    endTs: forecastRows.at(-1)?.ts ?? anchorBar.ts,
  })

  return {
    rows,
    microRows,
    yDomain: calculateVisiblePriceDomain(rows),
  }
}

function buildMicroRows({ rawRows, startTs, endTs }) {
  return rawRows
    .filter((row) => row.ts >= startTs && row.ts <= endTs)
    .map((row) => ({
      ts: row.ts,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    }))
}

function calculateVisiblePriceDomain(rows) {
  const values = rows
    .flatMap((row) => [row.close, row.forecastBase, row.forecastLower, row.forecastUpper, row.actualClose])
    .filter((value) => value !== null)

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min
  const pad = range === 0 ? Math.max(min * 0.0025, 1) : Math.max(range * 0.035, max * 0.0025, 1)

  return [Math.floor(min - pad), Math.ceil(max + pad)]
}

function interpolateForecast({ anchorBar, horizonRows, bars4h, anchorIndex }) {
  const horizonActuals = new Map(horizonRows.map((row) => [row.horizon, row.actualClose]))
  const points = [
    {
      step: 0,
      ts: anchorBar.ts,
      logBase: 0,
      sigma: 0,
      actual: anchorBar.close,
    },
    ...horizonRows.map((row) => ({
      step: row.horizon,
      ts: bars4h[anchorIndex + row.horizon]?.ts ?? anchorBar.ts + row.horizon * 4 * 60 * 60 * 1000,
      logBase: Math.log(row.predictedClose / anchorBar.close),
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
    const base = anchorBar.close * Math.exp(logBase)
    const lower = base * Math.exp(-sigma)
    const upper = base * Math.exp(sigma)
    const actualBar = bars4h[anchorIndex + step] ?? null
    const actualPoint = horizonActuals.get(step) ?? actualBar?.close ?? null

    rows.push({
      ts: bars4h[anchorIndex + step]?.ts ?? anchorBar.ts + step * 4 * 60 * 60 * 1000,
      base,
      lower,
      upper,
      actual: actualPoint,
      actualRangeStart: actualBar?.firstTs ?? null,
      actualRangeEnd: actualBar?.lastTs ?? null,
    })
  }

  return rows
}

function buildNarrative({ prediction, horizonRows }) {
  const selected = horizonRows.find((row) => row.horizon === prediction.selected_horizon) ?? horizonRows[0]
  const directionalWord = prediction.accepted_signal === false
    ? '見送りです。'
    : prediction.position < -0.25
      ? '下向きです。'
      : prediction.position > 0.25
        ? '上向きです。'
        : '横ばいです。'
  const overlayWord = prediction.overlay_action === 'hold'
    ? '短期判断は維持です。'
    : `短期判断は ${translateOverlay(prediction.overlay_action)} です。`
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
      prediction.accepted_signal === false
        ? `採用確率は ${(prediction.selection_probability * 100).toFixed(1)}%、閾値は ${(prediction.selection_threshold * 100).toFixed(1)}% です。`
        : `判断は ${translateOverlay(prediction.overlay_action)}、強さは ${Math.abs(prediction.position).toFixed(2)} です。`,
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
