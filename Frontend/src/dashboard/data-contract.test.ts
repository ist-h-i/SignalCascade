import assert from 'node:assert/strict'
import test from 'node:test'

import { normalizeDashboardData } from './data-contract.ts'
import type { DashboardDataInput } from './data-contract.ts'

function createDashboardData(runOverrides: Partial<DashboardDataInput['run']> = {}): DashboardDataInput {
  return {
    schemaVersion: 6,
    generatedAt: '2026-04-06T07:15:10.372Z',
    instrument: '金 / XAUUSD',
    artifacts: {
      metricsSchemaVersion: 4,
      predictionSchemaVersion: 6,
      forecastSchemaVersion: 6,
      sourceSchemaVersion: 2,
    },
    provenance: {
      rawRows: 2902,
      sourcePath: 'PyTorch/artifacts/gold_xauusd_m30/current/data_snapshot.csv',
      start: '2026-01-06T00:00:00.000Z',
      end: '2026-04-06T05:30:00.000Z',
    },
    run: {
      anchorTime: '2026-03-27T16:00:00.000Z',
      anchorClose: 4528.155,
      selectedHorizon: 2,
      executedHorizon: 2,
      selectedHours: 8,
      previousPosition: -0.42,
      position: -0.38,
      tradeDelta: 0.03,
      noTradeBandHit: false,
      gT: 0.5,
      selectedPolicyUtility: 0.35,
      overlayAction: 'reduce',
      policyStatus: 'active',
      stateResetMode: 'carry_on',
      costMultiplier: 1,
      gammaMultiplier: 1,
      minPolicySigma: 0.0001,
      interruptedTuning: false,
      runQuality: 'stale',
      trainSamples: 118,
      validationSamples: 36,
      sampleCount: 184,
      effectiveSampleCount: 154,
      purgedSamples: 30,
      bestValidationLoss: -0.39,
      bestEpoch: 13,
      convergenceGain: 0.19,
      generalizationGap: 0.03,
      sourceRows: 2902,
      ...runOverrides,
    },
    chart: {
      rows: [],
      microRows: [],
      yDomain: [4400, 4600],
    },
    horizons: [],
    metrics: {
      history: [],
      validation: null,
    },
    narrative: {
      title: 'SignalCascade',
      summary: 'test fixture',
      bullets: [],
    },
  }
}

test('normalizeDashboardData prefers effectivePriceScale over the legacy alias', () => {
  const normalized = normalizeDashboardData(
    createDashboardData({
      effectivePriceScale: 1,
      priceScale: 100,
    }),
  )

  assert.equal(normalized.run.effectivePriceScale, 1)
  assert.equal(normalized.run.priceScale, 1)
})

test('normalizeDashboardData falls back to priceScale when effectivePriceScale is missing', () => {
  const normalized = normalizeDashboardData(
    createDashboardData({
      priceScale: 25,
    }),
  )

  assert.equal(normalized.run.effectivePriceScale, 25)
  assert.equal(normalized.run.priceScale, 25)
  assert.equal(normalized.run.anchorCloseRaw, normalized.run.anchorClose * 25)
})
