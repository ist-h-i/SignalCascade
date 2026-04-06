import type { DashboardData } from './types'

type DashboardRunInput = Omit<DashboardData['run'], 'effectivePriceScale'> & {
  effectivePriceScale?: number
}

export type DashboardDataInput = Omit<DashboardData, 'run'> & {
  run: DashboardRunInput
}

export function normalizeDashboardData(data: DashboardDataInput): DashboardData {
  const effectivePriceScale = resolveEffectivePriceScale(data.run)

  return {
    ...data,
    run: {
      ...data.run,
      effectivePriceScale,
      priceScale: effectivePriceScale,
      anchorCloseRaw: resolveFiniteNumber(data.run.anchorCloseRaw) ?? data.run.anchorClose * effectivePriceScale,
    },
  }
}

export function resolveEffectivePriceScale(run: { effectivePriceScale?: number; priceScale?: number }): number {
  return resolvePositiveNumber(run.effectivePriceScale) ?? resolvePositiveNumber(run.priceScale) ?? 1
}

function resolvePositiveNumber(value: number | null | undefined): number | null {
  const numeric = Number(value)
  return Number.isFinite(numeric) && numeric > 0 ? numeric : null
}

function resolveFiniteNumber(value: number | null | undefined): number | null {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}
