# SignalCascade Research Report

- Generated (JST): `2026-04-06T05:35:59.996778+09:00`
- Policy mode: `shape_aware_profit_maximization`

## Artifact
- kind: `training_run`
- artifact id: `3d3da091d771ef7433fde1cb101d322987fab6ef4a5e5c8cc29a63f41844706d`
- source kind / path: `csv` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/data_snapshot.csv`
- config origin: `explicit_v2`
- git head / dirty: `d8ea7668f3e8f9adbd741cf315be3b28c4547df6` / `True`
- git tree sha: `07361b19559b65251dd30c713ee6084dae7abd6f`
- data snapshot sha256: `b851b48d449a5017`
- sub-artifact lineage: `analysis.json:regenerated, config.json:generated, data_snapshot.csv:generated, forecast_summary.json:generated, manifest.json:generated, metrics.json:generated, prediction.json:generated, research_report.md:regenerated, source.json:generated`

## Dataset
- sample_count: `181`
- effective_sample_count / purged_samples: `151` / `30`
- train / validation: `115` / `36`
- source_rows_original / used: `2886` / `2886`

## Training
- best_validation_loss: `-0.356716`
- best_epoch: `5`
- epochs / warmup_epochs: `8` / `2`
- shape_classes / state_dim: `6` / `24`

## Validation
- average_log_wealth: `0.006312`
- realized_pnl_per_anchor: `0.006654`
- cvar_tail_loss: `0.036532`
- max_drawdown: `0.080585`
- no_trade_band_hit_rate: `0.000000`
- exact_smooth_horizon_agreement / no_trade_agreement: `0.666667` / `1.000000`
- exact_smooth_position_mae / utility_regret: `0.672176` / `0.011478`
- shape_gate_usage: `0.524924`
- expert_entropy: `0.193249`
- mu_calibration / sigma_calibration: `0.112999` / `0.044620`
- log_wealth_clamp_hit_rate / state_reset_mode: `0.000000` / `carry_on`
- project_value_score / utility_score: `0.515672` / `0.635933`

## Evaluation
- carry_on average_log_wealth: `0.000000`
- reset_each_example average_log_wealth: `0.000000`
- reset_each_session_or_window average_log_wealth: `0.000000`
- policy sweep rows / pareto_optimal: `0` / `0`
- policy sweep selection basis / version: `-` / `-`
- selected policy sweep: none
- selected row key: `-`
- policy sweep rows sha256: `-`

## Forecast
- anchor_time_utc / jst: `2026-04-03T00:00:00+00:00` / `2026-04-03T09:00:00+09:00`
- anchor_close: `4675.8050`
- policy_horizon / executed_horizon: `18` / `18`
- previous_position / position / trade_delta: `0.0000` / `-1.0000` / `-1.0000`
- no_trade_band_hit: `False`
- g_t / shape_entropy / selected_policy_utility: `0.5246` / `0.2881` / `0.0850`
- h=1: mu_t=-0.1014, expected_return_pct=-0.0964, predicted_close=4224.9679, sigma_t=0.1033
- h=2: mu_t=-0.0720, expected_return_pct=-0.0695, predicted_close=4350.8970, sigma_t=0.0804
- h=3: mu_t=0.0104, expected_return_pct=0.0105, predicted_close=4724.8028, sigma_t=0.0910
- h=6: mu_t=-0.0869, expected_return_pct=-0.0832, predicted_close=4286.6187, sigma_t=0.0815
- h=12: mu_t=0.0903, expected_return_pct=0.0945, predicted_close=5117.5921, sigma_t=0.0687
- h=18: mu_t=-0.2259, expected_return_pct=-0.2022, predicted_close=3730.3145, sigma_t=0.1436
- h=30: mu_t=-0.0114, expected_return_pct=-0.0113, predicted_close=4622.7788, sigma_t=0.0778

## Assessment
- 新 spec の主経路は threshold policy ではなく `shape -> return distribution -> q_t*` です。 validation では average_log_wealth=0.0063, realized_pnl_per_anchor=0.0067, cvar_tail_loss=0.0365, no_trade_band_hit_rate=0.0000。 latest policy_horizon=18, q_t=-1.0000, g_t=0.5246, selected_policy_utility=0.0850。