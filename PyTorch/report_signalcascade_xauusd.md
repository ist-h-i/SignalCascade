# SignalCascade Research Report

- Generated (JST): `2026-04-06T17:19:27.010662+09:00`
- Policy mode: `shape_aware_profit_maximization`

## Artifact
- kind: `training_run`
- artifact id: `efcbd52f7214ce4afe4d8421ed23f9d5b1b92b40ad60ca66ef42fbda1737b0ad`
- source kind / path: `csv` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/data_snapshot.csv`
- config origin: `explicit_v2`
- git head / dirty: `47ee916cdfc551fc2edc6a89a5133fe22ff24c90` / `True`
- git tree sha: `677a7a7d02c731a4ee82f4879ddc1f57c5f6bef2`
- data snapshot sha256: `d50093565e1368c3`
- sub-artifact lineage: `analysis.json:regenerated, config.json:generated, data_snapshot.csv:generated, forecast_summary.json:generated, horizon_diag.csv:generated, manifest.json:generated, metrics.json:generated, policy_summary.csv:generated, prediction.json:generated, research_report.md:regenerated, source.json:generated, validation_rows.csv:generated, validation_summary.json:generated`

## Dataset
- sample_count: `184`
- effective_sample_count / purged_samples: `154` / `30`
- train / validation: `118` / `36`
- source_rows_original / used: `2902` / `2902`

## Training
- best_validation_loss: `-0.353946`
- best_epoch: `12`
- epochs / warmup_epochs: `14` / `2`
- shape_classes / state_dim: `6` / `24`

## Validation
- average_log_wealth: `0.005911`
- realized_pnl_per_anchor: `0.005996`
- cvar_tail_loss: `0.009092`
- max_drawdown: `0.035882`
- no_trade_band_hit_rate: `0.000000`
- exact_smooth_horizon_agreement / no_trade_agreement: `1.000000` / `1.000000`
- exact_smooth_position_mae / utility_regret: `0.085578` / `0.006091`
- shape_gate_usage: `0.499480`
- expert_entropy: `0.455483`
- mu_calibration / sigma_calibration: `0.033047` / `0.056408`
- log_wealth_clamp_hit_rate / state_reset_mode: `0.000000` / `carry_on`
- project_value_score / utility_score: `0.704763` / `0.651345`

## Evaluation
- carry_on average_log_wealth: `0.005911`
- reset_each_example average_log_wealth: `0.005582`
- reset_each_session_or_window average_log_wealth: `0.005731`
- policy sweep rows / pareto_optimal: `72` / `42`
- policy sweep selection basis / version: `pareto_rank_then_average_log_wealth_cvar_tail_loss_turnover_row_key` / `2`
- selected policy sweep: reset=`carry_on`, cost x`0.50`, gamma x`0.50`, min_sigma=`0.000100`, log_wealth=`0.011346`
- selected row key: `state_reset_mode=carry_on|cost_multiplier=0.5|gamma_multiplier=0.5|min_policy_sigma=0.0001`
- policy sweep rows sha256: `40173e3880cf2e09`

## Forecast
- anchor_time_utc / jst: `2026-03-27T16:00:00+00:00` / `2026-03-28T01:00:00+09:00`
- anchor_close_display / raw / price_scale: `4528.1550` / `4528.1550` / `1.0000`
- policy_horizon / executed_horizon: `18` / `18`
- previous_position / position / trade_delta: `0.4789` / `0.2781` / `-0.2008`
- no_trade_band_hit: `False`
- g_t / shape_entropy / selected_policy_utility: `0.4993` / `0.3295` / `0.0422`
- h=1: mu_t=-0.0233, expected_return_pct=-0.0230, predicted_close_display=4423.9354, sigma_t=0.0474
- h=2: mu_t=-0.0024, expected_return_pct=-0.0024, predicted_close_display=4517.1357, sigma_t=0.0570
- h=3: mu_t=0.0215, expected_return_pct=0.0217, predicted_close_display=4626.4056, sigma_t=0.0885
- h=6: mu_t=-0.0318, expected_return_pct=-0.0313, predicted_close_display=4386.2405, sigma_t=0.0649
- h=12: mu_t=-0.0556, expected_return_pct=-0.0541, predicted_close_display=4283.1293, sigma_t=0.0601
- h=18: mu_t=0.0508, expected_return_pct=0.0521, predicted_close_display=4764.0438, sigma_t=0.0873
- h=30: mu_t=0.0015, expected_return_pct=0.0015, predicted_close_display=4534.8628, sigma_t=0.0949

## Assessment
- 新 spec の主経路は threshold policy ではなく `shape -> return distribution -> q_t*` です。 validation では average_log_wealth=0.0059, realized_pnl_per_anchor=0.0060, cvar_tail_loss=0.0091, no_trade_band_hit_rate=0.0000。 latest policy_horizon=18, q_t=0.2781, g_t=0.4993, selected_policy_utility=0.0422。