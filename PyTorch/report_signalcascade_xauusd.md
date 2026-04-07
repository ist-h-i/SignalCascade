# SignalCascade Research Report

- Generated (JST): `2026-04-07T17:49:26.728497+09:00`
- Policy mode: `shape_aware_profit_maximization`

## Artifact
- kind: `training_run`
- artifact id: `69a9472e7bd04425b770d9983912e657e05d205aab2cda6a0c1a4946f216e17d`
- source kind / path: `csv` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/archive/session_20260407T041853Z/candidate_17/data_snapshot.csv`
- config origin: `explicit_v2`
- git head / dirty: `6183a6d4b44c9270756e1e501099cc1979a31ef7` / `True`
- git tree sha: `af6591a7c9ba9ddecdbc8cfe3471cd8ceae962fa`
- data snapshot sha256: `d50093565e1368c3`
- sub-artifact lineage: `analysis.json:regenerated, config.json:generated, data_snapshot.csv:generated, forecast_summary.json:generated, horizon_diag.csv:generated, manifest.json:generated, metrics.json:generated, policy_summary.csv:generated, prediction.json:generated, research_report.md:regenerated, source.json:generated, validation_rows.csv:generated, validation_summary.json:generated`

## Dataset
- sample_count: `184`
- effective_sample_count / purged_samples: `154` / `30`
- train / validation: `118` / `36`
- source_rows_original / used: `2902` / `2902`

## Contract
- current alias role: `production_current`
- authoritative runtime config: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/config.json`
- diagnostic recommendation pointer: `validation_summary.json.policy_calibration_summary.selected_row`
- top-level report role / path: `mirror_of_current_research_report` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/report_signalcascade_xauusd.md`
- current research report / prediction / forecast / source: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/research_report.md` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/prediction.json` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/forecast_summary.json` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/source.json`
- source-of-truth summary: `current/config.json, current/prediction.json, current/forecast_summary.json, current/source.json, current/research_report.md are authoritative. PyTorch/report_signalcascade_xauusd.md is a synchronized mirror of current/research_report.md.`

## Training
- best_validation_loss: `-0.173533`
- best_epoch: `1`
- best_epoch_by_exact_log_wealth: `10`
- best_epoch_by_exact_log_wealth_minus_lambda_cvar: `10`
- best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar: `10`
- selected_epoch_rank_by_exact_log_wealth / delta_to_best_exact_log_wealth: `2` / `0.003966`
- selected_epoch_rank_by_exact_log_wealth_minus_lambda_cvar / delta_to_best_exact_log_wealth_minus_lambda_cvar: `2` / `0.002833`
- selected_epoch_rank_by_blocked_objective_log_wealth_minus_lambda_cvar / delta_to_best_blocked_objective_log_wealth_minus_lambda_cvar: `2` / `0.003314`
- epochs / warmup_epochs / oof_epochs: `14` / `2` / `3`
- walk_forward_folds: `3`
- shape_classes / state_dim: `6` / `24`

## Validation
- average_log_wealth: `0.001921`
- realized_pnl_per_anchor: `0.001929`
- cvar_tail_loss: `0.005327`
- max_drawdown: `0.017009`
- no_trade_band_hit_rate: `0.361111`
- exact_smooth_horizon_agreement / no_trade_agreement: `0.972222` / `1.000000`
- exact_smooth_position_mae / utility_regret: `0.014458` / `0.000116`
- shape_gate_usage: `0.503692`
- expert_entropy: `0.894631`
- shape_posterior_top_class_share: `{'1': 1.0}`
- mu_calibration / sigma_calibration: `0.181520` / `0.134025`
- log_wealth_clamp_hit_rate / state_reset_mode: `0.000000` / `carry_on`
- project_value_score / utility_score: `0.606620` / `0.612404`

## Evaluation
- carry_on average_log_wealth: `0.001921`
- reset_each_example average_log_wealth: `0.000172`
- reset_each_session_or_window average_log_wealth: `0.000469`
- blocked_walk_forward_folds / best_state_reset_mode_by_mean_log_wealth: `3` / `carry_on`
- blocked carry_on mean/min/max average_log_wealth: `0.001763` / `-0.000395` / `0.003277`
- blocked reset_each_session_or_window mean average_log_wealth / turnover_mean: `0.000417` / `0.604192`
- policy sweep rows / pareto_optimal: `120` / `36`
- policy sweep selection basis / version: `pareto_rank_then_near_best_blocked_objective_mean_turnover_mean_blocked_objective_mean_average_log_wealth_mean_cvar_tail_loss_mean_row_key` / `4`
- selected policy sweep: reset=`carry_on`, cost x`6.00`, gamma x`0.50`, min_sigma=`0.000100`, q_max=`0.5000`, cvar_weight=`0.2000`, blocked_objective_mean=`0.009719`
- selected policy sweep blocked mean wealth / cvar / turnover: `0.010910` / `0.005953` / `0.595836`
- applied runtime policy: reset=`carry_on`, cost x`6.0000`, gamma x`4.0000`, min_sigma=`0.000100`, q_max=`0.5000`, cvar_weight=`0.2000`
- selected_row_matches_applied_runtime: `False`
- selected row key: `state_reset_mode=carry_on|cost_multiplier=6|gamma_multiplier=0.5|min_policy_sigma=0.0001|q_max=0.5|cvar_weight=0.2`
- policy sweep rows sha256: `a7891187f4198b26`

## Forecast
- anchor_time_utc / jst: `2026-04-06T08:00:00+00:00` / `2026-04-06T17:00:00+09:00`
- anchor_close_display / raw / price_scale: `4652.2580` / `4652.2580` / `1.0000`
- policy_horizon / executed_horizon: `18` / `18`
- previous_position / position / trade_delta: `-0.1292` / `-0.5000` / `-0.3708`
- no_trade_band_hit: `False`
- g_t / shape_entropy / selected_policy_utility: `0.5052` / `0.2702` / `0.0475`
- display forecast label / policy driver label / head relationship: `display forecast` / `policy driver` / `tied_to_forecast_head`
- overlay branch contract: `disabled_in_canonical_path`
- h=1: mu_t=-0.1255, expected_return_pct=-0.1180, predicted_close_display=4103.3568, sigma_t=0.0950
- h=2: mu_t=0.0141, expected_return_pct=0.0142, predicted_close_display=4718.1579, sigma_t=0.0932
- h=3: mu_t=0.0862, expected_return_pct=0.0900, predicted_close_display=5071.1549, sigma_t=0.0799
- h=6: mu_t=-0.0275, expected_return_pct=-0.0271, predicted_close_display=4526.2328, sigma_t=0.1166
- h=12: mu_t=-0.0215, expected_return_pct=-0.0213, predicted_close_display=4553.2224, sigma_t=0.0771
- h=18: mu_t=-0.2763, expected_return_pct=-0.2414, predicted_close_display=3529.1679, sigma_t=0.1052
- h=30: mu_t=0.2175, expected_return_pct=0.2430, predicted_close_display=5782.5366, sigma_t=0.0927

## Governance
- selection mode / rule / version: `auto_user_value_selection` / `optimization_gate_then_user_value_score` / `1`
- decision summary: `production current differs from the accepted candidate because chart fidelity, sigma-band reliability, and execution stability took priority over blocked-objective rank.`
- best / accepted / production current: `candidate_05` / `candidate_05` / `candidate_17`
- production current user_value_score / chart_fidelity / sigma_band / execution_stability: `0.670851` / `0.698648` / `0.329876` / `0.883697`
- accepted snapshot user_value_score / chart_fidelity / sigma_band / execution_stability: `0.559322` / `0.574995` / `0.864140` / `0.353427`
- production current blocked objective / blocked turnover / max_drawdown / exact_smooth_position_mae / trade_delta / policy_horizon: `0.001245` / `0.241325` / `0.017009` / `0.014458` / `-0.370761` / `18`
- accepted snapshot blocked objective / blocked turnover / max_drawdown / exact_smooth_position_mae / trade_delta / policy_horizon: `0.007317` / `1.458962` / `0.088695` / `0.161667` / `0.207183` / `2`
- production minus accepted delta (avg_log_wealth / blocked_objective / blocked_turnover / max_drawdown / exact_smooth_position_mae / trade_delta / policy_horizon): `-0.005602` / `-0.006072` / `-1.217636` / `-0.071686` / `-0.147208` / `-0.577944` / `16.000000`
- override priority metrics: `['blocked_directional_accuracy_mean', 'mu_calibration', 'blocked_exact_smooth_position_mae_mean', 'sigma_calibration', 'max_drawdown', 'blocked_turnover_mean', 'blocked_objective_log_wealth_minus_lambda_cvar_mean']`

## Claim Hardening
- supported claims: `['continuous posterior weighting is present in the current artifact', 'head coupling can move policy_horizon without restoring shape routing']`
- unsupported claims: `['shape-aware routing', 'regime-aware routing']`
- dominant shape class / share: `1` / `1.000000`
- restore-claim evidence gate: `blocked folds must show materially lower shape top-class concentration and a paired variant must improve frontier metrics through richer shape usage`

## Assessment
- None