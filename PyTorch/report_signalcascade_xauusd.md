# SignalCascade Research Report

- Generated (JST): `2026-04-08T20:39:12.566188+09:00`
- Policy mode: `shape_aware_profit_maximization`

## Artifact
- kind: `training_run`
- artifact id: `5c1e54f6411a4241f015b0412023ff1fef98c39d5eed050d296dc834b2910375`
- source kind / path: `csv` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/archive/session_20260408T075001Z/candidate_03/data_snapshot.csv`
- config origin: `explicit_v2`
- git head / dirty: `7988dd86eab7e75c8c4a3266780843f7f3707cbf` / `True`
- git tree sha: `b58fa64ee0bd77b74d903f602a020005ea9192a5`
- data snapshot sha256: `83aec529c968886d`
- sub-artifact lineage: `analysis.json:regenerated, config.json:generated, data_snapshot.csv:generated, forecast_summary.json:generated, horizon_diag.csv:generated, manifest.json:generated, metrics.json:generated, policy_summary.csv:generated, prediction.json:generated, research_report.md:regenerated, source.json:generated, validation_rows.csv:generated, validation_summary.json:generated`

## Dataset
- sample_count: `1392`
- effective_sample_count / purged_samples: `1362` / `30`
- train / validation: `1084` / `278`
- source_rows_original / used: `11608` / `11608`

## Contract
- current alias role: `production_current`
- authoritative runtime config: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/config.json`
- diagnostic recommendation pointer: `validation_summary.json.policy_calibration_summary.selected_row`
- top-level report role / path: `mirror_of_current_research_report` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/report_signalcascade_xauusd.md`
- current research report / prediction / forecast / source: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/research_report.md` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/prediction.json` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/forecast_summary.json` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/source.json`
- source-of-truth summary: `current/config.json, current/prediction.json, current/forecast_summary.json, current/source.json, current/research_report.md are authoritative. PyTorch/report_signalcascade_xauusd.md is a synchronized mirror of current/research_report.md.`

## Training
- best_validation_loss: `0.071385`
- best_epoch: `6`
- best_epoch_by_exact_log_wealth: `1`
- best_epoch_by_exact_log_wealth_minus_lambda_cvar: `6`
- best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar: `6`
- selected_epoch_rank_by_exact_log_wealth / delta_to_best_exact_log_wealth: `3` / `0.000233`
- selected_epoch_rank_by_exact_log_wealth_minus_lambda_cvar / delta_to_best_exact_log_wealth_minus_lambda_cvar: `1` / `0.000000`
- selected_epoch_rank_by_blocked_objective_log_wealth_minus_lambda_cvar / delta_to_best_blocked_objective_log_wealth_minus_lambda_cvar: `1` / `0.000000`
- epochs / warmup_epochs / oof_epochs: `14` / `2` / `3`
- walk_forward_folds: `3`
- shape_classes / state_dim: `6` / `24`

## Validation
- average_log_wealth: `0.000010`
- realized_pnl_per_anchor: `0.000010`
- cvar_tail_loss: `0.001834`
- max_drawdown: `0.019459`
- no_trade_band_hit_rate: `0.071942`
- exact_smooth_horizon_agreement / no_trade_agreement: `0.956835` / `1.000000`
- exact_smooth_position_mae / utility_regret: `0.010895` / `0.000392`
- shape_gate_usage: `0.493351`
- expert_entropy: `0.147324`
- shape_posterior_top_class_share: `{'1': 0.8525179856115108, '5': 0.1474820143884892}`
- mu_calibration / sigma_calibration: `0.054697` / `0.076400`
- log_wealth_clamp_hit_rate / state_reset_mode: `0.000000` / `carry_on`
- project_value_score / utility_score: `0.679322` / `0.593646`

## Evaluation
- carry_on average_log_wealth: `0.000010`
- reset_each_example average_log_wealth: `-0.000285`
- reset_each_session_or_window average_log_wealth: `-0.000149`
- blocked_walk_forward_folds / best_state_reset_mode_by_mean_log_wealth: `3` / `carry_on`
- blocked carry_on mean/min/max average_log_wealth: `0.000008` / `-0.000106` / `0.000139`
- blocked reset_each_session_or_window mean average_log_wealth / turnover_mean: `-0.000150` / `5.174468`
- policy sweep rows / pareto_optimal: `120` / `60`
- policy sweep selection basis / version: `pareto_rank_then_near_best_blocked_objective_mean_turnover_mean_blocked_objective_mean_average_log_wealth_mean_cvar_tail_loss_mean_row_key` / `4`
- selected policy sweep: reset=`carry_on`, cost x`0.50`, gamma x`4.00`, min_sigma=`0.000100`, q_max=`0.7500`, cvar_weight=`0.2000`, blocked_objective_mean=`-0.000274`
- selected policy sweep blocked mean wealth / cvar / turnover: `0.000062` / `0.001681` / `1.363700`
- applied runtime policy: reset=`carry_on`, cost x`6.0000`, gamma x`4.0000`, min_sigma=`0.000100`, q_max=`0.7500`, cvar_weight=`0.2000`
- selected_row_matches_applied_runtime: `False`
- selected row key: `state_reset_mode=carry_on|cost_multiplier=0.5|gamma_multiplier=4|min_policy_sigma=0.0001|q_max=0.75|cvar_weight=0.2`
- policy sweep rows sha256: `8b3e4adf5536cf2e`

## Forecast
- anchor_time_utc / jst: `2026-04-08T08:00:00+00:00` / `2026-04-08T17:00:00+09:00`
- anchor_close_display / raw / price_scale: `4819.6950` / `4819.6950` / `1.0000`
- policy_horizon / executed_horizon: `1` / `1`
- previous_position / position / trade_delta: `-0.0453` / `-0.0982` / `-0.0529`
- no_trade_band_hit: `False`
- g_t / shape_entropy / selected_policy_utility: `0.4923` / `0.0865` / `0.0248`
- display forecast label / policy driver label / head relationship: `display forecast` / `policy driver` / `separate_policy_head`
- overlay branch contract: `disabled_in_canonical_path`
- h=1: mu_t=0.0285, expected_return_pct=0.0290, predicted_close_display=4959.2337, sigma_t=0.0269
- h=2: mu_t=0.0081, expected_return_pct=0.0082, predicted_close_display=4859.1214, sigma_t=0.0247
- h=3: mu_t=-0.0358, expected_return_pct=-0.0352, predicted_close_display=4650.0053, sigma_t=0.0282
- h=6: mu_t=-0.0098, expected_return_pct=-0.0097, predicted_close_display=4772.7338, sigma_t=0.0546
- h=12: mu_t=0.0583, expected_return_pct=0.0600, predicted_close_display=5108.7897, sigma_t=0.0310
- h=18: mu_t=0.0779, expected_return_pct=0.0810, predicted_close_display=5210.2898, sigma_t=0.0374
- h=30: mu_t=0.0988, expected_return_pct=0.1038, predicted_close_display=5320.0075, sigma_t=0.0186

## Governance
- selection mode / rule / version: `accepted_candidate` / `optimization_gate_then_deployment_score` / `1`
- decision summary: `production current matches the accepted candidate after deployment-score selection.`
- best / accepted / production current: `candidate_03` / `candidate_03` / `candidate_03`
- production current user_value_score / chart_fidelity / sigma_band / execution_stability: `0.579331` / `0.699333` / `0.618001` / `0.692342`
- accepted snapshot user_value_score / chart_fidelity / sigma_band / execution_stability: `0.579331` / `0.699333` / `0.618001` / `0.692342`
- production current blocked objective / blocked turnover / max_drawdown / exact_smooth_position_mae / trade_delta / policy_horizon: `-0.000337` / `1.149111` / `0.019459` / `0.010895` / `-0.052850` / `1`
- accepted snapshot blocked objective / blocked turnover / max_drawdown / exact_smooth_position_mae / trade_delta / policy_horizon: `-0.000337` / `1.149111` / `0.019459` / `0.010895` / `-0.052850` / `1`
- production minus accepted delta (avg_log_wealth / blocked_objective / blocked_turnover / max_drawdown / exact_smooth_position_mae / trade_delta / policy_horizon): `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000`
- override priority metrics: `[]`

## Claim Hardening
- supported claims: `['continuous posterior weighting is present in the current artifact', 'head coupling can move policy_horizon without restoring shape routing']`
- unsupported claims: `['shape-aware routing', 'regime-aware routing']`
- dominant shape class / share: `1` / `0.852518`
- restore-claim evidence gate: `blocked folds must show materially lower shape top-class concentration and a paired variant must improve frontier metrics through richer shape usage`

## Assessment
- None