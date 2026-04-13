# SignalCascade Research Report

- Generated (JST): `2026-04-13T10:25:23.236446+09:00`
- Policy mode: `shape_aware_profit_maximization`

## Artifact
- kind: `training_run`
- artifact id: `0b2f918f2cf8f7668115b25676852d939208ac6419649a4363d28c5678c44819`
- source kind / path: `csv` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/data_snapshot.csv`
- config origin: `explicit_v2`
- git head / dirty: `54f67c0e943202c7f0bb7207864ca27b99b8af14` / `True`
- git tree sha: `bd79eb95a6235531ea4b0eb4bdfe8bcc568e64f4`
- data snapshot sha256: `a62bad3014419043`
- sub-artifact lineage: `analysis.json:regenerated, config.json:regenerated, data_snapshot.csv:generated, forecast_summary.json:regenerated, horizon_diag.csv:copied, manifest.json:generated, metrics.json:regenerated, policy_summary.csv:copied, prediction.json:regenerated, research_report.md:regenerated, source.json:regenerated, validation_rows.csv:copied, validation_summary.json:regenerated`

## Dataset
- sample_count: `1394`
- effective_sample_count / purged_samples: `1364` / `30`
- train / validation: `1086` / `278`
- source_rows_original / used: `11620` / `11620`

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
- applied runtime policy: reset=`carry_on`, cost x`0.5000`, gamma x`4.0000`, min_sigma=`0.000100`, q_max=`0.7500`, cvar_weight=`0.2000`
- selected_row_matches_applied_runtime: `True`
- selected row key: `state_reset_mode=carry_on|cost_multiplier=0.5|gamma_multiplier=4|min_policy_sigma=0.0001|q_max=0.75|cvar_weight=0.2`
- policy sweep rows sha256: `8b3e4adf5536cf2e`
- forecast quality score (selected_horizon / all_horizon / gap): `0.621600` / `0.426094` / `-0.195506`
- forecast quality directional_accuracy (selected_horizon / all_horizon): `0.535971` / `0.501028`
- forecast quality mu_calibration / probabilistic_score (selected_horizon / all_horizon): `0.054697` / `missing` / `0.076630` / `0.190461`

## Forecast
- anchor_time_utc / jst: `2026-04-08T16:00:00+00:00` / `2026-04-09T01:00:00+09:00`
- anchor_close_display / raw / price_scale: `4799.3250` / `4799.3250` / `1.0000`
- policy_horizon / executed_horizon: `1` / `1`
- previous_position / position / trade_delta: `-0.0411` / `-0.0997` / `-0.0585`
- no_trade_band_hit: `False`
- g_t / shape_entropy / selected_policy_utility: `0.4923` / `0.0925` / `0.0271`
- display forecast label / policy driver label / head relationship: `display forecast` / `policy driver` / `separate_policy_head`
- overlay branch contract: `disabled_in_canonical_path`
- h=1: mu_t=0.0159, expected_return_pct=0.0160, predicted_close_display=4876.1287, sigma_t=0.0292
- h=2: mu_t=-0.0078, expected_return_pct=-0.0077, predicted_close_display=4762.2670, sigma_t=0.0253
- h=3: mu_t=-0.0419, expected_return_pct=-0.0410, predicted_close_display=4602.5480, sigma_t=0.0309
- h=6: mu_t=-0.0087, expected_return_pct=-0.0086, predicted_close_display=4757.8354, sigma_t=0.0616
- h=12: mu_t=0.0568, expected_return_pct=0.0585, predicted_close_display=5079.8958, sigma_t=0.0329
- h=18: mu_t=0.0790, expected_return_pct=0.0822, predicted_close_display=5193.8640, sigma_t=0.0417
- h=30: mu_t=0.0830, expected_return_pct=0.0866, predicted_close_display=5214.8064, sigma_t=0.0174

## Governance
- selection mode / rule / version: `accepted_candidate` / `optimization_gate_then_deployment_score` / `1`
- decision summary: `production current matches the accepted candidate after deployment-score selection.`
- best / accepted / production current: `candidate_03` / `candidate_03` / `candidate_03`
- production current user_value_score / chart_fidelity / sigma_band / execution_stability: `0.579331` / `0.699333` / `0.618001` / `0.692342`
- production current selected_horizon / all_horizon forecast quality: `0.621600` / `0.426094`
- ranking split top candidate (current / selected / all): `candidate_03` / `candidate_01` / `candidate_05`
- ranking split Spearman (selected/current / all/current / selected/all): `0.547149` / `-0.050254` / `0.472614`
- accepted candidate rank (current / selected / all): `1` / `2` / `12`
- top-3 overlap with current (selected / all): `2` / `0`
- history sessions / accepted / production / diverged: `13` / `12` / `2` / `1`
- history accepted top-match rate (current / selected / all): `1.000000` / `0.375000` / `0.250000`
- history accepted median rank (current / selected / all): `1.000000` / `2.000000` / `2.500000`
- history production top-match rate (current / selected / all): `0.500000` / `0.000000` / `0.000000`
- history production median rank (current / selected / all): `1.500000` / `5.500000` / `15.500000`
- divergence scorecard coverage (full / partial): `8` / `5`
- divergence scorecard clusters (all sessions): `broad_alignment=4, insufficient_coverage=5, objective_evaluation_mismatch=2, rank_stability_coupled=1, stability_override=1`
- divergence scorecard clusters (full coverage): `broad_alignment=4, objective_evaluation_mismatch=2, rank_stability_coupled=1, stability_override=1`
- recent history snapshots: 20260407T041853Z:acc=candidate_05,prod=candidate_17,ranks=1/1/1 | 20260407T111619Z:acc=None,prod=None,ranks=missing/missing/missing | 20260408T075001Z:acc=candidate_03,prod=candidate_03,ranks=1/2/12
- divergence scorecard recent rows: 20260407T041853Z [full/stability_override] acc=`candidate_05` ranks=`1/1/1` turn=`1.458962` mae=`0.195734` mdd=`0.088695` prod=`candidate_17` ranks=`2/9/19` turn=`0.241325` mae=`0.018842` mdd=`0.017009` | 20260407T111619Z [partial/insufficient_coverage] acc=`-` ranks=`missing/missing/missing` turn=`missing` mae=`missing` mdd=`missing` prod=`-` ranks=`missing/missing/missing` turn=`missing` mae=`missing` mdd=`missing` | 20260408T075001Z [full/objective_evaluation_mismatch] acc=`candidate_03` ranks=`1/2/12` turn=`1.149111` mae=`0.011591` mdd=`0.019459` prod=`candidate_03` ranks=`1/2/12` turn=`1.149111` mae=`0.011591` mdd=`0.019459`
- accepted snapshot user_value_score / chart_fidelity / sigma_band / execution_stability: `0.579331` / `0.699333` / `0.618001` / `0.692342`
- accepted snapshot selected_horizon / all_horizon forecast quality: `0.621600` / `0.426094`
- production current blocked objective / blocked turnover / max_drawdown / exact_smooth_position_mae / trade_delta / policy_horizon: `-0.000337` / `1.149111` / `0.019459` / `0.010895` / `-0.052850` / `1`
- accepted snapshot blocked objective / blocked turnover / max_drawdown / exact_smooth_position_mae / trade_delta / policy_horizon: `-0.000337` / `1.149111` / `0.019459` / `0.010895` / `-0.052850` / `1`
- production minus accepted delta (avg_log_wealth / blocked_objective / blocked_turnover / max_drawdown / exact_smooth_position_mae / selected_horizon_quality / all_horizon_quality / trade_delta / policy_horizon): `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000`
- override priority metrics: `[]`

## Claim Hardening
- supported claims: `['continuous posterior weighting is present in the current artifact', 'head coupling can move policy_horizon without restoring shape routing']`
- unsupported claims: `['shape-aware routing', 'regime-aware routing']`
- dominant shape class / share: `1` / `0.852518`
- restore-claim evidence gate: `blocked folds must show materially lower shape top-class concentration and a paired variant must improve frontier metrics through richer shape usage`

## Assessment
- None