# SignalCascade Research Report

- Generated (JST): `2026-04-06T01:53:32.605024+09:00`
- Policy mode: `shape_aware_profit_maximization`

## Artifact
- kind: `training_run`
- artifact id: `a8421e752a1cde28f9622d39e21e2f65010870a08d6d56ddebad06406a119393`
- source kind / path: `csv` / `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/data_snapshot.csv`
- config origin: `explicit_v2`
- git head / dirty: `88c46878eaef49b5ad2ad402ce9bba28abf17251` / `True`
- git tree sha: `a8149ff65be671aea41cb03243b40f9bfca5b718`
- data snapshot sha256: `b851b48d449a5017`
- sub-artifact lineage: `analysis.json:regenerated, config.json:generated, data_snapshot.csv:generated, forecast_summary.json:generated, manifest.json:generated, metrics.json:generated, prediction.json:generated, research_report.md:regenerated, source.json:generated`

## Dataset
- sample_count: `119`
- effective_sample_count / purged_samples: `89` / `30`
- train / validation: `66` / `23`
- source_rows_original / used: `2886` / `2886`

## Training
- best_validation_loss: `-0.355344`
- best_epoch: `3`
- epochs / warmup_epochs: `6` / `2`
- shape_classes / state_dim: `6` / `24`

## Validation
- average_log_wealth: `0.014923`
- realized_pnl_per_anchor: `0.015250`
- cvar_tail_loss: `0.020756`
- max_drawdown: `0.053173`
- no_trade_band_hit_rate: `0.000000`
- exact_smooth_horizon_agreement / no_trade_agreement: `0.782609` / `1.000000`
- exact_smooth_position_mae / utility_regret: `0.490120` / `0.012436`
- shape_gate_usage: `0.514300`
- expert_entropy: `0.120315`
- mu_calibration / sigma_calibration: `0.173035` / `0.111195`
- log_wealth_clamp_hit_rate / state_reset_mode: `0.000000` / `carry_on`
- project_value_score / utility_score: `0.603435` / `0.743359`

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
- policy_horizon / executed_horizon: `6` / `6`
- previous_position / position / trade_delta: `0.0000` / `0.7095` / `0.7095`
- no_trade_band_hit: `False`
- tradeability_gate / shape_entropy / policy_score: `0.5114` / `0.6856` / `0.0207`
- h=1: expected_log_return=0.1160, expected_return_pct=0.1230, predicted_close=5250.9849, uncertainty=0.1995
- h=2: expected_log_return=0.0318, expected_return_pct=0.0323, predicted_close=4826.8368, uncertainty=0.1943
- h=3: expected_log_return=0.0346, expected_return_pct=0.0352, predicted_close=4840.5411, uncertainty=0.1936
- h=6: expected_log_return=0.1171, expected_return_pct=0.1242, predicted_close=5256.4693, uncertainty=0.1656
- h=12: expected_log_return=0.0380, expected_return_pct=0.0387, predicted_close=4856.7079, uncertainty=0.2034
- h=18: expected_log_return=0.0351, expected_return_pct=0.0357, predicted_close=4842.6370, uncertainty=0.1912
- h=30: expected_log_return=0.0294, expected_return_pct=0.0298, predicted_close=4815.1948, uncertainty=0.2168

## Assessment
- 新 spec の主経路は threshold policy ではなく `shape -> return distribution -> q_t*` です。 validation では average_log_wealth=0.0149, realized_pnl_per_anchor=0.0153, cvar_tail_loss=0.0208, no_trade_band_hit_rate=0.0000。 latest policy_horizon=6, position=0.7095, tradeability_gate=0.5114。