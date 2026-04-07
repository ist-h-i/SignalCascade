# SignalCascade Review Handoff

最終更新: 2026-04-07 22:30:08 JST

このファイルは、外部 reviewer / director に現状を短時間で伝えるための self-contained なプロンプトです。
生成元: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/reviewer_submission.md`

---

## 1. あなたへの依頼

あなたは `GPT-5.4 Pro` です。
`SignalCascade` の現状実装と観測結果を踏まえて、次に進むための判断材料を提供してください。
回答の最後に、ゴール到達までの残ステップと現在地を簡潔に示してください。

重視してほしい点:
- finance
- programming
- math

## 2. ゴール

- 現在の不明点とボトルネックを明文化する
- 外部 reviewer に判断してほしい論点を絞り込む
- 次の実装・検証・実験につながる提案を得る

## 3. 現状の重要観測

- Workspace: `/Users/inouehiroshi/Documents/GitHub/SignalCascade`
- Git branch: `main`
- Working tree: `dirty`
- 変更ファイル数: `36`

### Git status

```text
M Frontend/public/dashboard-data.json
 M Frontend/scripts/check-dashboard-data-contract.mjs
 M Frontend/scripts/dashboard-publish-regression.test.mjs
 M Frontend/scripts/sync-signal-cascade-data.mjs
 M Frontend/src/dashboard/DashboardPage.tsx
 M Frontend/src/dashboard/dashboard.css
 M Frontend/src/dashboard/types.ts
 M Frontend/src/dashboard/view-model.ts
 M Frontend/src/index.css
 M PyTorch/README.md
 M PyTorch/logic_multiframe_candlestick_model.md
 M PyTorch/report_signalcascade_xauusd.md
 M PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py
 M PyTorch/src/signal_cascade_pytorch/application/config.py
 M PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py
 M PyTorch/src/signal_cascade_pytorch/application/inference_service.py
 M PyTorch/src/signal_cascade_pytorch/application/policy_service.py
 M PyTorch/src/signal_cascade_pytorch/application/report_service.py
 M PyTorch/src/signal_cascade_pytorch/application/training_service.py
 M PyTorch/src/signal_cascade_pytorch/application/tuning_service.py
 M PyTorch/src/signal_cascade_pytorch/bootstrap.py
 M PyTorch/src/signal_cascade_pytorch/domain/entities.py
 M PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py
 M PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py
 M PyTorch/src/signal_cascade_pytorch/interfaces/cli.py
 M PyTorch/tests/test_artifact_schema.py
 M PyTorch/tests/test_policy_sweep.py
 M PyTorch/tests/test_policy_training.py
 M docs/README.md
 M docs/UI_UX_DESIGN_RULES.md
 M test-results/ui-quality/ui-quality-report.json
?? PyTorch/review_feedback_comment.md
?? PyTorch/review_followup_milestones.md
?? PyTorch/reviewer_submission.md
?? PyTorch/src/signal_cascade_pytorch/application/current_alias.py
?? docs/prompts/
```

### Diff summary

```text
Frontend/public/dashboard-data.json                | 1054 ++++++-------
 Frontend/scripts/check-dashboard-data-contract.mjs |   17 +
 .../scripts/dashboard-publish-regression.test.mjs  |  219 +++
 Frontend/scripts/sync-signal-cascade-data.mjs      |  170 ++-
 Frontend/src/dashboard/DashboardPage.tsx           | 1098 ++++++++++----
 Frontend/src/dashboard/dashboard.css               | 1273 +++++++++++-----
 Frontend/src/dashboard/types.ts                    |   12 +
 Frontend/src/dashboard/view-model.ts               |  108 +-
 Frontend/src/index.css                             |  159 +-
 PyTorch/README.md                                  |   30 +-
 PyTorch/logic_multiframe_candlestick_model.md      |  495 ++----
 PyTorch/report_signalcascade_xauusd.md             |  115 +-
 .../application/artifact_provenance.py             |    2 +-
 .../signal_cascade_pytorch/application/config.py   |   72 +-
 .../application/diagnostics_service.py             |  467 +++++-
 .../application/inference_service.py               |   70 +-
 .../application/policy_service.py                  |   79 +-
 .../application/report_service.py                  |  468 +++++-
 .../application/training_service.py                |  307 +++-
 .../application/tuning_service.py                  |  741 ++++++++-
 PyTorch/src/signal_cascade_pytorch/bootstrap.py    |  279 +++-
 .../src/signal_cascade_pytorch/domain/entities.py  |    2 +
 .../infrastructure/ml/losses.py                    |    2 +
 .../infrastructure/ml/model.py                     |   40 +-
 .../src/signal_cascade_pytorch/interfaces/cli.py   |   95 ++
 PyTorch/tests/test_artifact_schema.py              |  242 ++-
 PyTorch/tests/test_policy_sweep.py                 |  168 ++-
 PyTorch/tests/test_policy_training.py              | 1575 +++++++++++++++++++-
 docs/README.md                                     |    2 +
 docs/UI_UX_DESIGN_RULES.md                         |   13 +-
 test-results/ui-quality/ui-quality-report.json     |   68 +-
 31 files changed, 7547 insertions(+), 1895 deletions(-)
```

### Recent commits

```text
6183a6d Add dashboard publish regression test
543dfb3 Fix dashboard artifact publish lineage
aacda1b Consolidate docs and sync tuned dashboard artifacts
47ee916 Align profit policy contracts and add tuning acceptance gate
d8ea766 Update bootstrap.py
```

## 4. 自動収集した evidence

### 既存 handoff / review ファイル

- PyTorch/reviewer_submission.md
- docs/implementation-tasks/archive/review/reviewer_submission.md
- PyTorch/review_feedback_comment.md
- PyTorch/review_followup_milestones.md
- docs/implementation-tasks/archive/review/reviewer_submission_forecast_accuracy.md
- docs/prompts/signalcascade-dashboard-uiux-prompt.md

### 変更ファイル

- `Frontend/public/dashboard-data.json`
- `Frontend/scripts/check-dashboard-data-contract.mjs`
- `Frontend/scripts/dashboard-publish-regression.test.mjs`
- `Frontend/scripts/sync-signal-cascade-data.mjs`
- `Frontend/src/dashboard/DashboardPage.tsx`
- `Frontend/src/dashboard/dashboard.css`
- `Frontend/src/dashboard/types.ts`
- `Frontend/src/dashboard/view-model.ts`
- `Frontend/src/index.css`
- `PyTorch/README.md`
- `PyTorch/logic_multiframe_candlestick_model.md`
- `PyTorch/report_signalcascade_xauusd.md`
- `PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py`
- `PyTorch/src/signal_cascade_pytorch/application/config.py`
- `PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
- `PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `PyTorch/src/signal_cascade_pytorch/domain/entities.py`
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
- `PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
- `PyTorch/tests/test_artifact_schema.py`
- `PyTorch/tests/test_policy_sweep.py`
- `PyTorch/tests/test_policy_training.py`
- `docs/README.md`
- `docs/UI_UX_DESIGN_RULES.md`
- `test-results/ui-quality/ui-quality-report.json`
- `PyTorch/review_feedback_comment.md`
- `PyTorch/review_followup_milestones.md`
- `PyTorch/reviewer_submission.md`
- `PyTorch/src/signal_cascade_pytorch/application/current_alias.py`
- `docs/prompts/`

### Review marker hits

- `docs/implementation-tasks/archive/review/reviewer_submission_forecast_accuracy.md:38` - `rg -n "TODO|FIXME|QUESTION|TBD|UNSURE" PyTorch Frontend` は、実コード / 実 artifact 側では open decision marker をほぼ持たない。hit は handoff 文面が中心
- `docs/implementation-tasks/archive/review/reviewer_submission.md:68` - `rg -n "TODO|FIXME|QUESTION|TBD|UNSURE" PyTorch Frontend` は、この handoff 自身以外の実コード / 実 artifact 側 hit なし
- `PyTorch/reviewer_submission.md:64` - 未解決マーカー検索 (`rg -n "TODO|FIXME|QUESTION|TBD|UNSURE" PyTorch Frontend docs`) の hit は archive handoff 文書だけで、実コード・current artifact 側では open...
- `PyTorch/reviewer_submission.md:206` ### `rg -n "TODO|FIXME|QUESTION|TBD|UNSURE" PyTorch Frontend docs`
- `PyTorch/reviewer_submission.md:209` /Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/implementation-tasks/archive/review/reviewer_submission_forecast_accuracy.md:38:-...
- `PyTorch/reviewer_submission.md:210` /Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/implementation-tasks/archive/review/reviewer_submission.md:68:- `rg -n "TODO|FIXM...

### Diagnostics / report / artifact 候補

- `PyTorch/artifacts/research_shrink_smoke/validation_summary.json`
- `PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge/validation_summary.json`
- `PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge_head/validation_summary.json`
- `PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability/validation_summary.json`
- `PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability_head/validation_summary.json`
- `PyTorch/artifacts/research_shrink_smoke_diag_edge_correctness_product_head/validation_summary.json`
- `PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/validation_summary.json`
- `PyTorch/artifacts/research_shrink_smoke_diag_selector_probability_head/validation_summary.json`
- `PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
- `PyTorch/artifacts/gold_xauusd_m30/archive/manual_current_before_checkpoint_audit_20260407T103412+0900/validation_summary.json`
- `PyTorch/artifacts/gold_xauusd_m30/archive/manual_current_before_notrade_focus_20260407T102046+0900/validation_summary.json`
- `PyTorch/artifacts/gold_xauusd_m30/archive/manual_current_before_qmax075_20260407T094905+0900/validation_summary.json`

## 5. レビューしてほしい論点

### A. 不明点とブロッカー

1) 手元にコード上の open 決定マーカーはほぼない。`PyTorch` と `Frontend` で `TODO|FIXME|QUESTION|TBD|UNSURE` を再走査した際のヒットは主に handoff 文書側のみで、実装自体に保留が残っている形跡は見えにくい。
2) そのため、現実的なブロッカーは「仕様上の合意不足（採択条件と UI 運用整合）」であり、未実装ではなく意思決定設計です。
3) ここが明示できない限り、`accepted_candidate` と `production_current` の不一致（現在は `accepted:candidate_05 / production:candidate_17`）は、再現可能な運用判断の観点で再発リスクが残る。
4) `session_20260407T111619Z` で `candidate_01` が `max_drawdown` でしか落ちない状態なのに `current_updated` が `false` のままであることが、実運用上の blocker になっている（現状継続の根拠不足）。

ブロッキング判定のための具体質問:
- `optimization_gate` の閾値を固定するか、リスク重視モードに対して閾値と user-value の階層を分離すべきか。
- `accepted_candidate` の更新条件（`optimization_gate_passed` の有無、`current_updated` を false のまま維持する場合のガード）を `dashboard` 側が想定しているか。
- 再学習の `candidate_limit` が 2 のままの短絡実行は、`current_updated=false` を長期化する設計リスクかどうか。`quick_mode` 実験が採択不能を見落とす可能性をどう避けるか。

### B. ボトルネック / failure mode 候補

1) 主要 failure mode は `PyTorch/src/signal_cascade_pytorch/application/inference_service.py` の数値変換層で、`current_close` と `forecast_mu` を通じた再現可能性が崩れた場合、短時間観測では「予測値だけが 20-30% 以上ズレる」シグナルが出る。

具体確認観点:
- `PREDICTION_SCHEMA_VERSION` / `FORECAST_SCHEMA_VERSION` が同一実行履歴で混在していないか（`serialize` と `load` 経路の前後差分）。
- `expected_return_pct = exp(mu)-1` を前提にした時、`predicted_closes` の逆変換 (`exp`) を二重適用しないこと。過去 1 行目〜3行目のデータ点で比較差分を確認。
- `median_*` 系の正規化 (`current_close / price_scale`) が `dashboard` 表示値と `prediction.current_close_display` と整合しているか。

### C. 設計で不安な点

1) `Frontend/scripts/check-dashboard-data-contract.mjs` は `manifestGeneratedAt` と `artifactId` の同一一致を必須化している。`latest` と `current` が同居する過渡期に、更新順序が逆転すると false negative になる可能性がある。`run.sampleCount` と `effectiveSampleCount` の整合判定も同時に監査するべき。
2) `PyTorch/src/signal_cascade_pytorch/application/config.py` での既定値（`checkpoint_selection_metric`, `policy_sweep_defaults`, `tie_policy_to_forecast_head`, `disable_overlay_branch`）は、4xデータ化後も初期値依存で最適化されてしまうリスクがある。`gold_xauusd_m30/current` の選抜実績と照合して、どこまで最適値として固定化するべきかを明示する必要がある。
3) `PyTorch/src/signal_cascade_pytorch/application/policy_service.py` の policy パスは `max_drawdown` を直接抑制しにくく、`cvar` や `turnover` で間接制御される構造。`policy_cost_multiplier`, `policy_gamma_multiplier`, `q_max`, `min_policy_sigma` の同時変更が実務上のリスク許容度を十分に表現しているか再確認が必要。
4) `PyTorch/src/signal_cascade_pytorch/bootstrap.py` と `PyTorch/src/signal_cascade_pytorch/interfaces/cli.py` の CLI 互換性設計は、廃止予定オプションを warning で無視する方式。自動実行パイプラインでの誤設定が静かに温存される可能性があるため、`deprecated flags` の排除可否と CI 失敗化のレベルを明確化すべき。
5) テスト層（`PyTorch/tests/test_artifact_schema.py`, `PyTorch/tests/test_policy_sweep.py`, `PyTorch/tests/test_policy_training.py`）は挙動を広く守る一方、`max_drawdown` 閾値変更時の仕様テストが薄い。今回のボトルネックは値そのものより “失敗時のフォールバック挙動” なので、`candidate_limit` 拡張前提の回帰テストを追加しないと再発しやすい。

### D. 数式 / 閾値 / 指標レビュー候補

#### 1) `Frontend/scripts/sync-signal-cascade-data.mjs`
- 前処理定数: `defaultLookbackDays = 360`。`SIGNAL_CASCADE_LOOKBACK_DAYS` が設定されていれば `resolveLookbackDays()` で上書き。
- 署名一致検証:
  - `numbersMatch(left,right)` は `abs(left-right) <= 1e-9`。
- 同一ラン生成時の必須整合チェック:
  - `prediction.current_close_display == source.meta.current_close`。
  - ダッシュボードの `diagnosticsGeneratedAt` と現行 `validation_summary.generated_at_utc` が一致。
  - `effective_price_scale` と `price_scale` が現行値と一致。
- 予測更新フロー（ライブ）:
  - `spawnSync` で `python ... cli.py tune-latest --artifact-root ... --csv ... --csv-lookback-days <days>` を実行。
  - `--csv-lookback-days` には `configuredLookbackDays`（初期 360）を渡す。
- データ品質ガード:
  - `normalizeDukascopyCsv` で `rows.length < 512` は例外。
- 時間差・鮮度フラグ:
  - `artifactLagHours = (maxTs-minTs)`（h）、`forecastAgeHours` / `diagnosticsAgeHours` / `predictionLagHours` はダッシュボード生成時点との差分（絶対値）h。
  - `deriveRunQuality`:
    - `interrupted_tuning` なら `degraded`
    - `artifactLagHours >= 24 or predictionLagHours >= 48` なら `stale`
    - それ以外 `fresh`

#### 2) `PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
- `CONFIG`: `CHECKPOINT_SELECTION_METRICS = ('hybrid_exact','exact_log_wealth','exact_log_wealth_minus_lambda_cvar','validation_total')`。
- 最適化ゲート（最終採択前の必須条件）は以下すべて `optimization_gate_then_user_value_score` 系で再評価。
  - `average_log_wealth >= 0.0`
  - `realized_pnl_per_anchor >= 0.0`
  - `cvar_tail_loss <= 0.08`
  - `max_drawdown <= 0.15`
  - `directional_accuracy >= 0.50`
  - `no_trade_band_hit_rate <= 0.80`
  - 失敗時 `optimization_gate_status=failed`。
- 候補生成と離散化:
  - 基本候補をベースに `epochs/learning_rate` を2〜4倍変化、`hidden_dim` ±16、`batch_size` 2倍/半分、`dropout` ±0.05＋`weight_decay` 変更、`evaluation_state_reset_mode` 切替、`policy_cost_multiplier`±2倍、`policy_gamma_multiplier`±2倍、`q_max ±0.25`、`cvar_weight ×0.5/×2`、`min_policy_sigma ×0.5/×2`。
  - 追加候補 `_FALLBACK_PARAMETERS` を含め、重複を除去。
  - 量子化後キー域: `batch_size∈{8,16,32,64}`、`hidden_dim∈[32,96]を16刻み`、`min_policy_sigma∈{5e-5,1e-4,2e-4,4e-4}`、`policy_multiplier∈{0.5,1,2,4,6}`、`q_max∈{0.5,0.75,1.0,1.25,1.5}`、`cvar_weight∈{0.05,0.1,0.2,0.4,0.8}`。
- クイックモード: `quick_mode=True` だと `candidate_limit` は既定 `min(len(candidate),4)`。
  - 優先順位: `tie_policy_to_forecast_head`/`disable_overlay_branch` の順（(T,F)->(F,T)->(T,T)）で重複除去。
- 選定:
  - `leaderboard` はゲート通過フラグを最優先し、その後
    1. `-blocked_objective_log_wealth_minus_lambda_cvar_mean`（fallback `project_value_score`）
    2. `-blocked_average_log_wealth_mean`（fallback `average_log_wealth`）
    3. `+blocked_turnover_mean`
    4. `-project_value_score`
    5. `-average_log_wealth`
    6. `-realized_pnl_per_anchor`
    7. `+cvar_tail_loss` `+max_drawdown` 等

#### 3) `PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- 1 サンプルの評価式:
  - `pnl = position * realized_return - trade_cost`
  - `trade_cost = selected_row.cost * abs(trade_delta)`
  - `equity += pnl`、`max_drawdown = peak_equity - equity`。
  - `log_wealth = log1p(clamp(pnl,-0.95,inf))`。
  - `cvar_tail = cvar_tail_loss(-pnl_tensor, alpha=config.cvar_alpha)`。
- 主要集計:
  - `average_log_wealth`, `realized_pnl_per_anchor`, `no_trade_rate`, `mu_calibration`, `sigma_calibration`, `direction_accuracy` などを全サンプル平均。
- スコア:
  - `project_value_score = 0.28*wealth_score + 0.20*cvar_score + 0.18*drawdown_score + 0.14*calibration_score + 0.10*activity_score + 0.10*gate_score`
    - `wealth_score = clamp(0.5 + 10*avg_log_wealth)`
    - `cvar_score = clamp(1 - 8*cvar_tail)`
    - `drawdown_score = clamp(1 - 8*max_drawdown)`
    - `calibration_score = clamp(1 - 12*mu_calibration)`
    - `activity_score = clamp(1 - no_trade_rate)`
    - `gate_score = clamp(shape_gate_usage)`
  - `utility_score = 0.40*wealth_score + 0.25*direction_accuracy + 0.20*(1-no_trade_rate) + 0.15*shape_gate_usage`
- チェックポイント選択:
  - `config.checkpoint_selection_metric == exact_log_wealth` → `-avg_log_wealth`
  - `... == exact_log_wealth_minus_lambda_cvar` → `-(avg_log_wealth - cvar_weight*cvar)`
  - `validation_total` → `forecast_mae`
  - それ以外: `-avg_log_wealth + forecast_w*forecast_mae + calib_w*0.5*(forecast_sigma) + posgap_w*exact_smooth_position_mae`
- `split_examples`:
  - `train_end = sample_count - desired_validation - purge`
  - `desired_validation = max(1, int(n*(1-train_ratio)))`、`purge = min(purge_examples, n - desired_validation -1)`。

#### 4) `PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- パス項:
  - `sigma_sq = sigma^2` を `min_policy_sigma^2` で下限。
  - `g_t = tradeability_gate`
  - `mu_t_tilde = g_t * mean`
  - `scaled_costs = costs * policy_cost_multiplier`
  - `effective_gamma = risk_aversion_gamma * policy_gamma_multiplier`
  - `margin = mu_t_tilde - effective_gamma * sigma_sq * previous_position`
- Exact policy（horizon別）:
  - `abs(margin) <= cost` → no-trade (`position = previous_position`)
  - `margin > 0` → `(mu_t_tilde - cost)/(gamma*sigma_sq)`
  - `margin < 0` → `(mu_t_tilde + cost)/(gamma*sigma_sq)`
  - `position = clip(position, -q_max, q_max)`。
- utility:
  - `U = gated_mean*position - 0.5*gamma*sigma_sq*position^2 - cost*abs(position-previous_position)`。
- Smooth policy:
  - `smooth_excess = softplus(|margin|-scaled_cost, beta)/beta`
  - `direction = tanh(margin / policy_abs_epsilon)`
  - `delta_pos = direction * smooth_excess / (gamma*sigma_sq)`
  - `raw_position = previous_position + delta_pos`
  - `horizon_positions = q_max * tanh(raw_position / q_max)`
  - `turnover = sqrt((q_i-q_prev)^2 + eps)`
  - 軟選択 `softmax(utilities*policy_smoothing_beta)`。

#### 5) `PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
- Schema: `PREDICTION_SCHEMA_VERSION = 7`, `FORECAST_SCHEMA_VERSION = 7`。
- 予測値変換:
  - `predicted_closes[h] = current_close * exp(forecast_mu[h])`
- `serialize/predict` で `median_predicted_*` 系を `current_close / price_scale` 換算して保持。
- レポート出力は `display forecast` と `policy driver` を明示して分離。
- 期待値は log-return なので、`expected_return_pct = exp(mu_t)-1`。

#### 6) `PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `DIAGNOSTICS_SCHEMA_VERSION = 10`、`POLICY_SELECTION_RULE_VERSION = 4`、`POLICY_SELECTION_OBJECTIVE_FLOOR_RATIO = 0.7`。
- blocked walk-forward:
  - 連続 fold 分割して、各 `evaluation_state_reset_mode` で平均 `average_log_wealth`/`cvar`/`turnover` などを計算。
  - `best_state_reset_mode_by_mean_log_wealth` を最大平均 log-wealth で決定。
- policy sweep:
  - 軸は `cost_multipliers / gamma_multipliers / min_policy_sigma / q_max / cvar_weight / state_reset_mode`。
  - 各組み合わせで `evaluate_model` と blocked 指標を再計算し `objective_log_wealth_minus_lambda_cvar = avg_log_wealth - cvar_weight*cvar_tail_loss`。
  - Pareto 判定は
    - wealth が上か同等、turnover が下か同等、cvarが下か同等
    - 少なくとも1指標が strictly better
  - non-Pareto を除外後、候補を `objective >= best_objective*0.7`（`best_objective<=0` の場合はそのまま）でフィルタし、
    `turnover` 小さい順→`objective`高い順→`avg_log_wealth`高い順→`cvar`低い順。

#### 7) `PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py`
- `ARTIFACT_SOURCE_SCHEMA_VERSION = 2`。
- `build_artifact_source_payload` が付与する主キー:
  - `artifact_id`（source payload を固定キー順 JSON 文字列化して SHA-256）
  - `artifact_schema_version`,`artifact_dir`,`generated_at_utc`,`state_reset_boundary_spec_version`
  - 可能時 `data_snapshot_sha256`,`config_sha256`,`config_origin`
  - git: `head/commit_sha/git_tree_sha`,`git_dirty`,`dirty_patch_sha256`
- `artifact_id` seed field（固定順 JSON）:
  - `artifact_kind`, `generated_at_utc`, `data_snapshot_sha256`, `config_sha256`, `parent_artifact_id`, `source_kind`, `source_path`, `price_scale`。

#### 8) `PyTorch/src/signal_cascade_pytorch/application/config.py`
- `CONFIG_SCHEMA_VERSION = 7`。
- `TrainingConfig` の主要既定値:
  - `tie_policy_to_forecast_head=False`, `disable_overlay_branch=False`
  - `cvar_weight=0.20`, `cvar_alpha=0.10`, `risk_aversion_gamma=3.0`
  - `policy_cost_multiplier=1.0`, `policy_gamma_multiplier=1.0`, `q_max=1.0`, `min_policy_sigma=1e-4`
  - `checkpoint_selection_metric='exact_log_wealth_minus_lambda_cvar'`
  - policy sweep defaults: cost `(0.5,1,2,4)`, gamma `(0.5,1,2)`, min_sigma `(5e-5,1e-4,2e-4)`, q_max `(1.0,)`, cvar `(0.2,)`
  - checkpoint重み: `checkpoint_selection_forecast_weight=1.0`, `...calibration_weight=0.5`, `...position_gap_weight=0.25`。
  - `CHECKPOINT_SELECTION_METRICS = ('hybrid_exact','exact_log_wealth','exact_log_wealth_minus_lambda_cvar','validation_total')`。

#### 9) `PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `METRICS_SCHEMA_VERSION = 4`, `ANALYSIS_SCHEMA_VERSION = 10`。
- `_resolve_checkpoint_audit` は以下を再計算可能な形でレポートへ持ち込み:
  - selection metric でソートした選択epoch、`exact_log_wealth` 及び `exact_log_wealth - lambda*cvar` の最大 epoch、各種 rank と `delta_to_best`。
  - ランクは `validation_selection_score` と `epoch` の昇順、`validation_exact_log_wealth` の降順で定義。
- `required` ファイル差し戻し: `validation_summary.json`, `policy_summary.csv`, `horizon_diag.csv` がないと `load_required_diagnostics_summary` が失敗。
- リポートの数式値は、validation 指標、policy sweep 選定結果、checkpoint audit、期待リターン `exp(mu)-1` を集約。

#### 10) `PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- トレーニング/予測/診断で `limit_base_bars_to_lookback_days(..., lookback_days)` を適用（CSV時）。
- `tune-latest` 時に `csv_lookback_days` を受け取り、`tune_latest_dataset` へ渡す。
- `_apply_selected_policy_calibration_payload` は既存 `validation_summary.policy_calibration_summary.selected_row` を読み取り以下を上書き:
  - `evaluation_state_reset_mode`, `min_policy_sigma`, `policy_cost_multiplier`, `policy_gamma_multiplier`, `q_max`, `cvar_weight`。
- `predict`/`diagnostic` でも同様に `lookback_days` を source payload に反映。
- `train_command` の最終出力に best/selection指標を出す（`best_validation_loss`, `average_log_wealth`, `cvar_tail_loss`, `policy_horizon`）。

#### 11) `PyTorch/src/signal_cascade_pytorch/domain/entities.py`
- `TrainingExample.state_features` の長さは `STATE_FEATURE_NAMES`（regime 5 + 5 volatility 系）固定。
- `state_features` 冒頭が `regime_features` と完全一致。
- `trend_strength == regime_features[4] == state_features[4]`。
- `realized_volatility == state_features[5]`。
- 不整合時は `__post_init__` で `ValueError`。

#### 12) `PyTorch/report_signalcascade_xauusd.md`
- 実運用の「監査入口」として、`PyTorch/artifacts/gold_xauusd_m30/current` を mirror 出力する実体。
- Contract は `current alias` が `mirror_of_current_research_report`。
- ここは数式定義より、直近 artifact の実測値確認先として使う（`artifact id`, `best_epoch`, `selection mode`, `best/accepted/production current`, 各種 validation 指標の実測値が逐次格納）。

### E. 検証・テスト・診断の不足候補

1) `PyTorch/artifacts/research_shrink_smoke/validation_summary.json`
   - 目的: quick_mode・非 quick_mode での `optimization_gate.status` と `passed_candidate_count` の差を確認し、`candidate_limit=2` の再現性を証明。
   - 必須チェック: `optimization_gate_failed_rules`, `current_updated`, `accepted_candidate`, `production_current_candidate`。
2) `PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge/validation_summary.json`
   - 目的: `policy_sweep` での Pareto フィルタと `POLICY_SELECTION_OBJECTIVE_FLOOR_RATIO=0.7` がリスク指標に過敏か鈍感かを確認。
   - 必須チェック: `policy_cost_multiplier` / `policy_gamma_multiplier` / `q_max` を変更した時の `max_drawdown` 感度。
3) `PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge_head/validation_summary.json`
   - 目的: `tie_policy_to_forecast_head=True` と `False` の差分を再現し、`candidate_03/04` 付近の極端値挙動の再発有無を追跡。
   - 必須チェック: `max_drawdown`, `blocked_turnover_mean`, `no_trade_band_hit_rate`。
4) `PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability/validation_summary.json`
   - 目的: `cvar_weight` 強化時に `project_value_score` と `user_value_score` の乖離を把握し、スコア設計の安定性を検証。
   - 必須チェック: `cvar_tail_loss`, `max_drawdown`, `user_value_score`。
5) `PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability_head/validation_summary.json`
   - 目的: `forecast_head` 固定時のリスクリワード分解を点検し、`forecast_stability` と `expected_return` の関係を定量化。
   - 必須チェック: `forecast_sigma`, `expected_return_pct`, `directional_accuracy`。
6) `PyTorch/artifacts/research_shrink_smoke_diag_edge_correctness_product_head/validation_summary.json`
   - 目的: 相関の強い feature のエッジケースで `selection_mode` がどれだけ安定しているか評価。
   - 必須チェック: `optimization_gate_passed`, `user_value_economic_score`, `blocked_objective_log_wealth_minus_lambda_cvar_mean`。
7) `PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/validation_summary.json`
   - 目的: `state_reset_mode` 切替の有効性を確認し、`best_state_reset_mode_by_mean_log_wealth` が `candidate` 全体で一致するかを検証。
   - 必須チェック: `evaluation_state_reset_mode` ごとの `average_log_wealth` / `cvar_tail_loss` / `max_drawdown`。
8) `PyTorch/artifacts/research_shrink_smoke_diag_selector_probability_head/validation_summary.json`
   - 目的: head 指定時の selection 収束性を確認。`quick_mode` で `candidate_limit` 拡大した場合の階層順位を比較。
   - 必須チェック: `leaderboard` 上位順（`optimization_gate_passed` → `user_value_score`）と `production_current_candidate`。
9) `PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
   - 目的: 現行運用の真の SoT。`accepted_candidate` と `production_current` の差分が `delta_production_minus_accepted` で許容されているかを監査。
   - 必須チェック: `selection_rule`, `override_reason`, `run.sampleCount` 系。
10) `PyTorch/artifacts/gold_xauusd_m30/archive/manual_current_before_checkpoint_audit_20260407T103412+0900/validation_summary.json`
    - 目的: checkpoint-audit 手順時の運用整合性を比較し、`checkpoint_audit` のみで改善するか、候補選択自体を触るべきかを分離判断。
    - 必須チェック: `checkpoint_selection_metric`, `best_checkpoint`, `delta_to_best`。
11) `PyTorch/artifacts/gold_xauusd_m30/archive/manual_current_before_notrade_focus_20260407T102046+0900/validation_summary.json`
    - 目的: `no_trade_rate` 重視の実験比較。実働率低下が `max_drawdown` 抑制に対してどれだけ効くかを定量化。
    - 必須チェック: `no_trade_band_hit_rate`, `no_trade_rate`, `average_log_wealth`。
12) `PyTorch/artifacts/gold_xauusd_m30/archive/manual_current_before_qmax075_20260407T094905+0900/validation_summary.json`
    - 目的: `q_max` 下方制約の有無で `max_drawdown` と `trade_delta` がどう収束するかを見る。
    - 必須チェック: `q_max`, `blocked_turnover_mean`, `max_drawdown`, `trade_delta`。

## 6. 参照ファイル / アーティファクト

最優先で読む順に用途を固定します。外部 reviewer は追加検索なしで以下だけで意思決定できる想定です。

1. `PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`  
   現行の `selection_mode` と `override_reason` を読む最上位監査対象。UI が参照している `production_current` / `accepted_candidate` のズレを確認。
2. `PyTorch/artifacts/gold_xauusd_m30/archive/session_20260407T111619Z/manifest.json`  
   4倍拡張（360日）後の再実行の失敗時挙動（`optimization_gate.status`, `passed_candidate_count`, `current_updated`）を検証。
3. `PyTorch/artifacts/gold_xauusd_m30/archive/session_20260407T111619Z/leaderboard.json`  
   `candidate_01`〜`candidate_02` の比較。`max_drawdown` 落ちで採択不能になった理由を数式で再現。
4. `Frontend/scripts/sync-signal-cascade-data.mjs`  
   再学習起動 (`--csv-lookback-days` 360) と契約検証（`lookback`、`manifestGeneratedAt`、`diagnosticsGeneratedAt`）を確認。
5. `Frontend/scripts/check-dashboard-data-contract.mjs`  
   同一 artifact 検証と stale 判定の厳密条件を確認。`dashboard-data` と API 履歴ズレの再現条件を追えるようにする。
6. `Frontend/scripts/dashboard-publish-regression.test.mjs`  
   360日再取得から公開までの回帰点検。`download retry`、`runQuality`、`sampleCount` 更新が失敗時にどこで落ちるかを評価。
7. `Frontend/src/dashboard/DashboardPage.tsx` と `Frontend/src/dashboard/dashboard.css`  
   UI 表示上の `productionCurrentCandidate` と `acceptedCandidate` が同一の監査根拠で読まれているかを確認。
8. `Frontend/public/dashboard-data.json`  
   現行表示状態の正本。`run.sampleCount=184` / `effectiveSampleCount=154` の stale 判定が 360日再実行結果と齟齬がないか検証。
9. `Frontend/scripts/sync-signal-cascade-data.mjs` / `Frontend/scripts/check-dashboard-data-contract.mjs` のテスト依存ログ  
   フロー全体が自動化されているか（取得→同期→契約チェック→保存）を追跡し、UI 未反映の再現手順を確定。
10. `PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`  
   `optimization_gate` / `leaderboard` / `user_value` の最終判断ロジック本体。今回の提案検証が失敗する/通る根拠の中心。

## 7. 期待する出力形式

1. 結論
2. 根拠付きの主要 findings
3. 優先度付きの変更提案
4. 検証または実験計画
5. 追加で見るべき指標やファイル
6. 最後に `ゴールまでのステップと現在地`

- `ゴールまでのステップと現在地` は 3-6 ステップ以内で簡潔に書く
- `現在地` は `step N/M` か同等の表現で示す
- `次の一手` を 1-2 個だけ添える

## 8. 制約

- 一般論ではなく、このファイルに記載した evidence を起点に判断する
- 確認済みの事実と仮説を分けて書く
- 可能なら cheap で information gain が高い順に提案する
- Codex がそのまま実装や追加検証に移れる粒度で提案する
- 回答の最後で、ゴール到達までの残ステップと現在地を必ず明示する

## 9. Codex 追記メモ

- ここから下は Codex が repo 固有の観測、メトリクス、意思決定依頼に置き換える
- 汎用文のまま残さず、外部 reviewer が迷わない状態まで具体化する

## 10. 4倍拡張後の次アクション計画（データを閉じる）

### 直近事実（2026-04-07）

- 360日拡張は反映済み。`Frontend/scripts/sync-signal-cascade-data.mjs` は `--csv-lookback-days 360` を既定化済み。
- `PyTorch/artifacts/gold_xauusd_m30/archive/session_20260407T111619Z/manifest.json` の再実行:
  - `lookback_days`: `360`
  - `sample_count`: `1387`
  - `source_rows_used`: `11568`
  - `candidate_limit`: `2`, `quick_mode`: `true`
  - `optimization_gate.status`: `failed`
  - `optimization_gate.passed_candidate_count`: `0`
  - `current_updated`: `false`
- 主要候補（同 session）:
  - `candidate_01`: `average_log_wealth=0.000050`, `directional_accuracy=0.5090`, `max_drawdown=0.2153`, `cvar_tail_loss=0.01345`, `optimization_gate_failed_rules=["max_drawdown>0.150000"]`
  - `candidate_02`: `average_log_wealth=-0.002786`, `directional_accuracy=0.5018`, `max_drawdown=2.9477`
- 画面側の現行反映は未更新。`Frontend/public/dashboard-data.json` は `run.sampleCount=184`, `run.effectiveSampleCount=154` のまま、`governance.productionCurrentCandidate="candidate_17"`, `governance.acceptedCandidate="candidate_05"` を維持。

### 目標

1. 4倍データの恩恵を活かしつつ、`max_drawdown<=0.15` を満たす選抜候補を得る
2. `accepted_candidate`（採択）と `production current`（表示）を同一基準で再整合
3. 外挿リスクが小さい形で `forecast` の暴れを抑制した後、UIへ反映

### 実行順（短期）

1. `PyTorch/src/signal_cascade_pytorch/application/tuning_service.py` の候補選抜に、`max_drawdown` を user-value と統合スコアへ加点/減点（ゲート待ちではなく主要スコア化）する。
2. 同一 `lookback 360` 下で、`q_max`, `policy_cost_multiplier`, `policy_gamma_multiplier`, `min_policy_sigma` を 3〜5点ずつに絞って `candidate_limit` 拡張（2→8 など）し、`quick_mode` と通常比較。
3. `PyTorch/src/signal_cascade_pytorch/application/config.py` / `cli.py` に `risk budget`（`max_drawdown` 優先度）を明示化。overlay off は現時点の証拠では改善に寄与せず（極端値や calibration崩れ）を避け、`overlay on` のデフォルトを維持。
4. 上記結果で `manifest.optimization_gate` と `current/source` が更新される場合のみ、`Frontend/public/dashboard-data.json` を更新配信。

### 判定が難しいポイント（外部 reviewer に閉じて依頼）

- `candidate_01` は収益系指標が合格域寄りでも `max_drawdown` が超過している。**どこまでを `blocked` 判定、どこを `user_value` の中核に残すか**。
- `candidate_02` の `directional_accuracy` は合格線を満たしつつ `max_drawdown` が巨大。**短期改善を追うより、指標設計を再配置すべきか**。
- dashboard は過去 artifact (`accepted_candidate: candidate_05`, `production: candidate_17`) を参照し続けており、360日再実行結果が未反映。**現実運用上の SoT をどこに固定するか**。
- `overlay off` 実験では `candidate_03`/`candidate_04` で極端値・calibration悪化が報告されており、`tie/overlay_policy` の切り替え基準を統一する必要がある。

### ゴールまでのステップと現在地（要約）

- 現在地: `step 1/4`（データ増設は完了、選抜の再設計が未着手）
- 次の一手: `tuning_service` の risk-aware user-value スコア改修 → `quick` で 8候補比較 → `manifest.current_updated` が true なら dashboard refresh

## 11. GPT-5.4 PRO レビュー依頼プロンプト（finance / programming / math）

以下を、下記の証拠を前提に、次の出力形式で返してください（要約文字数は 1500 字以内を推奨）。

### あなたへの依頼

- 目的: `SignalCascade` を「4倍データ化（360日）後」に再運用可能な選抜ロジックへ戻すため、採択条件・数式・実装順を最終決定したい。
- 対象:
  - `lookback` 拡張後の再学習選抜
  - `accepted` と `production current` の整合
  - `forecast` 偏差抑制の数式設計

### 現時点の確定データ

- 参照先: `PyTorch/artifacts/gold_xauusd_m30/archive/session_20260407T111619Z/manifest.json`

```json
{
  "session_id": "20260407T111619Z",
  "lookback_days": 360,
  "quick_mode": true,
  "candidate_limit": 2,
  "source_rows_used": 11568,
  "sample_count": 1387,
  "generated_candidate_count": 22,
  "evaluated_candidate_count": 2,
  "optimization_gate": {
    "status": "failed",
    "passed_candidate_count": 0,
    "failed_candidate_count": 2,
    "thresholds": {
      "average_log_wealth": {"minimum": 0.0},
      "realized_pnl_per_anchor": {"minimum": 0.0},
      "cvar_tail_loss": {"maximum": 0.08},
      "max_drawdown": {"maximum": 0.15},
      "directional_accuracy": {"minimum": 0.5},
      "no_trade_band_hit_rate": {"maximum": 0.8}
    }
  },
  "current_updated": false,
  "production_current_candidate": null,
  "accepted_candidate": null
}
```

- 参照先: `.../archive/session_20260407T111619Z/leaderboard.json`

```json
{
  "candidate": "candidate_01",
  "project_value_score": 0.5689966209539452,
  "average_log_wealth": 0.00005003631007994594,
  "directional_accuracy": 0.5090252707581228,
  "max_drawdown": 0.21529955756395358,
  "cvar_tail_loss": 0.013448311015963554,
  "no_trade_band_hit_rate": 0.0,
  "policy_horizon": 2,
  "tie_policy_to_forecast_head": false,
  "disable_overlay_branch": false,
  "optimization_gate_status": "failed",
  "optimization_gate_failed_rules": ["max_drawdown>0.150000"],
  "user_value_score": 0.40527051625580496,
  "user_value_chart_fidelity_score": 0.5943790780401461,
  "user_value_sigma_band_score": 0.9363354527577521,
  "user_value_execution_stability_score": 0.0,
  "user_value_economic_score": 0.47788289454927,
  "user_value_forecast_stability_score": 0.20237795196544225
}
```

```json
{
  "candidate": "candidate_02",
  "average_log_wealth": -0.0027861036821830885,
  "directional_accuracy": 0.5018050541516246,
  "max_drawdown": 2.9476748471115446,
  "cvar_tail_loss": 0.06960450857877731,
  "no_trade_band_hit_rate": 0.010830324909747292,
  "policy_horizon": 30,
  "tie_policy_to_forecast_head": true,
  "disable_overlay_branch": false,
  "optimization_gate_status": "failed",
  "optimization_gate_failed_rules": [
    "average_log_wealth<0.000000",
    "realized_pnl_per_anchor<0.000000",
    "max_drawdown>0.150000"
  ],
  "user_value_score": 0.27477202038610304
}
```

- 参照先: `PyTorch/artifacts/gold_xauusd_m30/archive/session_20260407T111619Z/candidate_01/source.json`

```json
{
  "lookback_days": 360,
  "artifact_kind": "training_run",
  "data_snapshot_row_count": 11568,
  "data_snapshot_start_timestamp": "2025-04-13T22:00:00+00:00",
  "data_snapshot_end_timestamp": "2026-04-07T09:30:00+00:00",
  "artifact_dir": "/Users/.../archive/session_20260407T111619Z/candidate_01",
  "generated_at_utc": "2026-04-07T11:46:44.191079+00:00"
}
```

- `Frontend/scripts/sync-signal-cascade-data.mjs` の再学習起動引数

```javascript
const defaultLookbackDays = 360
const configuredLookbackDays = resolveLookbackDays()
...
"tune-latest",
"--artifact-root",
artifactRoot,
"--csv",
csvPath,
"--csv-lookback-days",
String(configuredLookbackDays),
```

- `PyTorch/src/signal_cascade_pytorch/application/tuning_service.py` の選抜核（要点）

```python
_OPTIMIZATION_GATE_RULES = (
    {"metric": "average_log_wealth", "operator": "minimum", "threshold": 0.0},
    {"metric": "realized_pnl_per_anchor", "operator": "minimum", "threshold": 0.0},
    {"metric": "cvar_tail_loss", "operator": "maximum", "threshold": 0.08},
    {"metric": "max_drawdown", "operator": "maximum", "threshold": 0.15},
    {"metric": "directional_accuracy", "operator": "minimum", "threshold": 0.50},
    {"metric": "no_trade_band_hit_rate", "operator": "maximum", "threshold": 0.80},
)

def _build_user_value_metrics(candidate):
    chart_fidelity_score = _build_chart_fidelity_score(candidate)
    sigma_band_score = _inverse_linear_score(candidate.get('sigma_calibration'), upper_bound=0.20, default=0.5)
    execution_stability_score = _build_execution_stability_score(candidate)
    economic_score = _wealth_like_score(
        candidate.get('blocked_objective_log_wealth_minus_lambda_cvar_mean'),
        candidate.get('average_log_wealth')
    )
    forecast_stability_score = _build_forecast_stability_score(candidate)
    base_user_value_score = (
        0.55 * chart_fidelity_score +
        0.10 * sigma_band_score +
        0.20 * execution_stability_score +
        0.15 * economic_score
    )
    user_value_score = 0.70 * base_user_value_score + 0.30 * forecast_stability_score

def _build_execution_stability_score(candidate):
    drawdown_score = _inverse_linear_score(candidate.get('max_drawdown'), upper_bound=0.15, default=0.0)
    turnover_score = _inverse_linear_score(candidate.get('blocked_turnover_mean'), upper_bound=2.0, default=0.0)
    return 0.60 * drawdown_score + 0.40 * turnover_score

def _select_production_current_candidate(leaderboard):
    passed_rows = [row for row in leaderboard if bool(row.get('optimization_gate_passed'))]
    return min(
      passed_rows,
      key=lambda row: (
        -_metric_with_fallback(
          row,
          primary_key="user_value_score",
          fallback_key="project_value_score",
          default=0.0,
        ),
        -_metric_with_fallback(
          row,
          primary_key="user_value_chart_fidelity_score",
          fallback_key="project_value_score",
          default=0.0,
        ),
        -_metric_with_fallback(
          row,
          primary_key="user_value_forecast_stability_score",
          fallback_key="user_value_chart_fidelity_score",
          default=0.0,
        ),
        _ascending_metric(row, "blocked_exact_smooth_position_mae_mean"),
        _ascending_metric(row, "sigma_calibration"),
        _ascending_metric(row, "blocked_turnover_mean"),
        _descending_metric(
          row,
          "blocked_objective_log_wealth_minus_lambda_cvar_mean",
          "average_log_wealth",
        ),
        str(row.get("candidate", "")),
      ),
    )
```

- `PyTorch/artifacts/gold_xauusd_m30/current/source.json` の accepted と production current の直近整合値

```json
{
  "current_selection_governance": {
    "selection_mode": "auto_user_value_selection",
    "selection_rule": "optimization_gate_then_user_value_score",
    "accepted_candidate": { "candidate": "candidate_05" },
    "production_current": { "candidate": "candidate_17" },
    "override_applied": true,
    "override_reason": "production current prioritizes chart fidelity, sigma-band reliability, and execution stability over blocked-objective winner",
    "decision_summary": "production current differs from the accepted candidate because chart fidelity, sigma-band reliability, and execution stability took priority over blocked-objective rank.",
    "paired_frontier": {
      "delta_production_minus_accepted": {
        "user_value_score": 0.11152906561806619,
        "average_log_wealth": -0.00560152233659186,
        "blocked_objective_log_wealth_minus_lambda_cvar_mean": -0.006071749320060633,
        "max_drawdown": -0.07168563052007085,
        "blocked_turnover_mean": -1.2176364214344293,
        "trade_delta": -0.5779438111643571
      }
    }
  }
}
```

- `Frontend/public/dashboard-data.json` の現行公開値（未反映警告: stale）

```json
{
  "governance": {
    "selectionMode": "auto_user_value_selection",
    "productionCurrentCandidate": "candidate_17",
    "acceptedCandidate": "candidate_05"
  },
  "run": {
    "sampleCount": 184,
    "effectiveSampleCount": 154,
    "trainSamples": 118,
    "validationSamples": 36,
    "runQuality": "stale",
    "modelDirectory": "PyTorch/artifacts/gold_xauusd_m30/current"
  }
}
```

### 判断したい論点（優先順）

1. **金融側**: `max_drawdown<=0.15` 失敗が主要ボトルネックとみなされるのは妥当か。収益/方向精度とのトレードオフでどの閾値を採用すべきか。
2. **数学側**: `user_value_score` に実際のリスクをどう入れるべきか。候補スコアへ `max_drawdown` を一次項として入れるのか、`forecast_return_pct_jump` を加えるのか。
3. **実装側**: `accepted_candidate` と `production current` の選抜ルールを、どのようなガード付き階層（must pass gate / tie-break）で固定するのが再現性が高いか。

### 出力で必須

1. まず結論を 3 点以内（採用順）で提示
2. 次に「短期A/Bテスト3件」「再現性確認指標3件」を提示
3. 最後に `ゴールまでの残ステップ` を `step N/M` 形式で 4 行以内で示す
