# SignalCascade Review Handoff

最終更新: 2026-03-27 JST

## 1. あなたへの依頼

あなたは、時系列予測・Quant ML・stateful sequence modeling・risk-aware policy optimization に強い reviewer です。
`/Users/inouehiroshi/Documents/GitHub/SignalCascade` で進めている `shape-aware profit-maximization` 移行の「現実装」と、2026-03-27 JST 時点で更新した artifact / diagnostics / Frontend contract を review してください。

今回は abstract な移行方針ではなく、すでに入った実装と artifact を前提に、次を判断してほしいです。

- `shape-aware distribution + state + profit policy` 中心の現実装が、canonical spec `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/shape_aware_profit_maximization_model.md` に対して筋の良い first implementation か
- 継続して tuning / artifact 更新 / Frontend 連携を進めてよいか、それとも先に止血すべき blocker があるか
- cheap かつ information gain が高い順に、次の修正・実験を rank できるか

一般論ではなく、このファイルに書いた `file / artifact / metric / command / date` を根拠に回答してください。

## 2. ゴール

- `threshold / selector` 中心の旧経路から `stateful shape-aware profit policy` 中心の新経路へ移した実装の妥当性を確定する
- train / inference mismatch、state carry semantics、policy calibration、artifact contract のうち、どれを先に止血すべきかを切り分ける
- Codex が次ターンでそのまま実装・検証できる粒度で、優先順位付きの変更案を得る

## 3. 現状の重要観測

### A. Confirmed facts

- workspace は `/Users/inouehiroshi/Documents/GitHub/SignalCascade`、branch は `main`、`git rev-parse --is-inside-work-tree` は `true` です。
- 2026-03-27 JST 時点の `git status --short` は dirty で、変更ファイルは 23 件です。
- `git diff --stat` は `19 files changed, 3293 insertions(+), 8379 deletions(-)` です。
- recent commits は次です。
  - `c645e99 Document roadmap for profit-maximization migration`
  - `ba54089 Clarify proposal and acceptance diagnostics semantics`
  - `61329bf Align chart and decision panel heights`
  - `86f97b2 Clarify proposal and acceptance diagnostics semantics`
  - `b34cce2 Add no-candidate policy diagnostics and review handoff`
- `TODO / FIXME / XXX / HACK / REVIEW / QUESTION / WIP / TBD / BUG` の marker は、`PyTorch/src`、`Frontend/src`、`PyTorch/tests` では自動検出されていません。

#### 実装で実際に変わった責務

- `TrainingConfig` は profit policy 用の config surface を持ち、さらに state reset と evaluation sweep 用の config を持つようになっています。
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
  - 主要項目:
    - `warmup_epochs`
    - `state_dim`
    - `shape_classes`
    - `return_loss_weight`
    - `profit_loss_weight`
    - `cvar_weight`
    - `cvar_alpha`
    - `risk_aversion_gamma`
    - `q_max`
    - `policy_abs_epsilon`
    - `policy_smoothing_beta`
    - `min_policy_sigma`
    - `training_state_reset_mode`
    - `evaluation_state_reset_mode`
    - `diagnostic_state_reset_modes`
    - `policy_sweep_cost_multipliers`
    - `policy_sweep_gamma_multipliers`

- exact policy と smooth surrogate は、比較可能な形まで寄せられています。
  - files:
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
  - exact path:
    - `gated_mean = tradeability_gate * mean`
    - `margin = gated_mean - gamma * sigma^2 * previous_position`
    - `abs(margin) <= cost` なら no-trade
    - そうでなければ closed form で `position`
    - horizon ごとに `policy_utility` を計算し max を選択
  - smooth path:
    - `smooth_policy_distribution()` が `selected_horizon_index`, `selected_position`, `selected_utility`, `selected_no_trade` を返す
    - train 側でも `combined_position` に加えて `selected_position` を利用可能になっています
  - `cost_multiplier` / `gamma_multiplier` を exact / smooth 両方へ注入できるようになっています。

- training / evaluation / diagnostics は、state carry semantics を mode で切り替えられるようになっています。
  - files:
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
  - modes:
    - `carry_on`
    - `reset_each_example`
    - `reset_each_session_or_window`
  - `_should_reset_recurrent_context()` は `regime_id` の session prefix、日付、4h を超える gap を reset 境界に使っています。

- diagnostics / artifact は policy consistency と calibration sweep を export するようになっています。
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
  - 追加された主要出力:
    - `exact_smooth_horizon_agreement`
    - `exact_smooth_no_trade_agreement`
    - `exact_smooth_position_mae`
    - `exact_smooth_utility_regret`
    - `log_wealth_clamp_hit_rate`
    - `stateful_evaluation`
    - `policy_calibration_sweep`

- inference / forecast / report / Frontend contract は `schema_version=4` に上がっています。
  - files:
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/src/App.tsx`
  - 現在の整理内容:
    - `prediction.json schema_version = 4`
    - `forecast_summary.json schema_version = 4`
    - `metrics.json schema_version = 4`
    - `dashboard-data.json schemaVersion = 4`
    - `predicted_close_semantics = "median_from_log_return"`
    - `median_predicted_closes` を追加
    - Frontend の `run.policyState` は `run.policyStatus` に改名
    - dashboard 側で artifact schema version を保持

- dataset count semantics も明示されました。
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/metrics.json`
  - 現在は
    - `sample_count = 164`
    - `effective_sample_count = 161`
    - `purged_samples = 3`
  - です。以前の `train_samples + validation_samples != sample_count` の曖昧さは、少なくとも field 名としては解消されています。

#### 最新 artifact の具体値

- current artifact root:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`

- session manifest:
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/manifest.json`
  - `session_id=20260327T041009Z`
  - `generated_at=2026-03-27T04:18:04.395085+00:00`
  - `evaluated_candidate_count=3`
  - `interrupted_tuning=true`

- metrics:
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/metrics.json`
  - `schema_version=4`
  - `train_samples=129`
  - `validation_samples=32`
  - `sample_count=164`
  - `effective_sample_count=161`
  - `purged_samples=3`
  - `best_validation_loss=-0.392638826277107`
  - `best_epoch=6`
  - validation metrics:
    - `average_log_wealth=0.00430597409890736`
    - `realized_pnl_per_anchor=0.004429503809889926`
    - `turnover=8.776042206270933`
    - `max_drawdown=0.05190163901661479`
    - `cvar_tail_loss=0.021612780168652534`
    - `no_trade_band_hit_rate=0.0`
    - `log_wealth_clamp_hit_rate=0.0`
    - `expert_entropy=0.14246521890163422`
    - `shape_gate_usage=0.48368864227086306`
    - `mu_calibration=0.09901334740987168`
    - `sigma_calibration=0.05038660046955356`
    - `directional_accuracy=0.71875`
    - `exact_smooth_horizon_agreement=0.84375`
    - `exact_smooth_no_trade_agreement=1.0`
    - `exact_smooth_position_mae=0.2989177416440315`
    - `exact_smooth_utility_regret=0.0070845208398817375`
    - `policy_score_mean=0.041306069724057204`
    - `utility_score=0.6694646927362589`
    - `project_value_score=0.5711067832502575`
    - `state_reset_mode="carry_on"`
    - `cost_multiplier=1.0`
    - `gamma_multiplier=1.0`
    - `policy_horizon_distribution={"1": 0.78125, "3": 0.21875}`

- stateful evaluation:
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
  - `schema_version=4`
  - `diagnostics_schema_version=4`
  - `generated_at_utc=2026-03-27T06:27:55.599148+00:00`
  - `carry_on`
    - `average_log_wealth=0.00430597409890736`
    - `turnover=8.776042206270933`
    - `exact_smooth_position_mae=0.2989177416440315`
  - `reset_each_example`
    - `average_log_wealth=0.006099512001465342`
    - `turnover=30.390098959236806`
    - `exact_smooth_position_mae=0.7027209445377646`
  - `reset_each_session_or_window`
    - `average_log_wealth=0.006918165313578291`
    - `turnover=18.855870526655696`
    - `exact_smooth_position_mae=0.4735008496955382`

- evaluation-only policy calibration sweep:
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
  - baseline `cost_multiplier=1.0`, `gamma_multiplier=1.0`
    - `average_log_wealth=0.00430597409890736`
    - `turnover=8.776042206270933`
    - `cvar_tail_loss=0.021612780168652534`
    - `no_trade_band_hit_rate=0.0`
  - best `average_log_wealth` in current sweep:
    - `cost_multiplier=4.0`
    - `gamma_multiplier=2.0`
    - `average_log_wealth=0.005547629695651211`
    - `turnover=6.061850592707009`
    - `cvar_tail_loss=0.01681150309741497`
    - `no_trade_band_hit_rate=0.03125`
  - lower-cost/lower-gamma 側では turnover は増えやすく、たとえば `cost_multiplier=0.5`, `gamma_multiplier=0.5` では `turnover=13.975500186057811` です。

- latest prediction:
  - files:
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/prediction.json`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/forecast_summary.json`
  - main values:
    - `schema_version=4`
    - `predicted_close_semantics="median_from_log_return"`
    - `anchor_time=2026-03-24T20:00:00+00:00`
    - `policy_horizon=1`
    - `executed_horizon=1`
    - `position=0.18051757675864127`
    - `previous_position=0.0`
    - `trade_delta=0.18051757675864127`
    - `no_trade_band_hit=false`
    - `tradeability_gate=0.4837523400783539`
    - `shape_entropy=0.6356882452964783`
    - `policy_score=0.0014695710552295508`
    - `expected_log_returns={"1": 0.03489750623703003, "3": -0.022616222500801086}`
    - `median_predicted_closes={"1": 4577.961999976752, "3": 4322.094800221551}`
    - `horizon_utilities={"1": 0.0014695710552295508, "3": 0.00045480530109543365}`
    - `shape_probabilities["3"]=0.6801014542579651`

- Frontend payload:
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
  - `schemaVersion=4`
  - `generatedAt=2026-03-27T06:28:13.255Z`
  - `run.policyStatus="active"`
  - `run.selectedHorizon=1`
  - `artifacts.metricsSchemaVersion=4`
  - `artifacts.predictionSchemaVersion=4`
  - `artifacts.forecastSchemaVersion=4`

#### 2026-03-27 JST に実行した検証

- 成功:
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m unittest PyTorch.tests.test_policy_training PyTorch.tests.test_math_formulas PyTorch.tests.test_policy_consistency PyTorch.tests.test_policy_sweep PyTorch.tests.test_stateful_evaluation PyTorch.tests.test_artifact_schema`
    - 結果: `Ran 25 tests ... OK`
  - `cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend && npm run build`
    - 結果: build 成功
    - ただし Vite の chunk size warning が 1 件あります
  - handoff の source of truth に合わせるため、current artifact は再学習なしで refresh 済みです。
    - `metrics.json / prediction.json / forecast_summary.json` はすべて `schema_version=4`
    - `exact_smooth_*` 4 指標が `metrics.json` に存在することを確認済みです

### B. Working hypotheses / suspected gaps

- train / inference mismatch は「定量化できる状態」にはなりましたが、まだ無視できるとまでは言えません。
  - 根拠:
    - `exact_smooth_horizon_agreement=0.84375`
    - `exact_smooth_no_trade_agreement=1.0`
    - `exact_smooth_position_mae=0.2989177416440315`
    - `exact_smooth_utility_regret=0.0070845208398817375`
  - horizon と no-trade は比較的近い一方、position の乖離はまだ大きめに見えます。

- state carry の意味づけは、まだ reviewer judgment が必要です。
  - 根拠:
    - `carry_on` より `reset_each_session_or_window` の方が `average_log_wealth` は高い
    - ただし turnover は `8.776 -> 18.856` に増えます
    - `reset_each_example` はさらに `turnover=30.390` です
  - これは「carry が悪い」のではなく、「state dependence が metrics と activity を強く動かす」ことの証拠に見えます。

- policy calibration は、設計破綻より「未調整」の可能性が高く見えます。
  - 根拠:
    - baseline では `no_trade_band_hit_rate=0.0`
    - しかし sweep では `cost_multiplier=4.0`, `gamma_multiplier=2.0` で
      - `average_log_wealth` 改善
      - `turnover` 低下
      - `cvar_tail_loss` 改善
      - `no_trade_band_hit_rate=0.03125`
  - つまり cost / gamma 調整で observable behavior はかなり動きます。

- artifact contract は前進しましたが、まだ public contract を完全固定する段階かは未確定です。
  - `schema_version=4` と `median_predicted_closes` は良い整理です。
  - 一方で reviewer には、`predicted_close_semantics="median_from_log_return"` を今のまま public field として維持するか、expected price を別で出すべきかを見てほしいです。

- tuning の信頼性には依然として制約があります。
  - 根拠:
    - `validation_samples=32`
    - `evaluated_candidate_count=3`
    - `interrupted_tuning=true`
  - 今回の改善は前向きですが、探索の浅さをまだ排除できません。

## 4. レビューしてほしい論点

### 論点 1: exact / smooth mismatch は「止血対象」か「許容できる first slice」か

- reviewer には次を判断してほしいです。
  - `exact_smooth_horizon_agreement=0.84375` と `exact_smooth_position_mae=0.2989` を、現段階で blocker と見るか
  - 最小修正は何か
  - utility を soft-select する train surrogate として十分か、それともさらに exact policy に寄せるべきか

### 論点 2: state carry / reset semantics のどこを先に固定すべきか

- reviewer には次を見てほしいです。
  - `carry_on` を default に据えたまま先へ進めるべきか
  - `reset_each_session_or_window` の方が validation behavior として自然か
  - training 側でも同じ reset semantics を固定してから tuning を再開すべきか
  - sequence batching や truncated BPTT より先に、reset definition を固めるべきか

### 論点 3: cost / gamma / sigma floor の calibration は先に止血すべきか

- reviewer には次を rank してほしいです。
  - いま最も plausible な failure mode
  - cheapest high-information experiment
  - next gating metric
- とくに
  - baseline `no_trade_band_hit_rate=0.0`
  - sweep best `cost_multiplier=4.0`, `gamma_multiplier=2.0`
  - `turnover` と `cvar_tail_loss` の改善
  をどう読むべきかを知りたいです。

### 論点 4: artifact / Frontend contract をこのまま固めてよいか

- reviewer には次を判断してほしいです。
  - `schema_version=4` で当面十分か
  - `predicted_close_semantics="median_from_log_return"` と `median_predicted_closes` の naming が適切か
  - Frontend が今の時点で依存してよい field はどこまでか
  - legacy compatibility property を次の段階でどこまで削ってよいか

### 論点 5: 現在の metrics を「安定した進歩」と見てよいか

- current metrics は旧 threshold infeasibility 地獄からは明らかに前進しています。
- ただし今回は
  - `validation_samples=32`
  - `evaluated_candidate_count=3`
  - `interrupted_tuning=true`
  です。
- reviewer には次を rank してほしいです。
  - 最も plausible な failure mode
  - 今すぐやるべき cheap experiment
  - 3-seed / 3-split walk-forward に入る前の最小 gating 条件

## 5. 参照ファイル / アーティファクト

### Canonical spec / 設計文書

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/shape_aware_profit_maximization_model.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/profit_maximization_migration_roadmap.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/reviewer_submission.md`

### 実装

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/dataset_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/domain/entities.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`

### テスト

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_math_formulas.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_consistency.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_sweep.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_stateful_evaluation.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py`

### Downstream consumer

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/src/App.tsx`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`

### Artifact

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/manifest.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/config.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/metrics.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/prediction.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/forecast_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/policy_summary.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/horizon_diag.csv`

## 6. 期待する出力形式

次の順で返してください。

1. 結論
   - `このまま進めてよい / 修正してから進める / 一度戻す` のどれか
2. 主要 findings
   - 優先度順
   - 各 finding に根拠ファイルか artifact を必ず付ける
3. 最優先の修正 or 実験
   - cheap -> information gain の順で 3 件まで
4. 数式 / contract に対する sanity check
   - 問題なし / 要修正箇所
5. Codex が次ターンで着手しやすい具体タスク
   - 変更対象ファイル
   - 期待する確認コマンド

## 7. 制約

- live chat transcript は参照不要です。このファイルだけで判断してください。
- 確認済み fact と hypothesis を分けてください。
- 一般的な trading advice ではなく、実装と artifact の整合レビューに集中してください。
- metrics の信頼性が足りないと判断するなら、「何が足りないか」と「最小追加実験」をセットで書いてください。
- 再現コマンドは `PyTorch/.venv/bin/python` 前提で書いてください。
