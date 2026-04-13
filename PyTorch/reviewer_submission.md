# SignalCascade 外部研究機関レビュー依頼書

数値スナップショット基準:

- `current` artifact: 2026-04-13 08:18:41 JST
- dashboard publish: 2026-04-13 08:18:54 JST
- Git HEAD: `54f67c0e943202c7f0bb7207864ca27b99b8af14`
- Working tree: `dirty`

この文書は、外部 reviewer がこの 1 ファイルだけで判断できるように作った self-contained handoff です。会話履歴やリポジトリの live 状態を前提にしません。第 5 節にパスとコマンドを載せますが、それらは provenance 用であり、この文書だけでレビュー可能な粒度まで必要情報を展開しています。

## 1. あなたへの依頼

あなたは、時系列予測、forecast evaluation、Quant ML、walk-forward validation、checkpoint selection、artifact governance に強い外部研究 reviewer です。以下を判断してください。

1. 2026-04-13 JST 時点の `current` artifact / dashboard は、model judgment を再開してよい段階に達しているか。
2. まだ blocker があるなら、その主因は何か。特に次を順位付けしてください。
   - `selection_diagnostics` と `runtime_current` の semantic 混在
   - `objective / checkpoint selection / candidate ranking` の不整合
   - `data / split / walk-forward variance`
   - `model architecture limitation`
   - その他
3. `history summary` と `blocked walk-forward` を同じ表に置いた現状 evidence は、`all-horizon rank` 悪化と stability 劣化の共起を支持しているか。それとも別の failure mode が混在しているか。
4. 現在の `validation metrics` は future multi-horizon forecast quality の proxy と言ってよいか。`Yes / No / Partially` のいずれかで明示してください。
5. 次に 1 タスクだけ選ぶなら何か。
6. cheap で information gain が高い実験を最大 3 件まで挙げてください。

回答では、必ず `Confirmed facts` と `Inferences` を分けてください。最後に、短い `ゴールまでのステップと現在地` を付けてください。

## 2. ゴール

このレビューで達成したいゴールは 3 つです。

1. 現在の主 blocker が、artifact / semantic separation なのか、objective / evaluation alignment なのか、variance なのかを切り分ける。
2. `all-horizon` 側の ranking disagreement が、stability 悪化と同じ問題なのか、別問題なのかを判断する。
3. 開発側が次に 1 つだけ実装・検証すべきタスクを確定する。

## 3. 現状の重要観測

### 3.1 現在地の要約

前回の external review では、主な論点は次の順でした。

1. `objective / checkpoint selection / candidate ranking` の不整合
2. `artifact promotion / predict / runtime policy` の不整合
3. `data / split / walk-forward variance`
4. `model architecture limitation`

その後、開発側は structural remediation と semantic lane separation を進めました。2026-04-13 JST 時点の判断材料として重要なのは次です。

- structural lineage はかなり回復しています。`current/source.json`、`manifest.json`、`analysis.json`、top-level report mirror、dashboard provenance の `artifact_id` / `generated_at` は整合しており、dashboard contract も通っています。
- semantic lane separation も前進しています。dashboard `run.costMultiplier` は `0.5` の runtime lane を、dashboard `metrics.validation.costMultiplier` は `6.0` の selection lane を表示します。
- ただし、研究上の主論点は消えていません。最新 accepted / production candidate `candidate_03` は current ranking では 1 位ですが、`all-horizon` ranking では 12 位です。
- history summary でも、accepted candidate の `current` top-match rate は `1.0` なのに、`all-horizon` top-match rate は `0.25` です。
- 一方で、`all-horizon rank` の悪化が常に blocked stability 悪化と一緒に起きているわけではありません。少なくとも 2026-04-08 session では、`all-horizon rank = 12` にもかかわらず blocked turnover / blocked MAE / max drawdown は破綻的ではありません。
- 逆に 2026-04-07 04:58 UTC session では、production override が `all-horizon rank = 19` の candidate を採用しており、ここでは stability / execution を優先して `all-horizon` を犠牲にしているように見えます。

開発側の暫定認識は「step 1 の structural consistency はかなり完了し、いまの主論点は objective / evaluation alignment と variance である可能性が高い」です。ただし、この認識が妥当かを external reviewer に判定してほしいです。

### 3.2 Confirmed facts

#### A. structural lineage と contract

- workspace: `/Users/inouehiroshi/Documents/GitHub/SignalCascade`
- current artifact path: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`
- dashboard path: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
- current artifact `artifact_id`: `246338cc6042945c5ec5c54bf58edb2005b6efce9147bd9ac630287f8b8f2c66`
- `current/source.json.generated_at_utc`: `2026-04-12T23:18:41.316351+00:00`
- `dashboard-data.json.generatedAt`: `2026-04-12T23:18:54.787Z`
- dashboard provenance `artifactId`: `246338cc6042945c5ec5c54bf58edb2005b6efce9147bd9ac630287f8b8f2c66`
- dashboard provenance `manifestGeneratedAt`: `2026-04-12T23:18:41.316351+00:00`
- dashboard provenance `diagnosticsGeneratedAt`: `2026-04-12T23:18:41.316351+00:00`
- dashboard provenance `forecastGeneratedAt`: `2026-04-12T23:18:41.306113+00:00`
- dashboard provenance `predictionAnchorTime`: `2026-04-08T16:00:00+00:00`
- dashboard contract command `npm run check:data:contract` は `dashboard-data contract OK`

`current/source.json` の `current_alias_contract` は次を明示しています。

- authoritative runtime config は `current/config.json`
- diagnostic recommendation pointer は `validation_summary.json.policy_calibration_summary.selected_row`
- `selected_row` は runtime config そのものではなく diagnostic recommendation

つまり、少なくとも contract 文言レベルでは「runtime truth」と「diagnostic recommendation」を分ける方向に進んでいます。

#### B. runtime lane と selection lane

現在の semantic lane は次のように分離されています。

| 項目 | 値 | 役割 |
| --- | --- | --- |
| `validation_summary.json.runtime_current.operating_point.cost_multiplier` | `0.5` | authoritative runtime lane |
| `validation_summary.json.policy_calibration_summary.selected_row.cost_multiplier` | `0.5` | diagnostic recommendation row |
| `validation_summary.json.policy_calibration_summary.applied_runtime_policy.cost_multiplier` | `0.5` | applied runtime policy |
| `validation_summary.json.policy_calibration_summary.selected_row_matches_applied_runtime` | `true` | selected row と runtime は一致 |
| `validation_summary.json.selection_diagnostics.validation.cost_multiplier` | `6.0` | selection / validation lane |
| `metrics.json.validation_metrics.cost_multiplier` | `6.0` | selection / validation lane |
| dashboard `run.costMultiplier` | `0.5` | runtime lane |
| dashboard `metrics.validation.costMultiplier` | `6.0` | selection / validation lane |

この点は重要です。dashboard が runtime lane を誤読していた状態ではなくなりました。現状は、runtime lane と selection lane が同居しており、それぞれ別の役割を持つように修正されています。

ただし、もう 1 つの事実もあります。

- `archive/session_20260408T075001Z/manifest.json` の accepted / production candidate `candidate_03` は `policy_cost_multiplier = 6.0`
- 一方、current runtime operating point は `cost_multiplier = 0.5`

つまり、`accepted_candidate` の archived validation lane と、live runtime lane は完全同一ではありません。これは dashboard 誤表示の問題ではなく、post-selection calibration をどう解釈するかという研究上の意味論の問題です。

#### C. dataset と current validation snapshot

最新 current artifact の dataset / training / validation snapshot は次です。

| 項目 | 値 |
| --- | --- |
| `sample_count` | `1394` |
| `effective_sample_count` | `1364` |
| `train_samples` | `1086` |
| `validation_samples` | `278` |
| `purged_samples` | `30` |
| `source_rows_original` | `11620` |
| `source_rows_used` | `11620` |
| `best_validation_loss` | `0.07138540796774755` |
| `best_epoch` | `6` |
| `best_epoch_by_exact_log_wealth` | `1` |
| `best_epoch_by_exact_log_wealth_minus_lambda_cvar` | `6` |
| `best_epoch_by_blocked_objective_log_wealth_minus_lambda_cvar` | `6` |
| checkpoint `selection_metric` | `exact_log_wealth_minus_lambda_cvar` |
| checkpoint `selected_epoch_rank_by_exact_log_wealth` | `3` |
| checkpoint `delta_to_best_exact_log_wealth` | `0.0002330950982166235` |

validation metrics:

| metric | value |
| --- | --- |
| `average_log_wealth` | `0.000009859592128159684` |
| `realized_pnl_per_anchor` | `0.00001048864025035419` |
| `cvar_tail_loss` | `0.0018339508678764105` |
| `turnover` | `3.2506622293699365` |
| `max_drawdown` | `0.01945894954516857` |
| `directional_accuracy` | `0.5359712230215827` |
| `mu_calibration` | `0.05469737889884449` |
| `sigma_calibration` | `0.07639970218959329` |
| `utility_score` | `0.593646427127219` |
| `project_value_score` | `0.6793216717779011` |

補足:

- `analysis.dataset.sample_count = 1394`
- `validation_summary.runtime_current.dataset.sample_count = 1392`

この mismatch はまだ残っています。構造面がかなり整った一方で、contract が完全に単一化されたとは言い切れません。

#### D. blocked walk-forward（最新 current artifact）

`blocked_walk_forward_evaluation` の主要値は次です。

| state_reset_mode | average_log_wealth_mean | turnover_mean | directional_accuracy_mean | exact_smooth_position_mae_mean |
| --- | --- | --- | --- | --- |
| `carry_on` | `0.000007943755435908524` | `1.149111051561954` | `0.5358812529219261` | `0.011591466342301243` |
| `reset_each_session_or_window` | `-0.00015049554150420717` | `5.1744675653153145` | `0.5468676951846657` | `0.052278375931428755` |
| `reset_each_example` | `-0.0002844693788457254` | `8.740124548387044` | `0.5540751129811438` | `0.08806577714517834` |

Confirmed facts:

- best state reset mode by mean log wealth は `carry_on`
- latest accepted / production candidate `candidate_03` の blocked metrics は `carry_on` の集計に一致
- stability 指標だけを見ると `carry_on` が最も良い

#### E. current policy calibration row と current accepted candidate

現在の runtime row と accepted candidate の比較です。

| 項目 | current runtime / selected row | accepted candidate `candidate_03` |
| --- | --- | --- |
| `cost_multiplier` | `0.5` | `6.0` |
| `gamma_multiplier` | `4.0` | `4.0` |
| `q_max` | `0.75` | `0.75` |
| `blocked_objective_log_wealth_minus_lambda_cvar_mean` | `-0.0002738745891528441` | `-0.00033674398364080517` |
| `blocked_turnover_mean` | `1.363700414008238` | `1.149111051561954` |
| `blocked_exact_smooth_position_mae_mean` | `0.013786676977842235` | `0.011591466342301243` |
| `blocked_average_log_wealth_mean` | `0.00006238665968407059` | `0.000007943755435908524` |

Confirmed facts:

- runtime row (`0.5`) は blocked objective / blocked average_log_wealth では accepted candidate (`6.0`) より良い
- しかし blocked turnover と blocked MAE は accepted candidate (`6.0`) の方が良い
- つまり post-selection calibration は「objective 改善」と「stability 悪化」のトレードオフに見える

#### F. current session (`20260408T075001Z`) の ranking disagreement

current selection session の candidate ranking diagnostics は次です。

| 項目 | 値 |
| --- | --- |
| `candidate_count` | `22` |
| `current_top_candidate` | `candidate_03` |
| `selected_horizon_top_candidate` | `candidate_01` |
| `all_horizon_top_candidate` | `candidate_05` |
| `selected_horizon_vs_current_spearman_rank_correlation` | `0.5471485036702428` |
| `all_horizon_vs_current_spearman_rank_correlation` | `-0.050254093732354566` |
| `selected_horizon_vs_all_horizon_spearman_rank_correlation` | `0.47261434217955955` |
| `selected_horizon_top_k_overlap_with_current_count` | `2 / 3` |
| `all_horizon_top_k_overlap_with_current_count` | `0 / 3` |
| `accepted_candidate_current_rank` | `1` |
| `accepted_candidate_selected_horizon_rank` | `2` |
| `accepted_candidate_all_horizon_rank` | `12` |
| `production_current_current_rank` | `1` |
| `production_current_selected_horizon_rank` | `2` |
| `production_current_all_horizon_rank` | `12` |

Confirmed facts:

- current ranking と `all-horizon` ranking はほぼ無相関 (`rho ≈ -0.05`)
- accepted / production current `candidate_03` は `all-horizon` では 12 位
- top-3 overlap も current vs `all-horizon` で `0`

#### G. session history summary（archive 13 session）

`current/analysis.json.selection_history_summary` の aggregate 値は次です。

| 項目 | 値 |
| --- | --- |
| `session_count` | `13` |
| `accepted_candidate_count` | `12` |
| `production_current_candidate_count` | `2` |
| `accepted_vs_production_divergence_count` | `1` |
| `accepted_current_top_match_ratio` | `1.0` |
| `accepted_selected_horizon_top_match_ratio` | `0.375` |
| `accepted_all_horizon_top_match_ratio` | `0.25` |
| `accepted_candidate_current_rank_median` | `1.0` |
| `accepted_candidate_selected_horizon_rank_median` | `2.0` |
| `accepted_candidate_all_horizon_rank_median` | `2.5` |
| `production_current_top_match_ratio` | `0.5` |
| `production_selected_horizon_top_match_ratio` | `0.0` |
| `production_all_horizon_top_match_ratio` | `0.0` |
| `production_current_current_rank_median` | `1.5` |
| `production_current_selected_horizon_rank_median` | `5.5` |
| `production_current_all_horizon_rank_median` | `15.5` |

Confirmed facts:

- accepted candidate は current ranking では常に 1 位
- accepted candidate は `all-horizon` では top 一致が 25% に落ちる
- production current は、観測できる 2 session では `all-horizon` で一度も 1 位になっていない

#### H. `all-horizon rank` と blocked metrics を並べた session 表

下表は、archive session ごとに accepted / production candidate の rank と blocked metrics を並べたものです。`FULL` は `selected-horizon` / `all-horizon` rank を backfill できた session、`PARTIAL` は古い artifact で raw diagnostics が不足し full backfill できない session です。

```text
FULL
| session_id | selection_mode | accepted_candidate | production_current_candidate | acc_cur | acc_sel | acc_all | acc_obj | acc_turn | acc_dir | acc_mae | acc_mdd | prod_cur | prod_sel | prod_all | prod_turn | prod_mae | rho_sel_cur | rho_all_cur |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260406T053046Z | - | candidate_03 | - | 1 | 1 | 2 | - | - | - | - | 0.1468 | - | - | - | - | - | -0.1167 | 0.7667 |
| 20260406T064716Z | - | candidate_06 | - | 1 | 2 | 7 | - | - | - | - | 0.0461 | - | - | - | - | - | 0.1500 | 0.2667 |
| 20260406T073432Z | - | candidate_06 | - | 1 | 5 | 2 | - | - | - | - | 0.0359 | - | - | - | - | - | 0.4833 | -0.0667 |
| 20260406T130645Z | - | candidate_01 | - | 1 | 5 | 1 | - | - | - | - | 0.0359 | - | - | - | - | - | 0.2333 | 0.6000 |
| 20260407T034224Z | - | candidate_03 | - | 1 | 1 | 4 | 0.0037 | 0.9598 | 0.5833 | 0.1960 | 0.1159 | - | - | - | - | - | 1.0000 | -0.8000 |
| 20260407T040723Z | - | candidate_04 | - | 1 | 4 | 3 | 0.0009 | 0.2098 | 0.7778 | 0.0172 | 0.0142 | - | - | - | - | - | -0.4000 | -0.4000 |
| 20260407T041853Z | auto_user_value_selection | candidate_05 | candidate_17 | 1 | 1 | 1 | 0.0073 | 1.4590 | 0.7222 | 0.1957 | 0.0887 | 2 | 9 | 19 | 0.2413 | 0.0188 | 0.0152 | -0.2648 |
| 20260408T075001Z | accepted_candidate | candidate_03 | candidate_03 | 1 | 2 | 12 | -0.0003 | 1.1491 | 0.5359 | 0.0116 | 0.0195 | 1 | 2 | 12 | 1.1491 | 0.0116 | 0.5471 | -0.0503 |

PARTIAL
- 20260405T203248Z: accepted=candidate_02, prod=-, acc_cur=1, acc_all=-, max_drawdown=0.0806
- 20260406T012659Z: accepted=candidate_01, prod=-, acc_cur=1, acc_all=-, max_drawdown=0.0663
- 20260406T035431Z: accepted=candidate_02, prod=-, acc_cur=1, acc_all=-, max_drawdown=0.0416
- 20260406T041657Z: accepted=candidate_01, prod=-, acc_cur=1, acc_all=-, max_drawdown=0.0416
- 20260407T111619Z: accepted=-, prod=-, acc_cur=-, acc_all=-, max_drawdown=-
```

Confirmed facts from this table:

- `20260407T041853Z` は明確な divergence session
  - accepted `candidate_05` は `acc_all_rank = 1`
  - production `candidate_17` は `prod_all_rank = 19`
  - ただし production `candidate_17` は `blocked_turnover = 0.2413`、`blocked_exact_smooth_position_mae = 0.0188` と、accepted `candidate_05` (`1.4590`, `0.1957`) より stability / execution 面で大幅に良い
- `20260408T075001Z` は別タイプの session
  - accepted / production `candidate_03` は `acc_all_rank = prod_all_rank = 12`
  - しかし blocked metrics は `turnover = 1.1491`、`blocked_exact_smooth_position_mae = 0.0116`、`max_drawdown = 0.0195`
  - `all-horizon rank` は悪いが、stability 指標は catastrophic ではない

#### I. 検証とテスト

2026-04-13 JST に実行した主要コマンドと結果は次です。

```bash
PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m unittest \
  PyTorch.tests.test_policy_training.PolicyAndTrainingTests.test_tune_latest_dataset_updates_current_with_deployment_score_override \
  PyTorch.tests.test_artifact_schema.ArtifactSchemaTests.test_generate_research_report_backfills_forecast_quality_ranking_diagnostics_from_leaderboard \
  PyTorch.tests.test_artifact_schema.ArtifactSchemaTests.test_predict_cli_refreshes_current_artifact_contract_outputs \
  PyTorch.tests.test_artifact_schema.ArtifactSchemaTests.test_build_current_alias_metadata_backfills_forecast_quality_from_candidate_diagnostics
```

結果:

- `Ran 4 tests in 0.390s`
- `OK`

artifact refresh:

```bash
PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli predict \
  --output-dir PyTorch/artifacts/gold_xauusd_m30/current \
  --csv PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv
```

最新 predict 出力:

- `policy horizon: 1`
- `executed horizon: 1`
- `position: -0.0997`
- `trade delta: -0.0585`
- `g_t: 0.4923`
- `selected policy utility: 0.027053`
- `no-trade-band hit: False`

dashboard publish / contract:

```bash
SIGNAL_CASCADE_DISABLE_TRAINING=1 SIGNAL_CASCADE_DISABLE_LIVE_SYNC=1 node ./scripts/sync-signal-cascade-data.mjs
npm run check:data:contract
```

結果:

- `Wrote Frontend/public/dashboard-data.json`
- `dashboard-data contract OK`

#### J. touched code に open marker はほぼない

`rg -n "TODO|FIXME|QUESTION|TBD|UNSURE" PyTorch/src/signal_cascade_pytorch PyTorch/tests Frontend/scripts`

この検索では、今回の touched runtime code / test code に open decision marker は見当たりませんでした。未解決論点は inline TODO ではなく、artifact semantics と evaluation interpretation の側にあります。

### 3.3 Inferences / working hypotheses

以下は confirmed fact ではなく、開発側の作業仮説です。これが妥当かどうかを external reviewer に判定してほしいです。

1. structural consistency はかなり回復しており、前回 blocker の中心だった「dashboard が runtime lane を誤表示している」という問題は主因ではなくなった可能性が高い。
2. いまの主論点は「semantic lane separation が未完了」というより、「runtime lane と selection lane をどう研究解釈すべきか」、および「objective / checkpoint selection / candidate ranking の alignment が崩れているか」に移っている可能性が高い。
3. `all-horizon rank` の悪化は、単純な stability deterioration と 1 対 1 では対応していない。
   - `20260407T041853Z` は stability 優先 override の事例に見える
   - `20260408T075001Z` は stability が比較的まともでも `all-horizon rank = 12` まで落ちる事例に見える
   - つまり少なくとも 2 種類の failure mode が混在している可能性がある
4. 最新 session では `current_top_candidate = candidate_03`、`all_horizon_top_candidate = candidate_05`、`all_vs_current_spearman ≈ -0.05` なので、current selection contract は `all-horizon` quality をほぼ反映していない可能性がある。
5. ただし、archive 13 session のうち early session の一部は `selected-horizon` / `all-horizon` rank を full backfill できておらず、history summary 全体の解釈には coverage limitation がある。

## 4. レビューしてほしい論点

### 論点 1. いまの primary blocker は何か

候補を順位付けし、各項目に confidence を付けてください。

- `selection_diagnostics` と `runtime_current` の semantic 混在
- `objective / checkpoint selection / candidate ranking` の不整合
- `data / split / walk-forward variance`
- `model architecture limitation`
- `artifact consistency` の未完了
- その他

特に知りたいのは、前回 review で優先度が高かった artifact consistency 問題が、今回の evidence ではどこまで下がったかです。

### 論点 2. semantic separation は十分に実装されたか

Confirmed facts:

- dashboard `run.costMultiplier = 0.5`
- dashboard `metrics.validation.costMultiplier = 6.0`
- `runtime_current.operating_point.cost_multiplier = 0.5`
- `policy_calibration_summary.selected_row.cost_multiplier = 0.5`
- `selected_row_matches_applied_runtime = true`
- archived accepted candidate config は `policy_cost_multiplier = 6.0`

ここから次のどれと読むべきかを判断してください。

1. semantic separation は実装として十分で、次の主論点は研究契約に移った
2. runtime lane と accepted candidate archive lane の乖離がなお blocker
3. まだ別の semantic 混在が残っている

### 論点 3. `all-horizon rank` 悪化と stability 劣化の共起は本当にあるか

上の session table を見て、次を判断してください。

1. `all-horizon rank` 悪化は blocked stability 悪化と同じ現象か
2. それとも `20260407T041853Z` のような deployment override と、`20260408T075001Z` のような objective / evaluation mismatch は別現象か
3. 開発側が次に組むべき表や実験は、どの列を足せば最も識別力が高いか

特に、下の 2 session を比較対象として重視してください。

- `20260407T041853Z`
  - accepted `candidate_05`: `acc_all_rank = 1`, `blocked_turnover = 1.4590`, `blocked_exact_smooth_position_mae = 0.1957`, `max_drawdown = 0.0887`
  - production `candidate_17`: `prod_all_rank = 19`, `blocked_turnover = 0.2413`, `blocked_exact_smooth_position_mae = 0.0188`, `max_drawdown = 0.0170`
- `20260408T075001Z`
  - accepted / production `candidate_03`: `acc_all_rank = 12`
  - ただし `blocked_turnover = 1.1491`, `blocked_exact_smooth_position_mae = 0.0116`, `max_drawdown = 0.0195`

### 論点 4. `validation metrics` は future multi-horizon quality の proxy か

開発側は現時点で `No` 寄りです。その根拠は次です。

- current selection session で `accepted_candidate_all_horizon_rank = 12`
- history summary で `accepted_all_horizon_top_match_ratio = 0.25`
- current validation metrics 自体は catastrophic ではない
- checkpoint selection metric は `exact_log_wealth_minus_lambda_cvar`
- selected epoch は `6`、best epoch by exact_log_wealth は `1`

この材料から、`validation metrics` を future multi-horizon forecast quality の proxy と見なしてよいかを判定してください。

### 論点 5. cheap experiment を 3 件以内で選んでほしい

開発側が候補として考えているのは次です。順位付けと、各実験の expected information gain を示してください。

1. `accepted_candidate` と `production_current` について、`all-horizon rank`, `blocked objective`, `blocked turnover`, `blocked exact_smooth_position_mae`, `max_drawdown`, `deployment_score`, `user_value_score` を session 横断で一枚の scorecard に固定し、divergence pattern を cluster 化する
2. 最新 session `20260408T075001Z` を `cost_multiplier = 0.5` と `6.0` の両 operating point で同一 candidate に対して再評価し、ranking / blocked metrics / displayed forecast の差を比較する
3. walk-forward fold 数か validation span を増やして、`all-horizon rank` と `blocked objective rank` の安定性を再評価する
4. architecture ablation

開発側の暫定順位は `1 > 2 > 3 > 4` です。これが妥当か見てください。

## 5. 参照ファイル / アーティファクト

この節は provenance 用です。レビューのために開く必要はありませんが、必要なら audit trail として使ってください。

- repo root: `/Users/inouehiroshi/Documents/GitHub/SignalCascade`
- current artifact directory: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`
- current `analysis.json`: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/analysis.json`
- current `validation_summary.json`: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
- current `metrics.json`: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/metrics.json`
- current `source.json`: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/source.json`
- top-level report mirror: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/report_signalcascade_xauusd.md`
- dashboard payload: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
- latest accepted session manifest: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/archive/session_20260408T075001Z/manifest.json`
- latest accepted session leaderboard: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/archive/session_20260408T075001Z/leaderboard.json`
- key divergence session manifest: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/archive/session_20260407T041853Z/manifest.json`
- key divergence session leaderboard: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/archive/session_20260407T041853Z/leaderboard.json`
- runtime refresh implementation: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- diagnostics lane implementation: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- current alias / governance snapshot: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/current_alias.py`
- ranking diagnostics implementation: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/selection_rank_diagnostics.py`
- tuning / session manifest writer: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
- report builder: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- dashboard publish: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
- dashboard contract: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/check-dashboard-data-contract.mjs`
- relevant tests:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`

## 6. 期待する出力形式

以下の順序で回答してください。

1. `結論`
   - 3 行以内で、いま model judgment を再開してよいか、主 blocker は何か、次の 1 タスクは何かを明示してください。
2. `Confirmed facts`
   - この文書にある evidence だけに基づく事実を列挙してください。
3. `Inferences`
   - 事実から導いた仮説・解釈を列挙してください。
4. `failure cause の順位付け`
   - 候補を優先度順に並べ、各項目に confidence を付けてください。
5. `cheap experiment`
   - 最大 3 件。期待できる information gain と、その実験で何が falsify / confirm されるかを書いてください。
6. `validation metrics は future multi-horizon quality の proxy か`
   - `Yes / No / Partially` のいずれかを明示し、理由を書いてください。
7. `ゴールまでのステップと現在地`
   - 3〜6 ステップ以内
   - 現在地を `step N/M` などで明示
   - immediate next action を 1 つに絞る

## 7. 制約

- この文書だけでレビューしてください。Section 5 のパス参照は audit trail 用であり、開くことを前提にしません。
- 一般論ではなく、この文書にある数値・表・コマンド・artifact snapshot を起点に判断してください。
- 事実と仮説を必ず分けてください。
- 不足 evidence があるなら、「何が不足しているためどこまでしか言えないか」を明示してください。
- cheap experiment は、開発側がそのまま着手できる粒度まで具体化してください。
- 日付は絶対日付で書いてください。

## 付録 A. semantic lane separation の実装要点

runtime lane と selection lane は現在、少なくとも payload 上は別物として持っています。実装の骨子は次です。

`bootstrap.py`

```py
forecast_quality_scorecards = _build_forecast_quality_scorecards(...)
refreshed_summary["forecast_quality_scorecards"] = forecast_quality_scorecards
refreshed_summary["selection_diagnostics"] = _build_selection_diagnostics_payload(...)
refreshed_summary["runtime_current"] = _build_runtime_current_payload(...)
```

`report_service.py`

```py
forecast_quality_ranking_diagnostics = _load_forecast_quality_ranking_diagnostics(
    current_selection_governance
)
selection_history_summary = build_selection_history_summary(
    _resolve_archive_root(output_dir, current_selection_governance)
)
```

`selection_rank_diagnostics.py`

```py
resolved_diagnostics = build_forecast_quality_ranking_diagnostics(
    backfill_forecast_quality_metrics_from_session(
        leaderboard_rows,
        leaderboard_path.parent,
    ),
    accepted_candidate=accepted_candidate,
    production_current_candidate=production_current_candidate,
)
```

Confirmed facts:

- 旧 session manifest に `forecast_quality_ranking_diagnostics` が無い場合でも、report は `leaderboard.json` と candidate diagnostics から backfill している
- したがって、history summary は retrospective reconstruction を含む

## 付録 B. current runtime / selection payload の exact values

```json
{
  "artifact_id": "246338cc6042945c5ec5c54bf58edb2005b6efce9147bd9ac630287f8b8f2c66",
  "source_generated_at_utc": "2026-04-12T23:18:41.316351+00:00",
  "dashboard_generated_at": "2026-04-12T23:18:54.787Z",
  "selected_row_key": "state_reset_mode=carry_on|cost_multiplier=0.5|gamma_multiplier=4|min_policy_sigma=0.0001|q_max=0.75|cvar_weight=0.2",
  "selected_row_matches_applied_runtime": true,
  "runtime_current_cost_multiplier": 0.5,
  "selection_validation_cost_multiplier": 6.0,
  "dashboard_run_costMultiplier": 0.5,
  "dashboard_validation_costMultiplier": 6,
  "dashboard_artifact_lag_hours": 103.32
}
```

## 付録 C. current latest candidate `candidate_03` の exact snapshot

latest accepted / production current `candidate_03` の主要値:

```json
{
  "candidate": "candidate_03",
  "policy_cost_multiplier": 6.0,
  "policy_gamma_multiplier": 4.0,
  "q_max": 0.75,
  "average_log_wealth": 0.000009859592128159684,
  "blocked_objective_log_wealth_minus_lambda_cvar_mean": -0.00033674398364080517,
  "blocked_turnover_mean": 1.149111051561954,
  "blocked_directional_accuracy_mean": 0.5358812529219261,
  "blocked_exact_smooth_position_mae_mean": 0.011591466342301243,
  "max_drawdown": 0.01945894954516857,
  "selected_horizon_forecast_quality_score": 0.6215998094867489,
  "all_horizon_forecast_quality_score": 0.4260937384640033,
  "accepted_candidate_current_rank": 1,
  "accepted_candidate_selected_horizon_rank": 2,
  "accepted_candidate_all_horizon_rank": 12
}
```

## 付録 D. 検証コマンド

```bash
PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m unittest \
  PyTorch.tests.test_policy_training.PolicyAndTrainingTests.test_tune_latest_dataset_updates_current_with_deployment_score_override \
  PyTorch.tests.test_artifact_schema.ArtifactSchemaTests.test_generate_research_report_backfills_forecast_quality_ranking_diagnostics_from_leaderboard \
  PyTorch.tests.test_artifact_schema.ArtifactSchemaTests.test_predict_cli_refreshes_current_artifact_contract_outputs \
  PyTorch.tests.test_artifact_schema.ArtifactSchemaTests.test_build_current_alias_metadata_backfills_forecast_quality_from_candidate_diagnostics

PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli predict \
  --output-dir PyTorch/artifacts/gold_xauusd_m30/current \
  --csv PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv

SIGNAL_CASCADE_DISABLE_TRAINING=1 SIGNAL_CASCADE_DISABLE_LIVE_SYNC=1 \
  node ./scripts/sync-signal-cascade-data.mjs

npm run check:data:contract
```

結果要約:

- unit test 4 件 `OK`
- `predict` refresh 成功
- dashboard publish 成功
- dashboard contract `OK`

## 付録 E. reviewer に特に見てほしい 2 つの対比

### 1. stability tradeoff 型の divergence

`20260407T041853Z`

- accepted `candidate_05`
  - `acc_all_rank = 1`
  - `blocked_objective = 0.0073`
  - `blocked_turnover = 1.4590`
  - `blocked_exact_smooth_position_mae = 0.1957`
  - `max_drawdown = 0.0887`
- production `candidate_17`
  - `prod_all_rank = 19`
  - `blocked_objective = 0.0012`
  - `blocked_turnover = 0.2413`
  - `blocked_exact_smooth_position_mae = 0.0188`
  - `max_drawdown = 0.0170`

この session は「all-horizon quality を捨てて execution stability を取る」型に見えます。

### 2. latest session の objective / evaluation mismatch らしさ

`20260408T075001Z`

- accepted / production current `candidate_03`
  - `current rank = 1`
  - `selected-horizon rank = 2`
  - `all-horizon rank = 12`
  - `blocked_objective = -0.0003367`
  - `blocked_turnover = 1.1491`
  - `blocked_exact_smooth_position_mae = 0.0116`
  - `max_drawdown = 0.0195`

この session は「stability は崩れていないのに all-horizon rank が悪い」型に見えます。

これが stability の問題ではなく objective / evaluation alignment の問題かどうかを判定してください。
