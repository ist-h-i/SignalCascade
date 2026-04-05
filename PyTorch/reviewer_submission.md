# SignalCascade Review Handoff

最終更新: 2026-04-05 18:37:00 JST

## 1. あなたへの依頼

あなたは、時系列予測・Quant ML・stateful sequence modeling・risk-aware policy optimization に強い reviewer です。  
`/Users/inouehiroshi/Documents/GitHub/SignalCascade` の current code、既存 artifact、frontend public payload を突き合わせて、次に進むための判断をしてください。

今回は一般論ではなく、次の 3 点を決めてほしいです。

- P0 実装後の versioned serialization / replay contract は、authoritative rerun に進める水準まで閉じたか
- `reset_each_session_or_window` を primary semantic に進めず、`carry_on` を default 維持にした判断は妥当か
- 次にやるべき cheapest / highest-information-gain な clean v2 rerun と overlay 実験は何か

## 2. ゴール

- code SoT と artifact SoT の境界を reviewer judgement ベースで確定する
- `Task C/D` に進む前の最小 Go / No-Go 条件を決める
- Codex がそのまま clean rerun / replay / artifact 再生成に移れる粒度で、次アクションを優先順位付きで返せる状態にする

## 3. 現状の重要観測

### Confirmed facts

- workspace: `/Users/inouehiroshi/Documents/GitHub/SignalCascade`
- branch: `main`
- `git status --short` は clean
- `git diff --stat` は差分なし
- `git log --oneline -5` は次
  - `395e394 Add artifact provenance overlays and dashboard diagnostics`
  - `7ca01ad Advance profit policy diagnostics and reset semantics`
  - `c645e99 Document roadmap for profit-maximization migration`
  - `ba54089 Clarify proposal and acceptance diagnostics semantics`
  - `61329bf Align chart and decision panel heights`
- `rg -n "TODO|FIXME|QUESTION|TBD|UNSURE" PyTorch Frontend` は code / artifact 側の hit なし

### 今の code SoT で入っているもの

- config / reset semantics
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
  - `config_schema_version=2`
  - primary default は `carry_on`
  - `training_state_reset_mode="carry_on"`
  - `evaluation_state_reset_mode="carry_on"`
  - `diagnostic_state_reset_modes=("carry_on", "reset_each_session_or_window", "reset_each_example")`

- artifact provenance
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py`
  - `ARTIFACT_SOURCE_SCHEMA_VERSION=2`
  - `STATE_RESET_BOUNDARY_SPEC_VERSION=1`
  - `source.json` に次を入れる code がある
    - `artifact_id`
    - `parent_artifact_id`
    - `artifact_kind`
    - `data_snapshot.csv`
    - `data_snapshot_sha256`
    - `config_sha256`
    - `config_origin`
    - sub-artifact ごとの `sha256` / `schema_version` / `materialization`
  - git provenance も追加済み
    - `git_commit_sha`
    - `git_tree_sha`
    - `git_dirty`
    - `dirty_patch_sha256`

- evaluation / replay boundary semantics
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
  - `evaluate_model()` が次を返す code になっている
    - `state_reset_boundary_spec_version=1`
    - `state_reset_count`
    - `session_count`
    - `window_count`

- policy selection contract
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
  - `DIAGNOSTICS_SCHEMA_VERSION=5`
  - `POLICY_SELECTION_RULE_VERSION=2`
  - `POLICY_SELECTION_BASIS="pareto_rank_then_average_log_wealth_cvar_tail_loss_turnover_row_key"`
  - `selected_row_key`
  - `policy_calibration_rows_sha256`
  - `row_key` を deterministic tie-breaker に使う sort へ更新済み
  - `no_trade_band_hit_rate` は primary selection basis から外している

- artifact kind / overlay materialization
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
  - `training_run` と `diagnostic_replay_overlay` を分離
  - `export-diagnostics` は親 artifact を上書きしない
  - default 出力先は sibling overlay
    - 例: `current_diagnostic_replay_overlay`
  - same-path overlay は reject する
  - overlay の `manifest.json` は regenerate
  - overlay の copy 対象は `prediction.json` と `forecast_summary.json`

- report / public payload
  - files:
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
  - `analysis.json` / `research_report.md` に provenance summary を出す code がある
  - frontend sync script は `schemaVersion=5`
  - `run.stateResetMode` は config default で埋めず、artifact 側値を優先
  - `artifact_id` / `parent_artifact_id` / `data_snapshot_sha256` / `gitCommitSha` / `gitDirty` は v2 provenance があるときだけ public payload に出す

- tests
  - files:
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_sweep.py`
  - `test_artifact_schema.py`
    - overlay default 出力先が sibling になること
    - same-path overlay が reject されること
    - provenance / selection fields が report へ流れること
  - `test_policy_sweep.py`
    - `selected_row_key` が row order を変えても不変であること
    - `policy_calibration_rows_sha256` が row order を変えても不変であること
    - `selection_basis` と rule version が固定されていること

### 2026-04-05 JST に再実行した確認コマンド

```bash
python3 -m py_compile \
  /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py \
  /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py \
  /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py \
  /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py \
  /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py \
  /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_sweep.py
```

- 結果: success

```bash
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_artifact_schema \
  tests.test_policy_sweep \
  tests.test_stateful_evaluation \
  tests.test_policy_training
```

- workdir: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch`
- 結果: `Ran 20 tests in 1.705s` / `OK`

```bash
node --check /Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs
```

- 結果: success

```bash
npm run build
```

- workdir: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend`
- 結果: success
- 備考: Vite の chunk size warning は残るが build failure ではない

### 既存 artifact / public payload はまだ stale

- current artifact root
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`

- replay artifact root
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/review_20260327_ab`

- `current/config.json`
  - `config_schema_version` の hit なし
  - `training_state_reset_mode` の hit なし
  - `evaluation_state_reset_mode` の hit なし
  - `policy_sweep_state_reset_modes` の hit なし
  - つまり config artifact はまだ legacy payload

- `current/source.json` と `review_20260327_ab/source.json`
  - どちらも旧形式のまま
  - 内容は `{"kind": "csv", "path": ".../live/xauusd_m30_latest.csv"}`
  - `artifact_id` / `parent_artifact_id` / `data_snapshot_sha256` / git provenance は absent

- `current/manifest.json`
  - `generated_at=2026-03-27T04:18:04.395085+00:00`
  - `source_path=/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv`
  - `schema_version` / `artifact_kind` は absent
  - まだ legacy manifest

- `current/validation_summary.json`
  - `schema_version=4`
  - `diagnostics_schema_version=4`
  - `generated_at_utc=2026-03-27T06:27:55.599148+00:00`
  - `primary_state_reset_mode` は absent
  - `selection_rule_version` / `selected_row_key` / `policy_calibration_rows_sha256` は absent
  - stateful evaluation
    - `carry_on`: `average_log_wealth=0.00430597409890736`, `turnover=8.776042206270933`, `cvar_tail_loss=0.021612780168652534`
    - `reset_each_example`: `average_log_wealth=0.006099512001465342`, `turnover=30.390098959236806`, `cvar_tail_loss=0.020319810137152672`
    - `reset_each_session_or_window`: `average_log_wealth=0.006918165313578291`, `turnover=18.855870526655696`, `cvar_tail_loss=0.02017613686621189`

- `review_20260327_ab/validation_summary.json`
  - `schema_version=4`
  - `diagnostics_schema_version=4`
  - `generated_at_utc=2026-03-27T08:37:49.244017+00:00`
  - `primary_state_reset_mode="reset_each_session_or_window"`
  - `policy_calibration_summary.row_count=72`
  - `policy_calibration_summary.pareto_optimal_count=30`
  - `selection_rule_version` / `selected_row_key` / `policy_calibration_rows_sha256` は absent
  - stateful evaluation
    - `carry_on`: `average_log_wealth=0.0030358684517417965`, `turnover=8.433028171769607`, `cvar_tail_loss=0.020039625465869904`
    - `reset_each_example`: `average_log_wealth=0.0026628230814526002`, `turnover=24.692878911587954`, `cvar_tail_loss=0.020220961421728134`
    - `reset_each_session_or_window`: `average_log_wealth=0.0023159739865122966`, `turnover=19.177541872854732`, `cvar_tail_loss=0.02017613686621189`
  - 同一 snapshot 内比較では `carry_on` が 3 軸で優位

- frontend public payload
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
  - `schemaVersion=4`
  - `generatedAt=2026-03-27T09:55:05.081Z`
  - `provenance.sourcePath=/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv`
  - `provenance.artifactKind` / `artifactId` / `parentArtifactId` / `dataSnapshotSha256` は null
  - `run.stateResetMode="carry_on"`
  - `run.runQuality="degraded"`
  - つまり sync script の code は更新済みだが、public payload 自体はまだ再生成前

### 重要な分離

- code SoT
  - current repo HEAD
  - P0 実装はかなり前進している

- artifact SoT
  - まだ未成立
  - `current/` と `review_20260327_ab/` は legacy / stale artifact と見るのが安全

- public / display payload
  - `Frontend/public/dashboard-data.json` は reviewer judgement 用の SoT ではない
  - 現時点では stale export

### Hypotheses

- reviewer judgement が要求した contract の主要フィールドは code 側ではほぼ揃った
- ただし authoritative artifact はまだ 1 本も clean v2 rerun されていないため、artifact SoT は未確立
- `carry_on` を primary default に戻した判断は、existing artifact evidence を見る限り妥当
- `current` alias は clean v2 `training_run` を 1 本 materialize するまで動かさない方が安全
- 次の cheapest / highest-information-gain step は次
  - frozen CSV を切る
  - その snapshot から clean v2 `training_run` を 1 本 materialize
  - その parent に対して single-point `diagnostic_replay_overlay` を 1 本だけ作る

## 4. レビューしてほしい論点

### 優先度サマリー

- `P0`: P0 実装後の versioned serialization / replay contract に、authoritative rerun 前の blocking gap がまだ残るか
- `P1`: `carry_on` default 維持と `reset_each_session_or_window` promotion rule
- `P1`: `training_run` と `diagnostic_replay_overlay` の責務境界
- `P2`: clean v2 rerun と single-point overlay の pinned A/B 実験設計

### [P0] 論点 1: authoritative rerun 前にまだ欠けている contract はあるか

- code には次が入っている
  - config versioning
  - artifact lineage ID
  - frozen snapshot hash
  - config hash / origin
  - git provenance
  - state reset boundary versioning
  - deterministic selection rule version / row key / candidate-set digest
  - artifact kind separation
- この時点で、authoritative rerun を止める最小 blocking gap はまだあるか
- もし残るなら、追加 field / test / invariant を優先度付きで知りたい

### [P1] 論点 2: `carry_on` default 維持は正しいか

- code default は `carry_on`
- existing evidence では
  - `current` では `reset_each_session_or_window` が trade-off frontier に乗る
  - `review_20260327_ab` 同一 snapshot では `carry_on` が strict domination
- reviewer として、現時点の promotion rule は次で足りるか
  - reject rule: pinned A/B で candidate が incumbent に strict domination されるなら却下
  - promote rule: strict domination するか、predeclared guardrail 付き trade-off のときだけ昇格

### [P1] 論点 3: `training_run` と `diagnostic_replay_overlay` の境界は妥当か

- current code は `artifact_kind="training_run"` と `artifact_kind="diagnostic_replay_overlay"` を分けている
- overlay は sibling dir に materialize し、親 artifact を上書きしない
- overlay は `manifest.json` を regenerate し、`prediction.json` と `forecast_summary.json` だけを copy する
- この boundary は妥当か
- reviewer に確認したい点
  - overlay は `current` alias を絶対に更新しない方針でよいか
  - overlay でも `data_snapshot.csv` を持つべきか、親 snapshot 参照だけで十分か
  - `source.json` という filename を維持したまま provenance manifest の責務を持たせる方針でよいか

### [P2] 論点 4: 次の cheapest / highest-IG な実験は何か

- 第一候補は次
  - frozen CSV を materialize
  - その frozen snapshot から clean v2 `training_run` を 1 本再生成
  - 親 artifact に対して single-point `diagnostic_replay_overlay` を 1 本だけ作る
- fixed-point 条件候補
  - `state_reset_mode in {carry_on, reset_each_session_or_window}`
  - `cost_multiplier=1.0`
  - `gamma_multiplier=1.0`
  - `min_policy_sigma=0.0001`
- baseline の `carry_on` は parent `training_run` 自身を使い、overlay は `reset_each_session_or_window` 側 1 本で十分か
- この single-point overlay でまだ曖昧なら、その次に full pinned 2-run A/B へ進むべきか

## 5. 参照ファイル / アーティファクト

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_sweep.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/config.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/source.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/manifest.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/review_20260327_ab/source.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/review_20260327_ab/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`

## 6. 期待する出力形式

1. 結論
2. Confirmed facts / Hypotheses の分離
3. 優先度付きの主要 findings
4. cheap で information gain が高い次アクション
5. reviewer としての No-Go / Go 条件

## 7. 制約

- 一般論ではなく、このファイルに書かれた evidence を起点に判断する
- code change と artifact freshness を混同しない
- 確認済みの事実と仮説を分ける
- cheap で information gain が高い順に提案する
- Codex がそのまま rerun / replay / 追加実装に移れる粒度で返す
