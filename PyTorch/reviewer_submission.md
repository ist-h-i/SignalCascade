# SignalCascade Review Handoff

最終更新: 2026-04-05 17:42:46 JST

## 1. あなたへの依頼

あなたは、時系列予測・Quant ML・stateful sequence modeling・risk-aware policy optimization に強い reviewer です。  
`/Users/inouehiroshi/Documents/GitHub/SignalCascade` の current code、既存 artifact、frontend public payload を突き合わせて、次に進むための判断をしてください。

今回は一般論ではなく、次の 4 点を決めてほしいです。

- versioned serialization / replay contract は、今回の code change で reviewer judgement に十分近づいたか
- `reset_each_session_or_window` を primary semantic に進めず、`carry_on` を default 維持に戻した判断は妥当か
- `review_20260327_ab` を successor ではなく `diagnostic_replay_overlay` として扱う方針は妥当か
- 次に最も cheap で information gain が高い pinned A/B は何か

## 2. ゴール

- code SoT と artifact SoT の境界を reviewer judgement ベースで確定する
- `Task C/D` に進む前の最小 gate を決める
- Codex がそのまま rerun / replay / artifact 再生成に移れる粒度で、次アクションを優先順位付きで返せる状態にする

## 3. 現状の重要観測

### Confirmed facts

- workspace: `/Users/inouehiroshi/Documents/GitHub/SignalCascade`
- `git status --short` は dirty。review relevant な変更は主に次
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_sweep.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_stateful_evaluation.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
- `git log --oneline -5` は次
  - `7ca01ad Advance profit policy diagnostics and reset semantics`
  - `c645e99 Document roadmap for profit-maximization migration`
  - `ba54089 Clarify proposal and acceptance diagnostics semantics`
  - `61329bf Align chart and decision panel heights`
  - `86f97b2 Clarify proposal and acceptance diagnostics semantics`
- `git diff --stat` は `13 files changed, 1113 insertions(+), 438 deletions(-)`。review handoff 以外では provenance / replay / reset semantics 周りの差分が中心
- `rg -n "TODO|FIXME|QUESTION|TBD|UNSURE" ...` は hit なし。明示的な unresolved marker は現時点で見つかっていない

### 今回までに入れた code change

- `TrainingConfig` は `config_schema_version=2` を持つ
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
  - current generation は `config_schema_version=2` を保存
  - legacy payload は `config_schema_version` 欠如時に fallback
  - reviewer judgement に合わせて primary default は `carry_on` に戻した
    - `training_state_reset_mode="carry_on"`
    - `evaluation_state_reset_mode="carry_on"`
    - `diagnostic_state_reset_modes=("carry_on", "reset_each_session_or_window", "reset_each_example")`
    - legacy payload の `policy_sweep_state_reset_modes=("carry_on",)` は維持

- artifact provenance contract を拡張した
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py`
  - `ARTIFACT_SOURCE_SCHEMA_VERSION=2`
  - `STATE_RESET_BOUNDARY_SPEC_VERSION=1`
  - `source.json` に次を materialize する code が入った
    - `artifact_id`
    - `parent_artifact_id`
    - `data_snapshot.csv`
    - `data_snapshot_sha256`
    - `config_sha256`
    - `config_origin`
    - sub-artifact ごとの `sha256` / `schema_version` / `materialization`
  - CSV source は `artifact_dir/data_snapshot.csv` に frozen snapshot として materialize する

- evaluation / replay semantics を versioned にした
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
  - `evaluate_model()` が次を返すようになった
    - `state_reset_boundary_spec_version=1`
    - `state_reset_count`
    - `session_count`
    - `window_count`
  - `reset_each_session_or_window` の boundary 判定は `_state_reset_boundaries()` に分離した

- policy selection summary を versioned にした
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
  - `DIAGNOSTICS_SCHEMA_VERSION=5`
  - `selection_rule_version=1`
  - `selected_row_key`
  - sweep row ごとに stable-ish な `row_key` を追加
  - Pareto domination と sort から `no_trade_band_hit_rate` を外した
    - 主要軸は `{average_log_wealth ↑, turnover ↓, cvar_tail_loss ↓}` に寄せている

- `training_run` / `diagnostic_replay_overlay` の materialization 導線を入れた
  - files:
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
    - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
  - `train` は `training_run` 用の `source.json` を 2-pass で再生成する
    - 理由: `analysis.json` / `research_report.md` など regenerated sub-artifact の hash を埋めたい
  - `export-diagnostics` は `diagnostic_replay_overlay` を親 artifact 参照つきで materialize する
  - overlay は `config.json` / `metrics.json` / `analysis.json` / `validation_summary.json` を replay 側で再生成し、`prediction.json` / `forecast_summary.json` / `manifest.json` は親から copy する

- report / analysis に provenance summary を出すようにした
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
  - `ANALYSIS_SCHEMA_VERSION=5`
  - `analysis.json` / `research_report.md` が次を載せる
    - `artifact_kind`
    - `artifact_id`
    - `parent_artifact_id`
    - `source_path`
    - `data_snapshot_sha256`
    - `config_origin`
    - `selection_rule_version`
    - `selected_row_key`

### 実行した確認コマンド

- 再実行日: 2026-04-05 JST

```bash
python3 -m py_compile \
  PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py \
  PyTorch/src/signal_cascade_pytorch/application/config.py \
  PyTorch/src/signal_cascade_pytorch/application/training_service.py \
  PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py \
  PyTorch/src/signal_cascade_pytorch/application/report_service.py \
  PyTorch/src/signal_cascade_pytorch/application/tuning_service.py \
  PyTorch/src/signal_cascade_pytorch/bootstrap.py \
  PyTorch/tests/test_artifact_schema.py \
  PyTorch/tests/test_stateful_evaluation.py \
  PyTorch/tests/test_policy_sweep.py
```

- 結果: 成功

```bash
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_artifact_schema \
  tests.test_stateful_evaluation \
  tests.test_policy_sweep \
  tests.test_policy_training
```

- workdir: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch`
- 結果: `Ran 18 tests in 1.088s` / `OK`

### 既存 artifact の観測

- current artifact root
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`

- replay artifact root
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/review_20260327_ab`

- `current/config.json` はまだ legacy payload
  - `config_schema_version`: absent
  - `training_state_reset_mode`: absent
  - `evaluation_state_reset_mode`: absent
  - `policy_sweep_state_reset_modes`: absent
  - `policy_sweep_min_policy_sigmas`: absent

- `current/source.json` と `review_20260327_ab/source.json` はどちらも旧形式のまま
  - `{"kind": "csv", "path": ".../live/xauusd_m30_latest.csv"}`
  - つまり新しい provenance contract は code に入ったが、artifact はまだ再生成していない

- `current/validation_summary.json`
  - `generated_at_utc=2026-03-27T06:27:55.599148+00:00`
  - `primary_state_reset_mode` は absent
  - `policy_calibration_summary.row_count` / `selection_rule_version` / `selected_row_key` も absent
  - stateful evaluation
    - `carry_on`: `average_log_wealth=0.00430597409890736`, `turnover=8.776042206270933`, `cvar_tail_loss=0.021612780168652534`
    - `reset_each_example`: `average_log_wealth=0.006099512001465342`, `turnover=30.390098959236806`, `cvar_tail_loss=0.020319810137152672`
    - `reset_each_session_or_window`: `average_log_wealth=0.006918165313578291`, `turnover=18.855870526655696`, `cvar_tail_loss=0.02017613686621189`

- `review_20260327_ab/validation_summary.json`
  - `generated_at_utc=2026-03-27T08:37:49.244017+00:00`
  - `primary_state_reset_mode="reset_each_session_or_window"`
  - `policy_calibration_summary.row_count=72`
  - `policy_calibration_summary.pareto_optimal_count=30`
  - ただし `selection_rule_version` / `selected_row_key` は absent
  - stateful evaluation
    - `carry_on`: `average_log_wealth=0.0030358684517417965`, `turnover=8.433028171769607`, `cvar_tail_loss=0.020039625465869904`
    - `reset_each_example`: `average_log_wealth=0.0026628230814526002`, `turnover=24.692878911587954`, `cvar_tail_loss=0.020220961421728134`
    - `reset_each_session_or_window`: `average_log_wealth=0.0023159739865122966`, `turnover=19.177541872854732`, `cvar_tail_loss=0.02017613686621189`
  - 同一 scratch snapshot 内比較では `carry_on` が 3 軸で優位

- frontend public payload
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
  - `generatedAt=2026-03-27T09:55:05.081Z`
  - `run.runQuality="degraded"`
  - `run.stateResetMode="carry_on"`
  - `provenance.sourcePath=/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv`
  - frontend も今は legacy artifact provenance を踏んで live CSV を向いている

### 重要な不一致

- code は reviewer judgement に沿って `carry_on` default、ID/hash/versioned selection、overlay lineage を持つように進んだ
- しかし existing artifact は再生成前なので、`current/` と `review_20260327_ab/` はまだ旧 contract のまま
- つまり今の SoT は次の分離が必要
  - code SoT: current repo HEAD
  - artifact SoT: まだ未確立
  - `current/`: legacy snapshot。authoritative SoT とまでは言えない

### Hypotheses

- reviewer judgement が要求した最小 contract の大半は code 側で入ったが、artifact SoT はまだ立っていない
- `reset_each_session_or_window` を primary に上げる evidence は、existing artifact からはまだ得られていない
- cheapest / highest-IG step は、reviewer judgement のまま
  - frozen CSV を切る
  - clean `training_run` を 1 本 materialize する
  - その parent に対して single-point `diagnostic_replay_overlay` A/B を打つ
- 追加の design uncertainty は次
  - `source.json` は実質 manifest だが、互換性のため名称を維持している。この naming を変えるべきか
  - overlay 側でも `data_snapshot.csv` を materialize している。親 snapshot を参照だけにすべきか、独立 replay artifact として snapshot を持つべきか

## 4. レビューしてほしい論点

### 優先度サマリー

- `P0`: versioned serialization / replay contract の closure と `code SoT / artifact SoT` の境界
  - 理由: code は v2 contract に進んだが、`current/` と `review_20260327_ab/` と frontend payload はまだ旧 contract のまま
- `P1`: `carry_on` default 維持判断と primary semantic の promotion rule
  - 理由: `current` と `review_20260327_ab` で `carry_on` と `reset_each_session_or_window` の相対評価が揺れている
- `P1`: `diagnostic_replay_overlay` の責務境界
  - 理由: `copied / regenerated` の線引きと `current` alias 非更新ルールが lineage contract に直結する
- `P2`: cheapest / highest-IG な pinned A/B と selection rule 周辺の詰め
  - 理由: `P0/P1` を前提に最小追加実験を決める段階で、selection basis の説明と実装軸の整合も確認したい

### [P0] 論点 1: versioned serialization / replay contract は、これで閉じたと言えるか

- 今回 code に追加した最小フィールド
  - `artifact_id`
  - `parent_artifact_id`
  - `data_snapshot_sha256`
  - `config_sha256`
  - `config_origin`
  - sub-artifact ごとの `sha256` / `schema_version`
  - `selection_rule_version`
  - `selected_row_key`
  - `state_reset_boundary_spec_version`
  - `state_reset_count` / `session_count` / `window_count`
- この時点で、reviewer judgement の `No-Go` 条件をどこまで解消できたと見るか
- まだ足りない最小フィールドがあれば、優先度付きで知りたい

### [P1] 論点 2: primary semantic を `carry_on` 維持に戻した判断は妥当か

- code default は `carry_on` に戻した
- existing evidence では
  - `current` では `reset_each_session_or_window` が frontier に乗る
  - `review_20260327_ab` 同一 snapshot では `carry_on` が strict domination
- この状況で `carry_on` 維持を default にしたのは妥当か
- reviewer として promotion rule をどう定義するか
  - reject rule: strict domination なら却下
  - promotion rule: domination もしくは predeclared guardrail 付き trade-off
  で足りるか

### [P1] 論点 3: `review_20260327_ab` を `diagnostic_replay_overlay` として扱う責務境界は妥当か

- 現在の code は `artifact_kind="training_run"` と `artifact_kind="diagnostic_replay_overlay"` を分けた
- overlay は親 artifact 参照つきで materialize する
- ただし existing `review_20260327_ab` artifact 自体は legacy payload のまま
- reviewer に確認したい点
  - overlay は `current` alias を絶対に更新しない方針でよいか
  - overlay が `prediction.json` / `manifest.json` を copy し、`metrics.json` / `validation_summary.json` / `analysis.json` を再生成する boundary は妥当か
  - `source.json` という名前を維持したまま manifest 的責務を持たせる方針でよいか

### [P2] 論点 4: cheapest / highest-IG な次実験は何か

- 現在の第一候補は reviewer judgement と同じ
  - frozen CSV を materialize
  - その frozen snapshot から clean `training_run` を 1 本再生成
  - parent artifact に対して single-point `diagnostic_replay_overlay` を 1 本だけ作る
- overlay の固定条件候補
  - `state_reset_mode in {carry_on, reset_each_session_or_window}`
  - `cost_multiplier=1.0`
  - `gamma_multiplier=1.0`
  - `min_policy_sigma=0.0001`
- diagnostics の `selection_basis` 文字列と実際の Pareto / sort 軸の説明が一致しているかも、この実験設計と一緒に sanity check したい
- この single-point overlay でまだ曖昧なら、その次に full pinned 2-run A/B に進むべきか

## 5. 参照ファイル / アーティファクト

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_sweep.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_stateful_evaluation.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/config.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/source.json`
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
- code change と artifact freshness の混同を避ける
- 確認済みの事実と仮説を分ける
- cheap で information gain が高い順に提案する
- Codex がそのまま rerun / replay / 追加実装に移れる粒度で返す
