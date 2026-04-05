# Profit Maximization Execution Schedule

最終更新: 2026-04-06 JST

この文書は、`profit_maximization_migration_roadmap.md` の実行版です。
目的は次の 3 点を 1 つにまとめることです。

- 12 個の移行タスクを 4 本の実装チケットへ束ねる
- `artifact contract 先行` を優先順位として確定する
- issue / PR 粒度まで分解し、日付付きの実行スケジュールに落とす

## 0. Status Snapshot

- 2026-04-06 JST: `PR 1` から `PR 6` までの core / contract / CLI migration を実装済み
- 2026-04-06 JST: `PR 7` として threshold-policy 由来の互換 field を `domain/historical_compatibility.py` へ隔離済み
- 2026-04-06 JST: `PR 8` として README / acceptance を更新し、unit 32件と `train -> predict -> export-diagnostics` smoke を再確認済み
- 2026-04-06 JST: `tune-latest` に optimization gate を導入し、実データ tuning で accepted candidate の `current` 反映、`Frontend/public/dashboard-data.json` 再同期、`npm run build` を確認済み

## 1. 現在地

2026-04-06 JST 時点の認識は次です。

- `Phase 1`: 完了
- `Phase 2`: 完了
- `Phase 3`: 完了
- `Phase 4`: 完了
- `Phase 5`: 完了
- `Phase 6`: 完了
- `Phase 7`: 完了

現行コードは、`q_t*` 主経路、profit objective、artifact / CLI / diagnostics の canonical contract、legacy compatibility の隔離、tuning acceptance gate、dashboard sync まで含めて schedule 上のスコープを一通り満たしています。

## 2. 優先順位

優先順位は次で固定します。

1. `Feature / State / Policy` の主経路整合
2. `Artifact / Diagnostics Contract` の切替
3. `CLI / Serving` の切替
4. `Legacy Retirement` の実施

`artifact contract` を `legacy isolation` より先に置く理由は、reviewer / downstream consumer / dashboard が読む契約を先に確定しないと、旧コードを切っても運用系が不安定になるためです。

## 3. 実装チケット

### Ticket A. Core Path Alignment

対象 phase:

- `Phase 1`
- `Phase 2`
- `Phase 3`
- `Phase 4`

目的:

- `X_t -> h_t -> s_t -> v_t -> (mu_t, sigma_t^2) -> q_t* -> pi_{t+1}` を主経路として確定する
- 学習時の主損失を `log wealth - CVaR` 中心へ寄せる

主要ファイル:

- `PyTorch/src/signal_cascade_pytorch/application/dataset_service.py`
- `PyTorch/src/signal_cascade_pytorch/domain/entities.py`
- `PyTorch/src/signal_cascade_pytorch/application/config.py`
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
- `PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
- `PyTorch/src/signal_cascade_pytorch/application/training_service.py`

Exit 条件:

- `q_t*` の exact inference path と differentiable training path が一致した意味で説明できる
- `profit objective` が主損失で、supervised loss は補助扱いになっている
- `state_vector` と `shape posterior` の意味が文書化されている

### Ticket B. Artifact / Diagnostics Contract Migration

対象 phase:

- `Phase 5`

目的:

- artifact / report / diagnostics の語彙を新 spec に合わせる

主要ファイル:

- `PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/training_service.py`

Exit 条件:

- `prediction.json`, `forecast_summary.json`, `metrics.json`, `research_report.md` が threshold-policy 語彙なしで読める
- `q_t*`, `mu_t`, `sigma_t^2`, `g_t`, `state_vector summary` が主語になっている

### Ticket C. CLI / Serving Migration

対象 phase:

- `Phase 6`

目的:

- 日常運用の CLI と replay 手順を新 contract 前提へ切り替える

主要ファイル:

- `PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
- `PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `PyTorch/src/signal_cascade_pytorch/application/inference_service.py`

Exit 条件:

- `train`, `predict`, `export-diagnostics` が新 contract で完結する
- `selection-threshold-mode` など legacy option の扱いが明示されている

### Ticket D. Legacy Retirement And Acceptance

対象 phase:

- `Phase 7`

目的:

- 旧 threshold-policy を主経路から完全に外し、受け入れ判定まで終える

主要ファイル:

- `PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `PyTorch/tests/test_policy_training.py`
- `PyTorch/tests/test_artifact_schema.py`
- `PyTorch/logic_multiframe_candlestick_model.md`
- `PyTorch/report_signalcascade_xauusd.md`

Exit 条件:

- 主 artifact / 主 CLI / 主文書に旧 threshold-policy 語彙が残っていない
- smoke training / replay diagnostics / integration test が新 contract で再成立している

## 4. Issue / PR 分解

各チケットは次の issue / PR 粒度で実行します。
PR はすべて `small-to-medium diff` を原則とし、レビュー単位で独立マージ可能にします。

### Issue 1 / PR 1. Data-State Contract Freeze

スコープ:

- `Phase 1` の棚卸し完了
- `TrainingExample` と state feature の命名確定
- `state_vector` 入力契約の明文化

依存:

- なし

### Issue 2 / PR 2. Encoder-State-Expert Alignment

スコープ:

- `shape posterior`, `state projection`, `expert heads` の責務整理
- `state_vector` の構成要素をコードと文書で一致させる

依存:

- `PR 1`

### Issue 3 / PR 3. Policy Path And Profit Objective Closure

スコープ:

- exact / differentiable policy の整合
- `log wealth - CVaR` 主損失化
- `q_t*` path の unit test 追加

依存:

- `PR 2`

### Issue 4 / PR 4. Metrics And Diagnostics Contract

スコープ:

- `metrics.json`
- `validation_summary.json`
- diagnostics export
- calibration / coverage / range 用語の整理

依存:

- `PR 3`

### Issue 5 / PR 5. Prediction / Forecast / Report Contract

スコープ:

- `prediction.json`
- `forecast_summary.json`
- `research_report.md`
- downstream consumer が読む主要語彙の切替

依存:

- `PR 4`

### Issue 6 / PR 6. CLI And Bootstrap Migration

スコープ:

- CLI option の整理
- legacy option の deprecated 境界明示
- `train` / `predict` / `export-diagnostics` の出力整合

依存:

- `PR 5`

### Issue 7 / PR 7. Legacy Isolation And Removal

スコープ:

- `selection_threshold`, `accepted_signal`, `selector_probability` など互換フィールドの隔離または削除
- historical module 化の最終判断

依存:

- `PR 6`

### Issue 8 / PR 8. Acceptance, Smoke, Docs

スコープ:

- unit / integration / replay / smoke
- README / roadmap / implementation logic 更新
- final acceptance checklist

依存:

- `PR 7`

## 5. 実行スケジュール

基準日:

- 計画開始: 2026-04-06 JST

### Wave 1. Core Path

- 2026-04-06 JST: `PR 1` 着手
- 2026-04-07 JST: `PR 1` 完了
- 2026-04-08 JST: `PR 2` 着手
- 2026-04-09 JST: `PR 2` 完了
- 2026-04-10 JST: `PR 3` 着手
- 2026-04-12 JST: `PR 3` 完了

Gate:

- `q_t*` path の exact / differentiable 整合 test が通ること
- synthetic smoke で発散しないこと

### Wave 2. Contract Migration

- 2026-04-13 JST: `PR 4` 着手
- 2026-04-14 JST: `PR 4` 完了
- 2026-04-15 JST: `PR 5` 着手
- 2026-04-16 JST: `PR 5` 完了

Gate:

- `prediction.json`, `forecast_summary.json`, `metrics.json`, `research_report.md` の新 contract が揃うこと
- dashboard / downstream consumer が新 field 名で読めること

### Wave 3. CLI Migration

- 2026-04-17 JST: `PR 6` 着手
- 2026-04-18 JST: `PR 6` 完了

Gate:

- `train`, `predict`, `export-diagnostics` の日常運用コマンドが新 contract 前提で成立すること

### Wave 4. Legacy Retirement And Acceptance

- 2026-04-19 JST: `PR 7` 着手
- 2026-04-20 JST: `PR 7` 完了
- 2026-04-21 JST: `PR 8` 着手
- 2026-04-22 JST: `PR 8` 完了

Gate:

- 主経路に legacy threshold-policy 語彙が残っていないこと
- smoke training / replay diagnostics / report review が通ること

### Buffer

- 2026-04-23 JST から 2026-04-24 JST: バッファ

用途:

- calibration 崩れ
- artifact schema 差し戻し
- dashboard 追随
- ドキュメント仕上げ

## 6. 依存関係

- `PR 1 -> PR 2 -> PR 3`
- `PR 3 -> PR 4 -> PR 5`
- `PR 5 -> PR 6`
- `PR 6 -> PR 7 -> PR 8`

並列化しない理由:

- 先に contract を固定しないと、後続 PR の diff がすべて schema 変更で衝突するため

## 7. 受け入れチェックリスト

- `Done 1`: profit objective が主損失
- `Done 2`: 推論主経路が `q_t*` 中心
- `Done 3`: threshold calibration が主経路から除外済み
- `Done 4`: artifact / report / diagnostics / CLI の語彙が新 spec と整合
- `Done 5`: unit / integration / replay / smoke が新 contract で通過

## 8. 運用ルール

- 各 PR の冒頭に「対象 phase」「対象 issue」「Exit 条件」を明記する
- 各 PR の最後に `research smoke` の結果を必ず添える
- `Frontend` が artifact contract を読む場合、`PR 5` の直後に追随差分を入れる
- `legacy removal` は `PR 6` 完了前に着手しない

## 9. 次の着手点

この execution schedule に載せた `Issue 1` から `Issue 8` までは一通り完了しました。
次の着手点は、新 canonical artifact を読む downstream (`Frontend/scripts/sync-signal-cascade-data.mjs` など) から互換 alias 依存を段階的に外す follow-up です。
