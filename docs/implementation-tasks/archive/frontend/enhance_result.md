# 予測精度改善 実装ログ

> 2026-04-06 JST 注記:
> この文書は 2026-03-24 時点の selector / threshold-policy 系の作業ログです。
> 現在の canonical 実装は `PyTorch/profit_maximization_migration_roadmap.md`,
> `PyTorch/profit_maximization_execution_schedule.md`, `PyTorch/README.md`
> にある profit-maximization / `q_t*` 主経路が正であり、
> `accepted_signal` や `selection_policy.json` は historical compatibility の文脈でのみ参照してください。

## 2026-03-24 JST Step 1: 現状調査と変更方針の固定

### Why
- `docs/enhance_plan.md` の中心は特徴量の刷新ではなく、教師ラベル・採用判定・閾値校正の刷新だった。
- 現行実装は `2値方向分類 + utility寄り horizon 選択 + 4クラス overlay` に寄っており、precision-first の要件とずれていた。
- 実装を途中でぶらさないために、最初にどの層を変えるべきかを固定する必要があった。

### What
- `PyTorch/src/signal_cascade_pytorch/application/dataset_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
- `PyTorch/src/signal_cascade_pytorch/domain/entities.py`

を主変更点として特定した。

### How
- `docs/enhance_plan.md` を最後まで読んで、Phase 1/2 の要件を既存コードへ対応付けた。
- 現行コードを確認し、既存の `returns_target` と `overlay_target` だけでは 3値方向ラベル、meta-label、selector、binary risk filter を表現できないことを確認した。
- 実装順を `dataset -> model/loss -> calibration/selector -> inference/CLI` に固定した。

## 2026-03-24 JST Step 2: データセット生成を clean-signal 前提へ変更

### Why
- `enhance_plan.md` の最優先項目は「全時点に無理に方向を付けない」ための 3値ラベル化だった。
- 既存の `returns_target + 4class overlay` だけでは、regime別 threshold と clean event の重み付けを後段で扱えなかった。
- まずデータ層で `direction_targets / weights / regime` を持たせないと、学習と採用判定の実装が後ろで破綻する。

### What
- `PyTorch/src/signal_cascade_pytorch/application/dataset_service.py`
- `PyTorch/src/signal_cascade_pytorch/domain/entities.py`
- `PyTorch/src/signal_cascade_pytorch/application/config.py`

を更新し、学習サンプルの表現を拡張した。

### How
- `4h` の forward return に対して、`realized volatility + regime(session/volatility/trend)` から `delta` と `eta` を決め、`-1 / 0 / +1` の 3値方向ラベルを作るように変更した。
- `MFE / MAE` と return excess を使って clean-signal の重みを計算し、曖昧なサンプルよりも「値幅が出て逆行が小さい」ケースを強く学習できるようにした。
- overlay は `hold / reduce / full_exit / hard_exit` ではなく、主方向が短期的に安全かどうかを見る `binary risk filter` に変更した。
- 推論用サンプルにも `regime_id`、`regime_features`、`horizon_costs` を載せ、学習時と同じ採用ロジックを推論時にも再利用できるようにした。

## 2026-03-24 JST Step 3: 学習器を 3値方向 + selector policy 前提へ変更

### Why
- データ層だけ 3値化しても、モデルが依然として `2値方向 + utility-based selection` のままでは計画書の目的に届かない。
- `80% precision を守りながら coverage を増やす` には、予測器と採用判定を分離した後段 policy が必要だった。
- 学習後に OOF ベースで threshold を決められるよう、校正ロジックを保存可能な形にする必要があった。

### What
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
- `PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/training_service.py`

を更新し、モデル出力・loss・OOF policy fitting を作り直した。

### How
- main head に `direction_logits[H,3]` を追加し、direction loss を `clean-signal weight` 付き focal CE に置き換えた。
- overlay は 4クラス softmax から 1logit の binary risk filter へ変更し、`hold_probability` を直接扱う形にした。
- walk-forward で OOF snapshot を作り、そこから `correctness_model(q)` と `selector_model(s)` を logistic fitting する `policy_service.py` を追加した。
- `J = q^alpha * max(|mu|-c, 0)^(1-alpha)` で horizon を選び、regime/horizon ごとの閾値を `precision >= 0.8` 制約で校正するようにした。

## 2026-03-24 JST Step 4: 推論・保存形式・フロント同期を新ポリシーへ接続

### Why
- 学習時だけ selector を導入しても、推論や可視化が旧フォーマットのままだと実運用に使えない。
- `accepted_signal` や `selection_probability` が出力されないと、「見送り」が改善されたのかを確認できない。
- tuning の選定軸も utility 一辺倒から precision-first に寄せる必要があった。

### What
- `PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
- `PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
- `Frontend/scripts/sync-signal-cascade-data.mjs`

を更新し、推論結果とダッシュボード連携を新形式に合わせた。

### How
- `prediction.json` に `accepted_signal / selection_probability / selection_threshold / correctness_probability / hold_probability / hold_threshold / regime_id` を追加した。
- `train` 時に `selection_policy.json` を保存し、`predict` 時には同ファイルを読み込んで同じ採用判定を再利用するようにした。
- `tune_latest` の候補比較を `selection_precision -> coverage_at_target_precision -> value_capture_ratio` 優先に変更した。
- フロント側の同期スクリプトに `selectionPrecision / coverageAtTargetPrecision / noTradeRate / turnover / maxDrawdown` を流し、`accepted_signal=false` のときは narrative を「見送り」として表示できるようにした。

## 2026-03-24 JST Step 5: 検証

### Why
- 大きく構造を変えたので、少なくとも静的検証と small smoke run を通しておかないと安全に終われない。
- 特に今回の変更は `dataset -> training -> policy -> inference -> frontend sync` を横断しているため、単一ファイルの lint では不足だった。

### What
- `py_compile` による Python 構文確認
- `node --check` による frontend sync script の構文確認
- `PyTorch/.venv` を使った synthetic smoke run

を実行した。

### How
- `python3 -m py_compile` で変更した Python ファイル一式の構文を確認した。
- `node --check Frontend/scripts/sync-signal-cascade-data.mjs` を実行し、フロント同期スクリプトの構文エラーがないことを確認した。
- `PyTorch/.venv/bin/python` で synthetic データを使った smoke run を実行し、`model.pt / metrics.json / selection_policy.json / prediction.json` が生成されることを確認した。
- smoke run の結果、`selection_policy.json` には regime/horizon 別 threshold が保存され、`prediction.json` には `accepted_signal=false` を含む新しい推論形式が出力されることを確認した。
