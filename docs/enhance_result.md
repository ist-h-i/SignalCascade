# 予測精度改善 実装ログ

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
