# SignalCascade XAUUSD Research Report

## Abstract
本レポートは、2026-03-24T19:12:17.269127+09:00 時点の最新成果物を用いて `SignalCascade` の再学習結果を再評価したものである。 採用モデルの `project_value_score` は 0.0625、 `utility_score` は 0.0920 であり、 段階評価は `low-confidence` となった。 `precision_feasible=False` で、 research 進捗として `best_selection_lcb=0.0362` を得た。 選択 horizon は 6 本先、 overlay 判定は `hold`、 ポジションは 0.0000 である。

## 1. Experimental Setup
- 生成時刻: `2026-03-24T19:12:17.269127+09:00`
- 成果物: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`
- サンプル数: `train=77`, `validation=26`, `total=133`
- 候補数: `9`
- best validation loss: `2.157878` at epoch `1`
- 入力ソース: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv`
- 期間: `2025-12-24T09:00:00+09:00` から `2026-03-24T09:30:00+09:00`
- 行数: `used=2840`, `original=2840`

## 2. Metric System
- `utility_score`: precision, coverage, capture, overlay F1, directional accuracy, drawdown, calibration の複合値。
- `project_value_score`: utility に加えて profit factor, sortino, calibration を強めた事業価値指標。
- `profit_factor`: 利益総額と損失総額の比率。数値安定化のため `10.0` で上限 clip する。
- `signal_sortino`: 下方リスクで調整した価値指標。数値安定化のため `10.0` で上限 clip する。
- `selection_brier_score`: 採用確率の calibration 誤差。0 に近いほど良い。
- `best_selection_lcb`: feasible 未達 run の比較用に、threshold 候補群で最大の Wilson LCB を追う研究指標。
- `pre_threshold_capture`: selector threshold を適用する前の仮想 capture。シグナル原石の有無を確認する。
- `alignment_rate` / `actionable_edge_rate`: return head と direction head の整合、およびコスト控除後の実行可能 edge の発生率を監視する。

## 3. Learning Analysis
| Metric | Value |
| --- | --- |
| best_epoch | 1 |
| epoch_count | 16 |
| best_epoch_ratio | 0.062500 |
| generalization_gap_at_best | 0.165028 |
| validation_drift_from_best_to_last | +3.142757 |
| early_peak | True |
- best validation loss は epoch 1/16 で最小化し、その後の validation total は +3.1428 悪化した。早期ピーク型で、direction head の汎化がまだ不安定である。
- selector calibration は `selection_brier_score=0.000002` まで改善した一方、`actionable_edge_rate=0.0000` のため、cost 控除後に採用可能な edge は validation 上で未出現である。
- threshold search の研究進捗として `best_selection_lcb=0.0362`、`support_at_best_lcb=5`、`precision_at_best_lcb=0.2000` を追跡する。feasible 未達でも threshold 面の改善度を比較できる。
- ラベル密度は horizon 30 で最大 (`nonflat_rate=0.9231`)、符号整合は horizon 1 が最大 (`alignment_rate=0.1923`) で、長短 horizon の学習難度が分かれている。

## 4. Validation Results
| Metric | Value |
| --- | --- |
| project_value_score | 0.062500 |
| utility_score | 0.092048 |
| precision_feasible | False |
| threshold_calibration_feasible | False |
| selection_precision | 0.000000 |
| selection_support | 0 |
| best_selection_lcb | 0.036223 |
| support_at_best_lcb | 5.000000 |
| precision_at_best_lcb | 0.200000 |
| tau_at_best_lcb | 0.332440 |
| coverage_at_target_precision | 0.000000 |
| value_capture_ratio | 0.000000 |
| pre_threshold_capture | 0.000000 |
| profit_factor | 0.000000 |
| signal_sortino | 0.000000 |
| alignment_rate | 0.049451 |
| actionable_edge_rate | 0.000000 |
| nonflat_rate | 0.609890 |
| selection_brier_score | 0.000002 |
| max_drawdown | 0.000000 |

## 5. Delta Vs Previous Current
| Metric | Delta | Previous | Current |
| --- | --- | --- | --- |
| project_value_score | +0.006022 | 0.056478 | 0.062500 |
| utility_score | +0.011509 | 0.080539 | 0.092048 |
| selection_precision | +0.000000 | 0.000000 | 0.000000 |
| coverage_at_target_precision | +0.000000 | 0.000000 | 0.000000 |
| value_capture_ratio | +0.000000 | 0.000000 | 0.000000 |
| profit_factor | +0.000000 | 0.000000 | 0.000000 |
| signal_sortino | +0.000000 | 0.000000 | 0.000000 |
| max_drawdown | +0.000000 | 0.000000 | 0.000000 |

## 6. Hyperparameter Optimization
| Candidate | Project Value | Utility | Precision | Best LCB | Align | Capture | Profit Factor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_04 | 0.062500 | 0.092048 | 0.000000 | 0.036223 | 0.049451 | 0.000000 | 0.000000 |
| candidate_02 | 0.059166 | 0.094208 | 0.000000 | 0.036223 | 0.032967 | 0.000000 | 0.000000 |
| candidate_01 | 0.062094 | 0.095763 | 0.000000 | 0.000000 | 0.016484 | 0.000000 | 0.000000 |
採用パラメータ: `{'epochs': 16, 'batch_size': 16, 'learning_rate': 0.00053125, 'hidden_dim': 64, 'dropout': 0.1, 'weight_decay': 5e-05}`

## 7. Horizon Diagnostics
| H(4h) | Nonflat | Up | Down | Align | Actionable Edge |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.615385 | 0.269231 | 0.346154 | 0.192308 | 0.000000 |
| 2 | 0.423077 | 0.115385 | 0.307692 | 0.000000 | 0.000000 |
| 3 | 0.423077 | 0.076923 | 0.346154 | 0.038462 | 0.000000 |
| 6 | 0.423077 | 0.000000 | 0.423077 | 0.115385 | 0.000000 |
| 12 | 0.576923 | 0.000000 | 0.576923 | 0.000000 | 0.000000 |
| 18 | 0.884615 | 0.000000 | 0.884615 | 0.000000 | 0.000000 |
| 30 | 0.923077 | 0.000000 | 0.923077 | 0.000000 | 0.000000 |

## 8. Forecast Estimation
- Anchor time JST: `2026-03-24T13:00:00+09:00`
- Anchor close: `4364.0350`
- Selected direction classifier: `long`
- Expected return direction: `long`
- Direction alignment: `True`
- Accepted signal: `False`
- Selection probability / threshold: `0.6013` / `null`
- Hold probability / threshold: `0.4115` / `0.2748`
- 選択 forecast: `6` 本先 (`2026-03-25T13:00:00+09:00`), 予測終値 `4374.1048`, 期待収益率 `0.23%`, 1σ帯 `[4322.8078, 4426.0105]`
| H(4h) | Forecast Time JST | Expected Return | Pred Close | 1σ Band |
| --- | --- | --- | --- | --- |
| 1 | 2026-03-24T17:00:00+09:00 | -0.03% | 4362.709830 | 4340.538218 .. 4384.994695 |
| 2 | 2026-03-24T21:00:00+09:00 | 0.10% | 4368.432389 | 4336.959183 .. 4400.133996 |
| 3 | 2026-03-25T01:00:00+09:00 | 0.04% | 4365.962731 | 4326.151943 .. 4406.139872 |
| 6 | 2026-03-25T13:00:00+09:00 | 0.23% | 4374.104799 | 4322.807786 .. 4426.010533 |
| 12 | 2026-03-26T13:00:00+09:00 | 0.14% | 4369.954561 | 4292.366984 .. 4448.944589 |
| 18 | 2026-03-27T13:00:00+09:00 | -0.48% | 4343.017506 | 4242.013097 .. 4446.426878 |
| 30 | 2026-03-29T13:00:00+09:00 | -0.60% | 4337.641405 | 4203.025468 .. 4476.568867 |

## 9. Project Value Assessment
project_value_score=0.0625、utility_score=0.0920 により、本 run の段階評価は `low-confidence` である。 precision_feasible=False、selection_precision=0.0000、value_capture_ratio=0.0000、signal_sortino=0.0000 が価値面の中心指標となる。 research progress として best_selection_lcb=0.0362、 support_at_best_lcb=5、 tau_at_best_lcb=0.332440、 selection_brier_score=0.000002、 actionable_edge_rate=0.0000 を併記する。 選択 horizon は 6 本先で、期待収益率は 0.23% と推定された。

### Limitations
- 検証サンプルは 26 件で、学習全体 133 件に対してまだ小さい。
- ハイパーパラメータ探索は 9 候補の近傍探索であり、広域探索ではない。
- project_value_score は validation ベースの内部指標であり、実約定ベースの live PnL ではない。

## Conclusion
現時点の `project_value_score=0.0625` は、本プロジェクトが `low-confidence` の段階にあることを示す。 価値指標の中心は `selection_precision`, `value_capture_ratio`, `profit_factor`, `signal_sortino`, `selection_brier_score` である。現状の主なボトルネックは `actionable_edge_rate` と `precision_feasible` であり、今後は validation の拡張と forward simulation を通じて外部妥当性を詰めるのが次段階である。
