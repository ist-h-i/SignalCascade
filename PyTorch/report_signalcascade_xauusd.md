# SignalCascade XAUUSD Research Report

## Abstract
本レポートは、2026-03-25T03:03:25.386505+09:00 時点の最新成果物を用いて `SignalCascade` の再学習結果を再評価したものである。 採用モデルの `project_value_score` は 0.0602、 `utility_score` は 0.1019 であり、 段階評価は `low-confidence` となった。 `precision_feasible=False` で、 research 進捗として `best_selection_lcb=0.0362` を得た。 proposal / accepted horizon は 18 / none、 overlay 判定は `reduce`、 ポジションは 0.0000 である。

## 1. Experimental Setup
- 生成時刻: `2026-03-25T03:03:25.386505+09:00`
- 成果物: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`
- サンプル数: `train=80`, `validation=27`, `total=137`
- 候補数: `9`
- best validation loss: `2.125586` at epoch `1`
- 入力ソース: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv`
- 期間: `2025-12-26T08:00:00+09:00` から `2026-03-25T01:30:00+09:00`
- 行数: `used=2834`, `original=2834`

## 2. Metric System
- `utility_score`: precision, coverage, capture, overlay F1, directional accuracy, drawdown, calibration の複合値。
- `project_value_score`: utility に加えて profit factor, sortino, calibration を強めた事業価値指標。
- `profit_factor`: 利益総額と損失総額の比率。数値安定化のため `10.0` で上限 clip する。
- `signal_sortino`: 下方リスクで調整した価値指標。数値安定化のため `10.0` で上限 clip する。
- `selector_head_brier_score`: selector head 自体の calibration 誤差。0 に近いほど良い。
- `best_selection_lcb`: feasible 未達 run の比較用に、threshold 候補群で最大の Wilson LCB を追う研究指標。
- `pre_threshold_capture`: selector threshold を適用する前の仮想 capture。シグナル原石の有無を確認する。
- `alignment_rate` / `actionable_edge_rate`: return head と direction head の整合、およびコスト控除後の実行可能 edge の発生率を監視する。

## 3. Learning Analysis
| Metric | Value |
| --- | --- |
| best_epoch | 1 |
| epoch_count | 12 |
| best_epoch_ratio | 0.083333 |
| generalization_gap_at_best | 0.129091 |
| validation_drift_from_best_to_last | +1.503392 |
| early_peak | True |
- best validation loss は epoch 1/12 で最小化し、その後の validation total は +1.5034 悪化した。早期ピーク型で、direction head の汎化がまだ不安定である。
- selector calibration は `selector_head_brier_score=0.091045` まで改善した一方、`actionable_edge_rate=0.0000` のため、cost 控除後に採用可能な edge は validation 上で未出現である。
- threshold search の研究進捗として `best_selection_lcb=0.0362`、`support_at_best_lcb=5`、`precision_at_best_lcb=0.2000` を追跡する。feasible 未達でも threshold 面の改善度を比較できる。
- ラベル密度は horizon 30 で最大 (`nonflat_rate=0.9259`)、符号整合は horizon 18 が最大 (`alignment_rate=0.1111`) で、長短 horizon の学習難度が分かれている。

## 4. Validation Results
| Metric | Value |
| --- | --- |
| project_value_score | 0.060224 |
| utility_score | 0.101870 |
| precision_feasible | False |
| threshold_calibration_feasible | False |
| acceptance_precision | 0.000000 |
| acceptance_support | 0.000000 |
| proposal_count | 27 |
| accepted_count | 0 |
| proposal_coverage | 1.000000 |
| best_selection_lcb | 0.036223 |
| support_at_best_lcb | 5.000000 |
| precision_at_best_lcb | 0.200000 |
| tau_at_best_lcb | 0.164426 |
| acceptance_coverage | 0.000000 |
| value_per_anchor | 0.000000 |
| value_per_proposed | 0.000000 |
| value_per_accepted | null |
| value_capture_ratio | 0.000000 |
| pre_threshold_capture | 0.000000 |
| profit_factor | 0.000000 |
| signal_sortino | 0.000000 |
| alignment_rate | 0.015873 |
| actionable_edge_rate | 0.000000 |
| nonflat_rate | 0.619048 |
| selector_head_brier_score | 0.091045 |
| max_drawdown | 0.000000 |

## 5. Delta Vs Previous Current
| Metric | Delta | Previous | Current |
| --- | --- | --- | --- |
| project_value_score | -0.002276 | 0.062500 | 0.060224 |
| utility_score | +0.009821 | 0.092048 | 0.101870 |
| acceptance_precision | +0.000000 | 0.000000 | 0.000000 |
| acceptance_coverage | +0.000000 | 0.000000 | 0.000000 |
| best_selection_lcb | +0.000000 | 0.036223 | 0.036223 |
| alignment_rate | -0.033578 | 0.049451 | 0.015873 |
| value_capture_ratio | +0.000000 | 0.000000 | 0.000000 |
| profit_factor | +0.000000 | 0.000000 | 0.000000 |
| signal_sortino | +0.000000 | 0.000000 | 0.000000 |
| max_drawdown | +0.000000 | 0.000000 | 0.000000 |

## 6. Hyperparameter Optimization
| Candidate | Project Value | Utility | Precision | Best LCB | Align | Capture | Profit Factor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_09 | 0.060224 | 0.101870 | 0.000000 | 0.036223 | 0.015873 | 0.000000 | 0.000000 |
| candidate_03 | 0.062500 | 0.092989 | 0.000000 | 0.000000 | 0.042328 | 0.000000 | 0.000000 |
| candidate_04 | 0.056996 | 0.088015 | 0.000000 | 0.000000 | 0.042328 | 0.000000 | 0.000000 |
採用パラメータ: `{'epochs': 12, 'batch_size': 16, 'learning_rate': 0.0005, 'hidden_dim': 48, 'dropout': 0.1, 'weight_decay': 5e-05}`

## 7. Horizon Diagnostics
| H(4h) | Nonflat | Up | Down | Align | Actionable Edge |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.592593 | 0.222222 | 0.370370 | 0.000000 | 0.000000 |
| 2 | 0.407407 | 0.111111 | 0.296296 | 0.000000 | 0.000000 |
| 3 | 0.407407 | 0.074074 | 0.333333 | 0.000000 | 0.000000 |
| 6 | 0.518519 | 0.000000 | 0.518519 | 0.000000 | 0.000000 |
| 12 | 0.592593 | 0.000000 | 0.592593 | 0.000000 | 0.000000 |
| 18 | 0.888889 | 0.000000 | 0.888889 | 0.111111 | 0.000000 |
| 30 | 0.925926 | 0.000000 | 0.925926 | 0.000000 | 0.000000 |

## 8. Forecast Estimation
- Anchor time JST: `2026-03-25T05:00:00+09:00`
- Anchor close: `4420.9580`
- Proposed / accepted horizon: `18` / `None`
- Proposed direction classifier: `flat`
- Expected return direction: `short`
- Direction alignment: `False`
- Accepted signal: `False`
- Acceptance score source / score / threshold: `selector_probability` / `0.0003` / `null`
- Selector probability: `0.0003`
- Threshold status / origin: `infeasible` / `stored_policy`
- Hold probability / threshold: `0.3552` / `0.4416`
- Proposal forecast: `18` 本先 (`2026-03-28T05:00:00+09:00`), 予測終値 `4409.7773`, 期待収益率 `-0.25%`, 1σ帯 `[4366.7920, 4453.1857]`
| H(4h) | Forecast Time JST | Expected Return | Pred Close | 1σ Band |
| --- | --- | --- | --- | --- |
| 1 | 2026-03-25T09:00:00+09:00 | -0.03% | 4419.814502 | 4409.393279 .. 4430.260356 |
| 2 | 2026-03-25T13:00:00+09:00 | 0.02% | 4421.908907 | 4406.456584 .. 4437.415418 |
| 3 | 2026-03-25T17:00:00+09:00 | -0.09% | 4417.096926 | 4399.833466 .. 4434.428122 |
| 6 | 2026-03-26T05:00:00+09:00 | 0.08% | 4424.526787 | 4402.908465 .. 4446.251254 |
| 12 | 2026-03-27T05:00:00+09:00 | 0.08% | 4424.371699 | 4386.090132 .. 4462.987385 |
| 18 | 2026-03-28T05:00:00+09:00 | -0.25% | 4409.777297 | 4366.791994 .. 4453.185733 |
| 30 | 2026-03-30T05:00:00+09:00 | -0.02% | 4420.253086 | 4362.399494 .. 4478.873926 |

## 9. Project Value Assessment
project_value_score=0.0602、utility_score=0.1019 により、本 run の段階評価は `low-confidence` である。 precision_feasible=False、acceptance_precision=0.0000、value_capture_ratio=0.0000、signal_sortino=0.0000 が価値面の中心指標となる。 research progress として best_selection_lcb=0.0362、 support_at_best_lcb=5、 tau_at_best_lcb=0.164426、 selector_head_brier_score=0.091045、 actionable_edge_rate=0.0000 を併記する。 proposal horizon は 18 本先で、期待収益率は -0.25% と推定された。

### Limitations
- 検証サンプルは 27 件で、学習全体 137 件に対してまだ小さい。
- ハイパーパラメータ探索は 9 候補の近傍探索であり、広域探索ではない。
- project_value_score は validation ベースの内部指標であり、実約定ベースの live PnL ではない。

## Conclusion
現時点の `project_value_score=0.0602` は、本プロジェクトが `low-confidence` の段階にあることを示す。 価値指標の中心は `acceptance_precision`, `value_capture_ratio`, `profit_factor`, `signal_sortino`, `selector_head_brier_score` である。現状の主なボトルネックは `actionable_edge_rate` と `precision_feasible` であり、今後は validation の拡張と forward simulation を通じて外部妥当性を詰めるのが次段階である。
