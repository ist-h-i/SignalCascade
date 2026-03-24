# SignalCascade XAUUSD Research Report

## Abstract
本レポートは、2026-03-24T17:25:42.689277+09:00 時点の最新成果物を用いて `SignalCascade` の再学習結果を再評価したものである。 採用モデルの `project_value_score` は 0.0565、 `utility_score` は 0.0805 であり、 段階評価は `low-confidence` となった。 選択 horizon は 1 本先、 overlay 判定は `hold`、 ポジションは 0.0000 である。

## 1. Experimental Setup
- 生成時刻: `2026-03-24T17:25:42.689277+09:00`
- 成果物: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`
- サンプル数: `train=77`, `validation=26`, `total=133`
- 候補数: `8`
- best validation loss: `-1.204061` at epoch `14`
- 入力ソース: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv`
- 期間: `2025-12-24T09:00:00+09:00` から `2026-03-24T09:30:00+09:00`
- 行数: `used=2840`, `original=2840`

## 2. Metric System
- `utility_score`: precision, coverage, capture, overlay F1, directional accuracy, drawdown, calibration の複合値。
- `project_value_score`: utility に加えて profit factor, sortino, calibration を強めた事業価値指標。
- `profit_factor`: 利益総額と損失総額の比率。数値安定化のため `10.0` で上限 clip する。
- `signal_sortino`: 下方リスクで調整した価値指標。数値安定化のため `10.0` で上限 clip する。
- `selection_brier_score`: 採用確率の calibration 誤差。0 に近いほど良い。

## 3. Validation Results
| Metric | Value |
| --- | --- |
| project_value_score | 0.056478 |
| utility_score | 0.080539 |
| precision_feasible | False |
| threshold_calibration_feasible | False |
| selection_precision | 0.000000 |
| selection_support | 0 |
| coverage_at_target_precision | 0.000000 |
| value_capture_ratio | 0.000000 |
| profit_factor | 0.000000 |
| signal_sortino | 0.000000 |
| selection_brier_score | 0.240896 |
| max_drawdown | 0.000000 |

## 4. Delta Vs Previous Current
| Metric | Delta | Previous | Current |
| --- | --- | --- | --- |
| project_value_score | -0.527476 | 0.583953 | 0.056478 |
| utility_score | -0.296282 | 0.376821 | 0.080539 |
| selection_precision | -0.200000 | 0.200000 | 0.000000 |
| coverage_at_target_precision | -0.192308 | 0.192308 | 0.000000 |
| value_capture_ratio | -0.097836 | 0.097836 | 0.000000 |
| profit_factor | -10.000000 | 10.000000 | 0.000000 |
| signal_sortino | -10.000000 | 10.000000 | 0.000000 |
| max_drawdown | +0.000000 | 0.000000 | 0.000000 |

## 5. Hyperparameter Optimization
| Candidate | Project Value | Utility | Precision | Capture | Profit Factor |
| --- | --- | --- | --- | --- | --- |
| candidate_03 | 0.056478 | 0.080539 | 0.000000 | 0.000000 | 0.000000 |
| candidate_02 | 0.053873 | 0.082605 | 0.000000 | 0.000000 | 0.000000 |
| candidate_04 | 0.052187 | 0.072984 | 0.000000 | 0.000000 | 0.000000 |
採用パラメータ: `{'epochs': 16, 'batch_size': 16, 'learning_rate': 0.000625, 'hidden_dim': 48, 'dropout': 0.1, 'weight_decay': 5e-05}`

## 6. Forecast Estimation
- Anchor time JST: `2026-03-24T13:00:00+09:00`
- Anchor close: `4364.0350`
- Selected direction classifier: `long`
- Expected return direction: `short`
- Direction alignment: `False`
- Accepted signal: `False`
- Selection probability / threshold: `0.3947` / `null`
- Hold probability / threshold: `0.4629` / `0.3107`
- 選択 forecast: `1` 本先 (`2026-03-24T17:00:00+09:00`), 予測終値 `4060.1421`, 期待収益率 `-6.96%`, 1σ帯 `[3601.3982, 4577.3204]`
| H(4h) | Forecast Time JST | Expected Return | Pred Close | 1σ Band |
| --- | --- | --- | --- | --- |
| 1 | 2026-03-24T17:00:00+09:00 | -6.96% | 4060.142065 | 3601.398243 .. 4577.320384 |
| 2 | 2026-03-24T21:00:00+09:00 | 0.27% | 4375.713910 | 3765.316483 .. 5085.063183 |
| 3 | 2026-03-25T01:00:00+09:00 | -9.29% | 3958.707166 | 3326.793435 .. 4710.650882 |
| 6 | 2026-03-25T13:00:00+09:00 | 12.88% | 4926.069374 | 4410.816381 .. 5501.512051 |
| 12 | 2026-03-26T13:00:00+09:00 | 12.87% | 4925.777087 | 4178.795289 .. 5806.285839 |
| 18 | 2026-03-27T13:00:00+09:00 | -9.79% | 3936.624656 | 3366.010303 .. 4603.970960 |
| 30 | 2026-03-29T13:00:00+09:00 | 2.06% | 4453.889761 | 3773.099492 .. 5257.516809 |

## 7. Project Value Assessment
project_value_score=0.0565、utility_score=0.0805 により、本 run の段階評価は `low-confidence` である。 precision_feasible=False、selection_precision=0.0000、value_capture_ratio=0.0000、signal_sortino=0.0000 が価値面の中心指標となる。 選択 horizon は 1 本先で、期待収益率は -6.96% と推定された。

### Limitations
- 検証サンプルは 26 件で、学習全体 133 件に対してまだ小さい。
- ハイパーパラメータ探索は 8 候補の近傍探索であり、広域探索ではない。
- project_value_score は validation ベースの内部指標であり、実約定ベースの live PnL ではない。

## Conclusion
現時点の `project_value_score=0.0565` は、本プロジェクトが `low-confidence` の段階にあることを示す。 価値指標の中心は `selection_precision`, `value_capture_ratio`, `profit_factor`, `signal_sortino`, `selection_brier_score` であり、今後は validation の拡張と forward simulation を通じて外部妥当性を詰めるのが次段階である。
