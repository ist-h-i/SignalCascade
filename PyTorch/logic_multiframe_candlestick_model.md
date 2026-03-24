# Logic Multiframe Candlestick Model

最終更新: 2026-03-24 JST

この文書は、`/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch` の現行実装を基準にした仕様書です。旧版にあった次の記述は、現在の実装では廃止または置換されています。

- overlay の 4 クラス分類 (`hold / reduce / full_exit / hard_exit`)
- regime x horizon ごとの selection threshold
- precision 未達 run でも `positive` に見えやすい旧スコア運用

現行版の主眼は、`precision-first` の採用判定を first-class にし、学習・評価・推論・保存フォーマットを同じポリシーで通すことです。

## 1. 目的

モデルは 30 分足を基底データとして、複数時間足のローソク特徴量から次を同時に学習します。

1. `4h / 1d / 1w` の main horizon return
2. 各 horizon の方向分類 `{-1, 0, +1}`
3. main timeframes の次バー shape
4. `30m / 1h` を使った binary overlay risk filter (`reduce / hold`)
5. OOF ベースの correctness / selector / threshold policy

最終的な採用判定は「予測した」かどうかではなく、「precision 下側信頼限界を満たす signal だけを accept したか」で決まります。

## 2. 時間足と horizon

### 2.1 時間足

- base: `30m`
- overlay: `30m`, `1h`
- main: `4h`, `1d`, `1w`

各時間足バーは base の `30m` から resample して作成します。

### 2.2 horizon

現行の forecast horizon は次の 7 本です。

`H = {1, 2, 3, 6, 12, 18, 30}`

ここで horizon は `4h` バー単位です。たとえば `h = 3` は 12 時間先、`h = 30` は 120 時間先に相当します。

## 3. Close-Anchor 特徴量

各バー `t` について、6 次元特徴量

`x_t = (z_t, dz_t, chi_t, g_t, rho_t, nu_t)`

を作ります。これは `close_anchor.py` と `candlestick.py` の実装に一致します。

### 3.1 Shape

OHLC を `O_t, H_t, L_t, C_t`、微小値を `eps` とすると、レンジ

`R_t = (H_t - L_t) + eps`

を用いて shape を

- upper shadow: `(H_t - max(O_t, C_t)) / R_t`
- body: `(C_t - O_t) / R_t`
- lower shadow: `(min(O_t, C_t) - L_t) / R_t`

で定義します。

### 3.2 Path-Averaged Directional Balance

Directional balance `chi_t` は

`x = (C_t - O_t) / (2 * (H_t - L_t) + eps)`

`x_clip = clip(x, -0.5, 0.5)`

`chi_t = clip(1.5 * (x_clip / (1 - x_clip^2 + eps)), -1, 1)`

で計算します。

### 3.3 Local Scale

各時間足ごとにレンジ `H_t - L_t` の EMA を取り、

`a_t = EMA(H_t - L_t) + eps`

を local scale とします。

### 3.4 Feedback Gate

1 本前の shape を `s_{t-1} = (u_{t-1}, b_{t-1}, l_{t-1})` とし、時間足ごとに設定された gate 重み `w = (w_u, w_b, w_l)` を用いて

`g_t = tanh(w_u u_{t-1} + w_b b_{t-1} + w_l l_{t-1} + bias)`

を計算します。既定値では `bias = 0.0` なので、明示的に上書きしない限り `tanh` の入力は 3 項の線形結合です。初期 shape は `(0, 0, 0)` です。

### 3.5 Additive Close Anchor

Residual 項を

`r_t = beta0 + beta_v g_t + beta_x chi_t + beta_vx g_t chi_t`

と置くと、close anchor `L0_t` は

`L0_t = C_t + a_t * r_t`

です。さらに `EMA(L0)_t` を使って

`z_t = (L0_t - EMA(L0)_t) / a_t`

`dz_t = z_t - z_{t-1}`

`rho_t = (H_t - L_t) / a_t`

`nu_t = (V_t / EMA(V)_t) - 1`

を作り、最終特徴量を

`x_t = (z_t, dz_t, chi_t, g_t, rho_t, nu_t)`

とします。

### 3.6 時間足別パラメータ

既定値は次のとおりです。

| Timeframe | EMA window | Gate weights |
| --- | ---: | --- |
| `30m` | 32 | `(0.60, 0.30, -0.20)` |
| `1h` | 32 | `(0.60, 0.30, -0.20)` |
| `4h` | 24 | `(0.75, 0.40, -0.25)` |
| `1d` | 20 | `(0.85, 0.45, -0.30)` |
| `1w` | 13 | `(0.95, 0.50, -0.35)` |

Residual の既定値は共通で `beta0 = 0.0`, `beta_v = 0.05`, `beta_x = 0.15`, `beta_vx = 0.05` です。

## 4. 学習サンプルの組み立て

### 4.1 必要データ量

学習サンプル構築には最低 `512` 本の `30m` base bar が必要です。

### 4.2 Sequence window

anchor は `4h` バーで取り、各時間足で anchor 時点までの sequence を集めます。

| Group | Timeframe | Window |
| --- | --- | ---: |
| main | `4h` | 48 |
| main | `1d` | 21 |
| main | `1w` | 8 |
| overlay | `1h` | 48 |
| overlay | `30m` | 96 |

main shape target は「各 main timeframe の次バー shape」です。

### 4.3 Regime 特徴

regime は `30m` ベースの情報から作ります。

- session: `asia / london / ny`
- realized volatility: 直近 48 本 `30m` の平均絶対 log return
- baseline volatility: 直近 192 本 `30m` の平均絶対 log return
- volatility ratio: `realized / baseline`
- trend strength: `abs(chi_4h) + 0.5 * abs(dz_4h)`

保存される regime feature は

`(session_asia, session_london, session_ny, clipped_volatility_ratio, trend_strength)`

です。`regime_id` は

`session|volatility_bin|trend_bin`

形式で保存されます。

重要なのは、現行版では regime は特徴量やラベル生成には使う一方、selection threshold を regime ごとに分ける用途には使っていない、という点です。

## 5. Main ラベル生成

各 anchor と horizon `h` について、`4h` future path

`r_{t,1}, ..., r_{t,h}`

を

`r_{t,k} = log(C_{t+k} / C_t)`

で作ります。

- target return: `r_{t,h}`
- long MAE: `max(0, max(-r_{t,k}))`
- short MAE: `max(0, max(r_{t,k}))`
- long MFE: `max(0, max(r_{t,k}))`
- short MFE: `max(0, max(-r_{t,k}))`
- cost: `c_h = base_cost * sqrt(h)`

### 5.1 方向ラベル

main move threshold は

`delta_h = c_h + m_delta * sigma_t * sqrt(h)`

で、`m_delta` は次の regime 補正付きです。

- まず `delta_multiplier = 1.35`
- high volatility なら `x 1.15`, low なら `x 0.90`
- trend なら `x 0.85`, range なら `x 1.05`
- asia session ならさらに `x 1.05`

main MAE threshold は

`eta_h = m_mae * sigma_t * sqrt(h)`

で、`m_mae` は次の補正付きです。

- まず `mae_multiplier = 0.95`
- high volatility なら `x 1.10`, low なら `x 0.95`
- trend なら `x 0.90`, range なら `x 1.05`
- asia session ならさらに `x 1.05`

方向ラベルは

- `+1` if `target_return >= delta_h` and `long_MAE <= eta_h`
- `-1` if `target_return <= -delta_h` and `short_MAE <= eta_h`
- `0` otherwise

です。

### 5.2 Direction Weight

学習時の direction loss には clean signal を重くする重みを掛けます。

`denom = sigma_t * sqrt(h) + 1e-6`

`excess = max(abs(target_return) - c_h, 0) / denom`

ここで `sqrt(h)` を掛けるのは、realized volatility を horizon 長に応じたスケールへ揃え、return head で使う `scale_h` と同じ基準にそろえるためです。

`w = 1 + clean_weight_return_scale * excess`

non-flat ラベルならさらに `clean_weight_bonus` を加えます。

そのうえで、long なら `long_MFE / long_MAE`、short なら `short_MFE / short_MAE` を最大 `4.0` まで使い、

`w += clean_weight_ratio_scale * ratio`

を加え、最終的に `[1, 6]` に clip します。

既定値は

- `clean_weight_return_scale = 0.75`
- `clean_weight_bonus = 0.65`
- `clean_weight_ratio_scale = 0.35`

です。

## 6. Overlay ラベル生成

overlay は多値 exit policy ではなく、binary risk filter です。

- `0 = reduce`
- `1 = hold`

### 6.1 Primary Direction

各 anchor で non-zero direction target のうち、`direction_weight` が最大のものを primary direction とします。non-zero がなければ overlay target は `reduce` です。

### 6.2 Binary Hold Label

anchor 以降 8 本の `30m` future bar を使います。primary direction を `d in {-1, +1}` とすると、directed path return は

`p_k = d * log(C_{t+k} / C_t)`

です。

- final return: `p_8`
- adverse excursion: `abs(min(min_k p_k, 0))`

overlay threshold は

`delta_overlay = c_1 + overlay_delta_multiplier * sigma_t * sqrt(8)`

`eta_overlay = overlay_mae_multiplier * sigma_t * sqrt(8)`

で、既定値は

- `overlay_delta_multiplier = 0.75`
- `overlay_mae_multiplier = 0.7`

です。overlay label は

- `hold` if `final_return >= delta_overlay` and `adverse_excursion <= eta_overlay`
- otherwise `reduce`

です。

## 7. モデル構造

現行版のネットワークは「各時間足 encoder + simple fusion」で、旧文書にあった coarse-to-fine gated fusion とは異なります。

### 7.1 TemporalEncoder

各時間足は共通構造の `TemporalEncoder` で処理します。

1. 入力 `feature_dim=6` を `hidden_dim` に線形射影
2. 1D residual temporal block を dilation `1, 2, 4` で 3 段適用
3. 各 block は `Conv1d(kernel=3) -> GELU -> Dropout -> residual`
4. 最終時点ベクトルに `LayerNorm`

### 7.2 Main Fusion

`4h / 1d / 1w` の latent を concatenate し、MLP で融合します。

`concat(main_4h, main_1d, main_1w) -> Linear -> GELU -> Dropout -> Linear -> GELU`

### 7.3 出力 head

main fusion から次を出力します。

- `mu`: 各 horizon の標準化 log return 平均
- `sigma`: 各 horizon の標準化不確実性。`softplus(head) + 1e-4`
- `direction_logits`: 各 horizon の 3 クラス logits

ここで return head は raw return ではなく、

`scale_h = sigma_realized * sqrt(h) + return_scale_epsilon`

で正規化した空間を学習します。学習時の target は

`y_std = clip(y_raw / scale_h, -standardized_return_clip, +standardized_return_clip)`

で、推論時・評価時には

`mu_raw = mu_std * scale_h`

`sigma_raw = sigma_std * scale_h`

で元の return 単位へ戻します。

各 main timeframe latent から次を出力します。

- `shape_predictions[tf]`: 次バー shape 3 成分

overlay 側は `1h / 30m` latent と edge を結合して binary head に入れます。

`edge = mu / sigma`

`overlay_input = concat(fused_main, latent_1h, latent_30m, edge)`

`overlay_logits = MLP(overlay_input)`

## 8. 損失関数

総損失は

`L = L_return + 0.35 L_dir + 0.25 L_shape + 0.10 L_overlay + 0.20 L_consistency`

です。

### 8.1 Return loss

Heteroscedastic Huber loss を使います。対象は raw return ではなく、上記の standardized return です。

`e = y - mu`

`q = min(|e|, delta)`

`l = |e| - q`

`Huber = 0.5 q^2 + delta l`

`L_return = mean(Huber / sigma^2 + log(sigma))`

既定値は `delta = 0.02`、`standardized_return_clip = 6.0`、`return_scale_epsilon = 1e-4` です。

### 8.2 Direction loss

3 クラス softmax に対する focal loss です。

- `gamma = 1.5`
- class alpha = `[1.2, 0.7, 1.2]`
- 各サンプルに `direction_weight` を掛ける

### 8.3 Shape loss

各 main timeframe の `smooth_l1_loss` の平均です。

### 8.4 Overlay loss

`binary_cross_entropy_with_logits` を使った binary loss です。

### 8.5 Direction Consistency loss

return head と direction head の整合を取るため、`(mu, sigma)` から implied direction probability を作り、direction logits と KL で近づけます。

flat band は固定値ではなく、horizon ごとの direction threshold を標準化した

`b_h = delta_h / scale_h`

を使います。したがって、

`p_up = 1 - Phi((b_h - mu) / sigma)`

`p_down = Phi((-b_h - mu) / sigma)`

`p_flat = 1 - p_up - p_down`

を作り、

`L_consistency = KL(log_softmax(direction_logits) || implied_probs)`

を最小化します。

## 9. 学習 split と OOF

### 9.1 Holdout split

sample 数を `N`、validation 希望数を

`n_valid = max(1, int(N * (1 - train_ratio)))`

とし、`n_valid <= N - 1` に切ります。

purge 数は

`purge = min(max_horizon, max(N - n_valid - 1, 0))`

です。最終的に

- train: `[0, train_end)`
- purge: `[train_end, validation_start)`
- validation: `[validation_start, N)`

となるように切ります。

これにより、purge と validation holdout が両立します。

### 9.2 Walk-forward OOF

policy fitting には `walk_forward_folds = 3` の OOF snapshot を使います。ここで fit するのは予測器本体ではなく、

- correctness model
- selector model
- selection threshold
- overlay threshold

です。

## 10. Selection Policy

現行版の policy は `policy_service.py` に実装されています。

### 10.1 Per-horizon row の特徴

各 horizon row から次を作ります。

- `edge = |mu| / sigma`
- `prob_gap = top1(direction_prob) - top2(direction_prob)`
- `sign_agreement = actionable horizons のうち現在 sign と一致する比率`
- `actionable_sign = argmax(p_down, p_up)` に基づく実行方向
- `actionable_edge = max(actionable_sign * mu - cost, 0)`

correctness model の入力は

- `edge`
- `p_down`, `p_flat`, `p_up`
- `prob_gap`
- `sign_agreement`
- session one-hot
- volatility ratio
- trend strength
- normalized horizon

selector model の入力は上記に加えて

- `q`
- `top_gap`

を含みます。

両モデルとも、標準化した特徴量に対する線形 logistic model です。学習には class imbalance を反映した `pos_weight` と BCE を使い、selector 側には `selector_brier_weight = 0.2` の Brier 項を追加します。

### 10.2 Correctness score と selector score

correctness model の出力を `q` とし、row score は

`score = q^alpha * actionable_edge^(1 - alpha)`

です。既定値は `alpha = 0.7` です。

ただし次の場合は score を `0` にします。

- `actionable_sign == 0`
- `actionable_edge <= 0`

selected horizon は

1. `score` 最大
2. tie の場合は `actionable_edge` 最大
3. さらに tie の場合は `|mu|` 最大

で決まります。

### 10.3 Threshold calibration

selection threshold は selected horizon に対する `selector_probability` 上で校正します。現行版で実際に使う scope は `global` のみで、`by_horizon` は保存フォーマット互換のために `null` placeholder を保持し、`by_regime` は空です。

候補 threshold `tau` ごとに

- `selected_count(tau)`
- `success_count(tau)`
- `precision(tau)`

を計算し、Wilson 下側信頼限界

`LCB(tau) = (p + z^2/(2n) - z * sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n)`

を使います。ここで `p = k / n`, `k = success_count`, `n = selected_count`, `z = 1.96` です。

feasible 条件は

- `selected_count >= selection_min_support`
- `LCB >= precision_target`

です。既定値は

- `selection_min_support = 5`
- `precision_target = 0.8`

です。

feasible な候補がある場合は coverage 最大、同率なら threshold が低い方を採用します。feasible 候補が 1 つもない場合は

- `selection_threshold = null`
- `selection_thresholds.meta.global.feasible = false`

さらに、infeasible run の比較用として

- `best_selection_lcb`
- `support_at_best_lcb`
- `precision_at_best_lcb`
- `tau_at_best_lcb`

を保存します。

となり、その run は accept できません。

### 10.4 Overlay threshold

overlay threshold は `hold_probability` に対する global threshold だけを持ちます。こちらは feasible gate ではなく、LCB と support が最大になる threshold を選び、`[0.05, 0.95]` に clip します。

### 10.5 Alignment gate

direction classifier と return head の矛盾 signal は reject します。

- `expected_direction = sign(mu_selected)`
- `direction_alignment = predicted_sign != 0 and predicted_sign == expected_direction`

accept 条件は次の全てです。

- `actionable_sign != 0`
- `actionable_edge > 0`
- `direction_alignment == true`
- `selection_threshold is not null`
- `selector_probability >= selection_threshold`

threshold を適用する前の `pre_threshold_eligible` と、そのときの `pre_threshold_position` も評価用に保存します。

## 11. 推論出力

推論時は selection policy を通したうえで次を返します。

- `selected_horizon`
- `selected_direction`
- `expected_log_returns`
- `predicted_closes`
- `uncertainties`
- `accepted_signal`
- `selection_probability`
- `selection_threshold`
- `correctness_probability`
- `hold_probability`
- `hold_threshold`
- `overlay_action`
- `direction_alignment`
- `regime_id`
- `current_close`

ポジションは

`raw_position = tanh(position_scale * mu_selected / sigma_selected)`

をベースにし、accept されなければ `0`、最終ポジションは

実装では `sigma_selected` をそのまま割らず、`max(sigma_selected, 1e-6)` を分母に使って極小分散時の不安定化を防ぎます。

`position = raw_position * hold_probability`

です。overlay action は

- `hold` if `hold_probability >= hold_threshold`
- `reduce` otherwise

で決まります。

## 12. 評価と tuning

### 12.1 主な validation metrics

現行版で重視するのは次です。

- `selection_precision`
- `selection_support`
- `precision_feasible`
- `threshold_calibration_feasible`
- `best_selection_lcb`
- `support_at_best_lcb`
- `precision_at_best_lcb`
- `coverage_at_target_precision`
- `pre_threshold_capture`
- `alignment_rate`
- `actionable_edge_rate`
- `selection_brier_score`
- `value_capture_ratio`
- `directional_accuracy`
- `overlay_accuracy`

さらに `horizon_diagnostics` として各 horizon ごとに

- `nonflat_rate`
- `up_rate`
- `down_rate`
- `alignment_rate`
- `actionable_edge_rate`

を保存します。

`profit_factor` と `signal_sortino` は保存はしますが、accepted signal が `selection_min_support` 未満なら `0` とし、sparse signal の極端値が tuning を壊さないようにしています。

### 12.2 Project value score

`utility_score` と `project_value_score` は依然として保存しますが、`precision_feasible == false` の run には強い penalty を掛けます。

- `utility_score *= 0.5`
- `project_value_score *= 0.25`

つまり、主 KPI 未達 run が見かけ上 `positive` になりにくい仕様です。

### 12.3 Leaderboard の並び順

tuning の candidate 比較は、現在は次の順です。

1. `precision_feasible` が真
2. `selection_precision` が高い
3. `coverage_at_target_precision` が高い
4. `best_selection_lcb` が高い
5. `support_at_best_lcb` が高い
6. `alignment_rate` が高い
7. `pre_threshold_capture` が高い
8. `selection_brier_score` が低い
9. `value_capture_ratio` が高い
10. `directional_accuracy` が高い
11. `best_validation_loss` が低い

`profit_factor` や `signal_sortino` は leaderboard の主ソートキーには使いません。

## 13. 保存物

artifact には少なくとも次が出力されます。

- `config.json`
- `metrics.json`
- `prediction.json`
- `forecast_summary.json`
- `analysis.json`
- `research_report.md`

これにより、モデル性能だけでなく

- threshold が feasible だったか
- signal が no-trade だったか
- direction の不一致が起きたか
- どの horizon で label 密度や整合率が偏ったか

まで追跡できます。

## 14. 現時点の制約

現行実装は前進していますが、まだ次の制約があります。

1. selection threshold は現状 `global` のみで、regime 別 partial pooling までは未実装です。
2. selector は軽量な線形 logistic model で、複雑な non-linear meta-model は使っていません。
3. standardized return 学習は導入済みですが、長期 forward simulation は未導入です。
4. overlay は binary risk filter であり、多段 exit policy ではありません。

## 15. 旧版からの読み替え

旧文書を読んでいる場合は、次の読み替えをしてください。

- `full_exit / hard_exit` は削除され、overlay は `reduce / hold` の 2 値です。
- regime 別 selection threshold は現行版では使いません。
- `accepted_signal` は単なる classifier 出力ではなく、selector threshold と alignment gate を通過したものだけです。
- `selection_threshold = 1.0` を既定 fallback とする運用はやめ、feasible でなければ `null` を返します。
- precision 未達 run は `project_value_score` が残っていても、研究上は low-confidence / infeasible として扱います。
