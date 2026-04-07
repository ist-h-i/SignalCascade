# Logic Multiframe Candlestick Model

最終更新: 2026-04-07 JST

この文書は、`/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch` の現行実装を基準にしたロジック説明です。

2026-03-26 JST 以降の canonical target spec は `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/shape_aware_profit_maximization_model.md` です。この文書は target architecture ではなく、現時点の reference implementation がどう動くかを説明します。

旧版にあった次の記述は、現在の実装では廃止または置換されています。

- overlay の 4 クラス分類 (`hold / reduce / full_exit / hard_exit`)
- regime x horizon ごとの selection threshold
- precision 未達 run でも `positive` に見えやすい旧スコア運用

現行版の主眼は、`shape -> forecast / policy -> q_t` を artifact / diagnostics / dashboard まで同じ語彙で通すことです。

## 0. Review Follow-up Decisions

2026-04-07 JST 時点で、review follow-up に基づく canonical 判断は次で固定します。

- `display forecast` は `forecast_mu/sigma` (`mu_t`, `sigma_t`) です。
- `policy driver` は `policy_mu_t/policy_sigma_t` です。dashboard と artifact では forecast と分離して表示します。
- 学習 loss の canonical 主経路は `return_loss + shape_loss + profit_loss` のみです。
- 旧文書にあった `direction loss / overlay loss / consistency loss` は historical note であり、現行 code path には入りません。
- overlay は live review の source of truth ではありません。diagnostic replay overlay は derived replay evidence として扱います。
- `tune-latest` の candidate 採用順は blocked walk-forward objective を優先します。
- checkpoint 選定は `exact_log_wealth_minus_lambda_cvar` を基準にし、blocked walk-forward が得られる場合は `oof_epochs` 本の rolling mean を優先します。
- `g_t` は当面 scalar broadcast を維持します。2026-04-07 JST の blocked ablation では、horizon collapse を崩した主因は `tie_policy_to_forecast_head` であり、`disable_overlay_branch` 単独でも horizon-specific gate 追加でもありませんでした。
- 2026-04-07 JST の full blocked-first run では structural 上位が `tie_policy_to_forecast_head=true && disable_overlay_branch=true` に収束したため、現行 production current もこの組み合わせを canonical に使います。
- `current` alias では `current/source.json` に governance metadata を残し、`best_candidate` / `accepted_candidate` / `production current` の関係を明示します。`accepted_candidate` は blocked-first winner、`production current` は chart fidelity / sigma-band reliability / execution stability を重み付けした `user_value_score` の自動選抜結果です。`report_signalcascade_xauusd.md` は `current/research_report.md` の mirror です。
- `shape_aware_profit_maximization` は互換 identifier として残りますが、current evidence は `continuous posterior weighting` と `head coupling` までです。`shape_posterior_top_class_share={'1':1.0}` が残る間は、`shape-aware routing` / `regime-aware selection` と読まない契約にします。

## 1. 目的

モデルは 30 分足を基底データとして、複数時間足のローソク特徴量から次を同時に学習します。

1. `forecast_mu/sigma` による horizon 別 forecast distribution
2. `policy_mu/sigma` による horizon 別 decision driver
3. main timeframes の次バー shape
4. recurrent state を使った context carry-on

最終的な採用判定は threshold classifier ではなく、`policy_mu/sigma -> utility -> q_t` の経路で決まります。

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

## 6. Overlay と旧ラベルの扱い

`TrainingExample` には historical compatibility のため `overlay_target` と direction 由来の項目が残っています。ただし 2026-04-07 JST 時点の canonical code path では、

- `overlay_target` は batch / loss に入りません
- direction head / consistency head は存在しません
- overlay replay artifact は live review SoT ではなく derived replay evidence です

つまり overlay は「artifact lineage 上の補助情報」であって、現行の主学習契約ではありません。

## 7. モデル構造

現行版のネットワークは multi-timeframe encoder + latent fusion + recurrent state + shape-conditioned experts です。

### 7.1 Encoder と latent fusion

- main encoders: `4h / 1d / 1w`
- overlay encoders: `1h / 30m`
- 各 encoder は `Linear -> residual causal conv blocks -> pooled latent`
- 全 latent を concatenate して `latent_fusion` に通します

overlay branch は `disable_overlay_branch=true` のときゼロ埋めされ、cheap ablation として切り離せます。

### 7.2 状態ベクトル

融合 latent `h_t`、shape posterior `s_t`、state features 射影 `z_t`、recurrent memory `m_t` を連結し、

`state_vector = concat(h_t, s_t, z_t, m_t)`

を作ります。

### 7.3 出力 head

現行の canonical 出力は次です。

- `forecast_mu`, `forecast_sigma`
- `policy_mu`, `policy_sigma`
- `shape_posterior`
- `tradeability_gate`
- `memory_state`
- `main_shape_predictions`

`tie_policy_to_forecast_head=true` の場合だけ `policy_mu/sigma = forecast_mu/sigma` になります。

## 8. 損失関数

現行の総損失は次です。

`L = w_return * L_return + w_shape * L_shape + w_profit * L_profit`

### 8.1 Return loss

`forecast_mu/sigma` に対する heteroscedastic Huber loss です。

### 8.2 Shape loss

各 main timeframe の次バー shape に対する `smooth_l1_loss` の平均です。

### 8.3 Profit loss

`policy_mu/sigma` を differentiable な `smooth_policy_distribution` に通し、

`profit_loss = -mean(log(1 + pnl)) + lambda * CVaR_tail`

を最小化します。

### 8.4 現在使っていない loss

次は historical note であり、現行 code path には入りません。

- direction loss
- overlay loss
- consistency loss

## 9. 学習 split / blocked walk-forward / OOF

### 9.1 Holdout split

train / purge / validation の contiguous split は従来どおりです。

### 9.2 Blocked walk-forward

`walk_forward_folds` は diagnostics だけでなく、checkpoint 選定と `tune-latest` candidate ranking に使います。validation block を contiguous folds に分け、fold 平均で

- `average_log_wealth`
- `cvar_tail_loss`
- `turnover`
- `objective_log_wealth_minus_lambda_cvar`

を集計します。

### 9.3 OOF epoch window

`oof_epochs` は blocked walk-forward 指標の rolling mean window として扱います。checkpoint 選定 metric が `exact_log_wealth` または `exact_log_wealth_minus_lambda_cvar` のとき、single split の 1 epoch 値よりも、直近 `oof_epochs` 本の blocked 指標平均を優先します。

## 10. Selection Policy

現行の policy は `policy_service.py` にある exact utility path が canonical です。

### 10.1 Horizon row

各 horizon について次を計算します。

- `g_t`: scalar tradeability gate
- `mu_t_tilde = g_t * mu`
- `sigma_sq = max(sigma^2, min_policy_sigma^2)`
- `cost_h = horizon_cost * policy_cost_multiplier`
- `policy_utility`
- `q_t_candidate`

`g_t` は horizon-specific ではなく scalar を全 horizon に broadcast します。

2026-04-07 JST の review follow-up では、この scalar broadcast を当面維持する判断にしました。blocked ablation では `tie_policy_to_forecast_head` を有効化しただけで horizon distribution が `18` 固定から `30` 主軸へ動いた一方、`shape_posterior_top_class_share` は全 variant で class `1` 固定のままでした。したがって現時点の第一ボトルネックは gate 次元ではなく head decoupling と small sample 下の branch specialization failure であり、horizon-specific `g_t` は future variant として保留します。

### 10.2 Exact policy

各 horizon の utility を比較し、最大 utility の horizon を `policy_horizon` とします。選ばれた horizon について

`q_t = clip(tanh(mu_t_tilde / (gamma * sigma_sq)), -q_max, +q_max)`

を基礎にし、no-trade band に入る場合は `q_t = q_t_prev` のまま維持します。

### 10.3 Smooth policy

学習時の `profit_loss` では softmax-like な smooth path を使い、推論・diagnostics では exact path を使います。`exact_smooth_horizon_agreement` などは両者の乖離監査用です。

## 11. 推論出力と表示契約

artifact では forecast と policy driver を分けます。

- `display forecast`: `mu_t`, `sigma_t`
- `policy driver`: `policy_mu_t`, `policy_sigma_t`

`prediction.json` と `forecast_summary.json` には両方を入れ、`display_forecast.label` と `policy_driver.label` で明示します。dashboard もこの分離に従います。

## 12. 評価と tuning

### 12.1 Validation / diagnostics

主に見る指標は次です。

- `average_log_wealth`
- `realized_pnl_per_anchor`
- `cvar_tail_loss`
- `turnover`
- `no_trade_band_hit_rate`
- `exact_smooth_*`
- `policy_horizon_distribution`
- `stateful_evaluation`
- `blocked_walk_forward_evaluation`

### 12.2 Checkpoint audit

各 epoch について

- single split exact wealth
- exact wealth minus lambda CVaR
- blocked objective mean

を監査し、`checkpoint_audit` に保存します。

### 12.3 tune-latest candidate ordering

candidate は次を優先して並べます。

1. optimization gate pass
2. `blocked_objective_log_wealth_minus_lambda_cvar_mean`
3. `blocked_average_log_wealth_mean`
4. `blocked_turnover_mean`
5. 補助的に single split 指標

network hyperparameters だけでなく、`evaluation_state_reset_mode`, `min_policy_sigma`, `policy_cost_multiplier`, `policy_gamma_multiplier`, `q_max`, `cvar_weight`, `tie_policy_to_forecast_head`, `disable_overlay_branch` も cheap sweep に含めます。

## 13. 保存物

canonical artifact には少なくとも次が出ます。

- `config.json`
- `metrics.json`
- `prediction.json`
- `forecast_summary.json`
- `validation_summary.json`
- `policy_summary.csv`
- `horizon_diag.csv`
- `analysis.json`
- `research_report.md`

これにより、display forecast と policy driver の分離、checkpoint audit、blocked walk-forward、policy calibration sweep を一貫して追跡できます。
