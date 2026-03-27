# Shape-Aware Profit Maximization Model

最終更新: 2026-03-26 JST

この文書は、`SignalCascade` の新しい canonical spec です。
目的は、初期アイデアだった

- 多層・多尺度 causal convolution
- shape 判定
- 内部状態ベクトル

を残したまま、最終目的関数だけを `誤差最小化` から `売買利益最大化` へ置き換えた、一貫したモデル契約を固定することです。

この文書は target architecture を定義するものであり、現時点の実装そのものを説明する文書ではありません。現行コードの説明は `logic_multiframe_candlestick_model.md` を参照してください。

## 1. モデル全体の流れ

全体像は次です。

```text
X_t
  -> multi-layer / multi-scale causal convolution
  -> h_t
  -> shape classification
  -> s_t
  -> state integration
  -> v_t
  -> return distribution estimation
  -> (mu_t, sigma_t^2)
  -> profit-maximizing policy
  -> q_t*
  -> market realization
  -> pi_{t+1}
```

各記号の意味は次です。

- `X_t`: 直近窓の入力テンソル
- `h_t`: 畳み込みで得た形状特徴
- `s_t`: shape class の確率ベクトル
- `v_t`: 内部状態ベクトル
- `(mu_t, sigma_t^2)`: 次期収益率の条件付き平均・分散
- `q_t*`: 最適ポジション
- `pi_{t+1}`: 実現損益

## 2. 入力表現

OHLCV をそのまま入れず、因果的に正規化した shape-aware feature を使います。

時刻 `t` のバー `(O_t, H_t, L_t, C_t, V_t)` に対し、たとえば次を定義します。

```text
x_t = [ell_t, b_t, u_t, d_t, nu_t, zeta_t]^T
```

```text
ell_t  = log(C_t / C_{t-1})
b_t    = (C_t - O_t) / ATR_n(t)
u_t    = (H_t - max(O_t, C_t)) / ATR_n(t)
d_t    = (min(O_t, C_t) - L_t) / ATR_n(t)
nu_t   = log(V_t / EMA_n(V)_t)
zeta_t = (C_t - EMA_n(C)_t) / ATR_n(t)
```

意味は次です。

- `ell_t`: 対数収益率
- `b_t`: 実体の大きさ
- `u_t`, `d_t`: 上ヒゲ・下ヒゲ
- `nu_t`: 出来高の異常度
- `zeta_t`: 移動平均からの乖離

直近 `L` 本を並べて

```text
X_t = [x_{t-L+1}, x_{t-L+2}, ..., x_t] in R^{C x L}
```

を作ります。すべて `t` 以前の情報のみで構成し、未来情報は使いません。

## 3. 多層・多尺度の畳み込み

短期・中期・長期の局所形状を別々に捉えるため、branch `b = 1, ..., B` を持つ multi-scale causal 1D convolution を使います。

各 branch は異なる kernel width や dilation を持ちます。

```text
H_t^{(b,0)} = X_t
H_t^{(b,l)} = rho(W^{(b,l)} *_d H_t^{(b,l-1)} + c^{(b,l)}) + R^{(b,l)} H_t^{(b,l-1)}
```

ここで

- `*_d`: dilation 付き causal convolution
- `rho(.)`: 非線形活性化
- `R^{(b,l)}`: 残差接続

です。

最後に各 branch を pooling して

```text
h_t^{(b)} = Pool(H_t^{(b,L_b)})
h_t = [h_t^{(1)}; h_t^{(2)}; ...; h_t^{(B)}]
```

を得ます。

この `h_t` は、反転、継続、ボラ収縮、ブレイク前圧縮、急変後ノイズのような局所形状の抽出結果です。

## 4. shape 判定

畳み込み特徴 `h_t` から、時刻 `t` の shape class を確率として出します。

```text
s_t = softmax(U_s h_t + b_s) in Delta^{M-1}
```

```text
s_t = [s_{t,1}, ..., s_{t,M}]^T
sum_m s_{t,m} = 1
```

`s_{t,m}` は「現在のパターンが shape `m` である確率」です。

shape class は事前定義でも潜在クラスでも構いませんが、この spec では重要なのは `s_t` を単なる補助ラベルではなく、後段の収益率分布推定に直接使うことです。

## 5. 内部状態ベクトル

shape feature `h_t`、shape posterior `s_t`、通常の統計特徴 `z_t` を統合し、内部メモリ `m_t` を更新します。

```text
m_t = tanh(A m_{t-1} + B[h_t; s_t; z_t] + b_m)
v_t = [h_t; s_t; z_t; m_t]
```

`z_t` の例は次です。

- realized volatility
- spread
- imbalance
- 長短移動平均差
- 直近 turnover

これにより `v_t` は

- 形状特徴
- 形状判定
- 統計特徴
- 時間的メモリ

を明示的に持つ状態ベクトルになります。

## 6. shape-conditioned return distribution

各 shape `m` ごとに、次期収益率の条件付き平均と分散を出します。

```text
mu_{t,m} = a_m^T v_t + alpha_m
log sigma_{t,m}^2 = g_m^T v_t + beta_m
```

shape posterior `s_t` を混合係数として使い、

```text
mu_t = sum_m s_{t,m} mu_{t,m}
E_t[r_{t+1}^2] = sum_m s_{t,m} (sigma_{t,m}^2 + mu_{t,m}^2)
sigma_t^2 = sum_m s_{t,m} (sigma_{t,m}^2 + mu_{t,m}^2) - mu_t^2
```

とします。

この構造により、

- 反転 shape は平均が小さく分散が大きい
- 継続 shape は平均が正で分散が比較的低い
- ノイズ shape は平均が 0 近く分散だけ高い

という振る舞いを学習できます。

## 7. tradeability gate

単に `mu_t > 0` だから買うのではなく、shape から「取るべき局面か」を決めます。

shape ごとの tradeability weight `omega_m in [0,1]` を学習し、

```text
g_t = sum_m omega_m s_{t,m}
mu_t_tilde = g_t mu_t
```

とします。

これで、平均が少し正でも shape がノイズならエントリー圧が下がります。

## 8. 利益最大化政策

時刻 `t` のポジションを `q_t`、次期収益率を `r_{t+1}`、売買コストを `c_t` とします。

1 期損益は

```text
pi_{t+1} = q_t r_{t+1} - c_t |q_t - q_{t-1}|
```

です。

厳密な目的は対数資産成長率の最大化です。

```text
q_t* = argmax_{|q| <= q_max} E_t[ log(1 + q r_{t+1} - c_t |q - q_{t-1}|) ]
```

短期収益率が小さいとみなして `log(1+x) ~= x - x^2/2` を使うと、実用上の policy objective は

```text
U_t(q) = mu_t_tilde q - (gamma / 2) q^2 sigma_t^2 - c_t |q - q_{t-1}|
```

となり、

```text
q_t* = argmax_{|q| <= q_max} U_t(q)
```

を解きます。

## 9. ノートレード帯

上の目的関数には `|q - q_{t-1}|` が入るため、no-trade band が自然に現れます。

最適条件は

```text
|mu_t_tilde - gamma sigma_t^2 q_{t-1}| <= c_t
```

のとき

```text
q_t* = q_{t-1}
```

です。

有界ポジションまで含めた最終形は

```text
q_t* = Projection_{[-q_max, q_max]}(q_hat_t)
```

```text
if mu_t_tilde > gamma sigma_t^2 q_{t-1} + c_t:
    q_hat_t = (mu_t_tilde - c_t) / (gamma sigma_t^2)
elif |mu_t_tilde - gamma sigma_t^2 q_{t-1}| <= c_t:
    q_hat_t = q_{t-1}
else:
    q_hat_t = (mu_t_tilde + c_t) / (gamma sigma_t^2)
```

です。

## 10. 実現損益と学習目的

実現損益は

```text
pi_{t+1} = q_t* r_{t+1} - c_t |q_t* - q_{t-1}*|
W_{t+1} = W_t (1 + pi_{t+1})
```

で、最終学習目的は

```text
theta* = argmax_theta [ sum_t log(1 + pi_{t+1}(theta)) - lambda_tail * CVaR_alpha(-pi(theta)) ]
```

です。

ここで `theta` は

- convolution kernel
- shape classifier `U_s`
- state update `(A, B)`
- shape expert `{a_m, g_m}`
- tradeability weight `omega_m`

などの全パラメータです。

この時点でモデルは「未来を当てるモデル」ではなく、「資産成長を最大化する政策モデル」になります。

## 11. 実装上の注意

学習時は、非滑らかな項を次で近似してよいものとします。

```text
|x| ~= sqrt(x^2 + epsilon)
Projection_{[-q_max, q_max]}(x) ~= q_max * tanh(x / q_max)
```

ただし推論時の実売買ロジックは、piecewise の厳密ルールに戻した方が解釈しやすいです。

## 12. 初期アイデアが残る位置

初期アイデアは役割分担して残ります。

- 多層畳み込み: `X_t -> h_t`
- shape 判定: `h_t -> s_t`
- 内部状態ベクトル: `v_t = [h_t; s_t; z_t; m_t]`

旧来の単一重み行列に相当する役割は、次へ分解されます。

- `U_s`: shape 判定
- `(A, B)`: 状態更新
- `{a_m, g_m}`: shape expert
- `omega_m`: tradeability weight

## 13. 完成形の要約

```text
X_t = [x_{t-L+1}, ..., x_t]
h_t = MSCNN_theta(X_t)
s_t = softmax(U_s h_t + b_s)
m_t = tanh(A m_{t-1} + B[h_t; s_t; z_t] + b_m)
v_t = [h_t; s_t; z_t; m_t]
mu_{t,m} = a_m^T v_t + alpha_m
log sigma_{t,m}^2 = g_m^T v_t + beta_m
mu_t = sum_m s_{t,m} mu_{t,m}
sigma_t^2 = sum_m s_{t,m} (sigma_{t,m}^2 + mu_{t,m}^2) - mu_t^2
g_t = sum_m omega_m s_{t,m}
mu_t_tilde = g_t mu_t
q_t* = argmax_{|q| <= q_max} { mu_t_tilde q - (gamma / 2) q^2 sigma_t^2 - c_t |q - q_{t-1}| }
pi_{t+1} = q_t* r_{t+1} - c_t |q_t* - q_{t-1}*|
theta* = argmax_theta [ sum_t log(1 + pi_{t+1}(theta)) - lambda_tail * CVaR_alpha(-pi(theta)) ]
```

## 14. このモデルの本質

この完成版の本質は次の 3 層を一体化した点にあります。

1. 形を読むネットワーク
2. 形ごとの収益率分布推定
3. コスト込み最適ポジション決定

旧来の

```text
v_t -> y_hat_{t+1}
```

ではなく、

```text
X_t -> h_t -> s_t -> v_t -> (mu_t, sigma_t^2) -> q_t* -> pi_{t+1}
```

が canonical flow です。
