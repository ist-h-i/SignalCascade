# ロジック説明書 v1.2（完全置換版）
## マルチタイムフレーム・Close Anchor 予測モデル
### 4時間足主モデル + 1時間足 / 30分足離脱オーバーレイ  
### Path-Averaged Directional Balance と加算型 \(L_0\) を組み込んだ再構築版

---

## 1. 文書の目的

本書は、v1.2 における数式仕様・構造仕様・学習仕様を統合的に記述する。  
v1.1 からの主要変更点は以下である。

1. close-only 改善案として Path-Averaged Directional Balance を導入した
2. \(L_0\) を乗算型から加算型の close anchor 残差構造へ変更した
3. 補正量を価格単位へ変換するスケール \(a_i\) を導入した
4. シミュレーションメソッド / パラメータ最適化メソッドを、v1.2 構造に合わせて再定義した

---

## 2. 記号と前提

### 2.1 時間足集合

主モデルで用いる時間足集合を

\[
\mathcal T_{\text{main}}=\{4h,1d,1w\}
\]

とする。

離脱オーバーレイで用いる時間足集合を

\[
\mathcal T_{\text{ov}}=\{1h,30m\}
\]

とする。

全時間足集合を

\[
\mathcal T=\mathcal T_{\text{main}}\cup\mathcal T_{\text{ov}}
\]

とする。

### 2.2 horizon

4 時間足基準の保有 horizon を

\[
\mathcal H=\{1,2,3,6,12,18,30\}
\]

とする。  
各値は 4h バー本数であり、対応する時間は以下である。

- 1: 4h
- 2: 8h
- 3: 12h
- 6: 1d
- 12: 2d
- 18: 3d
- 30: 5d

### 2.3 足データ

各時間足 \(\tau\) の \(i\) 本目のローソク足を

\[
B_i^{(\tau)}=
\left(
O_i^{(\tau)},H_i^{(\tau)},L_i^{(\tau)},C_i^{(\tau)},V_i^{(\tau)}
\right)
\]

とする。  
本理論における `d(i)` は常に終値である。

\[
d_i^{(\tau)}=C_i^{(\tau)}
\]

---

## 3. 足形状の自己教師ベクトル \(Q\)

各足から次の自己教師ベクトルを定義する。

\[
R_i^{(\tau)} = H_i^{(\tau)}-L_i^{(\tau)}+\varepsilon_R
\]

\[
u_i^{(\tau)}
=
\frac{
H_i^{(\tau)}-\max(O_i^{(\tau)},C_i^{(\tau)})
}{
R_i^{(\tau)}
}
\]

\[
b_i^{(\tau)}
=
\frac{
C_i^{(\tau)}-O_i^{(\tau)}
}{
R_i^{(\tau)}
}
\]

\[
\ell_i^{(\tau)}
=
\frac{
\min(O_i^{(\tau)},C_i^{(\tau)})-L_i^{(\tau)}
}{
R_i^{(\tau)}
}
\]

\[
Q_i^{(\tau)}
=
\begin{bmatrix}
u_i^{(\tau)}\\
b_i^{(\tau)}\\
\ell_i^{(\tau)}
\end{bmatrix}
\in\mathbb R^3
\]

意味は以下で固定する。

- 第1成分 \(u\): 上髭
- 第2成分 \(b\): 符号付き実体
- 第3成分 \(\ell\): 下髭

このとき理想的には

\[
u_i^{(\tau)}+|b_i^{(\tau)}|+\ell_i^{(\tau)}\approx 1
\]

が成り立つ。

---

## 4. Path-Averaged Directional Balance \(\chi\)

### 4.1 発想
OHLC しか観測されない場合、1本の足の内部経路は通常不明である。  
代表的な単純経路として、以下の2通りを考える。

1. \(O\to H\to L\to C\)
2. \(O\to L\to H\to C\)

この2経路において、  
\[
\frac{|上昇量|}{|上昇量|+|下落量|}
\]
に対応する量を作り、それを \([-1,1]\) の符号付き量に変換し、2経路平均した指標を close-only 改善量として使う。

### 4.2 補助記号

\[
A_i^{(\tau)} = 2\bigl(H_i^{(\tau)}-L_i^{(\tau)}\bigr)
\]

\[
B_i^{(\tau)} = C_i^{(\tau)}-O_i^{(\tau)}
\]

\[
X_i^{(\tau)}
=
\frac{B_i^{(\tau)}}{A_i^{(\tau)}+\varepsilon_X}
\]

理論上 \(|B_i|\le H_i-L_i\) なので、\(|X_i|\le \frac12\) が基本となる。  
数値安定のため、実装上は

\[
X_i^{(\tau)}
=
\operatorname{clip}
\left(
\frac{B_i^{(\tau)}}{A_i^{(\tau)}+\varepsilon_X},
-\frac12,\frac12
\right)
\]

とする。

### 4.3 2経路平均から得られる量

2経路平均から得られるスカラーを

\[
x_i^{(\tau)}
=
\frac{X_i^{(\tau)}}{1-\left(X_i^{(\tau)}\right)^2}
\]

と定義する。  
\(|X_i|\le \frac12\) より、\(x_i\) の理論値域は

\[
x_i^{(\tau)}\in\left[-\frac23,\frac23\right]
\]

となる。

### 4.4 正規化

v1.2 では、2/3 問題は正規化で対処する。  
最終的な方向バランス指標を

\[
\boxed{
\chi_i^{(\tau)}
=
\operatorname{clip}
\left(
\frac32\,x_i^{(\tau)},
-1,1
\right)
}
\]

と定義する。

\(\chi_i\) の意味は以下である。

- \(\chi_i>0\): 上昇優勢
- \(\chi_i<0\): 下降優勢
- \(\chi_i\approx0\): 中立

### 4.5 性質

\[
\chi_i^{(\tau)}
\]
は close-only に対する改善指標だが、情報源は OHLC に限られる。  
したがって、完全な intrabar path 再構成ではなく、**終値だけでは失われる方向性を低次元に補うためのスカラー**と位置づける。

---

## 5. ローカル価格スケール \(a_i\)

加算型 \(L_0\) を構築するため、補正項を価格単位へ変換するローカルスケールを定義する。

\[
a_i^{(\tau)}
=
\operatorname{EMA}_{m_\tau}
\left(
H^{(\tau)}-L^{(\tau)}
\right)_i
+
\varepsilon_a
\]

ここで \(m_\tau\) は時間足ごとの smoothing window である。  
例として、

- 30m, 1h: 24〜48 バー
- 4h: 20〜40 バー
- 1d: 20 バー
- 1w: 13 バー

程度を初期候補とする。

\(a_i^{(\tau)}\) は価格単位であり、以下の役割を持つ。

1. 無次元の補助信号を価格スケールへ変換する
2. 補正量を市場ボラティリティに応じて自然に変える
3. \(L_0\) の単位整合性を確保する

---

## 6. \(L_5\) の branch 出力と前時点状態ベクトル

各時間足 \(\tau\) の branch から得られる次足形状予測を

\[
L_{5,i}^{(\tau)}=\hat Q_i^{(\tau)}\in\mathbb R^3
\]

とする。  
inference では、前時点の shape 予測を

\[
v_{i-1}^{(\tau)} = L_{5,i-1}^{(\tau)}
\]

として使う。

学習初期には安定化のため teacher forcing を使い、

\[
v_{i-1}^{(\tau)} = Q_{i-1}^{(\tau)}
\]

としてよい。  
後半は scheduled sampling により \(\hat Q\) へ切り替える。

---

## 7. \(Fv\) による状態フィードバック

各時間足で、前時点の状態ベクトルをスカラーゲートに射影する。

\[
F^{(\tau)}\in\mathbb R^{1\times 3}
\]

\[
g_i^{(\tau)}
=
\tanh
\left(
F^{(\tau)}v_{i-1}^{(\tau)}
+
b_F^{(\tau)}
\right)
\]

ここで \(g_i^{(\tau)}\in[-1,1]\) である。  
役割は、前時点の足形状予測を close anchor 補正へ穏やかに反映することである。

---

## 8. 加算型 \(L_0\)（v1.2 の中心仕様）

### 8.1 基本思想
旧構造:
\[
L_0=d_i(1+\cdots)
\]

v1.2:
\[
L_0=d_i+\text{補正量}
\]

とする。  
これにより、close を主系列として保持しながら、OHLC 由来の情報を残差として加える。

### 8.2 定義

\[
\boxed{
L_i^{(0,\tau)}
=
d_i^{(\tau)}
+
a_i^{(\tau)}
\Bigl(
\beta_0^{(\tau)}
+
\beta_v^{(\tau)}g_i^{(\tau)}
+
\beta_x^{(\tau)}\chi_i^{(\tau)}
+
\beta_{vx}^{(\tau)}g_i^{(\tau)}\chi_i^{(\tau)}
\Bigr)
}
\]

### 8.3 各項の意味

- \(d_i^{(\tau)}\): 生の終値
- \(a_i^{(\tau)}\): 補正量の価格スケール
- \(\beta_0^{(\tau)}\): 時間足ごとのベース補正
- \(\beta_v^{(\tau)}g_i^{(\tau)}\): 前時点の状態による補正
- \(\beta_x^{(\tau)}\chi_i^{(\tau)}\): 今の足内部方向バランスによる補正
- \(\beta_{vx}^{(\tau)}g_i^{(\tau)}\chi_i^{(\tau)}\): 状態と今の足形状の相互作用

### 8.4 制約
close anchor の意味を壊さないため、以下を推奨する。

\[
|\beta_0^{(\tau)}|,\ |\beta_v^{(\tau)}|,\ |\beta_x^{(\tau)}|,\ |\beta_{vx}^{(\tau)}|
\]
は初期値を小さく置く。  
例えば 0 近傍から学習を開始し、補正量に正則化を入れる。

---

## 9. close anchor 入力ベクトル \(\xi\)

\(L_0\) を直接使うだけでなく、系列学習しやすいよう正規化入力を構築する。

\[
z_i^{(\tau)}
=
\frac{
L_i^{(0,\tau)}
-
\operatorname{EMA}_{m_\tau}(L^{(0,\tau)})_i
}{
a_i^{(\tau)}
}
\]

\[
\Delta z_i^{(\tau)}=z_i^{(\tau)}-z_{i-1}^{(\tau)}
\]

\[
\rho_i^{(\tau)}
=
\frac{
H_i^{(\tau)}-L_i^{(\tau)}
}{
a_i^{(\tau)}
}
\]

最終入力ベクトルを

\[
\boxed{
\xi_i^{(\tau)}
=
\begin{bmatrix}
z_i^{(\tau)}\\
\Delta z_i^{(\tau)}\\
\chi_i^{(\tau)}\\
g_i^{(\tau)}\\
\rho_i^{(\tau)}
\end{bmatrix}
}
\]

と定義する。

必要に応じて volume を使う場合は、

\[
\nu_i^{(\tau)}=\text{normalized volume feature}
\]

を付加し、

\[
\xi_i^{(\tau)}=
[z_i,\Delta z_i,\chi_i,g_i,\rho_i,\nu_i]^\top
\]

としてもよい。

---

## 10. 主モデルの branch encoder

各 \(\tau\in\mathcal T_{\text{main}}\) に対して、  
causal dilated TCN を用いて branch 表現を構築する。

### 10.1 履歴長
初期仕様では以下を採用する。

\[
L_{4h}=192,\qquad
L_{1d}=252,\qquad
L_{1w}=78
\]

### 10.2 hidden 次元

\[
d_h=64
\]

### 10.3 encoder

時刻 \(t\) における時間足 \(\tau\) の最後の確定足 index を \(n_\tau(t)\) とする。  
branch 表現は

\[
h_t^{(\tau)}
=
\mathrm{TCN}_\tau
\left(
\xi_{n_\tau(t)-L_\tau+1:n_\tau(t)}^{(\tau)}
\right)
\in\mathbb R^{d_h}
\]

とする。

---

## 11. 各 branch の shape head

各 branch は、次の足形状を予測する。

\[
\boxed{
L_{5,t}^{(\tau)}
=
\hat Q_t^{(\tau)}
=
\begin{bmatrix}
\sigma(a_{u,\tau}^{\top}h_t^{(\tau)}+c_{u,\tau})\\
\tanh(a_{b,\tau}^{\top}h_t^{(\tau)}+c_{b,\tau})\\
\sigma(a_{\ell,\tau}^{\top}h_t^{(\tau)}+c_{\ell,\tau})
\end{bmatrix}
}
\]

教師は

\[
\hat Q_t^{(\tau)} \approx Q_{n_\tau(t)+1}^{(\tau)}
\]

である。

---

## 12. 主モデルのマルチタイムフレーム統合

統合は coarse-to-fine のベクトルゲートで行う。  
週足を大局、日足を中間、4時間足を最終判断とする。

### 12.1 週足表現

\[
R_t^{(1w)} = U_w h_t^{(1w)}
\]

### 12.2 日足統合

\[
a_t^{(1d)}
=
\sigma
\left(
A_d h_t^{(1d)} + B_d R_t^{(1w)} + c_d
\right)
\in (0,1)^{d_h}
\]

\[
R_t^{(1d)}
=
a_t^{(1d)} \odot U_d h_t^{(1d)}
+
(1-a_t^{(1d)})\odot V_d R_t^{(1w)}
\]

### 12.3 4時間足統合

\[
a_t^{(4h)}
=
\sigma
\left(
A_h h_t^{(4h)} + B_h R_t^{(1d)} + c_h
\right)
\in (0,1)^{d_h}
\]

\[
\boxed{
R_t
=
a_t^{(4h)} \odot U_h h_t^{(4h)}
+
(1-a_t^{(4h)})\odot V_h R_t^{(1d)}
}
\]

\(R_t\) を主モデルの統合表現とする。

---

## 13. 融合後の総合 \(L_5^\ast\)

branch ごとの \(L_5\) とは別に、融合後のグローバルな足形状予測を持つ。

\[
\boxed{
L_{5,t}^{\ast}
=
\begin{bmatrix}
\sigma(w_U^\top R_t+b_U)\\
\tanh(w_B^\top R_t+b_B)\\
\sigma(w_L^\top R_t+b_L)
\end{bmatrix}
}
\]

この出力は、総合的な次足形状・方向性の説明ベクトルとして用いる。

---

## 14. multi-horizon return / uncertainty head

### 14.1 future return

4h 基準の forward return を

\[
r_{t,h}^{\mathrm{fwd}}
=
\log\frac{C_{t+h}^{(4h)}}{C_t^{(4h)}},
\qquad h\in\mathcal H
\]

とする。

### 14.2 期待値と不確実性

各 horizon ごとに

\[
\mu_{t,h}=w_{\mu,h}^{\top}R_t+b_{\mu,h}
\]

\[
\sigma_{t,h}
=
\operatorname{softplus}(w_{\sigma,h}^{\top}R_t+b_{\sigma,h})
+\sigma_{\min}
\]

を出力する。

### 14.3 予測終値

\[
\boxed{
\hat C_{t+h}^{(4h)}
=
C_t^{(4h)}\exp(\mu_{t,h})
}
\]

---

## 15. horizon 選択と主ポジション

### 15.1 コスト込み標準化エッジ

\[
S_{t,h}
=
\frac{
|\mu_{t,h}|-c_h
}{
\sigma_{t,h}
}
\]

ここで \(c_h\) は horizon ごとの round-trip cost 推定である。

### 15.2 最適 horizon

\[
h_t^\star = \arg\max_{h\in\mathcal H} S_{t,h}
\]

### 15.3 主ポジションサイズ

\[
\boxed{
\pi_t^{\text{main}}
=
\tanh
\left(
\gamma\frac{\mu_{t,h_t^\star}}{\sigma_{t,h_t^\star}}
\right)
}
\]

\[
\operatorname{sign}(\pi_t^{\text{main}})=\operatorname{sign}(\mu_{t,h_t^\star})
\]

である。

---

## 16. 1h / 30m 離脱オーバーレイ

### 16.1 役割
離脱オーバーレイは、主モデルで立てたシナリオを 4h レビュー間に監視し、途中での逸脱に対応する。  
新規エントリーは行わず、以下のみを担当する。

- hold
- reduce
- full exit
- hard exit

### 16.2 入力
オーバーレイも close anchor 入力 \(\xi\) を使う。

\[
z_u^{(1h)}
=
\mathrm{TCN}_{1h}^{\text{ov}}
\left(
\xi_{n_{1h}(u)-L_{1h}^{\text{ov}}+1:n_{1h}(u)}^{(1h)}
\right)
\]

\[
z_u^{(30m)}
=
\mathrm{TCN}_{30m}^{\text{ov}}
\left(
\xi_{n_{30m}(u)-L_{30m}^{\text{ov}}+1:n_{30m}(u)}^{(30m)}
\right)
\]

初期仕様:

\[
L_{1h}^{\text{ov}}=96,\qquad
L_{30m}^{\text{ov}}=160,\qquad
d_{\text{ov}}=32
\]

### 16.3 レビュー区間と進捗量

4h レビュー時刻を \(t_k\)、次レビューを \(t_{k+1}\)、現在監視時刻を \(u\in(t_k,t_{k+1}]\) とする。

主モデルが選んだ方向を

\[
s_k=\operatorname{sign}(\pi_{t_k}^{\text{main}})
\]

とする。

レビュー開始からの符号付き実現リターン:

\[
r_u^{\text{pos}}
=
s_k\log\frac{P_u}{P_{t_k}}
\]

経過率:

\[
\eta_u
=
\frac{u-t_k}{t_{k+1}-t_k}
\in(0,1]
\]

主モデルの次 4h 期待リターンを \(\mu_{k,1}=\mu_{t_k,1}\)、不確実性を \(\sigma_{k,1}=\sigma_{t_k,1}\) とすると、途中期待進捗を

\[
\bar r_u=\eta_u\mu_{k,1}
\]

逸脱 z スコアを

\[
\delta_u
=
\frac{
r_u^{\text{pos}}-\bar r_u
}{
\sigma_{k,1}\sqrt{\eta_u+\varepsilon_\eta}
}
\]

と定義する。

### 16.4 MFE / MAE / peak drawdown

\[
\mathrm{MFE}_u
=
\max_{v\in[t_k,u]}
s_k\log\frac{P_v}{P_{t_k}}
\]

\[
\mathrm{MAE}_u
=
-\min_{v\in[t_k,u]}
s_k\log\frac{P_v}{P_{t_k}}
\ge0
\]

\[
D_u^{\text{peak}}
=
\mathrm{MFE}_u-r_u^{\text{pos}}
\ge0
\]

### 16.5 オーバーレイ文脈ベクトル

\[
c_u=
\begin{bmatrix}
s_k\\
\eta_u\\
1-\eta_u\\
h_k^\star\\
S_{t_k,h_k^\star}\\
\mu_{k,1}\\
\sigma_{k,1}\\
r_u^{\text{pos}}\\
\delta_u\\
\mathrm{MFE}_u\\
\mathrm{MAE}_u\\
D_u^{\text{peak}}\\
L_{5,t_k}^{(4h)}\\
L_{5,t_k}^{(1d)}\\
L_{5,t_k}^{(1w)}
\end{bmatrix}
\]

### 16.6 1h / 30m 統合

\[
g_u^{\text{ov}}
=
\sigma
\left(
A_{\text{ov}}z_u^{(1h)}
+
B_{\text{ov}}z_u^{(30m)}
+
C_{\text{ov}}c_u
+
d_{\text{ov}}
\right)
\in(0,1)^{d_{\text{ov}}}
\]

\[
o_u
=
g_u^{\text{ov}}\odot U_{\text{ov}}z_u^{(1h)}
+
(1-g_u^{\text{ov}})\odot V_{\text{ov}}z_u^{(30m)}
\]

### 16.7 オーバーレイ出力

#### 残余リターン head

\[
\mu_u^{\text{rev}}
=
w_{\mu}^{\top}
\begin{bmatrix}
o_u\\
c_u
\end{bmatrix}
+b_\mu
\]

\[
\sigma_u^{\text{rev}}
=
\operatorname{softplus}
\left(
w_{\sigma}^{\top}
\begin{bmatrix}
o_u\\
c_u
\end{bmatrix}
+b_\sigma
\right)
+\sigma_{\min}
\]

#### adverse excursion 確率

レビュー区間の許容 DD 予算を

\[
d_k^{\max}
=
d_0+d_1\sigma_{k,1}+d_2\log(1+h_k^\star)
\]

とし、

\[
p_u^{\text{adv}}
=
\sigma
\left(
w_{\text{adv}}^{\top}
\begin{bmatrix}
o_u\\
c_u
\end{bmatrix}
+b_{\text{adv}}
\right)
\]

とする。

#### 短期逆向き形状スコア

1h:

\[
\chi_u^{(1h,\text{shape})}
=
-s_k \hat b_u^{(1h)}
+
\beta_w
\left(
\hat u_u^{(1h)}+\hat \ell_u^{(1h)}
\right)
\]

30m:

\[
\chi_u^{(30m,\text{shape})}
=
-s_k \hat b_u^{(30m)}
+
\beta_w
\left(
\hat u_u^{(30m)}+\hat \ell_u^{(30m)}
\right)
\]

混合係数:

\[
\omega_u
=
\sigma
\left(
a_\omega^\top
\begin{bmatrix}
o_u\\
c_u
\end{bmatrix}
+b_\omega
\right)
\]

\[
\chi_u^{\text{mix}}
=
\omega_u \chi_u^{(30m,\text{shape})}
+
(1-\omega_u)\chi_u^{(1h,\text{shape})}
\]

#### 離脱特徴と離脱確率

\[
\phi_u=
\begin{bmatrix}
[-\delta_u]_+\\
\left[
-\dfrac{\mu_u^{\text{rev}}-c_{\text{rev}}}{\sigma_u^{\text{rev}}}
\right]_+\\
p_u^{\text{adv}}\\
\chi_u^{\text{mix}}\\
\mathrm{MAE}_u\\
D_u^{\text{peak}}\\
\eta_u
\end{bmatrix}
\]

\[
p_u^{\text{exit}}
=
\sigma(w_{\text{exit}}^\top\phi_u+b_{\text{exit}})
\]

### 16.8 実行ルール

閾値を

\[
\theta_{\text{red}},\qquad \theta_{\text{full}}
\]

とする。

- \(p_u^{\text{exit}}<\theta_{\text{red}}\): hold
- \(\theta_{\text{red}}\le p_u^{\text{exit}}<\theta_{\text{full}}\): reduce
- \(p_u^{\text{exit}}\ge \theta_{\text{full}}\): full exit

reduce 率:

\[
\rho_u
=
\operatorname{clip}
\left(
\frac{p_u^{\text{exit}}-\theta_{\text{red}}}{\theta_{\text{full}}-\theta_{\text{red}}},
0,1
\right)
\]

\[
\pi_u=(1-\rho_u)\pi_{t_k}^{\text{main}}
\]

hard exit:

\[
r_u^{\text{pos}}\le -d_k^{\max}
\Rightarrow \pi_u=0
\]

---

## 17. シミュレーションメソッド

### 17.1 定義

\[
\boxed{
\mathrm{SimulateEnhancedCloseModel}
(\mathcal D,\Theta^\ast,\text{config})
\rightarrow
\mathcal R
}
\]

### 17.2 入力
- \(\mathcal D\): 原 OHLCV 系列
- \(\Theta^\ast\): 学習済みパラメータ
- config: 時間足、コスト、閾値、履歴長、初期状態など

### 17.3 出力
- 各 horizon の \(\hat C_{t+h}^{(4h)}\)
- 各時間足の \(L_5\)
- 融合後 \(L_5^\ast\)
- 選択 horizon \(h_t^\star\)
- 主ポジション \(\pi_t^{\text{main}}\)
- overlay 後の実効ポジション \(\pi_u\)
- 取引ログ
- PnL / utility / DD 系列

### 17.4 時系列前進手順

1. 原系列から 30m/1h/4h/1d/1w の確定足を生成
2. 各時間足で \(Q,\chi,a\) を計算
3. 前時点の \(L_5\) を使って \(g_i\) を計算
4. \(L_0\) を構築
5. \(\xi\) を生成
6. 4h レビュー時刻ごとに主モデル forward
7. horizon 選択と主ポジション決定
8. レビュー区間内で 30m overlay を forward
9. 実効ポジションを更新
10. 損益系列を更新

### 17.5 擬似コード

```python
def SimulateEnhancedCloseModel(raw_ohlcv, theta, config):
    bars = build_bars(raw_ohlcv, ["30m","1h","4h","1d","1w"])
    feat_state = init_feature_state()
    model_state = init_model_state()

    results = []

    for t in review_times_4h(bars["4h"]):
        update_bar_features_until(t, bars, feat_state, theta)

        x_main = build_close_anchor_inputs(
            bars, feat_state, model_state, t, ["4h","1d","1w"], theta
        )

        h_main = forward_main_branches(x_main, theta)
        R_t, L5_dict, L5_star, mu_vec, sigma_vec = forward_main_heads(h_main, theta)

        h_star = argmax((abs(mu_vec) - config.cost_vec) / sigma_vec)
        pi_main = tanh(config.gamma * mu_vec[h_star] / sigma_vec[h_star])

        model_state.set_main_review(t, h_star, pi_main, mu_vec, sigma_vec, L5_dict, L5_star)

        for u in overlay_times_30m(t, next_review_time(t)):
            update_bar_features_until(u, bars, feat_state, theta)

            x_ov = build_close_anchor_inputs(
                bars, feat_state, model_state, u, ["1h","30m"], theta
            )

            overlay_out = forward_overlay(x_ov, model_state, theta, u)
            model_state.apply_overlay_decision(u, overlay_out, config)

        results.append(snapshot(model_state, t))

    return finalize_simulation_results(results)
```

---

## 18. パラメータ最適化メソッド

### 18.1 定義

\[
\boxed{
\mathrm{OptimizeEnhancedCloseModelParameters}
(\mathcal D,\Theta_0,\text{config})
\rightarrow
\Theta^\ast
}
\]

### 18.2 役割
- 特徴量生成
- 学習 / 検証 split
- shape pretraining
- main training
- overlay training
- utility fine-tuning
- モデル選択

を一体で実行し、最終的な \(\Theta^\ast\) を返す。

---

## 19. loss 定義

## 19.1 主モデル return loss

forward return に対する heteroscedastic Huber を使う。

\[
\rho_\delta(x)=
\begin{cases}
\dfrac{x^2}{2\delta}, & |x|\le\delta\\[4pt]
|x|-\dfrac{\delta}{2}, & |x|>\delta
\end{cases}
\]

\[
\mathcal L_{\mathrm{ret}}
=
\sum_{h\in\mathcal H}
w_h
\left[
\rho_\delta
\left(
\frac{r_{t,h}^{\mathrm{fwd}}-\mu_{t,h}}{\sigma_{t,h}}
\right)
+
\log \sigma_{t,h}
\right]
\]

## 19.2 direction loss

\[
p_{t,h}
=
\sigma
\left(
\kappa\frac{\mu_{t,h}}{\sigma_{t,h}}
\right)
\]

\[
\mathcal L_{\mathrm{dir}}
=
\sum_{h\in\mathcal H}
w_h\,
\mathrm{BCE}
\left(
p_{t,h},
\mathbf 1[r_{t,h}^{\mathrm{fwd}}>0]
\right)
\]

## 19.3 shape loss

主モデルと overlay で共通に使う shape loss を

\[
\mathcal L_{\mathrm{shape}}
=
\sum_{\tau}
\lambda_\tau
\Bigl[
\rho(\hat u_t^{(\tau)}-u_{t+1}^{(\tau)})
+
2\rho(\hat b_t^{(\tau)}-b_{t+1}^{(\tau)})
+
\rho(\hat \ell_t^{(\tau)}-\ell_{t+1}^{(\tau)})
\Bigr]
\]

とする。

## 19.4 geometry loss

\[
\mathcal L_{\mathrm{geom}}
=
\sum_{\tau}
\eta_\tau
\left(
\hat u_t^{(\tau)}
+
|\hat b_t^{(\tau)}|
+
\hat \ell_t^{(\tau)}
-1
\right)^2
\]

## 19.5 close anchor 補正量正則化

今回新設した \(L_0\) 補正が暴走しないよう、  
補正量に直接正則化をかける。

\[
\boxed{
\mathcal L_{\mathrm{corr}}
=
\sum_{\tau,i}
\left(
\frac{
L_i^{(0,\tau)}-d_i^{(\tau)}
}{
a_i^{(\tau)}
}
\right)^2
}
\]

この loss が重要である理由は以下の通り。

- close anchor の主従関係を維持できる
- \(\chi\) や \(g\) の寄与が過大化しにくい
- 補正が「補助量」に留まりやすい

## 19.6 fusion smoothness

\[
\mathcal L_{\mathrm{fuse}}
=
\zeta_d\|a_t^{(1d)}-a_{t-1}^{(1d)}\|_1
+
\zeta_h\|a_t^{(4h)}-a_{t-1}^{(4h)}\|_1
\]

## 19.7 overlay residual return loss

\[
\mathcal L_{\mathrm{ov-rev}}
=
\rho_\delta
\left(
\frac{
r_u^{\mathrm{rev}}-\mu_u^{\mathrm{rev}}
}{
\sigma_u^{\mathrm{rev}}
}
\right)
+
\log \sigma_u^{\mathrm{rev}}
\]

## 19.8 overlay adverse excursion loss

\[
\mathcal L_{\mathrm{ov-adv}}
=
\mathrm{BCE}(p_u^{\mathrm{adv}},y_u^{\mathrm{adv}})
\]

## 19.9 overlay exit loss

\[
\mathcal L_{\mathrm{ov-exit}}
=
\mathrm{BCE}(p_u^{\mathrm{exit}},y_u^{\mathrm{exit}})
\]

## 19.10 overlay chatter loss

\[
\mathcal L_{\mathrm{ov-chat}}
=
|p_u^{\mathrm{exit}}-p_{u-30m}^{\mathrm{exit}}|
\]

## 19.11 utility loss

\[
\tilde w_{t,h}
=
\tanh
\left(
\gamma\frac{\mu_{t,h}}{\sigma_{t,h}}
\right)
\]

\[
\mathcal L_{\mathrm{util}}
=
-
\sum_{h\in\mathcal H}
\nu_h
\left[
\tilde w_{t,h}r_{t,h}^{\mathrm{fwd}}
-
c_h|\tilde w_{t,h}-\tilde w_{t-1,h}|
\right]
\]

---

## 20. 総損失

### 20.1 主モデル損失

\[
\boxed{
\mathcal J_{\mathrm{main}}
=
\mathcal L_{\mathrm{ret}}
+
\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{dir}}
+
\lambda_{\mathrm{shape}}\mathcal L_{\mathrm{shape,main}}
+
\lambda_{\mathrm{geom}}\mathcal L_{\mathrm{geom,main}}
+
\lambda_{\mathrm{corr}}\mathcal L_{\mathrm{corr}}
+
\lambda_{\mathrm{fuse}}\mathcal L_{\mathrm{fuse}}
+
\lambda_2\|\Theta\|_2^2
}
\]

### 20.2 オーバーレイ損失

\[
\boxed{
\mathcal J_{\mathrm{ov}}
=
0.30\,\mathcal L_{\mathrm{ov-rev}}
+
0.25\,\mathcal L_{\mathrm{ov-adv}}
+
0.25\,\mathcal L_{\mathrm{ov-exit}}
+
0.15\,\mathcal L_{\mathrm{shape,ov}}
+
0.05\,\mathcal L_{\mathrm{ov-chat}}
+
\lambda_{2,\mathrm{ov}}\|\Theta_{\mathrm{ov}}\|_2^2
}
\]

### 20.3 全体損失

\[
\boxed{
\mathcal J_{\mathrm{total}}
=
\mathcal J_{\mathrm{main}}
+
\lambda_{\mathrm{ov}}\mathcal J_{\mathrm{ov}}
+
\lambda_{\mathrm{util}}\mathcal L_{\mathrm{util}}
}
\]

---

## 21. 学習手順

### Step 0. バー生成
原 OHLCV から 30m / 1h / 4h / 1d / 1w の確定足を生成する。

### Step 1. 自己教師と close anchor 特徴生成
各時間足で以下を生成する。

- \(Q_i^{(\tau)}\)
- \(\chi_i^{(\tau)}\)
- \(a_i^{(\tau)}\)
- 初期 \(L_0\)
- \(\xi_i^{(\tau)}\)

### Step 2. 時系列分割
purged / embargoed walk-forward split を使う。  
各 fold で訓練・検証を時系列順に分ける。

### Step 3. shape pretraining
各 branch をまず

\[
\mathcal L_{\mathrm{shape}}+\mathcal L_{\mathrm{geom}}
\]

だけで学習し、足形状の意味を安定化させる。

### Step 4. main joint training
主モデルを

\[
\mathcal J_{\mathrm{main}}
\]

で学習する。  
このとき \(v_{i-1}\) は teacher forcing から scheduled sampling に移行する。

### Step 5. overlay training
主モデルを基本凍結し、overlay を

\[
\mathcal J_{\mathrm{ov}}
\]

で学習する。

### Step 6. utility fine-tuning
上位 fusion 層と head を中心に

\[
\mathcal J_{\mathrm{total}}
\]

で微調整する。

### Step 7. threshold calibration
validation fold を用いて以下を校正する。

- \(\theta_{\mathrm{red}}\)
- \(\theta_{\mathrm{full}}\)
- \(\gamma\)
- \(d_0,d_1,d_2\)
- cost model \(c_h\)

### Step 8. model selection
評価指標:

- net utility
- drawdown
- turnover
- no-trade 率
- directional accuracy
- shape accuracy

に基づいて \(\Theta^\ast\) を選ぶ。

---

## 22. 最適化メソッドの擬似コード

```python
def OptimizeEnhancedCloseModelParameters(raw_ohlcv, theta0, config):
    folds = make_purged_walkforward_splits(raw_ohlcv, config)
    best_theta = None
    best_score = -float("inf")

    for fold in folds:
        bars = build_bars(fold.train_and_valid, ["30m","1h","4h","1d","1w"])
        feats = make_close_anchor_features(bars, config)

        theta = init_or_copy(theta0)

        # A. shape pretraining
        theta = train_shape_pretraining(theta, feats, config)

        # B. main joint training
        theta = train_main_model(theta, feats, loss="J_main", config=config)

        # C. overlay training
        theta = train_overlay_model(theta, feats, loss="J_ov", config=config)

        # D. utility fine-tune
        theta = fine_tune_top_layers(theta, feats, loss="J_total", config=config)

        score = evaluate_validation(theta, fold.valid, config)

        if score > best_score:
            best_score = score
            best_theta = copy(theta)

    return best_theta
```

---

## 23. 実装上の注意

### 23.1 \(\chi\) の位置づけ
\(\chi\) は OHLC path の簡易代理量であり、close anchor 補正の補助に使う。  
主情報源は依然として close と multi-timeframe context である。

### 23.2 \(L_0\) は補正であって本体ではない
補正量が大きくなりすぎると close anchor の意味が崩れるため、\(\mathcal L_{\mathrm{corr}}\) を必ず入れる。

### 23.3 branch の因果性
上位足は必ず「その時点で確定している最後のバー」だけを使う。

### 23.4 overlay の役割制限
overlay は主モデルの代替ではない。  
役割は 4h シナリオの途中破綻検知に限定する。

### 23.5 過学習防止
utility fine-tune は最後に少量だけ行い、encoder 全体を過度に更新しない。

---

## 24. v1.2 の数理的な意義

v1.2 の改善は、単なる特徴量追加ではない。  
本質は以下の 3 点にある。

1. **close-only を捨てずに強化したこと**  
   中心は close のまま、OHLC を残差補正として組み込んだ。

2. **乗算型から加算型に変えたこと**  
   補正の意味が明確になり、数値的に安定した。

3. **補助量の単位を揃えたこと**  
   \(\chi\) と \(g\) をそのまま足すのではなく、\(a_i\) によって価格単位へ変換した。

---

## 25. 最終まとめ

v1.2 のモデルは、以下で要約できる。

\[
\boxed{
L_i^{(0,\tau)}
=
C_i^{(\tau)}
+
a_i^{(\tau)}
\Bigl(
\beta_0^{(\tau)}
+
\beta_v^{(\tau)}g_i^{(\tau)}
+
\beta_x^{(\tau)}\chi_i^{(\tau)}
+
\beta_{vx}^{(\tau)}g_i^{(\tau)}\chi_i^{(\tau)}
\Bigr)
}
\]

これを close anchor として各時間足の causal TCN に入力し、

- 4h / 1d / 1w で主シグナルを作る
- 1h / 30m で離脱監視を行う
- multi-horizon return と uncertainty を出す
- shape 自己教師で意味を固定する

というのが v1.2 の全体像である。

---

## 26. メソッド定義（ロジックレベル）

### 26.1 シミュレーションメソッド
\[
\mathrm{SimulateEnhancedCloseModel}
\]

役割:
- 学習済み \(\Theta^\ast\) を使って時系列を逐次前進させる
- 予測終値、\(L_5\)、ポジション、overlay 離脱、PnL を返す

### 26.2 パラメータ最適化メソッド
\[
\mathrm{OptimizeEnhancedCloseModelParameters}
\]

役割:
- 原 OHLCV から feature generation、学習、検証、モデル選択までを実行し、\(\Theta^\ast\) を返す

---

以上を v1.2 の正式ロジック仕様とする。
