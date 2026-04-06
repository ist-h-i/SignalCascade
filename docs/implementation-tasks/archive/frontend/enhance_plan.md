以下を **SignalCascade vNext の変更方針** としてまとめます。
一文で言うと、

**現行の「全時点に方向を付けて utility を最大化する」設計から、
「きれいなイベントだけを教師にし、当たりやすいシグナルだけを採用する」設計へ移す**
のが中心方針です。

---

# 1. 変更しない部分

まず、現時点では中核の価格表現は崩さなくてよいです。
当面は次を維持します。

## 1.1 Path-Averaged Directional Balance

[
\chi_i^{(\tau)}
===============

\operatorname{clip}
\left(
\frac32,x_i^{(\tau)},
-1,1
\right)
]

## 1.2 close anchor

[
L_i^{(0,\tau)}
==============

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
]

## 1.3 multi-timeframe main / overlay の骨格

* main: `4h / 1d / 1w`
* overlay: `1h / 30m`

つまり、**最初に変えるべきは特徴表現そのものではなく、教師ラベル・採用判定・閾値校正** です。

---

# 2. 変更の主目的

最適化目標を、平均的な directional accuracy や utility のみから、

[
\max_{\tau}
\ \mathrm{Coverage}(\tau)
\quad
\text{s.t.}
\quad
\mathrm{Precision}(\tau)\ge 0.8
]

へ変えます。

ここで、

[
a_t(\tau)=\mathbf{1}[s_t\ge \tau]
]

[
\mathrm{Coverage}(\tau)
=======================

\frac{1}{T}\sum_{t=1}^{T} a_t(\tau)
]

[
\mathrm{Precision}(\tau)
========================

\frac{\sum_{t=1}^{T} a_t(\tau),m_t}{\sum_{t=1}^{T} a_t(\tau)}
]

です。

* (s_t): そのシグナルを採用すべき確率
* (m_t): 実際に採用すべきだったかの教師

目的は、**採用精度80%以上を維持したまま、採用できる点数を増やすこと** です。

---

# 3. 変更方針の全体像

現行の流れは概ね

1. 将来リターンを予測する
2. edge の高い horizon を選ぶ
3. そのままポジションを出す

ですが、vNext は

1. 将来リターンを予測する
2. 方向ラベルを 3値化する
3. 各 horizon の「当たりやすさ」を校正する
4. そのシグナルを**採用すべきか**を別モデルで判定する
5. 80%精度制約の下で閾値を決める
6. overlay は 4クラスではなく risk filter として使う

に変えます。

---

# 4. 学習データ作成時の判定式を変更する

ここが最優先です。

## 4.1 基本記号

horizon を

[
h\in\mathcal H={1,2,3,6,12,18,30}
]

とし、forward return を

[
r_{t,h}
=======

\log\frac{C_{t+h}}{C_t}
]

と定義します。

過去情報だけから作るボラティリティ推定値を

[
\hat\sigma_t
]

とし、regime を

[
z_t\in\mathcal Z
]

とします。
(z_t) は最初は粗く、

* session: Asia / London / NY
* volatility bin: low / high
* trend bin: range / trend

程度で十分です。

---

## 4.2 adverse excursion の定義

ロング視点の adverse excursion を

[
\mathrm{MAE}^{+}_{t,h}
======================

\max_{1\le u\le h}
\log\frac{C_t}{C_{t+u}}
]

ショート視点の adverse excursion を

[
\mathrm{MAE}^{-}_{t,h}
======================

\max_{1\le u\le h}
\log\frac{C_{t+u}}{C_t}
]

とします。

参考までに favorable excursion も定義しておくと便利です。

[
\mathrm{MFE}^{+}_{t,h}
======================

\max_{1\le u\le h}
\log\frac{C_{t+u}}{C_t}
]

[
\mathrm{MFE}^{-}_{t,h}
======================

\max_{1\le u\le h}
\log\frac{C_t}{C_{t+u}}
]

---

## 4.3 ラベル閾値を固定値から horizon別・regime別へ変更する

正例閾値を固定値ではなく

[
\delta_{h,z_t}
==============

c_h + \lambda_{h,z_t},\hat\sigma_t\sqrt{h}
]

[
\eta_{h,z_t}
============

\rho_{h,z_t},\hat\sigma_t\sqrt{h}
]

で定義します。

* (c_h): コスト相当
* (\delta_{h,z_t}): 十分な値幅
* (\eta_{h,z_t}): 許容逆行
* (\lambda_{h,z_t},\rho_{h,z_t}): 学習 fold 内で決める係数

これにより、**その時のボラ・時間足に対して十分に大きく、途中逆行も小さい動きだけを正例化** できます。

---

## 4.4 main の方向ラベルを 2値から 3値へ変更する

現状の up/down 2値はノイズが多いので、次の 3値へ変更します。

[
\tilde y_{t,h}
==============

\begin{cases}
+1,
&
r_{t,h}\ge \delta_{h,z_t}
\ \land
\mathrm{MAE}^{+}*{t,h}\le \eta*{h,z_t}
[6pt]
-1,
&
r_{t,h}\le -\delta_{h,z_t}
\ \land
\mathrm{MAE}^{-}*{t,h}\le \eta*{h,z_t}
[6pt]
0,
&
\text{otherwise}
\end{cases}
]

意味は明確です。

* (+1): きれいな上昇
* (-1): きれいな下降
* (0): 曖昧、または値幅不足、または途中逆行が大きい

これで、**全時点に無理に方向を付けるのをやめる** ことができます。

---

## 4.5 サンプル重みを clean signal 優先にする

方向ラベルの重みは、最低でも次のように持たせる方がよいです。

[
w^{\mathrm{main}}_{t,h}
=======================

1
+
\alpha_1,
\frac{(|r_{t,h}|-c_h)*+}{\hat\sigma_t\sqrt{h}+\varepsilon}
+
\alpha_2,\mathbf{1}[\tilde y*{t,h}\neq 0]
]

必要なら、MFE/MAE 比も使えます。

[
w^{\mathrm{clean}}_{t,h}
========================

1
+
\alpha_3
\frac{\mathrm{MFE}^{(\tilde y)}*{t,h}}
{\mathrm{MAE}^{(\tilde y)}*{t,h}+\varepsilon}
]

---

# 5. ベースモデルの出力を整理する

ベースモデルは引き続き各 horizon に対して期待リターンと不確実性を出します。

[
\mu_{t,h}
=========

w_{\mu,h}^{\top}R_t+b_{\mu,h}
]

[
\sigma_{t,h}
============

\operatorname{softplus}(w_{\sigma,h}^{\top}R_t+b_{\sigma,h})
+\sigma_{\min}
]

加えて、方向 3値の確率を持たせます。

[
p_{t,h}^{(+1)},\quad p_{t,h}^{(0)},\quad p_{t,h}^{(-1)}
]

[
p_{t,h}^{(+1)}+p_{t,h}^{(0)}+p_{t,h}^{(-1)}=1
]

この段階では、ベースモデルはまだ「予測器」です。
採用可否の最終判断は別に持ちます。

---

# 6. 「当たりやすさ」を確率として校正する

ベースモデルから、その horizon の方向予測が当たる確率

[
q_{t,h}
=======

P!\left(
\operatorname{sign}(\hat d_{t,h})
=================================

\operatorname{sign}(r_{t,h})
\mid X_t
\right)
]

を作ります。
ここで (\hat d_{t,h}) はベースモデルの方向予測です。

最初は別 head を追加しなくてもよく、**OOF 予測に対する calibrator** で十分です。

[
q_{t,h}
=======

g_{\mathrm{cal}}(v_{t,h})
]

[
v_{t,h}
=======

\left[
\frac{|\mu_{t,h}|}{\sigma_{t,h}+\varepsilon},
\ p_{t,h}^{(+1)},
\ p_{t,h}^{(0)},
\ p_{t,h}^{(-1)},
\ z_t,
\ \text{agreement features}
\right]
]

agreement features には例えば、

* `4h / 1d / 1w` の符号整合
* top1 と top2 の edge 差
* session
* 過去ボラ

を入れます。

---

# 7. horizon 選択式を変更する

現行の

[
S_{t,h}
=======

\frac{|\mu_{t,h}|-c_h}{\sigma_{t,h}}
]

は utility 寄りです。
80%精度優先なら、confidence をより強く入れるべきです。

## 7.1 推奨する暫定式

[
J_{t,h}
=======

q_{t,h}^{\alpha}
\left(
\max(|\mu_{t,h}|-c_h,0)
\right)^{1-\alpha}
]

[
h_t^\star
=========

\arg\max_{h\in\mathcal H} J_{t,h}
]

ここで (\alpha) は precision-first なら大きめに置きます。

[
0.6 \le \alpha \le 0.8
]

くらいから開始するのが妥当です。

つまり、**値幅より「当たりやすさ」を優先しつつ、値幅ゼロの退化解は防ぐ** という設計です。

---

# 8. 最重要変更: meta-labeling を導入する

ここが vNext の中心です。
「予測する」ことと、「その予測を採用する」ことを分離します。

## 8.1 実現品質スコアを定義する

各 horizon の実現品質を

[
u^{\mathrm{real}}_{t,h}
=======================

## \operatorname{sign}(\mu_{t,h}),r_{t,h}

## c_h

\lambda_{\mathrm{ae}}
,
\mathrm{MAE}^{(\operatorname{sign}\mu_{t,h})}_{t,h}
]

で定義します。

* 符号が合えばプラス
* コストを引く
* 途中逆行が大きいものを罰する

です。

---

## 8.2 meta-label を定義する

[
m_{t,h}
=======

\mathbf{1}
\left[
u^{\mathrm{real}}*{t,h}
\ge
\kappa*{h,z_t}
\right]
]

ここで (\kappa_{h,z_t}) は horizon別・regime別の採用基準です。

これを展開形で書くと、

[
m_{t,h}
=======

\mathbf{1}!\left[
\operatorname{sign}(\mu_{t,h})=\operatorname{sign}(r_{t,h})
\ \land
|r_{t,h}|>c_h+\delta_{h,z_t}
\ \land
\mathrm{MAE}^{(\operatorname{sign}\mu_{t,h})}*{t,h}\le \eta*{h,z_t}
\right]
]

でも構いません。

この (m_{t,h}) が、**「この予測は採用すべきだったか」** の教師です。

---

## 8.3 selector を学習する

selector は

[
s_{t,h}
=======

P(m_{t,h}=1\mid \phi_{t,h})
]

を出します。
(\phi_{t,h}) には少なくとも次を入れます。

* (|\mu_{t,h}|/\sigma_{t,h})
* (q_{t,h})
* (p_{t,h}^{(+1)}, p_{t,h}^{(0)}, p_{t,h}^{(-1)})
* top1/top2 gap
* multi-timeframe の符号一致率
* session
* 過去ボラ
* range/trend 指標

ここで重要なのは、**selector は OOF 予測から学習する** ことです。
同 fold 内の in-sample 予測で (m_{t,h}) を作ると簡単にリークします。

---

# 9. 採用判定式を追加する

最終的な採用判定は

[
a_t
===

\mathbf{1}
\left[
s_{t,h_t^\star}
\ge
\tau_{z_t,h_t^\star}
\right]
]

です。

* (a_t=1): 採用
* (a_t=0): 見送り

この閾値 (\tau_{z,h}) は、global 1本ではなく
**regime別・horizon別** に持ちます。

---

# 10. ポジション式を変更する

現行の main position は

[
\pi_t^{\mathrm{main}}
=====================

\tanh
\left(
\gamma\frac{\mu_{t,h_t^\star}}{\sigma_{t,h_t^\star}}
\right)
]

ですが、vNext では selection gate を掛けます。

[
\pi_t^{\mathrm{raw}}
====================

a_t,
\tanh
\left(
\gamma\frac{\mu_{t,h_t^\star}}{\sigma_{t,h_t^\star}+\varepsilon}
\right)
]

これだけで、**no-trade を明示的に作れる** ようになります。

---

# 11. overlay は 4クラスではなく risk filter に簡略化する

現段階では、overlay は複雑すぎます。
まずは `hold / reduce / full_exit / hard_exit` から離れて、
**安全に保持できるかどうか** のフィルタに変えます。

## 11.1 overlay 教師

主ポジションの向きを

[
d_t = \operatorname{sign}(\pi_t^{\mathrm{raw}})
]

とし、overlay horizon (\Delta) に対する保持品質を

[
u_t^{\mathrm{ov}}
=================

## d_t,r_{t,\Delta}

\lambda_{\mathrm{ae}}^{\mathrm{ov}}
,
\mathrm{MAE}^{(d_t)}_{t,\Delta}
]

とします。

そして教師を

[
o_t
===

\mathbf{1}
\left[
u_t^{\mathrm{ov}}
\ge
\kappa_{z_t}^{\mathrm{ov}}
\right]
]

とします。

* (o_t=1): hold してよい
* (o_t=0): 危険なので exit/reduce 側

---

## 11.2 overlay 出力

[
e_t
===

P(o_t=1\mid X_t^{\mathrm{ov}})
]

として、

[
\rho_t
======

\mathbf{1}[e_t\ge \tau_{z_t}^{\mathrm{ov}}]
\quad
\text{または}
\quad
\rho_t=e_t
]

を定義し、

[
\pi_t
=====

\rho_t,\pi_t^{\mathrm{raw}}
]

とします。

つまり overlay は、**主シグナルの後段で掛かる risk gate** に変更します。

---

# 12. 損失関数の推奨形

## 12.1 return loss

[
\mathcal L_{\mathrm{return}}
============================

\frac{1}{N}
\sum_{t,h}
\left[
\frac{\mathrm{Huber}(r_{t,h}-\mu_{t,h})}{\sigma_{t,h}^2+\varepsilon}
+
\log(\sigma_{t,h})
\right]
]

## 12.2 direction 3値 loss

[
\mathcal L_{\mathrm{dir3}}
==========================

\frac{1}{N}
\sum_{t,h}
w^{\mathrm{main}}*{t,h}
,
\mathrm{FocalCE}!\left(
p*{t,h},
\tilde y_{t,h}
\right)
]

展開すると、

[
\mathrm{FocalCE}(p,y)
=====================

*

\sum_{k\in{-1,0,+1}}
\alpha_k,
\mathbf{1}[y=k],
(1-p^{(k)})^{\gamma_f}
\log p^{(k)}
]

です。

## 12.3 selector loss

[
\mathcal L_{\mathrm{sel}}
=========================

\frac{1}{N}
\sum_{t,h}
\Bigl(
------

## m_{t,h}\log s_{t,h}

(1-m_{t,h})\log(1-s_{t,h})
\Bigr)
+
\lambda_B
\frac{1}{N}
\sum_{t,h}
(s_{t,h}-m_{t,h})^2
]

これは BCE + Brier です。
**分類性能と確率校正の両方** を見ます。

## 12.4 overlay loss

[
\mathcal L_{\mathrm{ov}}
========================

\frac{1}{N}
\sum_t
\Bigl(
-o_t\log e_t-(1-o_t)\log(1-e_t)
\Bigr)
]

## 12.5 総損失

[
\mathcal L
==========

\lambda_r \mathcal L_{\mathrm{return}}
+
\lambda_d \mathcal L_{\mathrm{dir3}}
+
\lambda_s \mathcal L_{\mathrm{sel}}
+
\lambda_o \mathcal L_{\mathrm{ov}}
]

現段階では、旧 overlay 4クラス loss の重みは下げるか、一旦外す方が妥当です。

---

# 13. 閾値校正の式

閾値 (\tau_{z,h}) は学習後に OOF 予測で決めます。

[
\tau_{z,h}^{\star}
==================

\arg\max_{\tau}
\ \mathrm{Coverage}*{z,h}(\tau)
\quad
\text{s.t.}
\quad
\mathrm{Precision}*{z,h}(\tau)\ge 0.8
]

[
\mathrm{Coverage}_{z,h}(\tau)
=============================

\frac{
\sum_t
\mathbf{1}[z_t=z]
\mathbf{1}[h_t^\star=h]
\mathbf{1}[s_{t,h}\ge \tau]
}{
\sum_t
\mathbf{1}[z_t=z]
\mathbf{1}[h_t^\star=h]
}
]

[
\mathrm{Precision}_{z,h}(\tau)
==============================

\frac{
\sum_t
m_{t,h},
\mathbf{1}[z_t=z]
\mathbf{1}[h_t^\star=h]
\mathbf{1}[s_{t,h}\ge \tau]
}{
\sum_t
\mathbf{1}[z_t=z]
\mathbf{1}[h_t^\star=h]
\mathbf{1}[s_{t,h}\ge \tau]
}
]

これが、今回の要求そのものに対応する校正式です。

---

# 14. 評価指標も変更する

主指標は次です。

## 14.1 80%以上精度で採用できる点数

[
N_{80}^{\star}
==============

\max_{\tau:,\mathrm{Precision}(\tau)\ge 0.8}
\sum_{t=1}^{T} a_t(\tau)
]

## 14.2 Precision / Coverage curve

[
\mathrm{Precision}(\tau),\quad \mathrm{Coverage}(\tau)
]

を regime別・horizon別に見ます。

## 14.3 calibration

[
\mathrm{ECE}
============

\sum_{b=1}^{B}
\frac{|B_b|}{n}
\left|
\mathrm{acc}(B_b)-\mathrm{conf}(B_b)
\right|
]

## 14.4 運用系指標

forward simulation 上で

[
\mathrm{PnL}_t
==============

## \pi_{t-1},\Delta \log C_t

\lambda_{\mathrm{tc}}|\pi_t-\pi_{t-1}|
]

[
\mathrm{Turnover}
=================

\sum_t |\pi_t-\pi_{t-1}|
]

[
\mathrm{NoTradeRate}
====================

1-\frac{1}{T}\sum_t a_t
]

[
\mathrm{MaxDD}
==============

\max_{u<v}
\left(
\mathrm{Equity}_u-\mathrm{Equity}_v
\right)
]

を保存します。

---

# 15. 学習・検証プロトコル

これは必須です。

## 15.1 purged / embargoed walk-forward

各 fold (k) で

* 学習区間: (\mathcal T_k^{\mathrm{train}})
* purge / embargo
* 検証区間: (\mathcal T_k^{\mathrm{val}})

を持ち、
OOF 予測

[
\widehat\mu^{\mathrm{OOF}}*{t,h},
\quad
\widehat\sigma^{\mathrm{OOF}}*{t,h},
\quad
\widehat p^{\mathrm{OOF}}_{t,h}
]

を作ります。

この OOF 予測だけを使って

* calibrator (g_{\mathrm{cal}})
* meta-label (m_{t,h})
* selector (s_{t,h})
* 閾値 (\tau_{z,h})

を作ります。

これをやらないと、80%精度の見積もりが簡単に甘くなります。

---

# 16. 実装順の推奨

## Phase 1: 最小変更版

1. main ラベルを 3値化
2. (\delta_{h,z}, \eta_{h,z}) を導入
3. OOF 予測から (q_{t,h}) を校正
4. (J_{t,h}=q_{t,h}^{\alpha}(\max(|\mu|-c_h,0))^{1-\alpha}) に変更
5. (\tau_{z,h}) を 80%制約で校正

この段階では、selector をまだ入れずに (q_{t,h}) をそのまま使ってもよいです。

## Phase 2: 推奨完全版

1. meta-label (m_{t,h}) を作る
2. selector (s_{t,h}) を学習する
3. 採用判定 (a_t) を導入する
4. overlay を binary risk filter 化する

## Phase 3: 運用評価

1. forward simulation
2. PnL / DD / turnover / no-trade 率評価
3. regime別 threshold の再調整

---

# 17. 最終的な方針を一文でまとめると

**変更すべき方針は、
「予測器の平均精度を上げること」から
「きれいなイベントだけを学習し、当たりやすいシグナルだけを採用すること」へ軸を移すこと**
です。

そのための数式上の主変更点は次の5つです。

1. **ラベルを 2値から 3値へ**
   [
   \tilde y_{t,h}\in{-1,0,+1}
   ]

2. **閾値を固定値から horizon別・regime別へ**
   [
   \delta_{h,z_t}=c_h+\lambda_{h,z_t}\hat\sigma_t\sqrt h,\quad
   \eta_{h,z_t}=\rho_{h,z_t}\hat\sigma_t\sqrt h
   ]

3. **confidence を horizon 選択へ入れる**
   [
   J_{t,h}=q_{t,h}^{\alpha}\left(\max(|\mu_{t,h}|-c_h,0)\right)^{1-\alpha}
   ]

4. **meta-label と selector を導入する**
   [
   m_{t,h}=\mathbf{1}[u^{\mathrm{real}}*{t,h}\ge \kappa*{h,z_t}],
   \qquad
   s_{t,h}=P(m_{t,h}=1\mid \phi_{t,h})
   ]

5. **採用判定を明示する**
   [
   a_t=\mathbf{1}[s_{t,h_t^\star}\ge \tau_{z_t,h_t^\star}]
   ]
