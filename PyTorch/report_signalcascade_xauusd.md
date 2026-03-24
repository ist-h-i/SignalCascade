# SignalCascade XAUUSD 限界評価レポート

## Abstract

本レポートは、`SignalCascade` の XAUUSD 向け PyTorch 参照実装について、2026年3月24日 09:30 JST 時点までの最新学習データと最新成果物を用い、「すでに学習は限界まで進んだのか」「そもそもこの数理モデルの限界に到達したのか」を評価する内部技術報告である。結論は明確である。現時点の repo 内証拠だけからは、学習が限界まで進んだとも、この数理モデルの性能限界に到達したとも判定できない。理由は、現在の実装が `reference implementation` であり、仕様書が要求する `purged / embargoed walk-forward`、`forward simulation`、`drawdown`、`turnover`、`no-trade率`、`utility fine-tuning`、`threshold calibration` を未実装のまま、短期間データ・単一時系列 split・局所的な候補探索で評価しているためである。

本レポートの目的は、現状で何が示されたかを定量的に整理し、何がまだ示されていないかを仕様差分と検証差分の両面から明確化することである。外部文献レビューは行わず、`PyTorch/requirements_multiframe_candlestick_model.md`、`PyTorch/logic_multiframe_candlestick_model.md`、`PyTorch/artifacts/gold_xauusd_m30/current/metrics.json`、`PyTorch/artifacts/gold_xauusd_m30/current/manifest.json`、`PyTorch/artifacts/gold_xauusd_m30/archive/session_20260324T020618Z/leaderboard.json`、および現行コードのみを根拠とする。

## 1. Research Question

本報告で答える問いは次の二つである。

1. 現在の XAUUSD 学習は、利用可能な学習データと実装された最適化フローの範囲で見て、すでに飽和しているか。
2. `close anchor + Path-Averaged Directional Balance + multi-timeframe main + overlay` という数理モデル自体の限界に、現状の実装と実験で到達したと言えるか。

この問いに対し、本報告は「現時点の repo 内事実に限定した評価」を行う。したがって、ここでいう限界とは理論的上限の証明ではなく、少なくとも「追加学習・追加設計・追加検証を行っても改善余地が実質的に残っていない」と言える状態を指す。その意味で、現時点の証拠は限界到達を支持しない。

## 2. Model Specification

### 2.1 数理モデルの中心構造

仕様書 `PyTorch/logic_multiframe_candlestick_model.md` によれば、本モデルは終値系列を主系列に固定し、OHLC から得られる方向情報で終値を残差補正する `close anchor` 型モデルである。主な要素は以下の通りである。

- `Path-Averaged Directional Balance`:

\[
\chi_i^{(\tau)}
=
\operatorname{clip}
\left(
\frac32\,x_i^{(\tau)},
-1,1
\right)
\]

ここで \(\chi_i^{(\tau)}\) は 1 本の足の内部方向バランスを表し、\(\chi_i > 0\) は上昇優勢、\(\chi_i < 0\) は下降優勢、\(\chi_i \approx 0\) は中立を意味する。

- 加算型 `close anchor`:

\[
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
\]

この定義により、終値 \(d_i^{(\tau)}\) を主系列として保持しながら、OHLC 由来の補正を価格単位で残差的に加える。

- 学習入力:

\[
\xi_i^{(\tau)}
=
[z_i,\Delta z_i,\chi_i,g_i,\rho_i,\nu_i]^\top
\]

仕様書は volume 特徴 \(\nu_i\) を含む拡張形を許容しており、`PyTorch/README.md` でもこの 6 次元特徴量が現行参照実装の入力として説明されている。

### 2.2 主モデルと overlay

主モデルは `4h / 1d / 1w` を使って multi-horizon の期待リターンと不確実性を出力する。

\[
r_{t,h}^{\mathrm{fwd}}
=
\log\frac{C_{t+h}^{(4h)}}{C_t^{(4h)}}
\]

\[
\mu_{t,h}=w_{\mu,h}^{\top}R_t+b_{\mu,h}
\]

\[
\sigma_{t,h}
=
\operatorname{softplus}(w_{\sigma,h}^{\top}R_t+b_{\sigma,h})
+\sigma_{\min}
\]

\[
\hat C_{t+h}^{(4h)}
=
C_t^{(4h)}\exp(\mu_{t,h})
\]

horizon 選択はコスト込み標準化エッジ

\[
S_{t,h}
=
\frac{|\mu_{t,h}|-c_h}{\sigma_{t,h}}
\]

に基づき、

\[
h_t^\star = \arg\max_{h\in\mathcal H} S_{t,h}
\]

で決める。主ポジションサイズは

\[
\pi_t^{\text{main}}
=
\tanh
\left(
\gamma\frac{\mu_{t,h_t^\star}}{\sigma_{t,h_t^\star}}
\right)
\]

で与えられる。

これに対し overlay は `1h / 30m` を使って `hold / reduce / full_exit / hard_exit` を判定する補助モデルであり、主モデルの 4 時間レビュー間にポジションの逸脱管理を行う設計である。

### 2.3 仕様書が要求する完全版の学習・検証

`PyTorch/requirements_multiframe_candlestick_model.md` と `PyTorch/logic_multiframe_candlestick_model.md` は、完全版システムに対して少なくとも以下を要求している。

- 段階学習:
  `shape pretraining`、`main joint training`、`overlay training`、`utility fine-tuning`
- 時系列検証:
  `purged / embargoed walk-forward`
- モデル選択:
  `net utility`、`max drawdown`、`turnover`、`no-trade率`、`directional accuracy`、`shape prediction quality`
- 追加処理:
  `threshold calibration`
- 実運用評価:
  `SimulateEnhancedCloseModel` による時系列前進シミュレーション

したがって、仕様上の完全版は単なる予測モデルではなく、予測・ポジション決定・overlay・utility 評価まで含む統合予測・シミュレーション基盤として定義されている。

## 3. Current Implementation Scope

### 3.1 現在の実装が実際に行っていること

現行コード `PyTorch/src/signal_cascade_pytorch` を確認すると、現在の実装は仕様の中核を安全に実験するための参照実装であり、`PyTorch/README.md` でも明示的に `reference implementation` と位置付けられている。実際に実装されている流れは次の通りである。

1. 30 分足の生 OHLCV から `30m / 1h / 4h / 1d / 1w` を再集約する。
2. `build_close_anchor_features` を用いて各時間足の特徴量列を作る。
3. `4h / 1d / 1w` 主系列と `1h / 30m` overlay 系列から `TrainingExample` を構築する。
4. `SignalCascadeModel` により、各時間足の TCN エンコーダ、main fusion、`return mean / return scale / shape / overlay logits` を一括で出力する。
5. `train_model` は単一の `AdamW` 最適化で end-to-end 学習し、最小 `validation_total` の checkpoint を保存する。
6. `tune_latest_dataset` は既存ベスト近傍の 9 候補を評価し、`utility_score` 優先の leaderboard で `current` を更新する。

### 3.2 現在の実装がまだ行っていないこと

仕様書に対する未達項目は明確である。

- `shape pretraining` 未実装
- `overlay training` の独立段階未実装
- `utility fine-tuning` 未実装
- `threshold calibration` 未実装
- `purged / embargoed walk-forward` 未実装
- `forward simulation` 未実装
- `PnL / utility / DD` の時系列出力未実装
- `max drawdown / turnover / no-trade率` によるモデル選択未実装
- `baseline comparison` 未実装

### 3.3 現在の損失関数

現行 `losses.py` に実装されている損失は、

- `heteroscedastic_huber_loss`
- `directional_loss`
- `main_shape_loss`
- `overlay_classification_loss`

の 4 つであり、総損失は

\[
\mathcal L_{\text{current}}
=
\mathcal L_{\text{return}}
+0.2\,\mathcal L_{\text{direction}}
+0.3\,\mathcal L_{\text{shape}}
+0.3\,\mathcal L_{\text{overlay}}
\]

である。仕様書にある `geometry loss`、`close anchor 補正量正則化`、`fusion smoothness loss`、`residual return loss`、`adverse excursion loss`、`overlay chatter loss`、`utility loss` は現在の学習ループに含まれていない。

したがって、現状の学習は仕様完全版の最適化ではなく、あくまで簡略化された単段の参照学習である。

## 4. Dataset and Training Setup

### 4.1 最新学習データ

最新成果物 `PyTorch/artifacts/gold_xauusd_m30/current/metrics.json` および `Frontend/public/dashboard-data.json` から、最新の学習入力データは以下である。

- 対象銘柄: XAUUSD
- 入力粒度: 30 分足 OHLCV
- ソース CSV: `PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv`
- 使用行数: `2840`
- データ終端: `2026-03-24T00:30:00Z` = `2026年3月24日 09:30 JST`
- データ始端: `2025-12-24T00:00:00Z` = `2025年12月24日 09:00 JST`

したがって、最新学習はおよそ 3 か月の XAUUSD 30 分足を対象にしている。

### 4.2 学習例と split

`metrics.json` によれば、構築された学習例は `133` 件であり、`train_model` の `train_ratio = 0.8` と時系列順 split により、

- 学習例: `106`
- 検証例: `27`

に分割されている。ここで重要なのは、これは単一の時系列 split であって、`walk-forward` ではないことである。さらに `purge` と `embargo` は入っていない。

### 4.3 時間足・窓長・horizon

現行 `config.py` と `timeframes.py` による設定は以下である。

- main timeframes: `4h / 1d / 1w`
- overlay timeframes: `1h / 30m`
- horizon 集合:
  `1, 2, 3, 6, 12, 18, 30`（いずれも 4h 単位）
- main 窓長:
  `4h: 48`, `1d: 21`, `1w: 8`
- overlay 窓長:
  `1h: 48`, `30m: 96`

### 4.4 現在のハイパーパラメータ探索

最新セッション `PyTorch/artifacts/gold_xauusd_m30/archive/session_20260324T020618Z/leaderboard.json` では、候補は `9` 本である。探索はベスト近傍の局所的変形であり、探索されたのは主に以下である。

- `epochs`
- `batch_size`
- `learning_rate`
- `hidden_dim`
- `dropout`
- `weight_decay`

これは広域探索でも Bayesian optimization でもなく、狭い近傍でのヒューリスティック探索である。

## 5. Latest Empirical Results

### 5.1 最新ベスト候補

最新 `manifest.json` によれば、現在の採用候補は `candidate_06` であり、数値は以下の通りである。

| 項目 | 値 |
| --- | ---: |
| best candidate | `candidate_06` |
| best validation loss | `-2.407322` |
| utility score | `0.560945` |
| value capture ratio | `0.849089` |
| value per signal | `0.027062` |
| downside per signal | `-0.000785` |
| directional accuracy | `0.566138` |
| coverage at 1 sigma | `0.687831` |
| overlay accuracy | `0.666667` |
| overlay macro F1 | `0.200000` |
| selected horizon | `6` |
| position | `0.573864` |
| epochs | `15` |
| batch size | `8` |
| learning rate | `0.0005` |
| hidden dim | `48` |
| dropout | `0.15` |
| weight decay | `2.5e-05` |

アンカー時刻は `2026-03-24T04:00:00+00:00` であり、JST では `2026年3月24日 13:00 JST` に相当する。

### 5.2 leaderboard 上の位置付け

同セッションの 9 候補比較では、`candidate_06` は `utility_score` と `value_capture_ratio` を最優先にした並びで 1 位となっている。一方で `best_validation_loss` だけを見れば `candidate_03` の `-2.485300` の方が小さい。これは重要である。すなわち、現在の「最良」は損失最小ではなく、実装が定義した utility 近似指標で選ばれている。

ただし、この utility 指標は学習後に検証セット上で計算された近似指標であり、実際の前進シミュレーションから得られた `PnL / DD / turnover` ではない。したがって、ここで観測される 1 位は「現行近似指標の範囲での 1 位」であって、実運用 utility の 1 位ではない。

### 5.3 現結果の意味

現時点の結果から言えるのは次の程度である。

- 現行の参照実装は、短い最新 XAUUSD データに対して、完全なランダムではない方向性情報を捉えている可能性がある。
- `value_capture_ratio = 0.849089` は、現行の近似選択ルールの下で一定の価値回収が起きていることを示唆する。
- しかし `directional_accuracy = 0.566138` は高いとは言い切れず、`overlay_macro_f1 = 0.200000` は overlay のクラス品質がまだ弱いことを示す。

したがって、現結果は「有望な signal の兆候」ではあっても、「限界到達」や「十分な完成度」の証拠ではない。

## 6. Why Saturation Cannot Be Claimed

本節では、「すでに学習は限界まで進んでいる」と言えない理由を整理する。

### 6.1 探索空間が狭い

今回の探索は 9 候補のみであり、探索対象も `epochs / batch_size / learning_rate / hidden_dim / dropout / weight_decay` に限られる。仕様書が重視する `gamma`、`cost model c_h`、exit threshold、walk-forward 設計、utility fine-tune の有無、loss 構成そのものは未探索である。よって、現時点で「最適化余地が尽きた」とは言えない。

### 6.2 評価設計が単発 split である

現在の検証は単一の時系列 split であり、`purge` も `embargo` も `walk-forward` もない。非定常な金融時系列において、単発 split のスコアはその期間固有の条件に強く依存する。したがって、この split で改善が頭打ちに見えても、それは真の飽和を意味しない。

### 6.3 データ期間が短い

使用データは約 3 か月分であり、学習例は 133 件しかない。金価格のレジーム転換、マクロイベント、ボラティリティ環境の変化を十分に覆うには短い。少数サンプルで得られた plateau は、真の限界ではなくデータ不足による見かけの飽和である可能性が高い。

### 6.4 ベースライン比較がない

現在の repo には、少なくとも以下との定量比較がない。

- close-only baseline
- 単純 trend-following / mean-reversion baseline
- uncertainty を使わない horizon 選択 baseline
- overlay なし baseline

比較対象がない限り、現スコアが高いのか低いのか、また SignalCascade 数理の寄与がどれほどかを判定できない。

### 6.5 forward utility 系列がない

仕様書は `SimulateEnhancedCloseModel` による `PnL / utility / DD` 系列を要求しているが、現実装はそこまで到達していない。`utility_score` は検証セット上の近似再構成値であり、実際の時系列前進結果ではない。したがって、「運用上の飽和」を論じる土台が未完成である。

### 6.6 仕様書準拠の最適化が完了していない

仕様書が要求する 4 段階学習のうち、現実装は事実上単段の joint training である。特に `utility fine-tuning` と `threshold calibration` が未実装である以上、「仕様上のモデルを十分に学習した」とは言えない。

### 6.7 理論限界と実装限界が混同されている

現時点で観測されているのは「この reference 実装 + このデータ期間 + この split + この候補探索」の性能であって、SignalCascade 数理モデル一般の性能上限ではない。ここを混同すると、未実装要素や未検証条件を抱えたまま理論限界を語る誤りが生じる。

## 7. Model-Limit Assessment

次に、「この数理モデルの限界に到達したか」という問いに対する評価を整理する。結論は否定的である。より正確には、現時点では限界到達 여부を評価する証拠体系が成立していない。

### 7.1 非定常性

金価格はマクロ金利、米ドル、地政学、インフレ期待、リスクオフ需要などに敏感であり、統計的性質が時間とともに変化する。したがって、ある 3 か月区間での性能 plateau は、モデルの限界ではなく、そのレジーム下での局所最適に過ぎない可能性が高い。

### 7.2 マクロイベント依存

金価格の大きな変動は雇用統計、CPI、FOMC、地政学イベントなどの外生ショックに左右される。現モデルは OHLCV のみを使うため、これら外生変数を直接観測できない。この制約はモデル限界を規定する一因であるが、現時点ではその上限を定量化していない。

### 7.3 レジーム変化

上昇トレンド、レンジ、急落反発、高ボラ、低ボラで、`chi` や `close anchor` の有効性は変わり得る。長期の walk-forward 検証がない限り、レジームごとの性能上限は評価できない。

### 7.4 流動性時間帯差

金は欧州時間、NY 時間、アジア時間で流動性やボラティリティが異なる。現モデルは時間帯差を明示的な regime 変数としては扱っていないため、時間帯依存の性能差が限界要因になりうる。

### 7.5 OHLCV-only 情報制約

本モデルは order book、spread、ニュース、マクロ指標を持たず、OHLCV のみに基づく。これは実装の弱点ではなく、設計上の情報制約である。したがって、もし将来的に性能が頭打ちになったとしても、その要因が数理モデル自体なのか、観測情報の不足なのかを区別しなければならない。

### 7.6 ラベル設計依存

overlay 教師ラベルは、次の 4h 区間における符号付き path return と実現ボラティリティから合成されている。これは合理的な近似ではあるが、最適な exit 行動そのものではない。従って、overlay 性能の上限はラベル設計の上限にも束縛される。

### 7.7 overlay 教師生成バイアス

`hold / reduce / full_exit / hard_exit` は、将来 path の簡易閾値化による擬似教師である。もし閾値設計に系統誤差があれば、overlay はその誤差を忠実に学習してしまう。`overlay_macro_f1 = 0.200000` という低さは、モデル能力不足だけでなく教師設計の粗さを反映している可能性がある。

## 8. Required Experiments to Test the Limit

限界を本当に検証するには、次の順序で追加実験が必要である。

### 8.1 purged / embargoed walk-forward

最初に必要なのは、仕様書通りの `purged / embargoed walk-forward` である。これにより単発 split 依存を除去し、期間ごとの性能安定性を測る。

### 8.2 full forward simulation

次に、仕様書どおり `SimulateEnhancedCloseModel` を実装し、時系列前進で `horizon selection`、`overlay`、ポジション更新を逐次再現する必要がある。

### 8.3 PnL / DD / turnover / no-trade率

simulation の上で少なくとも以下を出力する必要がある。

- cumulative PnL
- net utility
- max drawdown
- turnover
- no-trade 率

これがなければ、現 `utility_score` が真の実務価値に対応しているか判断できない。

### 8.4 baseline 比較

限界を語る前に、以下の基準線が必要である。

- close-only baseline
- main-only baseline
- overlay なし baseline
- uncertainty を使わない horizon 選択 baseline
- 単純 trend-following / mean-reversion baseline

### 8.5 ablation

SignalCascade 数理の限界を評価するには、構成要素ごとの寄与を分離する必要がある。少なくとも次の ablation が必要である。

- `chi` なし
- `close anchor` 補正なし
- `1d / 1w` なし
- overlay なし
- uncertainty head なし

### 8.6 uncertainty calibration

仕様書は uncertainty を中核出力としている。したがって `coverage at 1 sigma` のみならず、複数水準での calibration、sharpness、miscalibration を継続的に測る必要がある。

### 8.7 長期期間拡張

最後に、複数年・複数レジームへの期間拡張が必要である。少なくとも 1 期間での頭打ちではなく、複数市場環境で plateau が再現されたときに初めて、「この設計は実質的に限界に近い」と議論する余地が生まれる。

## 9. Threats to Validity

本報告の妥当性には、少なくとも以下の制約がある。

1. 根拠を repo 内成果物に限定しており、外部再現や外部市場データで検証していない。
2. 現時点の数値は単一セッション `20260324T020618Z` に依存している。
3. `utility_score` は repo 独自の近似指標であり、厳密な取引損益ではない。
4. 最新 UI や補助 JSON は `current/metrics.json` を反映しているが、正本はあくまで `PyTorch/artifacts/...` の成果物である。
5. 現行 worktree には未整理の開発差分があるため、今後の実装更新で数値や構造が変わる可能性がある。

## 10. Conclusion

本レポートの結論は以下の 3 点に要約できる。

第一に、現時点の SignalCascade XAUUSD 学習について、「すでに限界まで学習した」とは言えない。データ期間は短く、学習例は 133 件、検証例は 27 件、探索候補は 9 本、評価設計は単一時系列 split に留まっており、最適化余地と検証余地が大きく残っているからである。

第二に、「この数理モデルの限界に到達した」とも言えない。現時点で観測されているのは仕様完全版ではなく、`reference implementation` の性能である。`utility fine-tuning`、`threshold calibration`、`walk-forward`、`forward simulation`、`drawdown`、`turnover`、`no-trade率` が未実装のままでは、モデル限界と実装限界を切り分けられない。

第三に、現状の結果は無価値ではない。`candidate_06` が `utility_score = 0.560945`、`value_capture_ratio = 0.849089`、`directional_accuracy = 0.566138` を示していることは、この設計が少なくとも最新の短期 XAUUSD データに対して何らかの有効 signal を持つ可能性を示唆する。しかしそれは「有望性の確認」であって、「限界到達の証明」ではない。現時点で妥当な結論は、SignalCascade はまだ評価途上にあり、限界到達を主張するには追加実験が必要、というものである。

## Appendix

### A. Latest Session Leaderboard (`session_20260324T020618Z`)

| Rank | Candidate | Utility | Value Capture | Directional Accuracy | Overlay Macro F1 | Best Val Loss | Selected Horizon |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `candidate_06` | `0.560945` | `0.849089` | `0.566138` | `0.200000` | `-2.407322` | `6` |
| 2 | `candidate_04` | `0.502885` | `0.756307` | `0.497354` | `0.206522` | `-2.300659` | `2` |
| 3 | `candidate_05` | `0.500394` | `0.735366` | `0.529101` | `0.206522` | `-2.077451` | `30` |
| 4 | `candidate_07` | `0.492282` | `0.762041` | `0.444444` | `0.206522` | `-2.341114` | `12` |
| 5 | `candidate_03` | `0.485511` | `0.681866` | `0.566138` | `0.200000` | `-2.485300` | `12` |
| 6 | `candidate_02` | `0.401047` | `0.470910` | `0.608466` | `0.200000` | `-2.418498` | `12` |
| 7 | `candidate_01` | `0.266129` | `0.135959` | `0.671958` | `0.200000` | `-2.467074` | `12` |
| 8 | `candidate_08` | `0.266129` | `0.135959` | `0.671958` | `0.200000` | `-2.467074` | `12` |
| 9 | `candidate_09` | `-0.229569` | `-0.819004` | `0.407407` | `0.206522` | `-2.072911` | `30` |

### B. Primary Evidence Files

- `PyTorch/requirements_multiframe_candlestick_model.md`
- `PyTorch/logic_multiframe_candlestick_model.md`
- `PyTorch/README.md`
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
- `PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
- `PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- `PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
- `PyTorch/artifacts/gold_xauusd_m30/current/metrics.json`
- `PyTorch/artifacts/gold_xauusd_m30/current/manifest.json`
- `PyTorch/artifacts/gold_xauusd_m30/archive/session_20260324T020618Z/leaderboard.json`
