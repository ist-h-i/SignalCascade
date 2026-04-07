以下、dossier だけを根拠にしたレビューです。根拠はすべて添付 markdown 内の artifact / source / metrics / 実装断片に依拠しています。全体の source of truth はこの dossier です。

## 1. 結論

最有力ボトルネックは、**model capacity 不足ではなく、実効自由度の崩壊**です。具体的には、validation でも live でも `policy_horizon` が 18 に完全固定、`shape_posterior` は class 1 に実質固定、`g_t` は 0.5 近傍に張り付き、複雑な shape-aware / multi-horizon / gate-aware 設計が、実運用上は「ほぼ単一 shape・単一 horizon の policy」に縮退しています。これは live 精度の上限を直接縛ります。

次点の問題は、**実装と仕様の乖離**です。dossier 上の旧 logic 文書では direction / overlay / consistency を含む設計が説明されていますが、現行の学習 loss 実装は実質 `return + shape + profit` のみで、`overlay_target` は batch に乗っておらず、overlay head も現行 forward 出力の主経路に存在しません。つまり overlay branch は複雑性を増やしている一方、直接 supervision をほぼ受けていない可能性が高いです。

3 点目は、**checkpoint 選択と tuning の最適化対象が live objective と十分に一致していない**ことです。checkpoint は `average_log_wealth` 最大で選ばれておらず、epoch 12 が採用されている一方で epoch 13/14 の方が `validation_exact_log_wealth` は高いです。さらに tuning は network hyperparameters 中心で、経済的に本質的な `cost / gamma / q_max / cvar / min_policy_sigma` は本体 tuning ではなく、ほぼ事後的な diagnostics sweep に追いやられています。

したがって優先順位は、**(a) collapse の正体を切る、(b) checkpoint / validation protocol を直す、(c) economic parameter を前面に出した cheap experiment を回す**、です。

---

## 2. Confirmed facts / hypotheses の分離

### Confirmed facts

validation の `policy_horizon_distribution` は `18: 1.0`、`shape_posterior_top_class_share` は `{'1': 1.0}`、`shape_posterior_mean` でも class 1 が約 0.796 を占めています。live でも `policy_horizon=18`、`executed_horizon=18`、`g_t=0.498986...`、shape posterior の class 1 は約 0.952 です。

dataset は `sample_count=184`、`train=118`、`validation=36`、`purged=30` です。validation は 1 本の holdout split で構成されています。

`walk_forward_folds=3` と `oof_epochs=3` は config に存在しますが、training/tuning の主経路で実利用されている証拠は dossier 内に見当たりません。少なくとも code search では config / bootstrap / tests には現れますが、学習ループや candidate selection の主経路には出ていません。

checkpoint は `checkpoint_selection_score` 最小で選ばれます。その score は `-average_log_wealth + forecast_mae + calibration項 + position_gap項` 型です。よって `average_log_wealth` 単独最大化ではありません。

実際に epoch 12 が採用されていますが、epoch 13 は `validation_exact_log_wealth=0.012409`、epoch 14 も `0.011408` で、採用 epoch 12 の `0.005911` より高いです。にもかかわらず epoch 12 が選ばれているため、selection rule は live wealth 最大化と一致していません。

現行 tuning が探索するのは `epochs, batch_size, learning_rate, hidden_dim, dropout, weight_decay` のみです。

一方で policy calibration sweep は `state_reset_mode, cost_multiplier, gamma_multiplier, min_policy_sigma` を 72 通りに後段評価し、selected row は `carry_on | cost x0.5 | gamma x0.5 | min_sigma 1e-4` です。

stateful evaluation では `carry_on` が `average_log_wealth=0.005911` で最良、`reset_each_session_or_window=0.005731`、`reset_each_example=0.005582` です。ただし turnover は `3.356 -> 9.378 -> 14.809` と急増し、directional_accuracy は `0.611 -> 0.500 -> 0.444` に低下します。

`exact_smooth_horizon_agreement=1.0`、`exact_smooth_no_trade_agreement=1.0` で、current config の exact/smooth gap は horizon/no-trade 判定では出ていません。ただし `exact_smooth_position_mae=0.08558` はゼロではありません。

`policy_summary.csv` 先頭行群では、全行で `policy_horizon=18` かつ `selected_g_t` が 0.499 近傍です。

現行 loss は `return_loss + shape_loss + profit_loss` です。overlay loss, direction loss, consistency loss は actual code path には入っていません。`examples_to_batch` に `overlay_target` もありません。

`build_policy_path_terms` では `g_t` は scalar gate を全 horizon にそのまま broadcast して使います。つまり gate は horizon-specific ではありません。

### Hypotheses

最も強い仮説は、**shape collapse / horizon collapse は market structure を捉えた結果というより、small sample + oversized architecture + weakly supervised branches + single split selection による degeneracy** です。これはかなり強いが、現データだけでは 100% 断定はしません。

`carry_on` の優位は genuine な temporal memory かもしれませんが、single contiguous validation block 上で recurrent context を引き継いでいるため、**evaluation artifact が混ざっている可能性**を高く見ます。

post-hoc sweep で `cost x0.5 / gamma x0.5` が勝っているのは、「policy が本当にそうあるべき」より、**validation block に過適合した aggressive setting** を選んでいる可能性があります。特に no-trade rate が現行でも 0 で、live でも `q_t_prev=0.2781 -> q_t=0.9364` の大きなリサイズが出ているため、この懸念は強いです。

---

## 3. 優先度付き findings (`P0 / P1 / P2`)

### P0-1. 実効自由度が崩壊しており、これが live 精度制約の本丸

**判断**
`shape-aware / multi-horizon / gate-aware` を名乗る設計に対し、validation と live の両方で実際の decision manifold が極端に狭いです。これは最優先で切るべきです。

**根拠**
artifact / metric: validation `policy_horizon_distribution={'18':1.0}`、`shape_posterior_top_class_share={'1':1.0}`、`g_t_mean=0.49948`。live でも `policy_horizon=18`, `executed_horizon=18`, `g_t=0.49899`, shape posterior class 1 約 0.952。
artifact: `policy_summary.csv` 先頭でも全行 `policy_horizon=18`, `selected_g_t≈0.4995`。

**意味**
ここでは forecast accuracy の改善余地より前に、**policy が「複数の選択肢の中から選ぶ系」になっていない**ことが問題です。horizon / shape / gate の分岐が働いていないと、追加 head や recurrent state の価値も検証不能になります。

**すぐ変更すべきか**
まず cheap experiment で原因分解。いきなり複雑化せず、むしろ簡素化 ablation が先です。

---

### P0-2. 実装と仕様がズレており、overlay branch の複雑性に対して supervision が乏しい

**判断**
現行 system は docs 上の “overlay / direction / consistency を持つ設計” と、実装上の “return + shape + profit のみ” がズレています。これは精度問題であると同時に、レビュー可能性・将来保守性の問題です。

**根拠**
spec/logic 文書では overlay label, overlay head, direction / consistency を含む loss が説明されています。
actual code では `examples_to_batch` に `overlay_target` がなく、`total_loss` は return / shape / profit のみです。
current model test でも主出力は `forecast_mu/sigma`, `policy_mu/sigma`, `shape_probs`, `state_vector` 等で、overlay logits は主経路に現れていません。

**意味**
overlay encoders は latent fusion に入っているので capacity と variance は増やしているのに、overlay 自体を正す direct gradient が弱いか存在しない。**情報利得の低い複雑性**です。

**すぐ変更すべきか**
これはコード/仕様のどちらかをすぐ揃えるべきです。

1. overlay を本当に使うなら supervision を戻す。
2. 当面使わないなら branch を落とす。
   中途半端が最悪です。

---

### P0-3. checkpoint 選択指標が live objective と非整合

**判断**
best checkpoint の selection rule が `average_log_wealth` 最大化と整合していません。これは live artifact の quality ceiling を直接縛ります。

**根拠**
`_checkpoint_selection_score` は `-average_log_wealth + forecast_mae + calibration_error + position_gap` 型です。
実際に epoch 12 が採用されていますが、epoch 13/14 の方が `validation_exact_log_wealth` は高いです。epoch 13 は `0.012409`、epoch 14 は `0.011408`、epoch 12 は `0.005911`。

**意味**
「forecast の見た目を良くするために、profit objective 上はより良い checkpoint を捨てている」可能性があります。live 精度制約のうち、最も cheap に直せる箇所です。

**すぐ変更すべきか**
はい。少なくとも `best_epoch_by_exact_log_wealth` を artifact に併記し、採用との差を常時監査すべきです。

---

### P1-1. single split bias が強く、carry-on 優位と policy sweep 優位が artifact である可能性

**判断**
validation protocol が脆いです。walk-forward / OOF を config に持つのに、実 training/tuning path は one split です。

**根拠**
config に `walk_forward_folds=3`, `oof_epochs=3`。
code search ではこれらは config/bootstrap/tests に現れる一方、学習本体の主経路では使われていません。
stateful evaluation で `carry_on` が最良ですが、validation examples は 1 本の連続系列として評価され、`previous_state` と `previous_position` を逐次 carry しています。

**意味**
carry-on の優位、post-hoc selected policy sweep row の優位、epoch 13/14 の優位/劣位、どれも block dependence の影響を受けています。今の validation からは「本当に live で効く」かを切りにくいです。

**すぐ変更すべきか**
cheap experiment を最優先。full retrain 前に blocked walk-forward replay を入れるべきです。

---

### P1-2. tuning が network hyperparameters に偏り、economically important parameters を後回しにしている

**判断**
現行 tuning の search space がズレています。

**根拠**
candidate search は `epochs, batch_size, learning_rate, hidden_dim, dropout, weight_decay` のみ。
一方で diagnostics sweep では `cost_multiplier, gamma_multiplier, min_policy_sigma, state_reset_mode` を変えるだけで `average_log_wealth` が `0.005911 -> 0.011346` まで動いています。

**意味**
“ネットワークの学習”より“policy economics の後段較正”の方が、現行 artifact では感度が大きい可能性があります。
この状態で hidden_dim や dropout をいじっても、information gain が低いです。

**すぐ変更すべきか**
はい。次週の探索予算はまず economic knobs に割くべきです。

---

### P1-3. forecast head と policy head の数値乖離が大きく、表示 forecast と trade decision の意味がズレている

**判断**
これは UI/interpretability 上の重大論点です。

**根拠**
推論 payload は `forecast_mu/sigma` 由来の `mu_t, sigma_t` と、`policy_mu_t, policy_sigma_t` を別々に保存しています。decision は `policy_mean/policy_sigma` で作られます。
live では表示 forecast の `h=18 mu_t=0.0834, sigma_t=0.0465` に対し、decision 側の `policy_mu_t=1.0607, policy_sigma_t=0.4330` です。これは同じ “18 horizon の見通し” としてはかなり別物です。

**意味**
dashboard で見せている forecast を operator が “decision basis” だと解釈すると誤読します。
精度の問題だけでなく、意思決定の説明可能性の問題です。

**すぐ変更すべきか**
はい。最低限、UI では “display forecast” と “policy driver” を分けて表示すべきです。理想は head の tying か、分離理由の監査です。

---

### P1-4. carry-on recurrent state の優位は genuine かもしれないが、現在の証拠では artifact を十分排除できない

**判断**
carry-on を採用してよいが、過信は危険です。

**根拠**
carry_on は `average_log_wealth` で最良。
ただし reset 系との差は絶対値で小さい一方、turnover・directional_accuracy・position gap は大きく変わります。
state reset 境界は session/day/gap 基準で、carry_on は contiguous validation sequence をほぼ通しで保持します。

**意味**
本当に alpha を持つ memory なのか、連続 holdout の文脈リークに近い挙動なのか、まだ切れていません。

**すぐ変更すべきか**
まず block-wise replay で確認です。

---

### P2-1. no-trade band が実質ほとんど機能していない

**判断**
金融実務上、これはかなり気になります。

**根拠**
current validation の `no_trade_band_hit_rate=0.0`。
policy sweep でも carry_on の no-trade rate は cost/gamma を上げてようやく 0.0278〜0.0833 程度です。
live でも `q_t_prev=0.2781 -> q_t=0.9364` と大きくリサイズしています。

**意味**
XAUUSD 30m でこの程度の friction model なのに “ほぼ毎回 trading” になるなら、cost model / gamma / min_policy_sigma のどれか、あるいは全部が甘い可能性があります。

**すぐ変更すべきか**
cheap experiment でよいですが、優先度は高めです。

---

### P2-2. sample size に対して architecture は重い

**判断**
capacity 過多は主因ではなく副因ですが、collapse を助長しています。

**根拠**
artifact 上の dataset は `train=118`, `validation=36`。
一方、現行 architecture は 5 encoder + latent fusion + shape head + memory update + 4 expert heads を持ちます。
埋め込まれた config/architecture から概算すると trainable parameter は約 19.5 万です。118 train samples に対して過重です。これは dossier 内の current config (`hidden_dim=32`, `state_dim=24`, `shape_classes=6`, horizons=7) と model 定義からの計算です。根拠の構造自体は dossier にあります。

**意味**
過学習というより、branch specialization が安定せず、一番楽な collapse 解に落ちている可能性が高いです。

**すぐ変更すべきか**
大規模 redesign の前に、簡素化 ablation を先にやるべきです。

---

## 4. 数学レビュー観点での重大論点

第一に、`g_t` が horizon-specific ではなく、1 個の scalar gate を全 horizon に broadcast しています。したがって “shape-aware horizon selection” というより、**shape で全 horizon の mean を一括減衰させた後、horizon ごとの utility 比較をしているだけ**です。これでは horizon collapse が起きても不思議ではありません。

第二に、`sigma_sq = sigma^2.clamp_min(min_policy_sigma^2)` で floor を入れたうえ、utility 最大の horizon を exact で選ぶので、small sample では「ある horizon の mean/sigma の相対順位」だけが安定して固定されやすいです。今その勝者が 18 に張り付いています。

第三に、`cvar_tail_loss` は validation 36 点に対して `alpha=0.1` なので、実際には worst 3〜4 点程度の平均です。これは ranking 指標としてノイジーです。現在の policy sweep / optimization gate で CVaR を強く読むと、かなり分散が大きいはずです。

第四に、`sigma_calibration` は “|forecast error| と sigma の平均絶対差” であって、proper probabilistic calibration ではありません。checkpoint selection がこの proxy に依存すると、**wealth に直結しない calibration shape** を拾うリスクがあります。

第五に、smooth/exact gap は horizon/no-trade 判定では今は出ていませんが、`exact_smooth_position_mae=0.0856` は無視できるほど小さくはありません。特に live で position がほぼ飽和域に入ると、small MAE が PnL に変換されたときの影響は非線形です。

---

## 5. 金融実務レビュー観点での重大論点

最も大きいのは、**post-hoc sweep で `cost x0.5 / gamma x0.5` が selected になっている点**です。これは「execution friction を半分、risk aversion を半分にした方が validation wealth が良かった」という意味で、実務的には危険信号です。もしこの倍率に実市場根拠がないなら、validation block に対する aggressive fit を選んでいるだけです。

次に、no-trade band が current validation で一度も発火していません。XAUUSD 30m で friction-aware policy を名乗るなら、全期間ほぼ毎回 trade する設計は慎重に疑うべきです。特に live で `trade_delta=0.6583` の大幅増しが出ている以上、slippage / spread widening / event-time liquidity の取り扱いが足りていない可能性があります。

さらに、表示 forecast と decision driver が乖離しているまま dashboard に出ると、運用者は “0.083 の expected log return を見て大きく買っている” と誤解しますが、実際には policy head の内部値で決めています。これは **operator risk** です。

最後に、carry-on state の優位を live にそのまま持ち込むのは尚早です。金融実務では “context persistence で validation が良く見える” パターンは珍しくありません。今の evidence だけで recurrent carry-on を production default として強く正当化するのは難しいです。

---

## 6. 最小実験プラン (3 件以内)

### 実験 1

**checkpoint 再選定監査: epoch 全件を exact objective で再順位付け**

やることは単純で、既存 history / checkpoints を replay し、
`A: 現行 selection_score`
`B: average_log_wealth`
`C: average_log_wealth - λ * cvar_tail_loss`
の 3 指標で best epoch を比較します。
これは最も cheap で、information gain が大きいです。すでに epoch 13/14 が epoch 12 より高い exact wealth を持つ証拠があるため、まずここを切るべきです。

### 実験 2

**structural simplification ablation: shape/policy 分離を縮める**

候補は 2 本で十分です。

1. forecast head と policy head を tie する。
2. overlay encoders を切った簡素版を回す。

もしこれで OOS wealth が落ちない、あるいは上がるなら、現在の複雑性はノイズ源です。collapse の本質が「branch が要らない」のか「branch は要るが supervision が足りない」のかを最短で見分けられます。根拠は、現状 overlay が direct supervision を欠き、shape/gate/horizon が collapse していることです。

### 実験 3

**network を固定して economic knobs を walk-forward で探索**

network hyperparameters ではなく、
`cost_multiplier, gamma_multiplier, min_policy_sigma, q_max, cvar_weight`
を主対象にして、3-fold blocked walk-forward で回します。
今の dossier では post-hoc sweep だけで wealth が大きく動いているので、次の 1 週間で最も information gain が大きいのはこれです。

---

## 7. 変更優先順位

1 位は **checkpoint 選択ルールの監査・是正** です。最も cheap で、artifact quality に直接効きます。

2 位は **実装と仕様の整合** です。overlay/direction/consistency を使うのか、使わないのかを決めて揃えるべきです。今の中間状態はレビュー不能性と branch collapse の両方を悪化させています。

3 位は **walk-forward / blocked evaluation 導入**。carry-on 優位や policy sweep 優位の真偽を切るために必要です。

4 位は **economic parameter を tuning の主軸へ昇格**。network hyperparameters はその後で十分です。

5 位は **UI/contract 修正**。forecast と policy driver を混同させない表示に変えるべきです。

---

## 8. ゴールまでのステップと現在地

ステップ 1は、現行 artifact の「本当に効いている自由度」を確定することです。今は horizon / shape / gate の collapse が強く、複雑な設計の大半が死んでいる可能性が高いです。

ステップ 2は、selection bias を外すことです。checkpoint と validation protocol を直し、`carry_on` や `cost x0.5 / gamma x0.5` の優位が genuine かを blocked walk-forward で確認します。

ステップ 3は、そのうえで branch を足すか削るかを決めることです。overlay を復活させるなら supervision を戻す。戻さないなら branch を削る。forecast/policy を分けるなら、その分離が wealth を改善していると示す必要があります。

**現在地**
「動く artifact はあるが、selection / shape / horizon の多くが collapse しており、validation protocol と checkpoint rule がその collapse を見逃している」段階です。live が悪いというより、**live をまだ正しく測れていない**色が強いです。

**immediate next action**
次にやるべき 1 手は、**epoch 再順位付け監査**です。具体的には current run の epoch 1–14 を `average_log_wealth` と `average_log_wealth - λ·CVaR` で再ランキングし、epoch 12 採用との差分を出してください。これが最短で selection rule のズレを可視化し、その後の simplified ablation と walk-forward 実験の優先順位を確定できます。
