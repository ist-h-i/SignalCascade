# SignalCascade Review Handoff

最終更新: 2026-03-26 JST

## 1. あなたへの依頼

あなたは、時系列予測・Quant ML・policy optimization・risk-aware training に強い reviewer です。
この handoff では、`/Users/inouehiroshi/Documents/GitHub/SignalCascade` の現行実装を前提に、固定済みの canonical target spec

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/shape_aware_profit_maximization_model.md`

へどう移行すべきかを review してほしいです。

- 初期アイデアの中核だった `multi-layer / multi-scale causal convolution` は残す
- `shape 判定` と `内部状態ベクトル` も残す
- ただし最終目的は `threshold を満たす signal 選別` ではなく、`売買利益最大化` に置き換える
- 具体的には、`shape-aware return distribution` と `cost-aware optimal position policy` を end-to-end に一貫化する

レビューで判断してほしい主論点は次です。

- 固定済み target spec を前提に、現行の `supervised predictor + post-hoc selection policy` からどう移すのが筋が良いか
- 現行実装の何を温存し、何を置き換えるべきか
- どの順で実装するとリスクが低く、検証効率が高いか
- 数式上の隠れた前提、単位不整合、境界条件の危険がどこにあるか

一般論ではなく、このファイルに記載した `file / artifact / metric / command` を根拠に答えてください。

## 2. ゴール

- `SignalCascade` の上位設計を、`precision-first threshold policy` 中心から `shape-aware profit objective` 中心へ切り替えるべきかを確定する
- 現行コードとの責務対応を明文化し、再実装ではなく移行計画として分解する
- reviewer に、`最小実装 slice`、`最初の検証順`、`消すべき旧 abstraction` を rank してもらう

## 3. 現状の重要観測

### A. Confirmed facts

- workspace は `/Users/inouehiroshi/Documents/GitHub/SignalCascade`、branch は `main` です。
- `git rev-parse --is-inside-work-tree` は `true`、`git log main..HEAD` は空で、`HEAD` に `main` 未取り込み commit はありません。
- evidence 収集時の `git status --short` は dirty で、変更は次の 3 ファイルです。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/reviewer_submission.md`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`
- recent commits は次です。
  - `ba54089 Clarify proposal and acceptance diagnostics semantics`
  - `61329bf Align chart and decision panel heights`
  - `86f97b2 Clarify proposal and acceptance diagnostics semantics`
  - `b34cce2 Add no-candidate policy diagnostics and review handoff`
  - `b114890 Add review diagnostics and formula regression tests`

#### 現行の「残せる要素」

- encoder はすでに `causal conv + residual + dilation` です。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
  - `TemporalEncoder` は `ResidualTemporalBlock(dilation=1,2,4)` を 3 層持ちます。
  - main encoder は `4h / 1d / 1w`、overlay encoder は `1h / 30m` です。
- shape head はすでに実装済みです。
  - 同ファイルで `main_shape_heads` があり、各 main timeframe に対して 3 成分 shape を出力しています。
- 入力特徴もすでに「価格そのもの」ではなく、close-anchor ベースの形状特徴です。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/logic_multiframe_candlestick_model.md`
  - `x_t = (z_t, dz_t, chi_t, g_t, rho_t, nu_t)` を使用しています。
- 評価側には価値指標がすでにあります。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
  - `project_value_score`, `profit_factor`, `signal_sortino`, `value_capture_ratio`, `turnover`, `max_drawdown` を保存しています。

#### 現行の「まだ一貫化されていない要素」

- 学習目的は profit objective ではなく、supervised loss の合成です。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
  - `total_loss = return_loss + 0.35*direction_loss + 0.25*shape_loss + 0.10*overlay_loss + 0.20*consistency_loss`
  - `log wealth`, `PnL`, `CVaR`, `turnover penalty` は train loss に入っていません。
- policy は予測器の外側にある post-hoc stage です。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
  - OOF snapshot から `correctness_model(q)` と `selector_model` を fit し、`selector_probability` threshold で accept / reject しています。
- position も cost-aware optimal policy ではありません。
  - 同ファイルで `raw_position = tanh(position_scale * mu_selected / max(sigma_selected, 1e-6))`
  - 最終 `position = raw_position * hold_probability`
  - `|q-q_{t-1}|` を含む最適化や no-trade band はありません。
- 内部状態ベクトルは明示的にはありません。
  - 現行 `model.py` は各 encoder latent を fuse しますが、`m_t = tanh(A m_{t-1} + B[...])` のような再帰 state は持っていません。
- shape posterior を return distribution の混合係数として使っていません。
  - shape head は auxiliary target であり、`mu, sigma` を出す head と直接結合されていません。

#### 現在の validation / diagnostics evidence

- baseline metrics は次です。
  - artifact: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/metrics.json`
  - `project_value_score=0.0562941825071729`
  - `utility_score=0.08644704399974731`
  - `selection_precision=0.0`
  - `selection_support=0`
  - `precision_feasible=False`
  - `threshold_calibration_feasible=False`
  - `best_selection_lcb=0.0`
  - `support_at_best_lcb=8.0`
  - `precision_at_best_lcb=0.0`
  - `tau_at_best_lcb=0.21013891952988573`
  - `actionable_edge_rate=0.041353383458646614`
  - `alignment_rate=0.43107769423558895`
  - `value_capture_ratio=0.0`
  - `profit_factor=0.0`
  - `signal_sortino=0.0`
  - `selection_brier_score=0.2482326997130845`
- fresh replay diagnostics 4 本も `accepted_row_count=0` のままです。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability_head`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability_head`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge_head`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_edge_correctness_product_head`
- 4 本に共通して次が確認されています。
  - `proposed_row_count=30`
  - `accepted_row_count=0`
  - `no_candidate_count=236`
  - `threshold_status=infeasible`
  - `threshold_origin=validation_replay`
  - `stored_threshold_compatibility=config_mismatch`
  - `selection_threshold_mode_requested=auto`
  - `selection_threshold_mode_resolved=replay`
- best-LCB row は次です。
  - `selector_probability`: `accepted_count_at_tau=7`, `precision_at_tau=0.7142857142857143`, `lcb=0.5255503826141334`
  - `correctness_probability`: `accepted_count_at_tau=1`, `precision_at_tau=1.0`, `lcb=0.5`
  - `actionable_edge`: `accepted_count_at_tau=2`, `precision_at_tau=1.0`, `lcb=0.6666666666666666`
  - `edge_correctness_product`: `accepted_count_at_tau=11`, `precision_at_tau=0.5454545454545454`, `lcb=0.39787687199922495`
- horizon ごとの candidate scarcity も fresh rerun 後に不変です。
  - `h=1 candidate_rate = strict_candidate_rate = 0.08270676691729323`
  - `h=3 candidate_rate = strict_candidate_rate = 0.041353383458646614`
  - `h=6 candidate_rate = strict_candidate_rate = 0.0`

#### 直近の検証コマンド

- `PyTorch/.venv/bin/python -m compileall PyTorch/src`
- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m unittest PyTorch.tests.test_policy_training`
  - 結果: `Ran 12 tests ... OK`
- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability_head --selection-threshold-mode auto --allow-no-candidate --selection-score-source selector_probability`
- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability_head --selection-threshold-mode auto --allow-no-candidate --selection-score-source correctness_probability`
- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge_head --selection-threshold-mode auto --allow-no-candidate --selection-score-source actionable_edge`
- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_edge_correctness_product_head --selection-threshold-mode auto --allow-no-candidate --selection-score-source edge_correctness_product`

### B. 今回 fixed したい target model contract

ここでは、初期アイデアを残したまま、目的関数だけを `売買利益最大化` に置き換えた完全版を canonical target contract として固定済みです。

#### Target contract の要約

- 入力:
  - 正規化 OHLCV 系特徴 `x_t = (ℓ_t, b_t, u_t, d_t, ν_t, ζ_t)`
  - causal window `X_t = [x_{t-L+1}, ..., x_t]`
- encoder:
  - multi-scale causal convolution により `h_t = MSCNN(X_t)`
- shape 判定:
  - `s_t = softmax(U_s h_t + b_s)`
- 内部状態:
  - `m_t = tanh(A m_{t-1} + B[h_t; s_t; z_t] + b_m)`
  - `v_t = [h_t; s_t; z_t; m_t]`
- shape-conditioned distribution:
  - `μ_{t,m} = a_m^T v_t + α_m`
  - `log σ_{t,m}^2 = g_m^T v_t + β_m`
  - `μ_t = Σ_m s_{t,m} μ_{t,m}`
  - `σ_t^2 = Σ_m s_{t,m}(σ_{t,m}^2 + μ_{t,m}^2) - μ_t^2`
- tradeability gate:
  - `g_t = Σ_m ω_m s_{t,m}`
  - `μ̃_t = g_t μ_t`
- cost-aware policy:
  - `q_t* = argmax_{|q|<=q_max} { μ̃_t q - (γ/2) q^2 σ_t^2 - c_t |q-q_{t-1}| }`
  - no-trade band:
    - `| μ̃_t - γ σ_t^2 q_{t-1} | <= c_t  =>  q_t* = q_{t-1}`
- realized PnL:
  - `π_{t+1} = q_t* r_{t+1} - c_t |q_t* - q_{t-1}*|`
- 学習目的:
  - `max Σ_t log(1 + π_{t+1}(θ)) - λ_tail CVaR_α(-π(θ))`

#### この contract が現行と異なる点

- 現行は `predict mu/sigma -> post-hoc policy fit -> threshold accept -> tanh position`
- target は `shape-aware distribution -> tradeability gate -> optimal position policy -> wealth objective`
- 現行の selector / threshold は `precision-first` の研究プロトコルとしては有効ですが、`PnL maximization` の主目的には直接つながっていません

### C. Working hypotheses

- 現行の conv encoder / shape head / sigma 推定 / value metrics は、profit-maximization 版の土台として再利用価値が高いです。
- 本当の置換点は `policy_service.py` と `losses.py` の責務です。
  - `post-hoc selector/correctness/threshold` を中心にした acceptance policy は、target contract では主役ではなくなります。
- current diagnostics が `proposal_count > 0` なのに `accepted_count = 0` で止まっているのは、threshold semantics の調整だけでは根治しない可能性が高いです。
  - つまり、`accept/reject` の設計を磨くより、`μ, σ, g_t, q_t*` を一体で最適化する方が大きな設計転換になると考えています。
- ただし、いきなり full end-to-end へ飛ぶと optimization が不安定になる可能性があります。
  - supervised warm start を残す 2-phase 移行の方が安全かもしれません。
- 入力特徴は、ユーザー提案の `ℓ,b,u,d,ν,ζ` へ全面移行しなくても、現行の close-anchor 特徴を first phase では維持できる可能性があります。
  - ここは reviewer に preserve / replace を判断してほしいです。

## 4. レビューしてほしい論点

### 論点 1: この target contract は数式として筋が通っているか

- `shape posterior -> mixture distribution -> tradeability gate -> q_t*` の流れに、単位不整合や hidden assumption はありますか。
- 特に reviewer に見てほしい点は次です。
  - `μ_t, σ_t` が現行コードの restore 後 return unit と整合するか
  - `c_t` と `γ σ_t^2 q^2 / 2` のスケール比較が現実的か
  - `log(1 + π)` 近似と no-trade band 導出に危険な境界条件がないか
  - `σ_t -> 0` や sparse trade 時の扱い

### 論点 2: 現行コードで本当に残すべきもの / 捨てるべきものは何か

- 残す候補:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/dataset_service.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/domain/close_anchor.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- 置換候補:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py` の validation metric contract
- reviewer には、「どこまで温存すると移行コストが下がり、どこを残すと設計が濁るか」を切ってほしいです。

### 論点 3: 実装順の最小 slice をどう切るべきか

次の候補を、`cheap / information gain / rollback のしやすさ` の順で rank してください。

1. 現行 encoder / close-anchor features を維持したまま、`post-hoc selection policy` を `differentiable q_t* policy head` に置き換える
2. `shape posterior s_t` を明示 head にし、`shape-conditioned mixture-of-experts` で `μ_t, σ_t` を再構成するが、入力特徴は現行のまま維持する
3. `m_t` を導入し、`v_t=[h_t;s_t;z_t;m_t]` へ拡張する
4. train loss を `supervised warm start -> profit objective fine-tune` の 2 段に分ける
5. 入力特徴を `close-anchor` から、今回の `ℓ,b,u,d,ν,ζ` 系へ全面移行する

私は現時点では `4 -> 1 -> 2 -> 3 -> 5` くらいを仮説にしていますが、確信はありません。

### 論点 4: いまの diagnostics / metrics contract をどう置き換えるべきか

現行は `selection_precision`, `threshold_calibration_feasible`, `best_selection_lcb`, `acceptance_coverage` が中心ですが、profit objective へ移るなら中心指標も変わるはずです。

reviewer には、first phase から最低限入れるべき metric / artifact を提案してほしいです。候補は次です。

- `average_log_wealth`
- `realized_pnl_per_anchor`
- `turnover`
- `cvar_alpha`
- `max_drawdown`
- `no_trade_band_hit_rate`
- `expert_entropy`
- `shape_gate_usage`
- `q_t*` と `q_{t-1}` の変更量分布
- `μ_t / σ_t / μ̃_t` の calibration

### 論点 5: 現行 evidence は「設計転換」を支持しているか

- fresh diagnostics 4 本とも `accepted_row_count=0` のままで、threshold infeasibility が続いています。
- ただし `proposal_count=30` はあり、完全に signal が無いわけではありません。
- reviewer には、これを
  - `threshold tuning の延長で直すべき問題`
  - `policy objective が目的とずれているサイン`
  のどちらとして読むべきかを判断してほしいです。

## 5. 参照ファイル / アーティファクト

### 実装

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/dataset_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/domain/close_anchor.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_math_formulas.py`

### 仕様・設計文書

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/logic_multiframe_candlestick_model.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/shape_aware_profit_maximization_model.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/requirements_multiframe_candlestick_model.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/enhance_result.md`

### Artifact

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/metrics.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/selection_policy.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/prediction.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability_head/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability_head/threshold_scan.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability_head/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge_head/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_edge_correctness_product_head/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/report_signalcascade_xauusd.md`

## 6. 期待する出力形式

次の順で返してください。

1. 結論
2. confirmed facts と working hypotheses の分離
3. 高優先度の review findings
4. 優先度付きの実装 slice 提案
5. cheap / high-information gain な最初の検証 3 件
6. 必要なら数式・metric・artifact contract の修正提案

各 finding では、可能なら次も含めてください。

- severity
- risky な理由
- どのファイルをどう変えるとよいか
- まず最初に取るべき検証

## 7. 制約

- live chat transcript には依存せず、この handoff だけで判断してください
- 確認済みの事実と仮説を分けて書いてください
- 一般論ではなく、ここに書いた `path / metric / command / artifact` を根拠にしてください
- `threshold tuning の延長` と `profit objective への設計転換` を混同せずに評価してください
- reviewer 自身が「唯一の正解」を押し付けるのではなく、`最も筋の良い移行順` を rank してください
