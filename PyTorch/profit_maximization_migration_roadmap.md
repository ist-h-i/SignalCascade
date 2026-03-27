# Profit Maximization Migration Roadmap

最終更新: 2026-03-27 JST

この文書は、`SignalCascade` を現行の

- `supervised predictor + post-hoc selection policy + threshold calibration`

から、新 canonical spec

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/shape_aware_profit_maximization_model.md`

へ完全移行するまでのロードマップです。

対象は `PyTorch/src/signal_cascade_pytorch` 配下の実装、関連ドキュメント、artifact contract、評価系です。

## 1. 現在地

現状の reference implementation は次を持っています。

- multi-timeframe causal encoder
- shape head
- return mean / sigma head
- OOF ベースの correctness / selector / threshold policy
- `project_value_score`, `profit_factor`, `signal_sortino` などの価値指標

一方で、新 spec に対する主な未移行点は次です。

- `shape posterior s_t` が return distribution の混合係数になっていない
- `内部状態ベクトル v_t=[h_t;s_t;z_t;m_t]` が明示されていない
- `q_t*` を直接解く cost-aware policy layer が存在しない
- `log wealth - CVaR` の profit objective が train loss に入っていない
- diagnostics / report / prediction 出力が旧 threshold-policy 語彙を引きずっている

## 2. ゴール

完全移行の Done 条件は次です。

1. 学習時の主目的関数が `log wealth - CVaR` を中心とした profit objective に置き換わっている
2. 推論時の主 decision path が `X_t -> h_t -> s_t -> v_t -> (mu_t, sigma_t^2) -> q_t* -> pi_{t+1}` になっている
3. `selection_policy.json` / `threshold calibration` / `selector_probability threshold` が主経路から外れている
4. 学習・推論・評価・report・diagnostics の語彙が新 spec と整合している
5. unit test / integration test / replay diagnostics / smoke training が新 contract で再成立している

## 3. 移行の基本方針

- 一気に全面置換しない
- 各 phase で `動く最小 slice` を作り、旧系と新系の責務を比較できる状態で止める
- 学習 objective の変更は、必ず metrics / diagnostics の変更と同時に入れる
- 旧 threshold-policy は、比較用 fallback として一定期間残すが、早い段階で主経路から外す
- 高 horizon の拡張より先に、`h={1,3}` など短い horizon で新 objective を安定化させる

## 4. フェーズ一覧

## Phase 0. Baseline Freeze

### 目的

移行開始前の基準点を固定し、以後の比較ができるようにする。

### 主作業

- 現行 reference implementation を baseline として固定
- 既存 diagnostics artifact の保存先と比較用指標を決める
- 新 spec / 現行ロジック / historical requirements / 本ロードマップの参照関係を整理する

### 主対象

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/shape_aware_profit_maximization_model.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/logic_multiframe_candlestick_model.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/requirements_multiframe_candlestick_model.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/profit_maximization_migration_roadmap.md`

### 検証

- baseline artifact が再参照可能であること
- 現行 spec と target spec の区別が文書上で明確であること

### Exit 条件

- 比較対象の artifact / metric / doc が固定されていること

## Phase 1. Data Contract Bridge

### 目的

新 spec が要求する入力 `x_t=[ell,b,u,d,nu,zeta]` と統計特徴 `z_t` を、現行 dataset pipeline から供給できるようにする。

### 主作業

- `dataset_service.py` に新 feature family を追加する
- 既存 close-anchor feature と新 feature を当面は併存させる
- `TrainingExample` に、profit policy で使う state feature を追加する
- 将来の `m_t` 更新に必要な sequence ordering / previous position 情報の扱いを定義する

### 主対象

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/dataset_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/domain/entities.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`

### 検証

- 新旧 feature を同時に出せる batch 生成テスト
- causal 性が崩れていないこと
- `ATR`, `EMA`, volume 正規化の NaN / zero division 耐性

### Exit 条件

- model 側が新 feature と state feature を受け取れる準備ができていること

## Phase 2. Encoder / State / Expert Heads

### 目的

`h_t`, `s_t`, `v_t`, shape expert に基づく `mu_t`, `sigma_t^2` を model 本体で出せるようにする。

### 主作業

- 現行 encoder を残しつつ、multi-scale branch の責務を明文化する
- `shape posterior head` を追加する
- `state projection` と `internal state update` の first implementation を追加する
- shape-conditioned expert head から `mu_{t,m}`, `sigma_{t,m}` を出し、混合後 `mu_t`, `sigma_t^2` を返す

### 主対象

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/model.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`

### 検証

- output tensor shape の unit test
- `sigma_t^2 >= 0` の保証
- `sum_m s_{t,m} = 1` の保証
- expert collapse や gate saturation を検知する debug metric

### Exit 条件

- 新 model が `mu`, `sigma`, `shape_probs`, `tradeability_gate`, `state_vector` を返せること

## Phase 3. Cost-Aware Policy Layer

### 目的

`post-hoc selection policy` を外し、`q_t*` を解く differentiable policy layer を入れる。

### 主作業

- `g_t = sum_m omega_m s_{t,m}` を実装する
- `mu_t_tilde = g_t * mu_t` を導入する
- 学習時は滑らかな `abs` / `projection` 近似で differentiable policy を計算する
- 推論時は piecewise な厳密 no-trade band policy を使う
- `q_{t-1}` の受け渡しを評価ループと推論パスに追加する

### 主対象

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`

### 検証

- no-trade band の unit test
- `sigma -> 0` 近傍での安定性 test
- `q_t*` の符号と `mu_t_tilde` の関係が直感に反しないこと
- turnover penalty が実際に position change を抑えること

### Exit 条件

- `selector_probability threshold` なしで `q_t*` が出せること
- `position` の主経路が新 policy layer に切り替わっていること

## Phase 4. Profit Objective Training

### 目的

train loss の主役を `log wealth - CVaR` に置き換える。

### 主作業

- `losses.py` に profit objective を追加する
- warm-start と fine-tuning の 2 段構成を入れる
- batch 内 rollout と epoch 内 sequence order の扱いを決める
- `CVaR_alpha`, `risk_aversion gamma`, `q_max`, `cost_scale` などの config を追加する

### 主対象

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/infrastructure/ml/losses.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`

### 検証

- synthetic データで loss が有限値を保つこと
- warm-start なし / ありで学習の発散性を比較すること
- `CVaR` 項の重みで過度にノートレードへ潰れないこと

### Exit 条件

- supervised loss は補助扱い、profit objective が主損失になっていること

## Phase 5. Metrics / Diagnostics / Artifact Contract

### 目的

新 objective に対して意味のある metrics / artifact へ入れ替える。

### 主作業

- `selection_precision`, `threshold_calibration_feasible` 中心の可視化から脱却する
- 新 artifact contract を定義する
- 最低限の新指標を保存する

### 最低限必要な新指標

- `average_log_wealth`
- `realized_pnl_per_anchor`
- `turnover`
- `max_drawdown`
- `cvar_alpha`
- `no_trade_band_hit_rate`
- `expert_entropy`
- `shape_gate_usage`
- `mu_calibration`
- `sigma_calibration`

### 主対象

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`

### 検証

- 新旧 artifact を比較して reviewer が誤読しないこと
- `prediction.json`, `metrics.json`, `research_report.md` の語彙が新 spec と整合すること

### Exit 条件

- reviewer / downstream consumer が threshold-policy 語彙なしで解釈できること

## Phase 6. CLI / Inference / Serving Migration

### 目的

CLI と推論出力を新 spec 中心に切り替える。

### 主作業

- `predict` / `train` / `export-diagnostics` の引数を見直す
- 旧 `selection-threshold-mode` 系は deprecated 扱いにする
- `prediction.json` と `forecast_summary.json` を `q_t*`, `mu_t`, `sigma_t^2`, `g_t`, `state_vector summary` 中心へ更新する

### 主対象

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/inference_service.py`

### 検証

- latest prediction が新 contract で JSON 出力されること
- backward compatibility をどこで切るかが明示されていること

### Exit 条件

- 日常運用の CLI が新 spec 前提で使えること

## Phase 7. Legacy Retirement

### 目的

旧 threshold-policy を比較用補助から完全撤去する。

### 主作業

- `selection_policy.json` の主経路依存を削除する
- `correctness_model`, `selector_model`, `threshold calibration` を remove または historical module へ隔離する
- 旧 test / 旧 docs / 旧 artifact example を整理する

### 主対象

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/logic_multiframe_candlestick_model.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/report_signalcascade_xauusd.md`

### 検証

- 旧語彙が主文書 / 主 artifact / 主 CLI に残っていないこと
- 新 spec 専用の smoke test が通ること

### Exit 条件

- 新 spec が実装・ドキュメント・artifact の唯一の主経路になっていること

## 5. フェーズ横断の検証戦略

各 phase で最低限次を回します。

### unit tests

- 数式の境界条件
- shape posterior の正規化
- `sigma` の非負制約
- no-trade band の場合分け
- turnover / CVaR の数値安定性

### replay diagnostics

- baseline との比較
- `h={1,3}` 先行で新 objective の安定性確認
- `h=6` 以上は短 horizon が安定してから戻す

### smoke training

- synthetic data
- 小さい CSV subset
- full research smoke

### report review

- metrics / prediction / diagnostics / report の語彙が一致しているか

## 6. 先にやるべき順序

着手順は次を推奨します。

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6
7. Phase 7

ただし実務上は、`Phase 2` と `Phase 3` の最小 slice を先に繋ぎ、`Phase 4` は warm-start 付きで入れるのが安全です。

## 7. 最小マイルストーン

完全移行までの途中で、最低限次の 3 つをマイルストーンとして置きます。

### Milestone A

- 新 feature / state contract が batch に載る
- model が `shape_probs`, `mu`, `sigma`, `tradeability_gate` を返す

### Milestone B

- `q_t*` の differentiable training path と exact inference path が両方動く
- `selector_probability threshold` なしで replay が回る

### Milestone C

- profit objective 学習 + 新 metrics/report + 旧 policy 退役まで完了する

## 8. 主なリスク

### リスク 1. sequence state の扱い

`m_t` と `q_{t-1}` は時間順序が必要なので、現在の example batching では情報の持ち方を明示しないと leakage か近似崩れを起こします。

### リスク 2. スケール不整合

`mu`, `sigma`, `cost`, `gamma`, `q_max` のスケールが揃わないと、ノートレード帯が常時発火するか、逆に常時フルポジションになります。

### リスク 3. expert collapse

shape posterior が一部 class に潰れると mixture-of-experts が意味を失います。entropy 監視が必要です。

### リスク 4. CVaR 過重

tail penalty を強くしすぎると、profit objective が trivial no-trade 解へ倒れます。

## 9. この文書の使い方

- 実装着手時は、まず対象 phase を 1 つ選ぶ
- その phase の対象ファイル、検証、Exit 条件を issue / PR に転記する
- phase 完了後は、この文書の Exit 条件を満たしたかを更新する

完全移行までは、常に

- canonical spec: `shape_aware_profit_maximization_model.md`
- current implementation logic: `logic_multiframe_candlestick_model.md`
- migration plan: `profit_maximization_migration_roadmap.md`

の 3 点セットで管理します。
