# SignalCascade PyTorch Reference

`PyTorch/` 配下には、現在 5 種類の文書があります。

- canonical target spec:
  - `shape_aware_profit_maximization_model.md`
- 移行ロードマップ:
  - `profit_maximization_migration_roadmap.md`
- 実行スケジュール:
  - `profit_maximization_execution_schedule.md`
- 現行実装のロジック説明:
  - `logic_multiframe_candlestick_model.md`
- 旧来の要件定義:
  - `requirements_multiframe_candlestick_model.md`

現行コードはまだ `shape_aware_profit_maximization_model.md` へ完全移行していません。`profit_maximization_migration_roadmap.md` は完全移行までの実施順を、`profit_maximization_execution_schedule.md` は issue / PR 粒度の実行順と日付計画を、`logic_multiframe_candlestick_model.md` は移行前の reference implementation が現在どう動くかを説明する文書です。

この実装は、以下を最小構成でカバーします。

- 30分足ベースの OHLCV 生成または CSV 読み込み
- 30m / 1h / 4h / 1d / 1w の再集約
- 足形状 `Q=[u,b,l]`
- bridge input `x_t=[ell,b,u,d,nu,zeta]`
- main branch (4h / 1d / 1w) の multi-horizon return / uncertainty / shape 学習
- overlay branch (1h / 30m) の exit action 学習

## Data-State Contract Freeze

`Issue 1 / PR 1` の bridge 契約は次で固定します。

- `TimeframeFeatureRow.vector`
  - `x_t=[ell_log_return,b_real_body,u_upper_shadow,d_lower_shadow,nu_volume_anomaly,zeta_ema_deviation]`
- `TrainingExample.state_features`
  - `z_t=[session_asia,session_london,session_ny,volatility_regime_offset,regime_trend_strength,realized_volatility_30m,baseline_volatility_30m,volatility_ratio_offset_30m,anchor_volume_anomaly_30m,anchor_ema_deviation_30m]`
- `state_vector`
  - model 側では `v_t=[h_t;s_t;z_t;m_t]` を維持し、`previous_position` は dataset contract に含めず policy/evaluation loop で別途受け渡します

この freeze は「後続 PR が同じ `TrainingExample` / `config.json` / artifact metadata を前提に進める」ためのもので、Phase 2 以降の `shape posterior` や `q_t*` 主経路移行の前提になります。

## Encoder-State-Expert Alignment

`Issue 2 / PR 2` では、model の責務を次で固定します。

- `h_t`
  - main / overlay encoder を fuse した shape feature。実装上は `shape_feature`
- `s_t`
  - `shape_head` の softmax 出力。実装上は `shape_posterior`
- `z_t`
  - `TrainingExample.state_features` を hidden 次元へ写像した統計特徴。実装上は `state_projection`
- `m_t`
  - `m_t = tanh(A m_{t-1} + B[h_t; s_t; z_t] + b_m)` に対応する recurrent memory。実装上は `memory_state`
- `v_t`
  - `v_t = [h_t; s_t; z_t; m_t]`。実装上は `state_vector` と `state_vector_components`
- expert heads
  - `expert_mu_by_shape`, `expert_sigma_by_shape` が `mu_{t,m}`, `sigma_{t,m}` を返し、`shape_posterior` を混合係数として `mu_t`, `sigma_t` を構成する

互換性のため `shape_probs`, `next_state`, `expert_mu`, `expert_sigma`, `shape_predictions` の alias は残しますが、新規コードは上記の spec 名を優先して参照します。

## Policy Path And Profit Objective

`Issue 3 / PR 3` では、exact policy と differentiable policy の共通核を `build_policy_path_terms` として揃えます。

- `g_t`
  - shape posterior から得る tradeability gate
- `mu_t_tilde`
  - `mu_t_tilde = g_t * mu_t`
- `sigma_sq`
  - policy で使う分散項
- `margin`
  - `margin = mu_t_tilde - gamma * sigma_sq * q_{t-1}`

exact path はこの shared terms から `q_t*` を piecewise に解き、smooth path は同じ shared terms を softplus / softmax 近似へ流します。学習 summary では `profit_objective_log_wealth_minus_cvar` を primary objective として明示し、return / shape loss は auxiliary として扱います。

## Diagnostics Contract

`Issue 4 / PR 4` では diagnostics artifact に次を追加します。

- `validation_rows.csv`
  - `mu_t`, `sigma_t`, `sigma_t_sq`, `g_t`, `mu_t_tilde`, `q_t_candidate`
- `policy_summary.csv`
  - `selected_g_t`, `selected_mu_t`, `selected_sigma_t_sq`, `selected_mu_t_tilde`, `q_t`
- `horizon_diag.csv`
  - `policy_horizon_share`
- `validation_summary.json`
  - `state_vector_summary`
  - `g_t_mean`, `policy_utility_mean`, `mu_t_calibration`, `sigma_t_calibration`

旧 `selection_rate` や `policy_score` 系のフィールドは当面 alias として残しますが、diagnostics の主語は `q_t*`, `mu_t`, `sigma_t^2`, `g_t`, `state_vector summary` へ寄せます。

## Forecast Contract

`Issue 5 / PR 5` では `prediction.json` と `forecast_summary.json` に次を追加します。

- `g_t`
- `selected_policy_utility`
- `q_t_prev`, `q_t`, `q_t_trade_delta`
- `mu_t`, `sigma_t`, `sigma_t_sq`
- `shape_posterior`

`research_report.md` もこれらの canonical 名を優先して参照し、旧 `tradeability_gate` / `policy_score` / `expected_log_returns` / `uncertainties` は compatibility alias として扱います。

## クイックスタート

```bash
cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
signal-cascade train --output-dir artifacts/demo
```

学習後、以下が生成されます。

- `artifacts/demo/model.pt`
- `artifacts/demo/config.json`
- `artifacts/demo/source.json`
- `artifacts/demo/metrics.json`
- `artifacts/demo/prediction.json`

追加推論:

```bash
source .venv/bin/activate
signal-cascade predict --output-dir artifacts/demo
```

## Local `current` promotion

accepted な `training_run` を local alias として `artifacts/gold_xauusd_m30/current` に昇格させるときは、partial overwrite をせず whole-directory replacement で更新します。

```bash
cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch
source .venv/bin/activate
signal-cascade promote-current \
  --artifact-root artifacts/gold_xauusd_m30 \
  --source-artifact-dir artifacts/gold_xauusd_m30/reruns/v2_parent_20260406T002338_JST
```

この command の運用上の前提は次です。

- `current` は新しい artifact ではなく accepted artifact の local alias view
- `current` は whole-directory replacement でのみ更新する
- `current` 配下を in-place mutate しない
- `artifacts/` は Git 非追跡なので、shared / upstream に渡すのは artifact bytes ではなく code / docs / exported payload

dashboard の再生成は promotion 後に行います。

```bash
cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend
npm run sync:data:fast
```

## CSV 入力

CSV を使う場合は 30 分足相当のデータを用意してください。

```csv
timestamp,open,high,low,close,volume
2024-01-01T00:30:00+00:00,100.0,101.0,99.5,100.7,1200
```

実行例:

```bash
signal-cascade train --csv /absolute/path/to/ohlcv_30m.csv --output-dir artifacts/from_csv
```

## 構成

```text
PyTorch/
├── README.md
├── pyproject.toml
├── requirements.txt
└── src/signal_cascade_pytorch
    ├── application
    │   ├── config.py
    │   ├── dataset_service.py
    │   ├── inference_service.py
    │   ├── ports.py
    │   └── training_service.py
    ├── domain
    │   ├── candlestick.py
    │   ├── close_anchor.py
    │   ├── entities.py
    │   └── timeframes.py
    ├── infrastructure
    │   ├── data
    │   │   ├── csv_source.py
    │   │   └── synthetic_source.py
    │   ├── ml
    │   │   ├── losses.py
    │   │   └── model.py
    │   └── persistence.py
    ├── bootstrap.py
    └── interfaces
        └── cli.py
```

## Clean Architecture の切り分け

- `domain`: 数式・OHLC 変換・時間足の純粋ロジック
- `application`: データセット構築、学習、推論のユースケース
- `infrastructure`: PyTorch モデル、データソース、永続化
- `interfaces`: CLI

## 実装メモ

- ドメイン層は `torch` に依存しません。
- 学習用データがなくても動作確認できるよう、synthetic data source を同梱しています。
- canonical spec の full 実装ではなく、移行前ロジックの中核を安全に実験できる reference 実装です。
- `1h / 30m` overlay の教師ラベルは、次の 4h 区間における符号付き path return と実現ボラティリティから生成しています。
