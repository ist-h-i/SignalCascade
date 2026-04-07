# SignalCascade PyTorch Reference

この README は、`PyTorch/` 配下の運用ガイド兼文書ハブです。

## Active docs

- `shape_aware_profit_maximization_model.md`
  - canonical target spec
- `logic_multiframe_candlestick_model.md`
  - 現在の実装ロジック説明
- `README.md`
  - 契約、CLI、tuning、promotion、frontend sync の運用手順

## Archived task docs

- 完了済みの migration plan、execution schedule、historical requirements、review handoff は
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/implementation-tasks/archive/`
  に移動しています。
- active docs の Source of Truth は上の 3 本だけです。

## Current Alias SoT

`artifacts/gold_xauusd_m30/current` は local alias view であり、新しい artifact kind ではありません。運用上の authoritative SoT は次で固定します。

- applied runtime config
  - `artifacts/gold_xauusd_m30/current/config.json`
- current forecast / policy driver / lineage
  - `artifacts/gold_xauusd_m30/current/prediction.json`
  - `artifacts/gold_xauusd_m30/current/forecast_summary.json`
  - `artifacts/gold_xauusd_m30/current/source.json`
  - `artifacts/gold_xauusd_m30/current/research_report.md`
- top-level report
  - `report_signalcascade_xauusd.md` は `current/research_report.md` の synchronized mirror
- accepted snapshot report
  - tuning session 直下の accepted candidate `research_report.md`

`validation_summary.json.policy_calibration_summary.selected_row` は diagnostics recommendation であり、applied runtime config ではありません。runtime は常に `current/config.json` を読む前提で扱います。

`accepted_candidate` と `production current` は同一とは限りません。`accepted_candidate` は blocked objective と optimization gate の勝者、`production current` は chart fidelity / sigma-band reliability / execution stability を加味した `user_value_score` の自動選抜結果です。

## Claim Hardening

`policy_mode=shape_aware_profit_maximization` は互換 identifier として残っていますが、2026-04-07 JST 時点の current evidence は `continuous posterior weighting` と `head coupling` までです。

- `shape_posterior_top_class_share={'1': 1.0}` が残っているため、current artifact を `shape-aware routing` や `regime-aware routing` と読まない
- `shape-aware` を再主張する条件は、blocked folds で top-class concentration が下がり、その差が frontier 改善に結び付く追加 evidence を得ること

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

## CLI And Bootstrap Migration

`Issue 6 / PR 6` では CLI と bootstrap の表示を新 contract に寄せます。

- `predict` / `train` / `export-diagnostics` は `q_t`, `g_t`, `selected policy utility` を主表示にする
- `--selection-score-source` と `--selection-threshold-mode` は deprecated compatibility option として警告のみ残す
- diagnostics replay は threshold replay ではなく `q_t` policy path の再計算として扱う

## Legacy Isolation

`Issue 7 / PR 7` では threshold-policy 時代の互換 field を main path から切り離します。

- `apply_selection_policy` の canonical 出力は `policy_horizon`, `executed_horizon`, `position`, `trade_delta`, `no_trade_band_hit`, `selected_policy_utility` を主に使う
- 旧 `accepted_signal`, `selection_probability`, `selection_threshold`, `correctness_probability` などは `domain/historical_compatibility.py` に集約し、`legacy_compatibility` view 経由でのみ参照する
- `PredictionResult` も canonical field を主とし、historical alias は `legacy_compatibility` へ隔離する

## Acceptance And Smoke

`Issue 8 / PR 8` の受け入れは次を基準にします。

- unit / integration: `tests/test_policy_training.py`, `tests/test_artifact_schema.py`, `tests/test_policy_consistency.py`, `tests/test_stateful_evaluation.py`, `tests/test_policy_sweep.py`
- replay diagnostics: `signal-cascade export-diagnostics --output-dir <artifact> --diagnostics-output-dir <overlay>`
- smoke training: `signal-cascade train --output-dir <artifact>` と `signal-cascade predict --output-dir <artifact>`
- acceptance checklist:
  - `Done 1`: primary objective は `profit_objective_log_wealth_minus_cvar`
  - `Done 2`: inference main path は `q_t*` 中心
  - `Done 3`: threshold calibration は主経路から除外
  - `Done 4`: artifact / report / diagnostics / CLI は canonical spec 優先
  - `Done 5`: unit / replay / smoke を新 contract で再確認

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

## Tuning Acceptance Gate

`signal-cascade tune-latest` は parameter sweep を回した後、`leaderboard` の 1 位を無条件で `current` に昇格しません。
`current` へ昇格するのは optimization gate を通過した candidate だけです。

- gate 指標
  - `average_log_wealth >= 0.0`
  - `realized_pnl_per_anchor >= 0.0`
  - `cvar_tail_loss <= 0.08`
  - `max_drawdown <= 0.15`
  - `directional_accuracy >= 0.50`
  - `no_trade_band_hit_rate <= 0.80`
- pass した candidate が 1 件以上ある場合:
  - blocked-first の best pass candidate を `accepted_candidate` と `best_params.json` に反映する
  - pass 済み candidate の中から `user_value_score` 最大の candidate を `production current` として `current` に反映する
- pass した candidate が 0 件の場合:
  - `current` は更新しない
  - session 配下の `leaderboard.json` / `manifest.json` に fail 理由を残す
  - CLI は non-zero exit で終了する

session ごとの gate 結果は `artifacts/gold_xauusd_m30/archive/session_<timestamp>/manifest.json` に保存されます。

accepted candidate が出た場合、`tune-latest` の時点で `artifacts/gold_xauusd_m30/current` は更新済みです。UI 反映はその後に `Frontend` 側で `dashboard-data.json` を再生成します。

```bash
cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch
source .venv/bin/activate
signal-cascade tune-latest --artifact-root artifacts/gold_xauusd_m30

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

## 関連文書

- 文書全体の整理方針:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/README.md`
- dashboard 開発手順:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/README.md`
- UI 実装ルール:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/UI_UX_DESIGN_RULES.md`

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
