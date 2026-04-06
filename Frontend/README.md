# SignalCascade Frontend

この README は、`Frontend/` の責務を `dashboard 開発` と `artifact sync` に限定して説明します。
Vite / React の汎用テンプレート説明は残しません。

## 役割

- `PyTorch/artifacts/gold_xauusd_m30/current` を読み、`public/dashboard-data.json` を再生成する
- 生成済み payload を使って dashboard を表示する
- build 前に data contract と alias migration の最低限テストを通す

## 主なコマンド

```bash
cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend
npm run dev
npm run dev:dashboard
npm run sync:data:fast
npm run sync:data
npm run check:data:contract
npm run test:data-contract
npm run build
```

## コマンドの使い分け

- `npm run dev`
  - 通常の Vite 開発起動
- `npm run dev:dashboard`
  - `current` から `public/dashboard-data.json` を再生成してから dashboard を起動
- `npm run sync:data:fast`
  - `PyTorch` 側で accepted candidate がすでに `current` に反映済みの前提で、frontend payload だけを更新
- `npm run sync:data`
  - 必要なら `PyTorch` 側の training / tuning を含めて同期
- `npm run check:data:contract`
  - `public/dashboard-data.json` と `current` の lineage / schema 整合を検証
- `npm run test:data-contract`
  - `effectivePriceScale` canonical / `priceScale` fallback の consumer 振る舞いを検証
- `npm run build`
  - contract check、consumer test、TypeScript build、Vite build をまとめて実行

## Data contract

- generated artifact:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
- canonical key:
  - `run.effectivePriceScale`
- legacy alias:
  - `run.priceScale`
- sync script は `prediction.json` / `forecast_summary.json` の canonical field
  - `mu_t`
  - `sigma_t`
  - `sigma_t_sq`
  - `g_t`
  - `selected_policy_utility`
  - `q_t_prev`
  - `q_t_trade_delta`
  を優先し、旧 alias は fallback としてのみ扱います。

## 関連文書

- `PyTorch` 運用手順:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/README.md`
- 文書全体の整理方針:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/README.md`
- UI 実装ルール:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/UI_UX_DESIGN_RULES.md`
