# Documentation Guide

このリポジトリの文書は、`active docs` と `implementation task archive` に分けて管理します。
新しい文書を追加するときは、まず既存文書の責務と重複しないことを確認してください。

## Active docs

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/README.md`
  - `PyTorch` 実装の運用ガイド、契約、CLI、tuning、同期手順
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/shape_aware_profit_maximization_model.md`
  - canonical target spec
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/logic_multiframe_candlestick_model.md`
  - 現在の実装ロジック説明
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/review_followup_milestones.md`
  - dossier review 指摘に対する現行の実行タスクとマイルストーン
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/README.md`
  - dashboard 開発とデータ同期の運用手順
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/UI_UX_DESIGN_RULES.md`
  - UI/UX 実装の上位ルール

## Implementation Task Archive

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/implementation-tasks/README.md`
  - 完了済みの計画書、実装ログ、review handoff のアーカイブ方針
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/docs/implementation-tasks/archive/`
  - すでに完了した実装タスク文書の保管場所

## Rules

- `report_signalcascade_xauusd.md` のような生成物は、運用 artifact として扱い、文書ハブには含めません。
- 完了済みの計画書、review handoff、旧要件書は active docs に残しません。
- 同じ責務の文書が複数必要になった場合は、新規作成より先に統合またはアーカイブを検討します。
