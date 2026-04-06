# SignalCascade Review Handoff

最終更新: 2026-04-06 16:28:51 JST

## 1. あなたへの依頼

あなたは、時系列予測・Quant ML・artifact contract 設計・diagnostics publication 設計・schema migration に強い reviewer です。  
`/Users/inouehiroshi/Documents/GitHub/SignalCascade` の **2026-04-06 16:28 JST 時点の dirty workspace** を前提に、次の 4 点をレビューしてください。

- diagnostics publication gap は、**live `current` に関しては閉じた** と見てよいか
- live review SoT は **`/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current` 単体** に固定してよいか
- `effective_price_scale` を唯一の canonical live key とし、`price_scale` を 1 migration cycle の read-only alias に落とす方針で十分か
- `Frontend/public/dashboard-data.json` を `schemaVersion=6` / `run.effectivePriceScale` 付きに再同期し、build-time gate と consumer test を足した今、**review cycle を閉じてよいか**。もしまだ blocker があるなら 1 つに絞ってください

## 2. ゴール

- live review SoT を `current` 単体に固定してよいかを判断する
- `effective_price_scale` / `price_scale` の canonical / alias / removal 条件を確定する
- stale generated dashboard payload 問題が **解消済み** か、まだ **構造的 blocker** が残るかを切り分ける
- review cycle を止める論点を P0/P1 と P2 cleanup に分離する

## 3. 現状の重要観測

### Confirmed facts

- workspace: `/Users/inouehiroshi/Documents/GitHub/SignalCascade`
- branch: `main`
- `git rev-parse --is-inside-work-tree` は `true`
- `git log main..HEAD --oneline` は空
- handoff 更新時点の `git status --short` は dirty
- `git diff --stat` は `23 files changed, 4213 insertions(+), 1806 deletions(-)`
- `git log --oneline -5`
  - `47ee916 Align profit policy contracts and add tuning acceptance gate`
  - `d8ea766 Update bootstrap.py`
  - `2e51e76 Wip`
  - `d612f5f Add completed SignalCascade UI template`
  - `88c4687 Archive reviewer outcome for shared-tracked completion`
- `rg -n "TODO|FIXME|QUESTION|TBD|UNSURE" PyTorch Frontend` は、実コード / 実 artifact 側では open decision marker をほぼ持たない。hit は handoff 文面が中心

### dirty workspace で入っている主な contract / consumer 修正

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
  - `validation_summary.json` / `policy_summary.csv` / `horizon_diag.csv` を required 化
  - diagnostics unpublished の report 生成を fail-fast 化
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
  - accepted candidate の `current` promote 前に diagnostics publication を検証
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
  - `promote-current` でも required diagnostics を検証
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
  - `current/validation_summary.json` / `policy_summary.csv` / `horizon_diag.csv` を required 化
  - payload に `schemaVersion=6` と `run.effectivePriceScale` を出力
  - `artifactId` / `tuningSessionId` / `diagnosticsGeneratedAt` / `effectivePriceScale` の lineage assertion を追加
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/check-dashboard-data-contract.mjs`
  - `public/dashboard-data.json` が `current` の `artifact_id`、`session_id`、`validation_summary.generated_at_utc`、`effectivePriceScale` と一致することを build 前に検証
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/src/dashboard/data-contract.ts`
  - consumer 読み込み時に `effectivePriceScale` を canonical として正規化
  - `priceScale` は fallback alias に限定
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/src/dashboard/data-contract.test.ts`
  - `effectivePriceScale` 優先
  - `effectivePriceScale` 欠落時のみ `priceScale` fallback

### 実行済み検証

```bash
PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m unittest PyTorch.tests.test_artifact_schema
```

- workdir: `/Users/inouehiroshi/Documents/GitHub/SignalCascade`
- 結果: `Ran 13 tests in 0.693s` / `OK`

```bash
cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend
npm run check:data:contract && npm run test:data-contract
```

- 結果:
  - `dashboard-data contract OK: /Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
  - Node test `2 passed / 0 failed`

```bash
cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend
npm run build
```

- 結果: success
- 備考: Vite の chunk size warning は残るが build failure ではない

### 2026-04-06 16:28 JST 時点の live `current` artifact facts

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/source.json`
  - `artifact_id="9f166350117c58735ea53306a8c819f203006efb350a9269cf2f4a861522a4fd"`
  - `effective_price_scale=1.0`
  - `price_scale_origin="default"`
  - `provider_scale_confirmed=false`
  - `requested_price_scale=null`
  - `generated_at_utc="2026-04-06T07:15:09.605152+00:00"`
  - `git.git_commit_sha="47ee916cdfc551fc2edc6a89a5133fe22ff24c90"`
  - `git.git_dirty=true`
  - `git.dirty_patch_sha256="b41ae548bd0e885f4ad13b91f1a129610a0eca5e8f5595e48f6b12bcaa8e1cba"`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/manifest.json`
  - `session_id="20260406T064716Z"`
  - `generated_at="2026-04-06T07:15:09.605152+00:00"`
  - `generated_at_utc=null`
  - `effective_price_scale=1.0`
  - `price_scale_origin="default"`
  - `provider_scale_confirmed=false`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
  - `schema_version=6`
  - `generated_at_utc="2026-04-06T07:11:23.246872+00:00"`
  - `primary_state_reset_mode="carry_on"`
  - `validation.average_log_wealth=0.0010135983592947807`
  - `stateful_evaluation.carry_on.average_log_wealth=0.0010135983592947807`
  - `stateful_evaluation.reset_each_example.average_log_wealth=0.0006248691828060796`
  - `stateful_evaluation.reset_each_session_or_window.average_log_wealth=0.0007911871755678176`
  - `policy_calibration_summary.row_count=72`
  - `policy_calibration_summary.pareto_optimal_count=63`
  - `policy_calibration_summary.selection_rule_version=2`
  - `policy_calibration_summary.selected_row_key="state_reset_mode=carry_on|cost_multiplier=0.5|gamma_multiplier=0.5|min_policy_sigma=0.0001"`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current` 直下に以下が存在する
  - `validation_summary.json`
  - `policy_summary.csv`
  - `horizon_diag.csv`
  - `validation_rows.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/report_signalcascade_xauusd.md`
  - `Generated (JST): 2026-04-06T16:15:09.776892+09:00`
  - `artifact id: 9f166350117c58735ea53306a8c819f203006efb350a9269cf2f4a861522a4fd`
  - `Evaluation` section は populated
    - `carry_on average_log_wealth: 0.001014`
    - `reset_each_example average_log_wealth: 0.000625`
    - `reset_each_session_or_window average_log_wealth: 0.000791`
    - `policy sweep rows / pareto_optimal: 72 / 63`
    - `policy sweep selection basis / version: pareto_rank_then_average_log_wealth_cvar_tail_loss_turnover_row_key / 2`

### dashboard generated artifact は latest code SoT と同期済み

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
  - `schemaVersion=6`
  - `generatedAt="2026-04-06T07:25:30.092Z"`
  - `provenance.artifactId="9f166350117c58735ea53306a8c819f203006efb350a9269cf2f4a861522a4fd"`
  - `provenance.diagnosticsGeneratedAt="2026-04-06T07:11:23.246872+00:00"`
  - `provenance.gitCommitSha="47ee916cdfc551fc2edc6a89a5133fe22ff24c90"`
  - `provenance.gitDirty=true`
  - `run.anchorClose=4528.155`
  - `run.anchorCloseRaw=4528.155`
  - `run.effectivePriceScale=1`
  - `run.priceScale=1`
  - `run.tuningSessionId="20260406T064716Z"`
  - `run.selectedHorizon=2`
  - `run.executedHorizon=2`
- `npm run check:data:contract` は、上記 payload と `current/source.json` / `current/manifest.json` / `current/validation_summary.json` の一致を確認して success
- `npm run build` は、その check と `effectivePriceScale` canonicalization test を build 前に必ず通す構成

### Working hypotheses

- diagnostics publication の本体問題は、少なくとも live `current` については閉じた可能性が高い
- `current` が live review SoT、overlay が derived replay evidence という役割分離で十分な可能性が高い
- `effective_price_scale` canonical / `price_scale` one-cycle alias という方針は、frontend consumer と generated payload の両方で now enforced されている
- 残る open item は `manifest.generated_at_utc=null` と dirty provenance の扱いで、review cycle を止める P0/P1 ではなく P2 cleanup の可能性が高い

## 4. レビューしてほしい論点

### 論点 A: review cycle はもう閉じてよいか

Confirmed facts:

- live `current` に required diagnostics がある
- root report の `Evaluation` は populated
- `Frontend/public/dashboard-data.json` は `schemaVersion=6` / `run.effectivePriceScale` に更新済み
- build 前の contract gate と consumer test は通っている
- `git log main..HEAD --oneline` は空だが、workspace 自体は dirty

判断してほしいこと:

- この状態で review cycle を **closed with P2 cleanup only** と見てよいか
- まだ blocker があるなら、それは publisher、consumer、provenance のどこか

### 論点 B: live review SoT と overlay の役割分担

Confirmed facts:

- live判断に必要な summary は `current` 直下で自己完結している
- overlay を見なくても `source.json` / `manifest.json` / `validation_summary.json` / root report / dashboard payload の整合が取れている
- overlay は historical replay evidence としては有用だが、current live review の必須 path ではない

判断してほしいこと:

- live review SoT を `current` 単体に固定してよいか
- overlay を今後は **derived replay evidence** と明示するだけで足りるか

### 論点 C: `effective_price_scale` canonicalization の終了条件

Confirmed facts:

- `current/source.json` と `current/manifest.json` は `effective_price_scale=1.0`
- `Frontend/public/dashboard-data.json` は `run.effectivePriceScale=1` と `run.priceScale=1` を併記
- `Frontend/src/dashboard/data-contract.ts` は `effectivePriceScale` 優先、`priceScale` fallback only
- `Frontend/src/dashboard/data-contract.test.ts` は上記優先順位を固定している

判断してほしいこと:

- `effective_price_scale` を唯一の canonical live key、`price_scale` を one-cycle alias と断定してよいか
- alias removal の終了条件を、`generated dashboard payload` と `consumer` の双方が `effectivePriceScale` 優先を満たすことに置いてよいか

### 論点 D: 残る secondary issue の優先度

Confirmed facts:

- `current/manifest.json.generated_at_utc` は `null`
- `current/source.json.git_dirty=true`
- `current/source.json.git.dirty_patch_sha256` は記録されている

判断してほしいこと:

- `manifest.generated_at_utc` は P2 cleanup で十分か
- dirty provenance は `git_commit_sha + dirty_patch_sha256` があれば reviewer / handoff 上は十分か

## 5. 参照ファイル / アーティファクト

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/source.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/manifest.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/report_signalcascade_xauusd.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/check-dashboard-data-contract.mjs`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/src/dashboard/data-contract.ts`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/src/dashboard/data-contract.test.ts`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/src/App.tsx`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/src/dashboard/types.ts`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/package.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/reviewer_submission.md`

## 6. 期待する出力形式

次の順で返してください。

1. 結論
2. Confirmed facts と hypotheses の分離
3. review cycle を止める blocker の有無
4. 次アクションの優先順位
5. cheap で certainty が上がる実験または整備を 3 件以内
6. `ゴールまでのステップと現在地`

補足:

- `effective_price_scale` / `price_scale` の推奨方針は一文で断定してください
- review cycle を閉じてよいなら、その条件と残る P2 cleanup を分けて書いてください
- 最後の `ゴールまでのステップと現在地` は、`step N/M` 相当の現在地と immediate next action を必ず含めてください

## 7. 制約

- 根拠は **2026-04-06 16:28 JST 時点の workspace facts** に限定してください
- 15:41 JST 時点の stale `schemaVersion=5` handoff 事実は superseded と扱ってください
- 旧 `scale=100` historical evidence を、現 live current artifact の主論点に戻さないでください
- 事実と仮説を混ぜないでください
- 一般論ではなく、この handoff に書いた file / metric / artifact / command を根拠にしてください
- Codex がそのまま cleanup / doc update / follow-up test に移れる粒度で提案してください
