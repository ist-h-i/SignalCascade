# SignalCascade Review Handoff

最終更新: 2026-04-06 01:45:51 JST

## 0. Reviewer Outcome

- reviewer judgement は `shared-tracked completion`。`shared artifact-byte completion` ではないが、この cycle は shared / upstream 観点でも閉じてよい
- `1497a93` は artifact provenance commit、`0589e67` は exporter / promotion code commit として分離して扱えば十分で、追加 metadata は不要
- `current` の local-only alias semantics は `signal-cascade promote-current` と README で十分に codify されている
- `best_params.json` は non-blocking cleanup、overlay rerun も不要扱いのままでよい

このファイルは reviewer に渡した evidence snapshot と、その reviewer judgement を残す review trail として保持する。以下の本文は、judgement 対象になった handoff 内容をそのまま残している。

## 1. 依頼時点の質問

あなたは、時系列予測・Quant ML・stateful sequence modeling・risk-aware policy optimization・artifact provenance 設計に強い reviewer です。  
`/Users/inouehiroshi/Documents/GitHub/SignalCascade` の current code、push 済みの tracked evidence、local-only artifact state、historical replay evidence を突き合わせて、**この cycle が shared / upstream 観点でも実質閉じたか**を判断してください。

今回は一般論ではなく、次の 4 点を決めてほしいです。

- `0589e67 Codify current promotion and dashboard sync` が `origin/main` に入った今、**この cycle を local completion ではなく shared-tracked completion と見てよいか**
- current code SoT は `0589e67` だが、accepted parent / `current` / `dashboard-data.json` の provenance は `gitCommitSha=1497a93...` を指しています。これは **artifact provenance commit** と **export / promotion code commit** の分離として十分か、それとも追加 metadata が要るか
- `signal-cascade promote-current` と README で `current` の alias semantics は codify されました。これで **local-only alias view** の説明は足りているか、それとも tracked manifest / doc をまだ足すべきか
- `best_params.json` は stale sidecar のままです。これは **non-blocking cleanup** と見てよいか。overlay rerun も引き続き不要でよいか

## 2. ゴール

- `origin/main` まで反映済みの tracked evidence を踏まえて、review cycle が実質完了したかを判断する
- `artifact provenance commit` と `dashboard exporter / promotion code commit` の責務分離が十分かを判断する
- もし残タスクがあるなら、それを rerun ではなく cleanup / doc / metadata decision のどこに置くべきかを決める

## 3. 現状の重要観測

### Confirmed facts

- workspace: `/Users/inouehiroshi/Documents/GitHub/SignalCascade`
- branch: `main`
- `git rev-parse --is-inside-work-tree` は `true`
- handoff 更新時点の `git status --short` は dirty だが、dirty file は `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/reviewer_submission.md` のみ
- `git log main..HEAD --oneline` は空
- `git log origin/main..HEAD --oneline` は空
- `HEAD == origin/main == 0589e671e956d6649af9df9e6c97c53bbe6d4745`
- `git log --oneline -5` は次
  - `0589e67 Codify current promotion and dashboard sync`
  - `1497a93 Add training run manifests`
  - `7467da0 Update reviewer handoff after rerun gate review`
  - `395e394 Add artifact provenance overlays and dashboard diagnostics`
  - `7ca01ad Advance profit policy diagnostics and reset semantics`
- `rg -n "TODO|FIXME|QUESTION|TBD|UNSURE" PyTorch Frontend` は、この handoff 自身以外の実コード / 実 artifact 側 hit なし
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current` は `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/.gitignore` により Git 非追跡

### code SoT

- current committed code SoT は `0589e67`
- `0589e67` で入った tracked change
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
    - `_promote_training_run_to_current()` を追加
    - `training_run` だけを `current` に whole-directory replacement で昇格
    - `current` source dir と同一 path、非 `training_run` source を reject
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
    - `signal-cascade promote-current` subcommand を追加
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py`
    - `promote-current` の success / reject を追加
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
    - `sourcePath` / `sourceOriginPath` / `run.modelDirectory` を repo-relative path に正規化
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/README.md`
    - `current` は local alias view であり、whole-directory replacement でのみ更新する、と明記
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/README.md`
    - dashboard 開発導線を追加
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/package.json`
    - `sync:data:fast`
    - `dev:dashboard`
- `1497a93` で入った `training_run manifest` 実装はそのまま `main` に入っている
- `395e394` の P0 contract 本体も維持されている
  - provenance / lineage / hash / git provenance
  - `POLICY_SELECTION_RULE_VERSION=2`
  - deterministic `selected_row_key`
  - `policy_calibration_rows_sha256`

### 実行済み検証

```bash
python3 -m py_compile \
  /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py \
  /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py \
  /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py
```

- 結果: success

```bash
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.test_artifact_schema \
  tests.test_policy_sweep \
  tests.test_stateful_evaluation \
  tests.test_policy_training
```

- workdir: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch`
- 結果: `Ran 24 tests in 0.919s` / `OK`

```bash
cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch
PYTHONPATH=src .venv/bin/python -m signal_cascade_pytorch.interfaces.cli promote-current \
  --artifact-root /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30 \
  --source-artifact-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260406T002338_JST
```

- 結果: success
- 出力要点
  - `current run dir: /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`
  - `artifact kind: training_run`
  - `artifact id: 4aebf56ac65444244b7d662d8df003641ed6de4858687a62ce3365a593ef28c9`
  - `git commit sha: 1497a9316538c893bb42b03be57164d5cd15d185`

```bash
cd /Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend
npm run sync:data:fast
npm run build
```

- `sync:data:fast`: success
- `build`: success
- 備考: Vite の chunk size warning は残るが build failure ではない

### accepted parent training_run

- frozen snapshot
  - file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/frozen_snapshots/xauusd_m30_20260405T185835_JST.csv`
  - shell SHA-256:
    - `3726a7f5775d1c1d907db86b8ee0f308823a3cb466aa7b235f707964df5b442b`

- accepted parent rerun dir
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260406T002338_JST`

- accepted parent の key facts
  - `config.json.config_schema_version=2`
  - `config.json.training_state_reset_mode="carry_on"`
  - `config.json.evaluation_state_reset_mode="carry_on"`
  - `source.json.artifact_schema_version=2`
  - `source.json.artifact_kind="training_run"`
  - `source.json.artifact_id="4aebf56ac65444244b7d662d8df003641ed6de4858687a62ce3365a593ef28c9"`
  - `source.json.parent_artifact_id=null`
  - `source.json.data_snapshot_sha256="b5d7a5dd5dde3780483e738016dffe77ed6759a7818fa2e3259a22f6a2b76dd3"`
  - `source.json.git.git_commit_sha="1497a9316538c893bb42b03be57164d5cd15d185"`
  - `source.json.git.git_dirty=false`
  - `manifest.json.schema_version=1`
  - `manifest.json.artifact_kind="training_run"`
  - `manifest.json.artifact_id="4aebf56ac65444244b7d662d8df003641ed6de4858687a62ce3365a593ef28c9"`
  - `manifest.json.parent_artifact_id=null`
  - `validation_summary.json.primary_state_reset_mode="carry_on"`
  - `validation_summary.json.policy_calibration_summary.selection_rule_version=2`
  - `validation_summary.json.policy_calibration_summary.selected_row_key="state_reset_mode=carry_on|cost_multiplier=1|gamma_multiplier=1|min_policy_sigma=0.0001"`
  - `validation_summary.json.validation.state_reset_boundary_spec_version=1`
  - artifact-local `data_snapshot.csv` の shell SHA-256 は
    - `b5d7a5dd5dde3780483e738016dffe77ed6759a7818fa2e3259a22f6a2b76dd3`
  - これは `source.json.data_snapshot_sha256` と一致

### `current` promotion 後の local artifact state

- `current` path
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`
- `current` は `signal-cascade promote-current` で whole-directory replacement された
- `current` の key facts
  - `source.json.artifact_kind="training_run"`
  - `source.json.artifact_id="4aebf56ac65444244b7d662d8df003641ed6de4858687a62ce3365a593ef28c9"`
  - `source.json.parent_artifact_id=null`
  - `source.json.data_snapshot_sha256="b5d7a5dd5dde3780483e738016dffe77ed6759a7818fa2e3259a22f6a2b76dd3"`
  - `source.json.git.git_commit_sha="1497a9316538c893bb42b03be57164d5cd15d185"`
  - `source.json.git.git_dirty=false`
  - `manifest.json.schema_version=1`
  - `validation_summary.json.primary_state_reset_mode="carry_on"`
- local filesystem 上では、同一 `artifact_id` を持つ path が 2 つある
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260406T002338_JST`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current`
- ただし `current` は Git 非追跡の local alias view であり、tracked artifact ではない

### 再生成済み `dashboard-data.json`

- file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
- `npm run sync:data:fast` 実行後の key facts
  - `schemaVersion=5`
  - `generatedAt="2026-04-05T16:22:14.080Z"`
  - `provenance.artifactKind="training_run"`
  - `provenance.artifactId="4aebf56ac65444244b7d662d8df003641ed6de4858687a62ce3365a593ef28c9"`
  - `provenance.parentArtifactId=null`
  - `provenance.dataSnapshotSha256="b5d7a5dd5dde3780483e738016dffe77ed6759a7818fa2e3259a22f6a2b76dd3"`
  - `provenance.sourcePath="PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260406T002338_JST/data_snapshot.csv"`
  - `provenance.sourceOriginPath="PyTorch/artifacts/gold_xauusd_m30/frozen_snapshots/xauusd_m30_20260405T185835_JST.csv"`
  - `provenance.gitCommitSha="1497a9316538c893bb42b03be57164d5cd15d185"`
  - `provenance.gitDirty=false`
  - `provenance.configOrigin="explicit_v2"`
  - `run.stateResetMode="carry_on"`
  - `run.modelDirectory="PyTorch/artifacts/gold_xauusd_m30/current"`
- つまり tracked dashboard payload は repo-relative path 化済みで、pre-promotion の `provenance.*=null` 状態は superseded されている

### historical fixed-point evidence

- clean parent
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260405T190223_JST`
- clean overlay
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260405T190223_JST_diagnostic_replay_overlay`
- commit: `7467da0`
- fixed-point comparison
  - parent `carry_on`
    - `average_log_wealth=-0.0005547871034936306`
    - `turnover=17.0`
    - `cvar_tail_loss=0.02641540765762329`
  - child `reset_each_session_or_window`
    - `average_log_wealth=-0.004066163070017062`
    - `turnover=24.0`
    - `cvar_tail_loss=0.033933743834495544`
- 同一 frozen snapshot・同一 fixed-point 条件で `carry_on` が 3 軸 strict domination
- `0589e67` / `1497a93` では overlay rerun をしていない

### stale sidecar

- file: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/best_params.json`
- current values
  - `updated_at="2026-03-27T04:18:04.395085+00:00"`
  - `source_path="/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/live/xauusd_m30_latest.csv"`
- ただし `parameters` 自体は current tunables と一致
  - `epochs=6`
  - `batch_size=16`
  - `learning_rate=0.0008`
  - `hidden_dim=48`
  - `dropout=0.1`
  - `weight_decay=0.0001`
- `_load_parameter_seed()` は `best_params.json.parameters` だけを seed に使う
- 現時点で挙動差は観測していない

### Hypotheses

- local authority gate は既に閉じており、tracked evidence も `0589e67` まで `origin/main` に反映済みなので、残論点は **contract gap** ではなく **publication / cleanup decision** にある
- `provenance.gitCommitSha=1497a93` は artifact bytes を作った commit、`0589e67` は promotion / dashboard sync の tracked code commit であり、層の違う provenance として共存していてもよい可能性が高い
- `current` の directory-copy alias は、`promote-current` CLI と README で whole-directory replacement only を明示したので、運用上は十分かもしれない
- `best_params.json` は cleanup 候補ではあるが、いまの cycle を reopen する blocker ではない可能性が高い

## 4. レビューしてほしい論点

### 優先度サマリー

- `P0`: shared / upstream 観点でもこの cycle を完了扱いしてよいか
- `P1`: `1497a93` artifact provenance と `0589e67` tracked code SoT の分離は十分か
- `P1`: `current` alias semantics は `promote-current` + README で足りているか
- `P2`: `best_params.json` cleanup と overlay rerun の優先度

### [P0] 論点 1: push 後の今、cycle は shared-tracked completion と見てよいか

- 現在そろっているもの
  - `origin/main` 上の code SoT: `0589e67`
  - `1497a93` 由来の accepted parent `training_run`
  - local `current` 昇格済み
  - tracked `dashboard-data.json` 再生成済み
  - `carry_on` strict domination の clean fixed-point evidence
- reviewer に確認したい点
  - ここで cycle を完了扱いしてよいか
  - まだ blocker があるなら、それは rerun ではなく metadata / doc / cleanup のどれか

### [P1] 論点 2: artifact provenance commit と exporter / promotion commit の分離は十分か

- 現状
  - accepted artifact / `current` / `dashboard` の provenance は `gitCommitSha=1497a93...`
  - しかし shared code SoT と tracked dashboard exporter code は `0589e67`
- reviewer に確認したい点
  - これは「artifact bytes を作った commit」と「tracked sync/promotion code commit」の自然な分離として十分か
  - それとも `dashboard-data.json` か doc に、exporter commit / promotion procedure version のような追加 field を持たせるべきか

### [P1] 論点 3: `current` alias semantics は今の tracked doc / command で足りているか

- 現状
  - `current` は local-only alias view
  - `signal-cascade promote-current` で whole-directory replacement する
  - `current` と rerun dir は同一 `artifact_id` を共有する
- reviewer に確認したい点
  - ここまで codify されていれば十分か
  - それとも alias manifest、pointer file、追加 README 注記などを足した方がよいか

### [P2] 論点 4: stale sidecar と overlay rerun は cleanup 優先度で扱ってよいか

- `best_params.json` は stale metadata を含むが、current tunables seed とは整合
- overlay rerun は `7467da0` 以降やっていないが、`carry_on` vs `reset_each_session_or_window` の判断を覆す新 evidence はない
- reviewer に確認したい点
  - `best_params.json` は separate cleanup / deprecation に回してよいか
  - overlay rerun は今後も不要扱いでよいか

## 5. 参照ファイル / アーティファクト

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/artifact_provenance.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_artifact_schema.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/README.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/scripts/sync-signal-cascade-data.mjs`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/README.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/package.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/Frontend/public/dashboard-data.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/.gitignore`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/frozen_snapshots/xauusd_m30_20260405T185835_JST.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260406T002338_JST/config.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260406T002338_JST/source.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260406T002338_JST/manifest.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260406T002338_JST/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/source.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/manifest.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/best_params.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260405T190223_JST/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/reruns/v2_parent_20260405T190223_JST_diagnostic_replay_overlay/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/reviewer_submission.md`

## 6. 期待する出力形式

1. 結論
2. Confirmed facts / Hypotheses の分離
3. 優先度付きの主要 findings
4. ここからの最小 next step
5. Go / No-Go 条件

## 7. 制約

- 一般論ではなく、この handoff に書かれた evidence を起点に判断する
- local artifact SoT と shared tracked evidence を混同しない
- `carry_on` vs `reset_each_session_or_window` を再度ゼロから議論しない
  - reopen するなら、`7467da0` の clean fixed-point strict domination evidence を覆す blocker がある場合に限る
- `1497a93` を artifact provenance commit、`0589e67` を tracked exporter / promotion code commit として分けて扱うかどうかを明示する
- dirty workspace はこの handoff 更新だけなので、artifact rerun cleanliness と混同しない
- cheap で information gain が高い順に提案する
- Codex がそのまま cleanup / doc / metadata decision に移れる粒度で返す
