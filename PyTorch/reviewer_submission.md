# SignalCascade Review Handoff

最終更新: 2026-03-25 JST

## 1. あなたへの依頼

あなたは時系列予測・Quant ML・threshold calibration・multi-stage policy 設計に強い reviewer です。
この handoff では、`/Users/inouehiroshi/Documents/GitHub/SignalCascade` の clean `HEAD` と、既存の replay diagnostics artifact を前提に、次の 4 点を判断してほしいです。

- 現在の `proposal / acceptance` schema split が reviewer / downstream consumer に十分明確か
- 既存 diagnostics artifact がまだ旧 schema のため、次の実験順位付けより先に freshness rerun を挟むべきか
- `accepted_signal=0` の主因が、現時点でも strict gate ではなく upstream candidate scarcity と見てよいか
- 追加で打つべきコマンドを cheap / information gain 順に 3 本 rank すると何か

一般論ではなく、このファイルに書いた commit・コード・artifact 数値・実行コマンドを根拠に答えてください。

## 2. ゴール

- `86f97b2 Clarify proposal and acceptance diagnostics semantics` の着地が妥当かを sanity-check する
- current code と既存 diagnostics artifact のズレが、次判断の blocker かどうかを決める
- freshness rerun 後に何を先に回すべきかを rank する

## 3. 現状の重要観測

### A. Confirmed facts

- Workspace は `/Users/inouehiroshi/Documents/GitHub/SignalCascade`、branch は `main` です。
- evidence 収集開始時の `git status --short` は空で、working tree は clean でした。
- `git log main..HEAD` は空で、`HEAD` に `main` 未取り込み commit はありません。
- recent commits は次です。
  - `86f97b2 Clarify proposal and acceptance diagnostics semantics`
  - `b34cce2 Add no-candidate policy diagnostics and review handoff`
  - `b114890 Add review diagnostics and formula regression tests`
  - `29545c1 Refine PyTorch training metrics and threshold calibration`
  - `8ac505d Add research report generation for training artifacts`
- `git show --stat --summary --oneline HEAD` では、今回の主変更は次です。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
    - summary を `proposed_row_count` / `accepted_row_count` 主体へ寄せる
    - run-level `threshold_status`, `stored_threshold_compatibility`, `selection_threshold_mode_requested/resolved` を追加
    - `threshold_scan.csv` を `accepted_count_at_tau`, `proposal_coverage`, `anchor_coverage` へ寄せる実装を追加
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
    - `threshold_status` を `ready | missing | infeasible` ベースへ整理
    - `stored_threshold_compatibility` を分離
    - `selection_thresholds.origin=validation_replay` を維持
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/domain/entities.py`
    - `PredictionResult.selected_horizon` を field ではなく read-side property alias に変更
    - writer 側の `prediction.json` / `forecast_summary.json` から `selected_horizon` を落とせる状態
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
    - `proposal_count`, `accepted_count`, `proposal_coverage`, `acceptance_precision`, `acceptance_coverage`
    - `value_per_anchor`, `value_per_proposed`, `value_per_accepted`
    - `selector_head_brier_score`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
    - report 表示を `proposal / acceptance` 命名へ寄せる
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
    - `export-diagnostics` summary に requested/resolved threshold mode を渡す
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
    - `--diagnostics-output-dir`
    - `--selection-threshold-mode {auto,stored,replay,none}`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`
    - `selected_horizon` alias が write-side に出ないことを固定
    - `ready` / `missing` + `stored_threshold_compatibility` を固定
- current `HEAD` に対して、次の検証を実行し成功しています。
  - `PyTorch/.venv/bin/python -m compileall PyTorch/src`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m unittest PyTorch.tests.test_policy_training`
  - 結果: `Ran 10 tests ... OK`

### B. 既存 artifact から確認できる数値

#### baseline artifact

- artifact: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/validation_summary.json`
  - `selected_row_count=266`
  - `threshold_scan_source=validation_selected_rows`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/metrics.json`
  - `selection_precision=0.0`
  - `selection_support=0`
  - `coverage_at_target_precision=0.0`
  - `actionable_edge_rate=0.041353383458646614`
  - `best_selection_lcb=0.0`
  - `support_at_best_lcb=8.0`
  - `precision_at_best_lcb=0.0`
  - `tau_at_best_lcb=0.21013891952988573`
  - `selection_brier_score=0.2482326997130845`

#### replay diagnostics artifact

- 既存 replay artifact は次の 3 本です。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge`
- 3 本に共通して `validation_summary.json` から読める数値は次です。
  - `anchor_sample_count=266`
  - `proposed_row_count=30`
  - `selected_row_count=30`
  - `accepted_row_count=0`
  - `no_candidate_count=236`
  - `no_strict_candidate_count=236`
  - `candidate_but_no_strict_count=0`
  - `any_candidate_rate=0.11278195488721804`
  - `any_strict_candidate_rate=0.11278195488721804`
  - `threshold_origin=validation_replay`
  - `threshold_scan_source=policy_calibration:validation_replay`
  - `threshold_status_counts={"infeasible": 266}`
- source ごとの best threshold row は次です。
  - `selector_probability`
    - `tau=0.316361621015634`
    - `selected_count=7`
    - `success_count=5`
    - `precision=0.7142857142857143`
    - `lcb=0.5255503826141334`
    - `feasible=False`
    - `coverage=0.23333333333333334`
  - `correctness_probability`
    - `tau=0.8505217862026361`
    - `selected_count=1`
    - `success_count=1`
    - `precision=1.0`
    - `lcb=0.5`
    - `feasible=False`
    - `coverage=0.03333333333333333`
  - `actionable_edge`
    - `tau=0.00042977907732129095`
    - `selected_count=2`
    - `success_count=2`
    - `precision=1.0`
    - `lcb=0.6666666666666666`
    - `feasible=False`
    - `coverage=0.06666666666666667`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/horizon_diag.csv`
  - `h=1 candidate_rate=0.08270676691729323`
  - `h=1 strict_candidate_rate=0.08270676691729323`
  - `h=1 selected_rate=0.08270676691729323`
  - `h=3 candidate_rate=0.041353383458646614`
  - `h=3 strict_candidate_rate=0.041353383458646614`
  - `h=3 selected_rate=0.03007518796992481`
  - `h=6 candidate_rate=0.0`
  - `h=6 strict_candidate_rate=0.0`
  - `h=6 selected_rate=0.0`

### C. freshness gap に関する confirmed fact

- current `HEAD` code は新 schema を writer 側に実装していますが、既存 artifact はまだ旧 schema のままです。
- 具体例:
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/validation_summary.json`
    - まだ `selected_row_count` と `threshold_status_counts` を含む
    - `threshold_status`, `stored_threshold_compatibility`, `selection_threshold_mode_requested/resolved` は未反映
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/threshold_scan.csv`
    - header は `tau,selected_count,success_count,precision,lcb,feasible,coverage`
    - current code が intended writer として持つ `accepted_count_at_tau`, `success_count_at_tau`, `precision_at_tau`, `proposal_coverage`, `anchor_coverage` は未反映
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/horizon_diag.csv`
    - `selected_rate` が残っており、`proposed_rate` はない

### D. Working hypotheses

- 既存 artifact の数値だけを見る限り、現ボトルネックは strict gate ではなく upstream candidate scarcity です。
- `candidate_but_no_strict_count=0`、かつ horizon 別でも candidate / strict candidate が一致しているため、strict gate 緩和の初手価値は低い可能性があります。
- `selector_probability` / `correctness_probability` / `actionable_edge` の並べ替えだけでは support shortage は崩れていません。
- `h=6` は current evidence では dead weight に見えますが、fresh rerun 後も同じかは未確認です。
- current code が schema を直しても artifact が stale のままだと、reviewer が naming 問題と bottleneck 問題を混同する恐れがあります。
- したがって、次実験の順位付け前に「freshness rerun を 3 本入れるか」を reviewer に先に判断してもらう価値があります。

## 4. レビューしてほしい論点

### 論点 1: freshness rerun は experiment ranking の前提条件か

- current code は `proposal / acceptance` schema に進んでいますが、既存 diagnostics artifact は旧 schema のままです。
- reviewer として、まず replay diagnostics を現 `HEAD` で rerun してから議論すべきですか。
- それとも、既存 artifact の数値だけでも「candidate scarcity が主因」という判断と次実験の rough ranking は可能ですか。

### 論点 2: schema / naming はここで止めてよいか

- `PredictionResult.selected_horizon` は field ではなく property alias です。
- writer 側は `proposed_horizon` / `accepted_horizon` へ寄せています。
- この read-side alias だけ残す形で十分ですか。
- それとも `selected_*` をさらに purge して、CLI flag や legacy metric 名も追加で整理したほうがよいですか。

### 論点 3: threshold semantics の責務分離は十分か

- current code では `threshold_status` と `stored_threshold_compatibility` を分離しています。
- reviewer として、この split で十分ですか。
- あるいは `disabled | missing | infeasible` の扱い、`threshold_origin`, `selection_threshold_mode_requested/resolved`, compatibility fingerprint まで追加すべきですか。

### 論点 4: next 3 commands をどう rank するか

- 候補は次です。
  - current `HEAD` で既存 3 replay diagnostics を fresh rerun
  - `edge_correctness_product` の 4 本目 replay
  - `h={1,3}` + `consistency_loss=0` smoke retrain
  - `cost_scale=0.75`
  - `delta_scale=eta_scale=0.75`
  - `cost + delta/eta` combined sweep
- reviewer には、cheap / high-information gain 順に top 3 を rank してほしいです。
- 可能なら exact command も返してください。

### 論点 5: h6 を次 smoke から落としてよいか

- 既存 artifact では `h=6 candidate_rate=0.0`, `strict_candidate_rate=0.0`, `selected_rate=0.0` です。
- `h={1,3}` retrain を第一候補に上げてよいか、fresh rerun 後まで待つべきかを判断してください。

## 5. 参照ファイル / アーティファクト

### コード

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/tuning_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/domain/entities.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`

### 既存 artifact

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/metrics.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/selection_policy.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/research_report.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/threshold_scan.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/horizon_diag.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability/threshold_scan.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge/threshold_scan.csv`

### 実行済みコマンド

- current `HEAD` で実行済み
  - `git status --short`
  - `git diff --stat`
  - `git log --oneline -5`
  - `git show --stat --summary --oneline HEAD`
  - `git log --oneline main..HEAD`
  - `PyTorch/.venv/bin/python -m compileall PyTorch/src`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m unittest PyTorch.tests.test_policy_training`
- 既存 replay artifact に対応する過去コマンド
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability --selection-threshold-mode auto --allow-no-candidate --selection-score-source selector_probability`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability --selection-threshold-mode auto --allow-no-candidate --selection-score-source correctness_probability`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge --selection-threshold-mode auto --allow-no-candidate --selection-score-source actionable_edge`
- 未実行の candidate commands
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability_head --selection-threshold-mode auto --allow-no-candidate --selection-score-source selector_probability`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability_head --selection-threshold-mode auto --allow-no-candidate --selection-score-source correctness_probability`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge_head --selection-threshold-mode auto --allow-no-candidate --selection-score-source actionable_edge`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_edge_correctness_product_head --selection-threshold-mode auto --allow-no-candidate --selection-score-source edge_correctness_product`

## 6. 期待する出力形式

次の順で返してください。

1. 結論
2. confirmed facts と working hypotheses の分離
3. 高優先度の review findings
4. cheap / high-information gain な次アクション 3 件
5. 必要なら metrics / schema / CLI の修正提案

各 finding では、可能なら次も含めてください。

- severity
- risky な理由
- どのファイルをどう直すとよいか
- replay で足りるか、fresh rerun / retrain が必要か

## 7. 制約

- live chat transcript には依存せず、この handoff だけで判断してください
- current `HEAD` code と既存 artifact のズレを明示的に考慮してください
- 確認済みの事実と仮説を分けて書いてください
- 「まず打つ 3 コマンド」を必ず rank してください
- 数式の全面再証明は不要です。今回は schema semantics、validation freshness、threshold provenance、next experiment ordering を重視してください
