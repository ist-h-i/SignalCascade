# SignalCascade Review Handoff

最終更新: 2026-03-25 JST

## 1. あなたへの依頼

あなたは時系列予測・Quant ML・threshold calibration・multi-stage policy 設計に強い reviewer です。
この handoff では、`SignalCascade` の現行 workspace と 2026-03-25 JST に freshly rerun した replay diagnostics を前提に、次の 3 点を判断してほしいです。

- 現在の dirty diff が cheap diagnostic lever として十分か
- `proposed_horizon` / `accepted_horizon` / `selected_horizon` / threshold provenance の設計がまだ危ういか
- replay の次に回すべき 3 実験を cheap / information gain 順に rank すると何か

一般論ではなく、このファイルに書いた code diff・artifact・metrics・commands を根拠に答えてください。

## 2. ゴール

- stale artifact 上の `accepted_signal=0` が、現在はどこまで分解できたかを確認する
- current bottleneck が `strict gate` ではなく upstream candidate scarcity なのかを sanity-check する
- replay のあとに `edge_correctness_product` 追加 replay、`h={1,3}` + `consistency_loss=0` smoke retrain、`cost / delta / eta` sweep のどれを先に回すべきかを決める

## 3. 現状の重要観測

### A. Confirmed facts

- Workspace は `/Users/inouehiroshi/Documents/GitHub/SignalCascade`、branch は `main` です。
- `git log main..HEAD` は空で、未取り込み commit はありません。
- 現在の working tree は dirty で、未コミット変更は 10 ファイルです。
- dirty diff は主に次を追加しています。
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
    - `build_replay_selection_policy()`
    - `proposed_horizon`, `accepted_horizon`
    - `threshold_status`, `threshold_origin`, `threshold_score_source`
    - stored policy 非互換時の replay-calibrated threshold 利用
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
    - `no_strict_candidate_count`
    - `candidate_but_no_strict_count`
    - `threshold_status_counts`
    - `build_validation_snapshots()`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
    - `--diagnostics-output-dir`
    - `--selection-threshold-mode {auto,stored,replay,none}`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
    - `export-diagnostics` が artifact dir と diagnostics dir を分離
    - `auto` / `replay` 時に validation replay-calibration を使える
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
    - `acceptance_precision`
    - `value_per_anchor`, `value_per_proposed`
    - `selector_head_brier_score`
    - `acceptance_score_source`
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
    - forecast に `proposed_horizon`, `accepted_horizon`, `threshold_status`, `threshold_origin` を表示
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/domain/entities.py`
    - prediction schema に stage/provenance 項目を追加
  - `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`
    - `config_mismatch`
    - `validation_replay` origin
    - `no_candidate`
    - `selection_score_below_threshold`
- 検証として次を実行済みです。
  - `python3 -m compileall PyTorch/src/signal_cascade_pytorch`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m unittest discover -s PyTorch/tests -p 'test_policy_training.py'`
- stale artifact で確認済みの旧症状は次です。
  - artifact: `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke`
  - validation anchors: `266`
  - old `selected_row_count=266`
  - `accepted_signal=0`
  - `selection_support=0`
  - `selection_threshold_missing=266`
  - `selector_probability_below_threshold=0`
  - `actionable_edge_rate=0.041353383458646614`
- 2026-03-25 JST に、以下 3 本の replay diagnostics を fresh 実行しました。
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability --selection-threshold-mode auto --allow-no-candidate --selection-score-source selector_probability`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability --selection-threshold-mode auto --allow-no-candidate --selection-score-source correctness_probability`
  - `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --diagnostics-output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge --selection-threshold-mode auto --allow-no-candidate --selection-score-source actionable_edge`
- 3 本の replay で共通して観測された数値は次です。
  - `proposed_row_count=30`
  - `selected_row_count=30`
  - `accepted_row_count=0`
  - `no_candidate_count=236`
  - `no_strict_candidate_count=236`
  - `candidate_but_no_strict_count=0`
  - `any_candidate_rate=0.11278195488721804`
  - `any_strict_candidate_rate=0.11278195488721804`
  - `threshold_scan_source=policy_calibration:validation_replay`
  - `threshold_origin=validation_replay`
  - `threshold_status_counts={"infeasible": 266}`
- つまり、fresh replay 後は `selection_threshold_missing` ではなく `validation_replay` provenance 付きの `infeasible` に置き換わり、candidate funnel は見えるようになりました。
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
- selector baseline の horizon diagnostics では次が見えています。
  - `h=1 candidate_rate=0.08270676691729323`
  - `h=3 candidate_rate=0.041353383458646614`
  - `h=6 candidate_rate=0.0`
  - `h=1 strict_candidate_rate=0.08270676691729323`
  - `h=3 strict_candidate_rate=0.041353383458646614`
  - `h=6 strict_candidate_rate=0.0`
  - `h=6 selected_rate=0.0`

### B. Working hypotheses

- `accepted_signal=0` の一番大きい理由は、少なくとも現在の replay 上では threshold semantics そのものより `candidate scarcity` です。
- `strict gate` は今の replay ではほぼ差分を作っていません。`candidate_but_no_strict_count=0` なので、`strict` を緩めても初手の情報利得は低い可能性があります。
- `selection_score_source` を変えると rank quality は少し変わりますが、support 不足は崩れていません。`actionable_edge` は best `lcb` が一番高い一方、coverage は still thin です。
- `h=6` は現状 dead weight である可能性が高いです。ただしこれが model/loss の問題か label/cost 設計の問題かは replay だけでは切れません。
- stage 分離は前進しましたが、`selected_horizon` を `proposed_horizon` の backward-compatible alias として残している点は、なお reviewer に sanity-check してほしいです。
- `realized_return=0` を abstain に入れる設計は `per-anchor` utility では自然ですが、`per-proposed` / `per-accepted` 系の読み方が十分に分離できているかは未確信です。

## 4. レビューしてほしい論点

### 論点 1: stage separation と naming / schema

- `proposed_horizon` / `accepted_horizon` を入れましたが、`selected_horizon` を alias のまま残しています。
- この構成で reviewer 視点に十分明確ですか。
- それとも `selected_horizon` を完全に廃止し、`proposal` と `execution` の 2 段だけにしたほうが事故が減りますか。
- もし rename するなら、どのファイルから先に直すべきかも示してください。

### 論点 2: threshold provenance と `selection-threshold-mode`

- `predict` / live path は stored-policy fail-closed のままです。
- `export-diagnostics` は `--selection-threshold-mode auto` で、config mismatch なら validation replay-calibration に落とします。
- この split は妥当ですか。
- さらに `threshold_status` を `stored_compatible | config_mismatch | missing | infeasible | disabled` のように増やすべきですか。
- `threshold_origin` / `threshold_score_source` / compatibility hash のどこまで必要かも判断してほしいです。

### 論点 3: metric meaning の分離

- いまは `selector_head_brier_score` と `acceptance_precision` を分けました。
- ただし `selection_precision` など旧名称も残っており、report / metrics schema が二重化しています。
- `return_per_anchor` / `return_per_proposed` / `return_per_accepted` の粒度まで強制的に寄せるべきか、現状で十分かを見てください。
- non-probability score source に対して calibration metric をどう扱うべきかも確認したいです。

### 論点 4: 次の実験順序

- fresh replay 後の現状では、`strict gate` より `candidate scarcity` が支配的に見えます。
- この前提で、次の候補を cheap / information gain 順に rank してください。
  - `edge_correctness_product` を 4 本目の replay に追加
  - `h={1,3}` + `consistency_loss=0` の smoke retrain
  - `cost_scale=0.75`
  - `delta_scale=eta_scale=0.75`
  - `cost + delta/eta` の combined sweep
- 可能なら「まず打つ 3 コマンド」を exact command level で返してください。

### 論点 5: h6 の扱い

- replay 上では `h=6 candidate_rate=0.0` です。
- この時点で `h=6` を切る判断は早すぎますか。
- それとも `h={1,3}` retrain はもはや cheap enough な第一候補ですか。

## 5. 参照ファイル / アーティファクト

コード:

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

baseline artifact:

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/selection_policy.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/research_report.md`

fresh replay artifacts:

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/threshold_scan.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_selector_probability/horizon_diag.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_correctness_probability/threshold_scan.csv`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke_diag_actionable_edge/threshold_scan.csv`

## 6. 期待する出力形式

次の順で返してください。

1. 結論
2. confirmed facts と working hypotheses の分離
3. 高優先度の review findings
4. cheap / high-information gain な次アクション 3 件
5. 必要なら metrics / schema / CLI の修正提案

各 finding について、可能なら次も含めてください。

- severity
- risky な理由
- どのファイルをどう直すとよいか
- replay で足りるか、retrain が必要か

## 7. 制約

- live chat transcript には依存せず、この handoff だけで判断してください
- 一般論ではなく、上記の code diff と artifact 数値を根拠にしてください
- 「まずどの 3 コマンドを打つべきか」を必ず rank してください
- 数式の全面再証明は不要です。今回は semantics、metric meaning、validation freshness、next experiment ordering を重視してください
