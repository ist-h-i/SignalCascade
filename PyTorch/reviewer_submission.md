# SignalCascade Review Handoff

最終更新: 2026-03-24 JST

## あなたへの依頼

あなたは時系列予測・Quant ML・threshold calibration・multi-stage policy 設計に強い reviewer です。
この handoff では、`SignalCascade` の既存 artifact が示したボトルネックと、2026-03-24 JST 時点で入れた未コミットの code diff を前提に、次の一手を判断してほしいです。

今回ほしいのは一般論ではなく、次の 3 点です。

- 今回の diff が cheap diagnostic lever として妥当か
- どこが設計上まだ危ういか
- retrain / replay / metrics 更新をどの順で回すべきか

## ゴール

- `accepted_signal=0` の主因切り分けで、今回の変更がどこまで情報利得を増やせるかを判断する
- `allow_no_candidate` と `selection_score_source` 周りの semantics が妥当かを sanity-check する
- 次にやる `replay diagnostics` と `smoke retrain` の優先順位を決める

## 現状の重要観測

### 1. 今回の code diff で実際に変えたこと

- 変更日は 2026-03-24 JST。branch は `main`、未コミット差分あり。
- 変更ファイルは 11 件、追加 handoff が 1 件。
- 主変更は以下です。
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
  - `allow_no_candidate: bool = False`
  - `selection_score_source: str = "selector_probability"`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
  - `train` / `predict` / `export-diagnostics` / `tune-latest` に `--allow-no-candidate` と `--selection-score-source` を追加
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
  - horizon row に `candidate`, `strict_candidate`, `selection_score` を追加
  - `allow_no_candidate=True` かつ candidate 0 本なら `selected_horizon=None`, `accept_reject_reason=no_candidate`
  - threshold 判定は `selector_probability` 固定ではなく `selection_score_source` に切替
  - 既存 policy 保存値と config が不一致なら stored threshold を無効化して `selection_threshold=None`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
  - `selected_row_count` を「anchor 数」ではなく実選択数に修正
  - `no_candidate_count`, `any_candidate_rate`, `candidate_count_per_anchor`, `selection_score` を summary / CSV に追加
  - config と policy が不一致なら threshold scan source を saved policy ではなく validation replay に切替
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
  - `selected_horizon=None` を許容
  - `selection_score_source` / `selection_score` を表示
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
  - `selected_horizon=None` を許容
  - `no_candidate_rate` を validation metrics に追加
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`
  - `no_candidate` と `selection_score_below_threshold` を追加

### 2. 今回の diff が狙っている仮説

- 確認済み事実:
  - 既存 artifact は forced-choice 前提で、validation anchor `266` に対して selected row count も `266`
  - 既存 summary では `selection_threshold_missing=266`、`selector_probability_below_threshold=0`
  - 既存 report では `accepted_signal=0`, `precision_feasible=False`, `actionable_edge_rate=0.041353383458646614`
- 作業仮説:
  - 現状は threshold が厳しいというより、forced-choice が diagnostics を汚している
  - `selector_probability` を acceptance score に固定しているため、upstream quality と selector 自体の寄与が切り分けにくい

### 3. 既存 artifact から確定している数値

対象 artifact:

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/metrics.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/research_report.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/selection_policy.json`

確定している観測:

- validation anchors: `266`
- train samples: `1063`
- `selection_precision=0.0`
- `selection_support=0`
- `precision_feasible=false`
- `coverage_at_target_precision=0.0`
- `actionable_edge_rate=0.041353383458646614`
- `alignment_rate=0.43107769423558895`
- `selected_horizon_distribution = {1: 0.6992481203007519, 3: 0.2819548872180451, 6: 0.018796992481203006}`
- report では `best_selection_lcb=0.0000`, `support_at_best_lcb=8`, `tau_at_best_lcb=0.210139`

### 4. stale / 未検証の点

- まだ fresh rerun はしていません。
- つまり、今回の code diff 後に以下は未実施です。
- `export-diagnostics --allow-no-candidate --selection-score-source ...`
- 新しい `validation_summary.json` / `threshold_scan.csv` の生成
- `h={1,3}` や `consistency_loss=0` の smoke retrain
- したがって、現時点の artifact は「変更前 semantics」のものです。

### 5. 検証済みコマンド

- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m unittest discover -s PyTorch/tests -p 'test_policy_training.py'`
  - `Ran 7 tests in 0.018s`, `OK`
- `python3 -m compileall PyTorch/src/signal_cascade_pytorch`
  - success
- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --help | rg "allow-no-candidate|selection-score-source"`
  - 新 CLI option の露出を確認済み

## レビューしてほしい論点

### 1. `allow_no_candidate` の semantics

- 確認済み事実:
  - 現実装の `candidate` は `actionable_sign != 0 && actionable_edge > 0`
  - `strict_candidate` はそれに `direction_alignment` を加えたもの
  - chooser は `allow_no_candidate=True` でも `candidate` ベースで選び、`strict_candidate` は threshold 前 gate にだけ使っています
- 聞きたいこと:
  - `no_candidate` 判定は `candidate` ベースでよいですか。それとも `strict_candidate` ベースにすべきですか。
  - `candidate` と `strict_candidate` の 2 層は reviewer 観点で理解しやすいですか。それとも naming / schema を整理すべきですか。

### 2. `selection_score_source` と threshold 互換性

- 確認済み事実:
  - `selection_score_source` は `selector_probability`, `correctness_probability`, `actionable_edge`, `edge_correctness_product`
  - 既存 policy 保存値と config が不一致のとき、stored threshold は使わず `selection_threshold=None` にしています
  - diagnostics scan はその場合 `validation_selected_rows:<score_source>` に切り替わります
- 聞きたいこと:
  - mismatch 時に threshold を無効化する安全策は妥当ですか
  - それとも `export-diagnostics` で validation predictions から threshold を自動 replay-calibrate すべきですか
  - 非 probability source で threshold clip を外した判断は妥当ですか

### 3. metrics の意味づけ

- 確認済み事実:
  - `selection_brier_score` と `selection_calibration_error` は引き続き `selector_probability` ベースです
  - 一方で accept/reject は `selection_score_source` ベースに切り替え可能です
  - `selected_horizon=None` のとき `training_service.py` では `realized_return = 0.0` として処理しています
- 聞きたいこと:
  - calibration metric を `selector_probability` に残したままで問題ないですか
  - `selection_score_source != selector_probability` のとき、acceptance metric と calibration metric を分離表示すべきですか
  - `selected_horizon=None` 時に `realized_return=0` と置くのは value metrics を甘くしませんか

### 4. stale artifact と rerun 優先順位

- 確認済み事実:
  - 現在の `research_shrink_smoke` artifact は今回の code diff 前に生成されたものです
  - したがって、今回の新 diagnostics 項目は artifact 側にまだ存在しません
- 聞きたいこと:
  - 次の最優先は replay diagnostics ですか。それとも `h={1,3}` + `consistency_loss=0` の smoke retrain ですか
  - cheap / high-information gain の観点で、まず回すべき 3 コマンドを順位付けしてください

### 5. こちらの現時点の暫定優先順位

- 仮説としては次の順です。
- 1. `export-diagnostics` replay で forced-choice 汚染を切る
- 2. `selection_score_source` を `correctness_probability` / `actionable_edge` / `edge_correctness_product` で比較する
- 3. `h={1,3}` + `consistency_loss=0` の smoke retrain に進む
- これが妥当か、別順がよいかを rank してほしいです

## 参照ファイル / アーティファクト

コード:

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/config.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/interfaces/cli.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/bootstrap.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/policy_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/diagnostics_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/training_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/report_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/application/inference_service.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/src/signal_cascade_pytorch/domain/entities.py`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/tests/test_policy_training.py`

artifact:

- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/metrics.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/validation_summary.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/selection_policy.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke/research_report.md`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/metrics.json`
- `/Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/gold_xauusd_m30/current/selection_policy.json`

実行候補コマンド:

- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --allow-no-candidate --selection-score-source selector_probability`
- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --allow-no-candidate --selection-score-source correctness_probability`
- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --allow-no-candidate --selection-score-source actionable_edge`
- `PYTHONPATH=PyTorch/src PyTorch/.venv/bin/python -m signal_cascade_pytorch.interfaces.cli export-diagnostics --output-dir /Users/inouehiroshi/Documents/GitHub/SignalCascade/PyTorch/artifacts/research_shrink_smoke --allow-no-candidate --selection-score-source edge_correctness_product`

## 期待する出力形式

次の順で返してください。

1. 結論
2. confirmed facts と working hypotheses の分離
3. 高優先度の review findings
4. cheap / high-information gain な次アクション 3 件
5. 必要なら metrics / schema / CLI の修正提案

可能なら、各 finding について以下も書いてください。

- severity
- なぜそれが risky か
- どのファイルをどう直すとよいか
- replay で足りるか、retrain が必要か

## 制約

- この handoff だけで判断できるように、live chat への依存は避けてください
- 一般論ではなく、上記の code diff と artifact 数値を根拠にしてください
- 「まずどの 3 コマンドを打つべきか」を必ず rank してください
- 数式の全面再証明は不要です。今回は semantics, metric meaning, validation freshness, next experiment ordering を重視してください
