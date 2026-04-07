# Review Follow-up Milestones

最終更新: 2026-04-07 JST

この文書は、dossier review で確定した指摘事項を

- すでに取り込み済みの項目
- 進行中の項目
- 未完了の項目

に分けたうえで、次に進めるべき実行順とマイルストーンを固定するための active doc です。

## 0. Status Snapshot

2026-04-07 JST 時点の整理は次です。

初回 follow-up で `T0-T7` は完了扱いまで進んだが、同日付の追加 external review で

- `accepted` と `production current` の意味の分離不足
- `current` / `report` / consumer schema の authoritative SoT 不一致
- `candidate_04` vs `candidate_05` の promote 根拠が governance override のまま未 codify
- `shape-aware` claim が current evidence より強すぎる可能性

が新たな open item として返ってきたため、`T8-T11` を追加して完了扱いを解除した。2026-04-07 JST の contract closure 実装で、これら 4 件も反映完了した。

### 取り込み済み

- 2026-04-07 JST の再監査で、`review_feedback_comment.md` の論点が `T0-T7` / 完了条件 / `logic_multiframe_candlestick_model.md` に反映済みであることを確認した
- `checkpoint` 再順位付け監査を実装し、artifact 化済み
- `checkpoint_audit` を `validation_summary.json` に常設済み
- `economic knobs` を tuning / diagnostics / runtime に通す変更を反映済み
- `economic knobs` の blocked walk-forward 集計を policy sweep / leaderboard に反映済み
- `tie_policy_to_forecast_head` / `disable_overlay_branch` の cheap ablation を実装済み
- `tie` variant の replay overlay を `current` 候補として扱えるよう整備済み
- `tune-latest` に `candidate_limit` / `quick_mode` を追加し、blocked-first smoke run を切れるようにした
- `tune-latest` seed は `current / best_params` を優先し、`policy_calibration_summary.selected_row` の暗黙上書きをやめた
- `quick_mode` は inherited economic knobs を保った structural quartet (`baseline / tie / overlay_off / both`) を先頭評価する
- `walk_forward_folds` を checkpoint 選定と training 主経路へ昇格済み
- `oof_epochs` を blocked walk-forward 指標の rolling window として主経路へ反映済み
- `overlay` は `Option A` を採用し、canonical 主経路から外す判断を docs / code contract に反映済み
- `display forecast` と `policy driver` の artifact / dashboard 分離を反映済み
- `no-trade` を増やす focused sweep を実施し、現 `current` は `cost x6 / gamma x4 / q_max 0.75 / cvar 0.2` 系へ更新済み
- blocked-first full session `session_20260407T041853Z` を完走し、`generated=22 / evaluated=22 / quick_mode=false` を確認済み
- full session の accepted は `candidate_05` で、その後 `current_selection_governance` を `user_value_score` 自動選抜へ切り替え、現 production current は `candidate_17` に更新済み
- `current/source.json` に `current_selection_governance` / `current_alias_contract` を追加し、`accepted_candidate` / `best_candidate` / `production current` と selection payload を codify 済み
- `report_signalcascade_xauusd.md` を `current/research_report.md` の synchronized mirror として固定し、top-level report と current alias の SoT を一致させた
- `accepted winner` / `production current` の paired frontier comparison を `current/research_report.md` / `report_signalcascade_xauusd.md` / `current/source.json` に固定した
- `policy_calibration_summary.selected_row` は diagnostic recommendation、`current/config.json` は applied runtime config として命名分離した
- claim hardening を docs / report に反映し、current evidence は `continuous posterior weighting` と `head coupling` までであることを明示した

### 進行中

- なし

### 未完了

- なし

## 1. 優先順位

優先順位は次で固定します。

1. `current` / `report` / `schema` の authoritative SoT を 1 つに固定する
2. `accepted` と `production current` の意味を governance rule として codify する
3. `accepted winner` と `production current` の paired frontier replay を固定し、risk budget を数値で示す
4. `shape-aware` / `regime-aware` の claim を current evidence に合わせて harden する
5. 新しい live data 向け rerun は、上の governance / contract cleanup を閉じてから再開する

## 2. やるべきことリスト

### T0. `collapse` の原因分解を blocked ablation で固定する

Status:

- 完了

目的:

- reviewer が最上位で指摘した `shape / horizon / g_t` collapse が、
  `small sample + oversized branch structure` による degeneracy なのか、
  それとも現行 market structure を捉えた genuine collapse なのかを切る

やること:

- `baseline`
- `tie_policy_to_forecast_head=true`
- `disable_overlay_branch=true`
- `tie_policy_to_forecast_head=true && disable_overlay_branch=true`
  を同一 blocked walk-forward 条件で比較する
- 各 variant で
  `policy_horizon_distribution`
  `shape_posterior_top_class_share`
  `g_t_mean`
  `blocked_objective_log_wealth_minus_lambda_cvar_mean`
  を並べる
- `sample_count=184` に対して branch complexity を維持する根拠が残るかを判断する

Done 条件:

- collapse が「複雑性を減らすと改善する」のか、
  「複雑性を残しても genuine に固定される」のかを blocked 指標で説明できる
- `logic_multiframe_candlestick_model.md` と review follow-up の両方で、
  cheap ablation の結論が一意に読める

結論:

- `baseline` は `policy_horizon_distribution[18]=1.0`、blocked objective mean `0.0101`、`g_t_mean=0.5002`
- `tie` は `policy_horizon_distribution[30]=0.6667`、blocked objective mean `0.0081`、`g_t_mean=0.5453`
- `overlay_off` は `policy_horizon_distribution[18]=1.0` のままで、blocked objective mean `0.0047` まで悪化した
- `tie + overlay_off` は `12 / 18` に分散したが、blocked objective mean `0.0047` で `tie` 単独より弱い
- 4 variant すべてで `shape_posterior_top_class_share={'1':1.0}` は変わらず、shape collapse 自体は残る
- 結論として、horizon collapse を崩す主因は `tie_policy_to_forecast_head` であり、`disable_overlay_branch` 単独は本命ではない

### T1. `tune-latest` の blocked-first 採用を本番 artifact で完走確認する

Status:

- 完了

目的:

- `candidate selection` が `single split average_log_wealth` ではなく
  `blocked_objective_log_wealth_minus_lambda_cvar_mean`
  を主基準にしていることを、実 run で確定する

やること:

- `candidate_limit` か `quick_mode` で smoke run を切り、blocked-first leaderboard の形を先に固定する
- `tune-latest` を 1 run 完走させる
- `leaderboard.json` に `blocked_average_log_wealth_mean`
  `blocked_turnover_mean`
  `blocked_objective_log_wealth_minus_lambda_cvar_mean`
  が並ぶことを確認する
- `accepted_candidate` と `best_params.json` と `current/config.json` が blocked-first の candidate を指すことを確認する

Done 条件:

- `archive/session_*` に完整な `leaderboard.json` が生成されている
- `best_candidate` と `accepted_candidate` の選定理由を blocked 指標で説明できる
- smoke run と full run の両方で candidate 並び順の規則が変わらない

現時点:

- `session_20260407T034224Z` で blocked-first sorting と `leaderboard.json` / `manifest.json` の生成自体は確認済み
- ただし first-4 candidates が structural quartet になっておらず、seed も `selected_row` で上書きされていた
- 両方を修正し、`session_20260407T040723Z` では `baseline / tie / overlay_off / both` を `cost x6 / gamma x4 / q_max 0.75` のまま smoke 評価した
- 同 session は `generated=22 / evaluated=4 / quick_mode=true` で完走し、accepted は `candidate_04` (`tie=true / overlay_off=true`)
- accepted row の blocked 指標は `objective_mean=0.000919`、`average_log_wealth_mean=0.001186`、`turnover_mean=0.2098`、`policy_horizon=30`
- full session `session_20260407T041853Z` も `generated=22 / evaluated=22 / quick_mode=false` で完走した
- full session の accepted は `candidate_05` (`tie=true / overlay_off=true / epochs=16 / learning_rate=0.000421875`) で、blocked objective mean `0.007317` を確認した
- run 完了直後は `current/config.json` も accepted candidate を指すことを確認済みで、その後 `T4` の保守 current を経て、最終的に `user_value_score` 自動選抜で production current を `candidate_17` へ更新した

### T2. `walk_forward_folds / oof_epochs` を training 主経路へ昇格する

Status:

- 完了

目的:

- diagnostics の後段 replay ではなく、training / tuning の main path 自体を `one split` から外す

やること:

- `walk_forward_folds` を candidate 評価の主ループで使う
- `oof_epochs` の役割を明文化し、使うなら主経路に入れる
- current holdout split は fallback か smoke 用へ格下げする

Done 条件:

- `train` / `tune-latest` の主経路で blocked walk-forward の fold 集計が行われる
- `carry_on` 優位や `aggressive sweep` 優位を fold 平均で比較できる

実装メモ:

- epoch 選定は blocked walk-forward 指標を優先し、`oof_epochs` は rolling mean window として扱う
- `checkpoint_audit` / `analysis.json` / `research_report.md` に blocked 指標を常設した

### T3. `overlay branch` の canonical 判断を完了する

Status:

- 完了

目的:

- 現在の「latent には入るが direct supervision が薄い」中途半端な状態を終わらせる

やること:

- 次の 2 択を明示的に選ぶ
- `Option A`: overlay を canonical から外す
- `Option B`: overlay label / head / loss を戻す
- 選んだ方に合わせて `logic_multiframe_candlestick_model.md` と code path を一致させる

決定:

- `Option A` を採用済み
- overlay は live review SoT ではなく、derived replay evidence として扱う
- 現行 canonical loss path は `return + shape + profit` のみ

Done 条件:

- docs と actual loss path が一致している
- reviewer が `overlay は使っている / 使っていない` を一意に読める

### T4. `no-trade` と turnover を production 向けにもう 1 段詰める

Status:

- 完了

目的:

- `position` の飽和と過剰売買を、`tie` で得た自由度を殺さずに抑える

やること:

- `q_max`
- `min_policy_sigma`
- `policy_gamma_multiplier`
- 必要なら `policy_cost_multiplier`
  の狭い grid を current 周辺で再探索する
- live replay を複数点で比較し、`trade_delta` と `no_trade_band_hit` を観察する

Done 条件:

- validation の `no_trade_band_hit_rate` が現水準より改善している
- live replay の `trade_delta` が current より安定して小さい
- 保守的 current 候補を経由して production current の user-value 自動選抜へ接続できている

結論:

- full session の accepted `candidate_05` は blocked objective では最良だったが、live `policy_horizon=2` まで短縮したため production current には採らない
- gate 通過候補のうち `candidate_04` は `policy_horizon_distribution[30]=0.8611`、live `policy_horizon=30`、`blocked_turnover_mean=0.2098`、`no_trade_band_hit_rate=0.4444`
- これは tie-only snapshot の `turnover=3.4635 / no_trade_band_hit_rate=0.3889` や、review で問題化された `trade_delta=0.6583` より保守的で、live `trade_delta=0.4567` まで縮んだ
- その後 chart fidelity / sigma-band reliability / execution stability を統合した `user_value_score` を導入し、`candidate_17` が `candidate_04` の `0.661377` を上回る `0.670851` を出したため、現 production current は `candidate_17` になった

### T5. `carry_on` の genuine 優位を切り分ける

Status:

- 完了

目的:

- recurrent context の優位が holdout artifact か、本当に live 有効な memory かを分ける

やること:

- blocked walk-forward で `carry_on / reset_each_session_or_window / reset_each_example` を固定比較する
- fold ごとの sign consistency を確認する
- 単一 block だけで勝つ mode は canonical から外す

Done 条件:

- `carry_on` 優位が fold 単位でも再現するか、または artifact だったと結論できる

結論:

- `archive/session_20260407T034224Z/previous_current` (`cost x6 / gamma x4 / q_max 0.75 / tie=true`) では `carry_on` の blocked objective mean が `0.0115`
- 同 snapshot で `reset_each_session_or_window` は `0.0090`、`reset_each_example` は `0.0040`
- fold sign は `carry_on=['+','+','+']`、他 2 mode はどちらも `['-','+','+']`
- objective mean と sign consistency を合わせると `carry_on` が最も安定しているため、canonical は `carry_on` を維持する

### T5b. `sample-size` に対する branch complexity の妥当性を決める

Status:

- 完了

目的:

- `train=118 / validation=36` に対して現行の `5 encoder + latent fusion + shape head + memory + expert heads`
  を canonical に維持するのか、簡素版へ寄せるのかを判断する

やること:

- `T0` の ablation 結果を `sample_count=184` の条件と一緒に読む
- cheap ablation が blocked 指標で同等以上なら、不要 branch を canonical から外す候補として扱う
- cheap ablation が悪化するなら、complexity を残す代わりに supervision / contract を明文化する

Done 条件:

- `capacity 過多が副因か主因か` を説明できる
- canonical path に残す branch と、diagnostic only に落とす branch を区別できる

結論:

- `overlay_off` 単独は blocked objective mean `0.0047` まで落ちるため、単純な branch 削減は悪化要因だった
- `tie` は blocked objective mean `0.0081` を保ちながら `18` 固定を崩して `30` 主軸へ動かした
- hardened smoke `session_20260407T040723Z` と full session `session_20260407T041853Z` の両方で、gate 通過候補の上位は `tie + overlay_off` に寄った
- full session では `tie + overlay_off` の `candidate_05 / candidate_04 / candidate_16 / candidate_17` が gate 通過上位を占め、tie-only の `candidate_02` より一貫して優位だった
- よって canonical structural choice は `tie_policy_to_forecast_head=true && disable_overlay_branch=true` とする

### T7. `g_t` scalar broadcast を維持するか、horizon-specific へ進むかを決める

Status:

- 完了

目的:

- 数学レビューで指摘された
  `g_t` scalar broadcast が horizon collapse を助長している可能性を、
  speculation ではなく設計判断として閉じる

やること:

- 現行の `scalar -> all horizon broadcast` を SoT として維持する理由を整理する
- 進めるなら `horizon-specific gate` の cheap variant と比較軸を定義する
- `sigma floor`
  `exact/smooth gap`
  `small-sample CVaR noise`
  を含めて、checkpoint / policy objective への影響を整理する

Done 条件:

- `g_t` の設計判断が docs と code contract で一意に読める
- future task として残す場合でも、なぜ今は未着手なのかを説明できる

結論:

- `g_t` scalar broadcast は当面維持する
- 根拠は、`tie_policy_to_forecast_head` だけで horizon distribution が大きく動いた一方、4 variant すべてで shape top class は `1.0` のままだったこと
- つまり現時点の第一ボトルネックは `g_t` の次元不足ではなく、head decoupling と small sample 下の branch specialization failure と読む
- `sample_count=184 / validation=36` では horizon-specific gate を追加する information gain がまだ低いため、future variant として保留する

### T6. `forecast` と `policy driver` の UI / contract を分離する

Status:

- 完了

目的:

- operator が `display forecast` をそのまま decision basis と誤読するリスクを消す

やること:

- `prediction.json` と dashboard data の表示項目を再点検する
- `forecast_mu/sigma` と `policy_mu_t/policy_sigma_t` を別ラベルで出す
- dashboard 上の文言を `display forecast` / `policy driver` に分ける

Done 条件:

- UI と artifact の両方で、`forecast` と `decision basis` の混同が起きない

実装メモ:

- `prediction.json` / `forecast_summary.json` に `display_forecast` と `policy_driver` を追加済み
- dashboard に head relationship と overlay contract を表示する形へ更新済み

### T8. `accepted` と `production current` の governance rule を codify する

Status:

- 完了

目的:

- blocked-objective winner と user-facing production current が
  別 objective で選ばれることを曖昧さなく表現する

やること:

- `accepted_candidate`
  `best_candidate`
  `production current`
  `selection payload`
  の意味を docs / artifact contract / follow-up に明示する
- `blocked_directional_accuracy_mean`
  `mu_calibration`
  `sigma_calibration`
  `blocked_exact_smooth_position_mae_mean`
  `max_drawdown`
  `blocked_turnover_mean`
  を `production current` 選定の secondary sort または acceptance gate 候補として整理する
- accepted winner と production current の差分理由を、
  paired comparison 付きで operator / reviewer が追える場所へ固定する
- rule-based selection を採る場合は、`selection_mode / selection_rule / reason / metric snapshot / timestamp` を残す metadata 形式を決める

Done 条件:

- `accepted != current` の理由が blocked objective から user-value objective へ切り替わったこととして一意に読める
- `production current` の選定が rule-based selection か explicit governance override かのどちらかに固定される

結論:

- `current/source.json` に `current_selection_governance` を追加し、`selection_mode=auto_user_value_selection` / `selection_rule=optimization_gate_then_user_value_score` / `override_priority_metrics` を保存するようにした
- `accepted_candidate=candidate_05` と `production_current=candidate_17` の paired metric snapshot を同 payload に固定し、decision summary を `chart fidelity, sigma-band reliability, and execution stability took priority over blocked-objective rank` として明文化した
- 選抜時刻は `selection_timestamp_utc`、自動選抜根拠は `decision_summary` と paired frontier metric 群で追えるようにした

### T9. `current` / `report` / `schema` の authoritative SoT を 1 つに固定する

Status:

- 完了

目的:

- producer 側の nested schema と consumer 側の旧 flat key 読みを解消し、
  `current` alias / top-level report / artifact JSON の authoritative source を揃える

やること:

- `prediction.json`
  `forecast_summary.json`
  `source.json`
  の canonical key を 1 系統に固定し、旧 flat key consumer を削除または明示 alias 化する
- `PyTorch/report_signalcascade_xauusd.md` を
  `current report`
  ではなく `accepted snapshot report` として扱うのか、
  それとも `current/research_report.md` と artifact id 一致を強制するのかを決める
- `current/research_report.md`
  top-level report
  dashboard/export consumer
  の責務分担を文書と生成コードの両方で揃える
- `config.json` の applied runtime config と
  `validation_summary.json.policy_calibration_summary.selected_row`
  の diagnostic recommendation を名前で分離する
- artifact id / report path / non-null canonical key を検証する test か CI guard を追加する

Done 条件:

- consumer が obsolete flat key を読まない
- `current` / top-level report / artifact JSON の authoritative path と役割が一意に読める
- `selected_row` が applied config と誤読されない

結論:

- `current/source.json.current_alias_contract` に authoritative paths を固定し、`current/config.json` / `prediction.json` / `forecast_summary.json` / `source.json` / `research_report.md` を canonical SoT とした
- `report_signalcascade_xauusd.md` は `current/research_report.md` の synchronized mirror として再生成するようにし、manual promote 後も SoT がずれないようにした
- `validation_summary.json.policy_calibration_summary` に `applied_runtime_policy` / `selected_row_role` / `selected_row_matches_applied_runtime` を追加し、`selected_row` を diagnostics recommendation として分離した

### T10. `accepted winner` と `production current` の paired frontier replay を固定する

Status:

- 完了

目的:

- `current` が blocked-objective winner ではなく user-facing profile を優先した判断であることを、
  retrain なしで再現可能な frontier として示す

やること:

- 既存 artifact のみを使って
  accepted winner / production current
  の paired comparison を 1 枚にまとめる
- 最低でも
  `average_log_wealth`
  `blocked_objective_log_wealth_minus_lambda_cvar_mean`
  `turnover`
  `max_drawdown`
  `directional_accuracy`
  `exact_smooth_position_mae`
  `latest trade_delta`
  を同一表に並べる
- どの指標が optimization objective で、
  どの指標が governance / execution risk budget なのかを分けて定義する
- accepted winner を current に戻すか、user-value winner を維持するかの判断条件を明文化する

Done 条件:

- accepted winner / production current の frontier comparison が 1 箇所で再現可能になる
- current promote の根拠が reviewer / operator 向けに数値で説明できる

結論:

- `current_selection_governance.paired_frontier` に accepted winner / production current の snapshot と delta を保存した
- report では `average_log_wealth` / `blocked_objective_log_wealth_minus_lambda_cvar_mean` を optimization objective、`user_value_score` / `chart_fidelity` / `sigma_band` / `execution_stability` と `policy_horizon` / `blocked_turnover_mean` / `max_drawdown` / `exact_smooth_position_mae` / `trade_delta` を production selection contract として並列表示する
- これにより `candidate_05` が blocked objective winner、`candidate_17` が user-value production current であることを current report 1 本で再現できる

### T11. `shape-aware` / `regime-aware` の claim を harden する

Status:

- 完了

目的:

- `shape_posterior_top_class_share={'1': 1.0}` の current evidence に対して、
  docs / report / dashboard が `shape-aware routing` を過剰に主張しない状態にする

やること:

- docs / report / dashboard の文言で
  `shape-aware`
  `regime-aware`
  と読める箇所を洗い出す
- 現時点で言ってよいことを
  `continuous posterior weighting`
  `head coupling`
  `shape top-class collapse remains`
  の粒度で言い換える
- それでも `shape-aware` を維持したい箇所があれば、必要な追加 evidence か cheap variant を明記する

Done 条件:

- current evidence を超える claim が docs / report / dashboard から消える
- 次に `shape-aware` を再主張するための追加 evidence 条件が明文化される

結論:

- docs と report の文言を `continuous posterior weighting` / `head coupling` / `shape top-class collapse remains` の粒度へ寄せた
- `shape_aware_profit_maximization` は互換 identifier として残す一方、current artifact を `shape-aware routing` / `regime-aware routing` と読まないことを契約化した
- 再主張条件は「blocked folds で top-class concentration が下がり、その差が frontier 改善に結び付く追加 evidence を得ること」として明文化した

## 3. マイルストーン

日付は 2026-04-07 JST 時点の目安です。

### M1. Selection Bias Closure

対象期間:

- 2026-04-07 JST から 2026-04-08 JST

含むタスク:

- `T0`
- `T1`

Exit:

- blocked-first の `tune-latest` 完走 artifact が 1 本あること
- collapse ablation の smoke run 比較が 1 本あること

### M2. Training Protocol Upgrade

対象期間:

- 2026-04-08 JST から 2026-04-10 JST

含むタスク:

- `T2`
- `T5`
- `T5b`

Exit:

- `walk_forward_folds` が main training / tuning path に入っていること
- `carry_on` 優位を fold 平均で説明できること
- branch complexity を維持するか減らすかを blocked evidence で説明できること

現在地:

- `T2` は完了
- `T5` は完了
- `T5b` も完了

### M3. Structural Contract Decision

対象期間:

- 2026-04-10 JST から 2026-04-11 JST

含むタスク:

- `T3`
- `T7`

Exit:

- `overlay` の canonical 扱いが docs と code で一致していること
- `g_t` の設計判断が docs と code contract で一致していること

現在地:

- `T3` は完了
- `T7` も完了

### M4. Economic Hardening

対象期間:

- 2026-04-11 JST から 2026-04-12 JST

含むタスク:

- `T4`

Exit:

- `no-trade` / `turnover` を改善した current 候補が 1 本あり、その後の user-value selection に接続できること

現在地:

- `candidate_04` で保守 current の条件を満たし、その後 `user_value_score` 自動選抜で `candidate_17` へ更新した

### M5. Operator Contract Cleanup

対象期間:

- 2026-04-12 JST から 2026-04-13 JST

含むタスク:

- `T6`

Exit:

- dashboard と prediction artifact が `display forecast` と `policy driver` を分離していること

現在地:

- 完了

### M6. Governance And Contract Closure

対象期間:

- 2026-04-07 JST から 2026-04-09 JST

含むタスク:

- `T8`
- `T9`
- `T10`

Exit:

- `accepted` / `production current` の意味が rule か explicit override metadata として固定されていること
- `current` / report / artifact JSON の authoritative SoT が 1 つに決まっていること
- accepted winner / production current の paired frontier comparison が 1 本あること

現在地:

- 完了

### M7. Claim Hardening

対象期間:

- 2026-04-09 JST から 2026-04-10 JST

含むタスク:

- `T11`

Exit:

- `shape-aware` / `regime-aware` の claim が current evidence に一致していること

現在地:

- 完了

## 4. 完了条件

この review follow-up を完了扱いにしてよい条件は次です。

1. `checkpoint` と `candidate selection` が blocked walk-forward 基準で一貫している
2. `shape / horizon / g_t` collapse に対して cheap ablation の説明が付いている
3. `overlay` の canonical 判断が終わっている
4. `no-trade` / `turnover` の改善が validation と live replay の両方で確認できる
5. `forecast` と `policy driver` の表示契約が分離されている
6. `g_t` scalar broadcast を維持するか、次 variant へ進むかの理由が明文化されている
7. `accepted` と `production current` の関係が rule か explicit governance override として固定されている
8. `current` / `report` / consumer schema の authoritative SoT が 1 つに固定されている
9. accepted winner / production current の paired frontier comparison が current promote 根拠として残っている
10. `shape-aware` / `regime-aware` の claim が current evidence に一致している

2026-04-07 JST 時点で、旧 follow-up の 1-6 は満たしていた。今回の contract closure 実装で、追加 review intake で定義した 7-10 も満たした。

## 5. Immediate Next Action

この review follow-up は close してよい。

- 直近の次アクションは、new live data rerun 前に current governance metadata が想定どおり更新されることを smoke で再確認すること
- その次は、`shape-aware` を再主張するための cheap variant を別タスクとして切り出すこと
