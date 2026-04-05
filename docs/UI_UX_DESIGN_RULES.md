# UI/UX Design Rules

`SignalCascade` のフロントエンドを作り直す前提で、`ユーザーに与える価値`、`判断を助ける情報設計`、`デザインとUI/UXのオリジナリティ` を優先するための実務ルールをまとめる。AI に UI を作らせるときも、この文書を上位制約として使う。

## 1. 最優先原則

1. UI は「見た目」ではなく「ユーザーの判断をどう速く、どう確かにするか」で評価する。
2. 最初の 5-10 秒で、`何が起きているか`、`いま何をすべきか`、`なぜそう言えるのか` が読める構成を優先する。
3. 各画面には、`brand promise`、`decision support`、`proof` のいずれか 1 つを主役として割り当て、役割を混線させない。
4. オリジナリティは装飾量ではなく、`構図`、`情報の見せ順`、`素材感`、`モーションの意味` のどこで差別化するかを先に定義する。
5. KPI、チャート、カードは「あるから置く」のではなく、「ユーザーの次の行動を強くする」場合だけ置く。

## 2. SignalCascade で与える価値

- 予測を見ること自体ではなく、`次の判断に再利用できる確率の高い示唆` を返す。
- 数字を並べることではなく、`信頼してよい条件` と `警戒すべき条件` を明示する。
- 1 回の予測結果ではなく、`予測の鮮度`、`根拠`、`不確実性`、`検証履歴` を近接配置して信頼を作る。
- ダッシュボードの主価値は、`読むこと` よりも `迷いを減らすこと` に置く。

## 3. AI支援UI生成規約（GPT-5.4）

### 3.1 使い方

1. まず画面種別を `LP`、`Brand Site`、`Dashboard`、`App Surface`、`Form`、`Docs` から 1 つ選ぶ。
2. `user value thesis`、`visual thesis`、`content plan`、`interaction thesis`、`originality thesis` を空欄のままにしない。
3. 指示文は日本語を基本にし、英語キーワードは必要な場合だけ補助的に使う。
4. 仮テキストより、実在する情報、実データ、実際の制約を渡す。
5. 画像が必要なら、アップロード済み画像、事前生成画像、またはムードボード指示を明記する。
6. 最後に `desktop` と `mobile` の検証条件を明記する。
7. 画面実装前に、必ず `visual direction`、`interaction strategy`、`originality point` を 1 行ずつ宣言する。

### 3.2 必須で埋める項目

- `user value thesis`: この画面がユーザーの何を前進させるのか
- `decision moment`: ユーザーがこの画面で最終的に下す判断
- `trust strategy`: 何を近くに置くとユーザーが信頼できるか
- `originality thesis`: 他の SaaS テンプレートと何が違うのか
- `signature element`: その画面で一番記憶に残る構図、面、演出、または操作

### 3.3 コピペ用テンプレート（SignalCascade 既定値入り完成版）

```text
あなたは GPT-5.4 ベースの UI デザイナー兼実装者です。
既存のデザインシステムや指定制約がある場合はそれを優先し、ない場合は以下の条件で UI を設計・実装してください。
回答は日本語を基本にし、必要な英語キーワードだけ補助的に使ってください。

[タスク設定 (Task Setup)]
- 画面種別 (task type): Dashboard
- 推論レベル (reasoning level): high
- 想定ユーザー (target user): 金融時系列モデルの研究者、裁量トレーダー、検証担当者
- 最重要タスク (primary task): いまの予測を意思決定に再利用してよいかを短時間で判断する
- 成功条件 (success criteria): 初見でも 10 秒以内に「推奨行動」「予測の強さ」「信頼してよい理由または見送る理由」が読めること

[価値仮説 (Value Thesis)]
- user value thesis: 数字の羅列ではなく、再利用可能な示唆だけを短時間で見極められる判断面を作る
- decision moment: いまシグナルに追従するか、観察を続けるか、見送るかを決める
- trust strategy: 予測値の近くに、信頼度、不確実性、鮮度、検証指標、直近実績を置いて根拠を分断しない

[ビジュアル方針 (Visual Thesis)]
- visual thesis: 静かな緊張感を持つ市場監視室のように、重心の低い構図で判断の解像度を上げる
- visual direction: 大きな時系列面を核にし、右側に判断面を縦に束ねる二層構成

[構成案 (Content Plan)]
- content plan:
  主作業領域（Primary workspace） -> ナビゲーション（Navigation） -> 補助情報（Secondary context）
  主作業領域: 過去、予測、実績を同じ時間軸で比較できる主チャート
  ナビゲーション: 将来的な時間軸切替やシンボル切替が入る前提で上部または左側に最小限で置く
  補助情報: いまの行動、信頼度、予測幅、鮮度、モデル品質の要点を近接配置する

[体験方針 (Interaction Thesis)]
- interaction thesis:
  1. 主要面はフェードとわずかな移動だけで立ち上げ、過剰演出を避ける
  2. チャート hover 時は対象系列だけを強調し、比較対象は弱める
  3. 信頼度や freshness の更新時は数値だけを短くトランジションさせる
- interaction strategy: 動きは「目線誘導」と「更新の気づき」にだけ使う

[独自性方針 (Originality Thesis)]
- originality thesis: 金融ダッシュボードの定番である KPI モザイクではなく、判断面と根拠面の近接関係で独自性を作る
- originality point: editorial な面構成で、主チャートと判断面の主従を強く分ける
- signature element: 右側の判断面で `Action`, `Confidence`, `Why`, `Freshness` が一続きに読めること
- avoid imitation: 暗色ネオンの HUD 風演出、意味のない発光、カードの均一反復

[デザインシステム (Design System)]
- 背景 (background): 暖かいアイボリーのベースに、アンバーとブルーの薄い放射グラデーションを重ねた静かな計測面
- 面 (surface): 情報差を背景差で整理した半透明アイボリー面と、主チャートだけを受け止める濃色サーフェス
- 主要テキスト (primary text): 深いチャコールブラウン `#151210`
- 補助テキスト (muted text): 青みを含んだウォームグレー `rgba(21, 18, 16, 0.62)`
- アクセント (accent): 予測はアンバー `#DAA23A`、実績はブルー `#7F8CFF`、警戒はバーントトーン
- display typography: `Space Grotesk`
- headline typography: `Space Grotesk`
- body typography: `IBM Plex Sans`
- caption typography: `IBM Plex Sans` をベースに数値は tabular figure を優先する

[入力素材 (Content Inputs)]
- ブランド/製品名 (brand/product name): SignalCascade
- 核となるメッセージ (core message): 予測の可視化ではなく、再利用できる判断を返す
- CTA ラベル (CTA label): 詳細検証を見る
- 補足事実 (supporting facts):
  - selected horizon
  - expected return
  - prediction band
  - trust score
  - freshness
  - validation metrics
- 必須セクション (required sections):
  - Main price chart
  - Decision panel
  - Confidence and uncertainty
  - Provenance and freshness
  - Validation summary
- 除外したい要素 (forbidden sections):
  - マーケティング用ヒーロー
  - 装飾だけの KPI カード列
  - ネオン調の未来風 UI
  - 情報を囲いすぎる多重ボーダー

[画像方針 (Asset Guidance)]
- 画像はアップロード済み、または事前生成済みのものを優先する。
- 使える画像がない場合は、先にムードボードか 2-3 案の方向性を作る。
- 明示指示がない限り、任意の Web 画像は使わない。

[厳守条件 (Hard Constraints)]
- ファーストビューに置く主CTAは 1 つに絞る。
- 各セクションには 1 つの役割だけを持たせる。
- 汎用的な SaaS のカードモザイクは作らない。
- 枠線の入れ子を避ける。
- 枠線を足す前に、余白・背景差・タイポグラフィで整理する。
- hover / focus / active / disabled の状態差を必ず見せる。
- モーションは抑制しつつ、意味のある変化だけに使う。
- `prefers-reduced-motion` を尊重する。
- コントラストとセマンティック HTML を確保する。
- すべての主要指標、チャート、図版に「なぜ重要か」を短く添える。
- Recommendation と Evidence を視覚的に混同させない。
- 指標の近くに、必要なら freshness / source / confidence を置く。
- 色だけで状態を伝えない。ラベル、形、位置でも意味を補う。

[画面種別ルール (Screen-Type Rules)]
- 画面種別が LP または Brand Site の場合:
  - 最初の表示面は 1 つのまとまりとして読めること。
  - `brand first`、`headline second`、`support copy third`、`CTA fourth` の順で伝える。
  - `full-bleed hero` または支配的なビジュアル面を優先する。
  - ヒーローカード、フローティングバッジ、統計帯、プロモチップは既定で置かない。
  - 世界観の演出は「空気感」と「約束」に集中させ、機能の羅列で崩さない。

- 画面種別が Dashboard または App Surface の場合:
  - 明示指定がない限り、マーケティング用のヒーローは入れない。
  - まず作業面そのものを見せる。
  - ブランド訴求よりも、orientation / status / action を優先する。
  - カードは、ユーザー操作の器として必要なときだけ使う。
  - 第一視界で `現状`、`推奨行動`、`根拠` の 3 点が分かるようにする。
  - 不確実性、鮮度、検証結果は深い階層へ追いやらず、判断面に近接させる。
  - 「観測」「示唆」「実行」を視覚的に分離し、同一面に混ぜない。

[検証 (Verification)]
- desktop と mobile のレイアウトを確認する。
- mobile で横スクロールが起きないようにする。
- fixed / sticky / floating 要素が本文、CTA、フォームに重ならないようにする。
- 主要な CTA と操作要素で hover / focus / active / disabled を確認する。
- 初見ユーザーが 10 秒以内に「いま何を見るべきか」「次に何をするか」を答えられるか確認する。
- 画面の独自性が、文言を読まなくても構図や面の設計から伝わるか確認する。
- 可能なら、実際に描画して見え方を確認し、必要に応じて調整する。

[出力形式 (Output Format)]
- 最初に、次の 5 点を日本語で短く再掲する:
  - user value thesis
  - visual thesis
  - content plan
  - interaction thesis
  - originality thesis
- その後に UI 実装を出力する。
- プロンプトの指示や設計メモを、画面上の文言に混ぜ込まない。
```

## 4. 価値と信頼を崩さないための追加ルール

### 4.1 User Value First

- 各画面には、`この画面を開く理由` を 1 文で定義する。
- 各セクションには、`読むと何が楽になるか` を説明できる役割だけを持たせる。
- 不要な情報は「見せない」ことも価値とみなす。
- 価値説明が抽象的な場合は、`時間短縮`、`認知負荷削減`、`判断精度向上`、`安心感の補強` のどれに効くのかへ分解する。

### 4.2 Recommendation vs Evidence

- 推奨アクションを見せるなら、少なくとも 1 つの証拠面を近接配置する。
- 証拠は `単一数値` ではなく、`範囲`、`比較`、`履歴`、`鮮度` のうち 2 つ以上で支える。
- 信頼度が中以下のときは、強い CTA よりも `保留理由` や `観察継続` を優先する。
- 「自信があるように見える UI」が「実際に信頼できる UI」を上回らないようにする。

### 4.3 Dashboard の判断導線

- 第一視界で `status`、`confidence`、`recommended action` を読む順番を固定する。
- 一番大きい面は、最も重要な判断のために使う。最も派手な面を、最も重要でない要素に使わない。
- チャートは装飾ではなく、`変化の方向`、`変化の幅`、`例外` のどれを示すかを明確にする。
- KPI 群は最大 4-6 個を基本とし、判断を強めない指標は折り畳むか二次面へ送る。

## 5. オリジナリティ規約

### 5.1 毎画面で 1 つの独自軸を持つ

以下のうち最低 1 つを明示的に採用する。

1. `Composition`: 非対称グリッド、支配的な主面、読み順を制御する大きなタイポ
2. `Material`: 紙、金属、霧、ガラス、計測機器、取引画面など、素材感のある面設計
3. `Narrative`: 結果を見せる順番そのものに物語性を持たせる
4. `Motion`: 状態変化にだけ意味のある動きを使う
5. `Data Framing`: 予測、実績、信頼度、鮮度の近接関係に独自性を持たせる

### 5.2 避けるべき既視感

- どの SaaS にも見える薄いカードの均等グリッド
- 強い理由のない紫系グラデーション
- 意味のないガラスモーフィズム
- 情報密度を上げたつもりで枠線だけが増える構成
- アイコン、バッジ、数値、補足の全部を同じ視覚強度で並べること
- 「見栄えの良い金融 UI」を真似て、判断面よりムード面が勝つこと

### 5.3 オリジナリティと可用性の両立

- 独自性は `読む順番` を壊さない範囲で入れる。
- 特徴的な面は 1 つでよい。全要素を主張させない。
- アクセシビリティを削ってまで独自表現を優先しない。
- 実験的なビジュアルほど、操作部品は標準的に保つ。

## 6. SignalCascade 向けダッシュボード追加規約

- 主役は `いま取るべき判断` とし、予測そのものを主役にしすぎない。
- `Action`, `Confidence`, `Prediction Band`, `Freshness`, `Why` を近くに束ねる。
- `過去`, `予測`, `実績` は同じ面に置く場合でも、色、線種、ラベルで厳密に区別する。
- モデル品質の説明は、研究報告のように深く埋めず、再利用可否の判断に必要な粒度へ要約する。
- 予測が強いときほど UI は静かにし、赤や警告色の多用で煽らない。
- 不確実性が高い局面では、派手な推奨より `見送りの正当性` を示せる構成を優先する。

## 7. 記述例: SignalCascade ダッシュボード

```text
あなたは GPT-5.4 ベースの UI デザイナー兼実装者です。
既存のデザインシステムや指定制約がある場合はそれを優先し、ない場合は以下の条件で UI を設計・実装してください。
回答は日本語を基本にし、必要な英語キーワードだけ補助的に使ってください。

[タスク設定 (Task Setup)]
- 画面種別: Dashboard
- 推論レベル: high
- 想定ユーザー: 金融時系列モデルの研究者、裁量トレーダー、検証担当者
- 最重要タスク: いまの予測を意思決定に再利用してよいかを短時間で判断する
- 成功条件: 初見でも 10 秒以内に「推奨行動」「予測の強さ」「信頼してよい理由または見送る理由」が読めること

[価値仮説 (Value Thesis)]
- user value thesis: 数字の羅列ではなく、再利用可能な示唆だけを短時間で見極められる判断面を作る
- decision moment: いまシグナルに追従するか、観察を続けるか、見送るかを決める
- trust strategy: 予測値の近くに、信頼度、不確実性、鮮度、検証指標、直近実績を置いて根拠を分断しない

[ビジュアル方針 (Visual Thesis)]
- visual thesis: 静かな緊張感を持つ市場監視室のように、重心の低い構図で判断の解像度を上げる
- visual direction: 大きな時系列面を核にし、右側に判断面を縦に束ねる二層構成

[構成案 (Content Plan)]
- content plan:
  主作業領域（Primary workspace） -> ナビゲーション（Navigation） -> 補助情報（Secondary context）
  主作業領域: 過去、予測、実績を同じ時間軸で比較できる主チャート
  ナビゲーション: 将来的な時間軸切替やシンボル切替が入る前提で上部または左側に最小限で置く
  補助情報: いまの行動、信頼度、予測幅、鮮度、モデル品質の要点を近接配置する

[体験方針 (Interaction Thesis)]
- interaction thesis:
  1. 主要面はフェードとわずかな移動だけで立ち上げ、過剰演出を避ける
  2. チャート hover 時は対象系列だけを強調し、比較対象は弱める
  3. 信頼度や freshness の更新時は数値だけを短くトランジションさせる
- interaction strategy: 動きは「目線誘導」と「更新の気づき」にだけ使う

[独自性方針 (Originality Thesis)]
- originality thesis: 金融ダッシュボードの定番である KPI モザイクではなく、判断面と根拠面の近接関係で独自性を作る
- originality point: editorial な面構成で、主チャートと判断面の主従を強く分ける
- signature element: 右側の判断面で `Action`, `Confidence`, `Why`, `Freshness` が一続きに読めること
- avoid imitation: 暗色ネオンの HUD 風演出、意味のない発光、カードの均一反復

[デザインシステム (Design System)]
- 背景: 温度の低いアイボリーグレーか、わずかに霧感のあるニュートラル
- 面: 情報差を背景差で整理した半不透明サーフェス
- 主要テキスト: 深いスレート
- 補助テキスト: 青みを含んだ中間グレー
- アクセント: 予測はアンバー、実績はブルー、警戒は赤ではなく抑えたバーントトーン
- display typography: 使いすぎないが、必要なら存在感のある grotesk
- headline typography: 締まった sans-serif
- body typography: 可読性の高い sans-serif
- caption typography: 等幅または準等幅の technical face を限定利用

[入力素材 (Content Inputs)]
- ブランド/製品名: SignalCascade
- 核となるメッセージ: 予測の可視化ではなく、再利用できる判断を返す
- CTA ラベル: 現在は不要。必要なら「詳細検証を見る」程度に抑える
- 補足事実:
  - selected horizon
  - expected return
  - prediction band
  - trust score
  - freshness
  - validation metrics
- 必須セクション:
  - Main price chart
  - Decision panel
  - Confidence and uncertainty
  - Provenance and freshness
  - Validation summary
- 除外したい要素:
  - マーケティング用ヒーロー
  - 装飾だけの KPI カード列
  - ネオン調の未来風 UI
  - 情報を囲いすぎる多重ボーダー

[検証 (Verification)]
- desktop と mobile のレイアウトを確認する。
- mobile で横スクロールが起きないようにする。
- first screen で推奨行動と根拠が読めるかを確認する。
- hover / focus / active / disabled の状態差を確認する。
- charts, labels, confidence 表現が色だけに依存していないことを確認する。
```

## 8. レビューチェックリスト

- `user value thesis` が見た目の説明ではなく、ユーザー成果の説明になっているか
- 主要面の最初の 3 秒で、画面の役割が分かるか
- `originality point` が具体的で、画面に視覚的に現れているか
- 主CTAまたは主判断が 1 つに絞られているか
- Recommendation と Evidence が混線していないか
- 不確実性、鮮度、ソースが必要な場所に近接しているか
- `border-depth <= 1` を守れているか
- `desktop` と `mobile` で階層の読み順が壊れていないか
