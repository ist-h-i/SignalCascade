# SignalCascade PyTorch Reference

`PyTorch/` 配下には、現在 4 種類の文書があります。

- canonical target spec:
  - `shape_aware_profit_maximization_model.md`
- 移行ロードマップ:
  - `profit_maximization_migration_roadmap.md`
- 現行実装のロジック説明:
  - `logic_multiframe_candlestick_model.md`
- 旧来の要件定義:
  - `requirements_multiframe_candlestick_model.md`

現行コードはまだ `shape_aware_profit_maximization_model.md` へ完全移行していません。`profit_maximization_migration_roadmap.md` は完全移行までの実施順を、`logic_multiframe_candlestick_model.md` は移行前の reference implementation が現在どう動いているかを説明する文書です。

この実装は、以下を最小構成でカバーします。

- 30分足ベースの OHLCV 生成または CSV 読み込み
- 30m / 1h / 4h / 1d / 1w の再集約
- 足形状 `Q=[u,b,l]`
- Path-Averaged Directional Balance `chi`
- ローカル価格スケール `a_i`
- 加算型 close anchor `L0`
- `xi=[z,dz,chi,g,rho,nu]` の特徴量化
- main branch (4h / 1d / 1w) の multi-horizon return / uncertainty / shape 学習
- overlay branch (1h / 30m) の exit action 学習

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
