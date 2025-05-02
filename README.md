# pubcom-seiri

パブリックコメントの類似性分析ツール

## 概要

このリポジトリは、パブリックコメントのデータを分析し、類似したコメントをクラスタリングするためのツールを提供します。

## 機能

### v1 (従来の実装)

#### tool1
- データ処理を3ステップに分割
  1. データの単純なロード：CSVファイルからデータを読み込む
  2. 重複除去：完全一致の入力を検出し、同一内容のコメントを1つにまとめる
  3. 除去済みのデータからのN件取得：分析対象とするデータを選択（デフォルト10000件）
- CSVデータの改行は空白を入れずに結合
- 完全一致の入力を検出し、同一内容のコメントを記録
- クラスタリング結果のファイル出力
- 近い順に併合過程を可視化（最初の50件をMarkdownとCSVで出力）
- 結果を見やすくHTML形式で表示
  - 2つの文字単位diffを取り可視化、共通部の多さをパーセント表示
  - クラスタ内の最も遠いデータとの比較を表示
  - 各データには元CSVでのIDを併記
  - 完全一致のコメントを一覧表示

#### tool2
- tool1の結果から類似度の閾値を設定
- クラスタリング結果を2件以上のクラスタと独立点に分類
- クラスタの中央を計算
- 除外されていたデータの振り分け
- 出力データの生成（独立点リスト、各クラスタリスト、代表点リスト）
- 分析レポートの生成（Markdown形式）
  - 「ID Xは同一内容がN件あった」という完全一致情報を含む

### v2 (フェーズ分割実装)

#### フェーズ1 (phase1.py)
- CSVデータの読み込みと重複除去
  - 完全一致の重複除去（既存機能）
  - 文字ベースのdiffを使用して50%以上共通しているコメントをグループ化
  - 処理結果をCSVとJSONとして出力
  - 類似度グループのMarkdownレポートを生成

#### フェーズ2 (phase2.py)
- フェーズ1の出力を読み込み
- 埋め込みベクトルを生成
- クラスタリングを実行
- HTMLレポートを生成（文字ベースの類似度グループ情報を含む）

#### フェーズ3 (tool2.py)
- フェーズ2の結果から類似度の閾値を設定
- クラスタリング結果を2件以上のクラスタと独立点に分類
- クラスタの中央を計算
- 除外されていたデータの振り分け
- 出力データの生成（独立点リスト、各クラスタリスト、代表点リスト）
- 分析レポートの生成（Markdown形式）

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### v1 (従来の実装)

#### tool1

```bash
python -m pubcom_seiri.tool1 --input input.csv --output output_dir --limit 10000 --threshold 0.4
```

#### tool2

```bash
python -m pubcom_seiri.tool2 --input output_dir/clusters.json --threshold 0.4 --output result_dir
```

### v2 (フェーズ分割実装)

#### 全フェーズを一括実行

```bash
python -m pubcom_seiri.run_experiment --input input.csv --output output_dir --limit 10000 --threshold 0.4 --similarity 50.0
```

#### 各フェーズを個別に実行

##### フェーズ1

```bash
python -m pubcom_seiri.phase1 --input input.csv --output phase1_output --limit 10000 --similarity 50.0
```

##### フェーズ2

```bash
python -m pubcom_seiri.phase2 --input phase1_output --output phase2_output --threshold 0.4
```

##### フェーズ3

```bash
python -m pubcom_seiri.tool2 --input phase2_output --original input.csv --threshold 0.4 --output phase3_output
```

## 依存関係

- Python 3.12
- sentence-transformers
- scikit-learn
- numpy
- pandas
- scipy
- difflib
- jinja2

## ライセンス

MIT

