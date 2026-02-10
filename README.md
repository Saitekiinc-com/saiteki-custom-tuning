# Saiteki Custom Gemini Tuning

このリポジトリは、Saiteki独自のAIモデル（Gemini）の構築・ファインチューニング・比較検証を行うためのプロジェクトです。

## ディレクトリ構成

- **`tuning/`**: モデルの学習（ファインチューニング）に関するスクリプト群
  - `start_tuning.py`: Vertex AI 上で学習ジョブを実行するスクリプト
  - `convert_data.py`: 学習データをJSONL形式に変換するツール
  - `data/`: 学習用データ置き場
- **`ci_scripts/`**: GitHub Actions 等で使用する自動検証・比較用スクリプト
  - `compare_models.py`: ベースモデルと独自モデルの回答を比較するスクリプト
- **`.github/workflows/`**: GitHub Actions の設定ファイル
  - `model_comparison.yml`: Issue作成時にモデル比較を自動実行するワークフロー

---

## 使い方 (Workflow)

### 1. 学習データの準備
`tuning/data/source.jsonl` (元データ) を配置し、変換スクリプトを実行して Vertex AI 用の形式 (`training.jsonl`) を生成します。

```bash
python3 tuning/convert_data.py
# -> tuning/data/training.jsonl が生成されます
```

### 2. ファインチューニングの実行
Vertex AI 上で学習ジョブを開始します。
※ Google Cloud SDK (`gcloud`) の認証が必要です。

```bash
python3 tuning/start_tuning.py
```
実行すると、Google Cloud コンソールのURLが表示されます。学習完了まで待ちます。

### 3. デプロイと設定 (Google Cloud Console)
学習が完了したら、コンソール上で「モデルのエンドポイントへのデプロイ」を手動で行ってください。
新しいエンドポイントIDが発行された場合は、GitHub Secrets (`VERTEX_ENDPOINT_ID`) を更新します。

### 4. モデルの検証 (GitHub Issues)
このリポジトリの **Issues** にて、「New Issue」から `Model Comparison` テンプレートを選択して投稿すると、自動的に比較テストが走ります。

- **トリガー**: `model-comparison` ラベルが付いたIssueが作成・更新された時
- **動作**: 
  1. 入力されたプロンプトに対して、ベースモデル (Gemini 2.5 Flash等) と チューニング済みモデル の両方が回答を生成
  2. 結果がIssueのコメントとして自動投稿される

これにより、チーム全員でモデルの挙動変化を確認できます。

## 必要な環境変数 / Secrets

GitHub Actions (`Settings > Secrets and variables > Actions`) に以下を設定してください。

| Secret名 | 説明 |
| :--- | :--- |
| `GCP_PROJECT_ID` | Google Cloud プロジェクトID |
| `GCP_REGION` | リージョン (例: `us-central1`) |
| `VERTEX_API_KEY` | Vertex AI (Gemini) APIキー |
| `VERTEX_ENDPOINT_ID` | デプロイしたチューニング済みモデルのエンドポイントID |
