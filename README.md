# Dahlia

ローカル環境で人物Identityを管理する、AIクリエイター向けデスクトップアプリ。

写真から顔を検出し、特徴量で同一人物を分類して、再利用可能な Person Identity として保存します。画像生成機能は持ちません。

## 構成

- `apps/studio/` - Tauri + React + TypeScript デスクトップアプリ
- `core/` - Rust コアライブラリ (storage / identity / jobs)
- `workers/` - Python AIワーカー (face / embedding / clustering) - ONNX Runtime
- `database/` - SQLite スキーマ
- `tests/` - テストとサンプルデータ
- `docs/` - 仕様と設計メモ（仕様の正本は `docs/SPEC.md`、構成は `docs/ARCHITECTURE.md`）
- `scripts/` - バージョン整合チェックとポータブル版ビルド手順
- `packaging/` - Windows ポータブル版の同梱物（起動 bat・README・自己診断）

## 開発

```bash
cd apps/studio
npm install
npm run tauri dev
```

データはOSのアプリデータフォルダ内の `dahlia.db` (SQLite) に保存されます。

## AIワーカーのセットアップ（開発環境）

顔検出・特徴量・クラスタリングは Python ワーカーが担当します。

```bash
python3 -m venv .venv
.venv/bin/pip install onnxruntime pillow numpy
# モデル配置: workers/face/models/det_500m.onnx (SCRFD)
#             workers/embedding/models/w600k_mbf.onnx (ArcFace)
# 出典: InsightFace buffalo_s パック
```

アプリ実行時は環境変数でワーカーを指定できます（未指定時は実行ファイル/カレント配下の `workers/` を自動探索）:

```bash
DAHLIA_PYTHON=$PWD/.venv/bin/python DAHLIA_WORKERS_DIR=$PWD/workers npm run tauri dev
```

## リリースビルド（Windows ポータブル版）

```bat
cd apps\studio
npm install
npm run tauri build
```

で `apps/studio/src-tauri/target/release/` に実行ファイルができます。
ポータブル版フォルダの組み立ては `scripts/build-portable.sh` が正本です
（組み立て済み Python ランタイムと実行ファイルを入力に、検証つきで
ZIP まで作成します。入力の詳細はスクリプト冒頭のコメント参照）。
バージョン表記は `scripts/check-version.sh` で整合を確認できます。

## テスト

```bash
cd core && cargo test                      # ストレージ層
cd apps/studio/src-tauri && cargo test     # パイプラインE2E（モデル+venv必須、無い場合はスキップ）
.venv/bin/python -m unittest tests.test_workers   # ワーカー単体
```
