# Dahlia

ローカル環境で人物Identityを管理する、AIクリエイター向けデスクトップアプリ。

写真から顔を検出し、特徴量で同一人物を分類して、再利用可能な Person Identity として保存します。画像生成機能は持ちません。

取り込み対応形式: JPEG / PNG / WebP / BMP / GIF / HEIC / HEIF（iPhone の標準形式。HEIC の読み込みには pillow-heif が必要で、無い環境では HEIC の写真だけが失敗として記録されます）

## 構成

- `apps/studio/` - Tauri + React + TypeScript デスクトップアプリ
- `core/` - Rust コアライブラリ (storage / identity / jobs)
- `workers/` - Python AIワーカー (face / embedding / clustering) - ONNX Runtime
- `database/` - SQLite スキーマ
- `tests/` - テストとサンプルデータ
- `docs/` - 仕様と設計メモ（仕様の正本は `docs/SPEC.md`、構成は `docs/ARCHITECTURE.md`、精度の実測と閾値判断は `docs/EVALUATION.md`）
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
.venv/bin/pip install onnxruntime pillow numpy pillow-heif
mkdir -p workers/face/models workers/embedding/models
curl -fL 'https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' \
  -o workers/face/models/face_detection_yunet_2023mar.onnx
curl -fL 'https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' \
  -o workers/embedding/models/face_recognition_sface_2021dec.onnx
```

モデルは OpenCV Zoo 配布の YuNet (MIT) と SFace (Apache-2.0) です。
SHA-256（ポータブル版の組み立て時にも検証します）:

| モデル | SHA-256 |
| --- | --- |
| YuNet | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| SFace | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

**0.5.0 への更新:** 旧版の ArcFace 特徴量は SFace と互換性がなく、混在しません。
既存プロジェクトで「特徴量を取得する」を実行して SFace の特徴量を作り直し、
「人物を分類する」を実行してください。顔検出結果も YuNet に更新したい場合は、
先に「全件を再検出」を実行してください。この操作は取得済み特徴量を消去します。

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
