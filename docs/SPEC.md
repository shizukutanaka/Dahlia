# Dahlia 仕様書（正本）

このファイルが Dahlia の仕様の正本である。もともと仕様は開発指示
（Dahlia Development Master Prompt、2026-11-09）にのみ存在し repo の
外にあったため、記録から転記して repo に移設した（W89）。仕様を変える
ときは、このファイルを更新し、変更理由をコミットメッセージに残すこと。

## 製品定義

Dahlia は、ローカル環境で人物 Identity を管理する AI クリエイター向け
デスクトップアプリである。

- 写真から顔を検出し、特徴量で同一人物を分類して、再利用可能な
  Person Identity として保存する
- Dahlia 自体は画像生成 AI ではない（生成は利用者側のツールの仕事で、
  Dahlia はその材料となる Identity を管理する）

## 技術構成（固定）

- Tauri + React / TypeScript（フロントエンド）
- Rust（バックエンド／コア）
- SQLite（保存）
- Python AI ワーカー（JSON Lines プロトコル、workers/README.md）
- ONNX Runtime（推論）
- 顔検出: SCRFD / RetinaFace 系
- 特徴量: ArcFace 系
- クラスタリング: DBSCAN

## アーキテクチャ規則

Frontend → Rust Core → Job Queue → Python AI Worker → AI Models

UI から AI 処理を直接呼ばない。必ず Rust Core の Job Queue を経由する。

## MVP スコープ（10項目）

1. アプリ起動
2. Project 作成
3. 写真フォルダ取り込み
4. 画像一覧
5. 顔検出
6. 特徴量生成
7. 同一人物クラスタリング
8. Person Identity 生成
9. SQLite 保存
10. Identity 表示

## Non-Goals（やらないこと）

- 自前の画像生成 AI
- 動画生成
- VR
- 3D モデル生成
- 音声生成
- AI チャット
- クラウド同期
- SNS
- 課金
- サーバ
- Plugin Marketplace

## 開発フェーズ（順序厳守、当時の完了条件）

Phase 1 環境確認 → Phase 2 Tauri 起動 → Phase 3 SQLite →
Phase 4 Project 管理 → Phase 5 画像 Import → Phase 6 顔検出 →
Phase 7 Embedding → Phase 8 Clustering → Phase 9 Identity UI →
Phase 10 テスト。V0.1 はこの順序で完成した。

## 作業規律（Autonomous Development Agent Prompt より）

- 自律実行は許可されるが、仕様変更・MVP 拡張・技術変更は勝手に行わない
- フェーズ順序を飛ばさない
- 報告形式: Completed / Changed Files / Test Result / Problems / Next Action

## 承認済みの拡張（MVP 完了後の追加）

いずれも利用者の明示指示で追加されたもので、Non-Goals との関係も
記録されている。

- **V0.2: 3D Identity Space（内蔵）** — 特徴量の 3D 可視化（PCA/UMAP）と
  AI 深度レリーフ。「3D モデル生成」ではなく可視化であり Non-Goals と
  衝突しないという整理で追加
- **V0.3: 3D 顔プレビュー（内蔵）** — MediaPipe FaceLandmarker による
  顔メッシュの可視化（同様に可視化の範囲）
- **市販レベル化バッチ1〜3** — データ保全・誤分類の修正・規模・配布の
  品質底上げ。計画の正本は goal 側の
  「Dahlia市販レベル化-ギャップ監査と実装計画.md」。方針変更が要る項目
  （モデルの権利確認、保存時暗号化、多言語化、自動更新、人物の書き出し、
  コード署名）は同文書の第8節に分離され、判断なしに着手しない
