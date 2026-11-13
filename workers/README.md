# AI Workers

Rust Core からサブプロセスとして起動される Python ワーカー群。

通信: 標準入出力の JSON Lines (1リクエスト = 1行 JSON、1レスポンス = 1行 JSON)。

- リクエスト: `{"id": 1, ...}`（id は Rust 側が付ける）
- 進捗 (任意・何行でも): `{"id": 1, "progress": {"done": d, "total": t}}`
  — ハンドラは `common.report_progress(done, total)` で送る。
  Rust 側は UI へ転送し、生存信号としても扱う（長い計算は黙らず
  定期的に進捗を出すこと）
- 最終レスポンス: `{"id": 1, "ok": true, ...}` または
  `{"id": 1, "ok": false, "error": "..."}`

Rust 側の監督 (core/jobs): ワーカーが一定時間（既定300秒、
`DAHLIA_WORKER_TIMEOUT_SECS` で変更可）沈黙するとハングとみなして
終了させる。利用者が中断した場合はワーカーごと止める。中断・
タイムアウト・切断のあと、そのセッションは再利用されない。

- 共通        画像の読み込みは common.load_image_rgb に集約（EXIF 回転の
              適用込み）。HEIC/HEIF (W87) は pillow-heif が必要:
              pip install pillow-heif
              無い環境では HEIC の写真だけが理由つきで失敗する。
- heif_convert.py  HEIC → JPEG 変換の一回限りスクリプト。Rust 側は
              HEIC を直接読めないため、取り込み時にアプリデータ内へ
              作業コピー (プロキシ) を作り、サムネイルと顔の切り抜きを
              そこから作る。Rust 側 (src-tauri の heif.rs) が
              SRC/DST の組を引数に一括で呼ぶ。
- face/       顔検出 (SCRFD, ONNX)
- embedding/  顔特徴量 (ArcFace系, ONNX)
- clustering/ DBSCAN による同一人物クラスタリング
              大量の顔は `{"op": "add_items", "items": [...]}` を
              繰り返してから `{"op": "cluster", ...}` で分類する
              分割送信にも対応（従来の一括形式も可）

モデルファイル (*.onnx) は各ワーカーの models/ に置く (Git 管理外)。
- facemesh/   MediaPipe FaceLandmarker による478点顔ランドマーク
              (3D顔プレビュー用、モデルは models/ に同梱)
- space/      埋め込みの3次元化 (PCA / UMAP) と単眼深度推定
              (Depth Anything V2, 任意)。内蔵3D空間ビュー用。
              clustering と同じ分割送信 (`add_items` → `reduce`)
              にも対応（従来の一括形式も可）
              深度モデルは <repo>/models/depth-anything-v2-small.onnx
              または環境変数 DAHLIA_DEPTH_MODEL で指定。
              facemesh は mediapipe (Tasks API) と scipy が必要:
              pip install mediapipe scipy
              (Linux では libEGL/libGL のシステムライブラリも必要)
