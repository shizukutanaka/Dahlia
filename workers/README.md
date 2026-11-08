# AI Workers

Rust Core からサブプロセスとして起動される Python ワーカー群。

通信: 標準入出力の JSON Lines (1リクエスト = 1行 JSON、1レスポンス = 1行 JSON)。

- face/       顔検出 (SCRFD, ONNX)
- embedding/  顔特徴量 (ArcFace系, ONNX)
- clustering/ DBSCAN による同一人物クラスタリング

モデルファイル (*.onnx) は各ワーカーの models/ に置く (Git 管理外)。
