# AI Workers

Rust Core からサブプロセスとして起動される Python ワーカー群。

通信: 標準入出力の JSON Lines (1リクエスト = 1行 JSON、1レスポンス = 1行 JSON)。

- face/       顔検出 (SCRFD, ONNX)
- embedding/  顔特徴量 (ArcFace系, ONNX)
- clustering/ DBSCAN による同一人物クラスタリング

モデルファイル (*.onnx) は各ワーカーの models/ に置く (Git 管理外)。
- space/      埋め込みの3次元化 (PCA / UMAP) と単眼深度推定
              (Depth Anything V2, 任意)。内蔵3D空間ビュー用。
              深度モデルは <repo>/models/depth-anything-v2-small.onnx
              または環境変数 DAHLIA_DEPTH_MODEL で指定。
