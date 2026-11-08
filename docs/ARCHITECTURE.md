# Architecture

Frontend (React) -> Tauri Commands -> Rust Core (dahlia-core) -> Job Queue -> Python AI Worker -> ONNX Models

- UI から AI 処理を直接呼ばない。必ず Rust Core の Job Queue を経由する。
- AI ワーカーは標準入出力の JSON Lines プロトコルで Rust と通信する (workers/README.md)。
- AI モデルは ONNX ファイル差し替え可能な前提で、モデル名を embeddings.model に記録する。
- DB は SQLite 単一ファイル。スキーマは database/schema.sql が正。
