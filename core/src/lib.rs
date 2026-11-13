//! Dahlia core library.
//!
//! The module directories live at the repository-specified locations
//! (core/storage, core/identity, core/jobs) and are wired in via
//! #[path] so the on-disk layout matches docs/ARCHITECTURE.md.

#[path = "../storage/mod.rs"]
pub mod storage;

#[path = "../identity/mod.rs"]
pub mod identity;

#[path = "../jobs/mod.rs"]
pub mod jobs;

#[path = "../logging/mod.rs"]
pub mod logging;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("データベースの処理で問題が発生しました: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("データの形式で問題が発生しました: {0}")]
    Json(#[from] serde_json::Error),
    #[error("ファイルの読み書きで問題が発生しました: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Worker(String),
    #[error("{0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, Error>;
