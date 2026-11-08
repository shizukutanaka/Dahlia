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

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("database error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("worker error: {0}")]
    Worker(String),
    #[error("not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, Error>;
