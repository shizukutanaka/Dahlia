//! Minimal file logging for diagnostics (W12/W13).
//!
//! Design constraints: no external crates, no network — lines are
//! appended to one local file (logs/dahlia.log under the app data
//! directory) so a user can find it and attach it to a problem
//! report. Logging must never break the app: every failure to write
//! is silently dropped.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Append-only line logger over a single file. Kept separate from
/// the global state so it can be unit-tested directly.
pub struct FileLog {
    file: File,
    path: PathBuf,
}

impl FileLog {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    pub fn write_line(&mut self, level: &str, message: &str) {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // One line per event: keep embedded newlines from breaking
        // the log's line structure.
        let flat = message.replace(['\n', '\r'], " ");
        let _ = writeln!(self.file, "[{secs}] {level} {flat}");
        let _ = self.file.flush();
    }
}

static GLOBAL: Mutex<Option<FileLog>> = Mutex::new(None);

/// Start logging to `path`. Called once from app setup; later calls
/// replace the target. Returns the path on success.
pub fn init(path: &Path) -> Option<PathBuf> {
    let log = FileLog::open(path).ok()?;
    let p = log.path.clone();
    if let Ok(mut slot) = GLOBAL.lock() {
        *slot = Some(log);
    }
    Some(p)
}

/// The file logs are written to, if logging was initialised.
pub fn log_path() -> Option<PathBuf> {
    GLOBAL
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|l| l.path.clone()))
}

fn write(level: &str, message: &str) {
    if let Ok(mut slot) = GLOBAL.lock() {
        if let Some(log) = slot.as_mut() {
            log.write_line(level, message);
        }
    }
}

pub fn info(message: &str) {
    write("INFO", message);
}

pub fn warn(message: &str) {
    write("WARN", message);
}

pub fn error(message: &str) {
    write("ERROR", message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_log_appends_timestamped_lines() {
        let dir = std::env::temp_dir().join(format!("dahlia-log-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("logs").join("dahlia.log");
        {
            let mut log = FileLog::open(&path).unwrap();
            log.write_line("INFO", "起動しました");
            log.write_line("ERROR", "複数行\nメッセージ");
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "embedded newline must be flattened: {text}");
        assert!(lines[0].contains("INFO 起動しました"), "{text}");
        assert!(lines[1].contains("ERROR 複数行 メッセージ"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
