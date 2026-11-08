//! Job execution + Python AI worker bridge.
//!
//! A "job" here is one batch session with a worker process: the worker
//! is spawned once, reads JSON-lines requests on stdin and writes
//! JSON-lines responses on stdout, and exits when stdin closes. Model
//! loading therefore happens once per batch, not once per image.
//!
//! Protocol (see workers/README.md):
//!   -> {"id": 1, ...request fields...}
//!   <- {"id": 1, "ok": true, ...result fields...}
//!   <- {"id": 1, "ok": false, "error": "..."}

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use serde_json::Value;

use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub python: String,
    pub workers_dir: PathBuf,
}

impl WorkerConfig {
    /// Locate the workers directory and Python interpreter.
    ///
    /// Overrides: DAHLIA_PYTHON, DAHLIA_WORKERS_DIR.
    /// Otherwise searches ancestors of the current executable and of
    /// the current directory for a `workers/` tree containing
    /// face/worker.py (repo layout in development; workers shipped
    /// next to the executable in packaged builds).
    pub fn locate() -> Self {
        let python = std::env::var("DAHLIA_PYTHON").unwrap_or_else(|_| {
            if cfg!(windows) { "python".to_string() } else { "python3".to_string() }
        });
        if let Ok(dir) = std::env::var("DAHLIA_WORKERS_DIR") {
            return Self { python, workers_dir: PathBuf::from(dir) };
        }
        let mut starts: Vec<PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            starts.push(exe);
        }
        if let Ok(cwd) = std::env::current_dir() {
            starts.push(cwd);
        }
        for start in starts {
            let mut cur: Option<PathBuf> = Some(start);
            while let Some(d) = cur {
                let candidate = d.join("workers");
                if candidate.join("face").join("worker.py").exists() {
                    return Self { python, workers_dir: candidate };
                }
                cur = d.parent().map(|p| p.to_path_buf());
            }
        }
        Self { python, workers_dir: PathBuf::from("workers") }
    }

    pub fn worker_script(&self, name: &str) -> PathBuf {
        self.workers_dir.join(name).join("worker.py")
    }
}

pub struct WorkerSession {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: i64,
}

impl WorkerSession {
    pub fn start(cfg: &WorkerConfig, worker: &str) -> Result<Self> {
        let script = cfg.worker_script(worker);
        if !script.exists() {
            return Err(Error::Worker(format!(
                "worker script not found: {}",
                script.display()
            )));
        }
        let mut child = Command::new(&cfg.python)
            .arg("-u")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| Error::Worker(format!("failed to start python worker '{worker}': {e}")))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        Ok(Self { child, stdin, stdout, next_id: 0 })
    }

    /// Send one request, read its single response line.
    pub fn call(&mut self, mut request: Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        if let Some(obj) = request.as_object_mut() {
            obj.insert("id".into(), Value::from(id));
        }
        let line = serde_json::to_string(&request)?;
        writeln!(self.stdin, "{line}").map_err(|e| Error::Worker(format!("worker write: {e}")))?;
        self.stdin.flush().ok();

        let mut buf = String::new();
        let n = self.stdout.read_line(&mut buf)
            .map_err(|e| Error::Worker(format!("worker read: {e}")))?;
        if n == 0 {
            return Err(Error::Worker("worker closed the connection unexpectedly".into()));
        }
        let resp: Value = serde_json::from_str(buf.trim())?;
        if resp.get("ok").and_then(|v| v.as_bool()) == Some(true) {
            Ok(resp)
        } else {
            let msg = resp
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown worker error")
                .to_string();
            Err(Error::Worker(msg))
        }
    }
}

impl Drop for WorkerSession {
    fn drop(&mut self) {
        // Closing stdin would end the worker's read loop; kill is
        // immediate and certain for a batch session that is done.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
