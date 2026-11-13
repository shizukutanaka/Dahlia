//! Job execution + Python AI worker bridge.
//!
//! A "job" here is one batch session with a worker process: the worker
//! is spawned once, reads JSON-lines requests on stdin and writes
//! JSON-lines responses on stdout, and exits when stdin closes. Model
//! loading therefore happens once per batch, not once per image.
//!
//! Protocol (see workers/README.md):
//!   -> {"id": 1, ...request fields...}
//!   <- {"id": 1, "progress": {"done": 3, "total": 10}}   (any number,
//!                                                         interim)
//!   <- {"id": 1, "ok": true, ...result fields...}        (final)
//!   <- {"id": 1, "ok": false, "error": "..."}            (final)
//!
//! Long calls are supervised (W8/W9/W10): a call whose worker stays
//! silent past the inactivity timeout is treated as hung and the
//! worker is killed; a shared cancel flag aborts the wait (and the
//! worker) promptly; progress lines are forwarded to the caller's
//! hook and also reset the inactivity clock. After a timeout, a
//! cancel or a disconnect the session is "broken" and refuses
//! further calls — its state can no longer be trusted.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::{Error, Result};

/// Interim progress for a long job, in whatever unit the job counts
/// (photos processed, faces clustered...). `phase` optionally names
/// the current sub-step when a job has several.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgressInfo {
    pub done: u64,
    pub total: u64,
    pub phase: Option<String>,
}

/// Hooks a long job reports through: a cancel flag the UI can set,
/// and a progress sink. Cloning shares the same flag and sink.
#[derive(Clone, Default)]
pub struct JobHooks {
    pub cancel: Option<Arc<AtomicBool>>,
    pub progress: Option<Arc<dyn Fn(ProgressInfo) + Send + Sync>>,
}

impl JobHooks {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    pub fn report(&self, done: u64, total: u64, phase: Option<&str>) {
        if let Some(sink) = &self.progress {
            sink(ProgressInfo {
                done,
                total,
                phase: phase.map(|s| s.to_string()),
            });
        }
    }

    /// Derive hooks whose progress reports are transformed first —
    /// e.g. a worker reporting rows of a sub-computation whose
    /// counts must be offset into the job's overall units.
    pub fn mapped(
        &self,
        f: impl Fn(ProgressInfo) -> ProgressInfo + Send + Sync + 'static,
    ) -> JobHooks {
        JobHooks {
            cancel: self.cancel.clone(),
            progress: self.progress.as_ref().map(|sink| {
                let sink = sink.clone();
                Arc::new(move |p: ProgressInfo| sink(f(p))) as Arc<dyn Fn(ProgressInfo) + Send + Sync>
            }),
        }
    }
}

/// How long a worker may stay silent (no final response and no
/// progress line) before it is treated as hung. Generous on
/// purpose: a false kill of a slow-but-working worker is worse
/// than a late catch of a hung one. Overridable for tests and
/// support via DAHLIA_WORKER_TIMEOUT_SECS.
pub fn default_timeout() -> Duration {
    std::env::var("DAHLIA_WORKER_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(300))
}

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
    lines: Receiver<String>,
    next_id: i64,
    timeout: Duration,
    /// Why the session died (timeout / cancel / disconnect / bad
    /// output). Once set, every further call fails fast.
    broken: Option<String>,
}

impl WorkerSession {
    pub fn start(cfg: &WorkerConfig, worker: &str) -> Result<Self> {
        let script = cfg.worker_script(worker);
        if !script.exists() {
            return Err(Error::Worker(format!(
                "AI処理の部品（ワーカー）が見つかりません: {}",
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
            .map_err(|e| Error::Worker(format!("Python（AI処理の実行環境）を起動できませんでした（{worker}）: {e}")))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let mut stdout = child.stdout.take().expect("piped stdout");
        // A reader thread owns stdout so call() can wait with a
        // deadline (and watch the cancel flag) instead of blocking
        // on read_line forever.
        let (tx, lines) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let mut reader = BufReader::new(&mut stdout);
            let mut buf = String::new();
            loop {
                buf.clear();
                match reader.read_line(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if tx.send(buf.clone()).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            lines,
            next_id: 0,
            timeout: default_timeout(),
            broken: None,
        })
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    pub fn is_broken(&self) -> bool {
        self.broken.is_some()
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn break_with(&mut self, why: &str) {
        if self.broken.is_none() {
            self.broken = Some(why.to_string());
        }
        self.kill();
    }

    /// Send one request, read its response. Interim progress lines
    /// are forwarded to the hooks; the hooks' cancel flag aborts
    /// the wait. Equivalent to the original protocol when the
    /// worker never sends progress lines.
    pub fn call(&mut self, request: Value) -> Result<Value> {
        self.call_hooked(request, &JobHooks::none())
    }

    pub fn call_hooked(&mut self, mut request: Value, hooks: &JobHooks) -> Result<Value> {
        if let Some(why) = &self.broken {
            return Err(Error::Worker(format!(
                "ワーカーとの通信が切れているため処理を続けられません（{why}）"
            )));
        }
        if hooks.is_cancelled() {
            self.break_with("中断されました");
            return Err(Error::Cancelled);
        }
        self.next_id += 1;
        let id = self.next_id;
        if let Some(obj) = request.as_object_mut() {
            obj.insert("id".into(), Value::from(id));
        }
        let line = serde_json::to_string(&request)?;
        if let Err(e) = writeln!(self.stdin, "{line}").and_then(|_| self.stdin.flush()) {
            self.break_with("送信に失敗しました");
            return Err(Error::Worker(format!("ワーカーへの送信に失敗しました: {e}")));
        }

        let mut deadline = Instant::now() + self.timeout;
        loop {
            if hooks.is_cancelled() {
                self.break_with("中断されました");
                return Err(Error::Cancelled);
            }
            let wait = deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(100));
            match self.lines.recv_timeout(wait) {
                Ok(raw) => {
                    deadline = Instant::now() + self.timeout;
                    let resp: Value = match serde_json::from_str(raw.trim()) {
                        Ok(v) => v,
                        Err(e) => {
                            self.break_with("応答を解釈できませんでした");
                            return Err(Error::Worker(format!(
                                "ワーカーの応答を解釈できませんでした: {e}"
                            )));
                        }
                    };
                    let resp_id = resp.get("id").and_then(|v| v.as_i64());
                    if let Some(p) = resp.get("progress") {
                        if resp_id == Some(id) || resp_id.is_none() {
                            hooks.report(
                                p.get("done").and_then(|v| v.as_u64()).unwrap_or(0),
                                p.get("total").and_then(|v| v.as_u64()).unwrap_or(0),
                                None,
                            );
                        }
                        continue;
                    }
                    if resp_id.is_some() && resp_id != Some(id) {
                        // A final response to some other request:
                        // cannot happen with the shipped workers;
                        // ignore rather than misattribute it.
                        continue;
                    }
                    if resp.get("ok").and_then(|v| v.as_bool()) == Some(true) {
                        return Ok(resp);
                    }
                    let msg = resp
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("ワーカーが不明なエラーを返しました")
                        .to_string();
                    return Err(Error::Worker(msg));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if Instant::now() >= deadline {
                        self.break_with("応答がタイムアウトしました");
                        return Err(Error::Worker(format!(
                            "ワーカーが{}秒以上応答しないため、処理を中止しました",
                            self.timeout.as_secs()
                        )));
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    self.break_with("ワーカーが終了しました");
                    return Err(Error::Worker(
                        "処理中にワーカーが終了してしまいました".into(),
                    ));
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write as _;
    use std::sync::Mutex;

    /// Write a fake worker (Python, no repo imports) into a fresh
    /// temp workers dir; returns the config to start it with.
    fn fake_worker(name: &str, body: &str) -> Option<WorkerConfig> {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            return None; // no python3 in this environment: skip
        }
        let dir = std::env::temp_dir().join(format!(
            "dahlia-jobs-test-{}-{name}",
            std::process::id()
        ));
        let script_dir = dir.join(name);
        std::fs::create_dir_all(&script_dir).unwrap();
        let mut f = std::fs::File::create(script_dir.join("worker.py")).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        Some(WorkerConfig {
            python: "python3".to_string(),
            workers_dir: dir,
        })
    }

    const ECHO: &str = r#"
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    if req.get("fail"):
        print(json.dumps({"id": req["id"], "ok": False, "error": "boom"}), flush=True)
    else:
        print(json.dumps({"id": req["id"], "ok": True, "echo": req.get("x")}), flush=True)
"#;

    const CHATTY: &str = r#"
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    for d in (1, 2, 3):
        print(json.dumps({"id": req["id"], "progress": {"done": d, "total": 3}}), flush=True)
    print(json.dumps({"id": req["id"], "ok": True}), flush=True)
"#;

    const SLEEPY: &str = r#"
import json, sys, time
for line in sys.stdin:
    req = json.loads(line)
    time.sleep(30)
    print(json.dumps({"id": req["id"], "ok": True}), flush=True)
"#;

    #[test]
    fn plain_call_roundtrip_and_request_error_keeps_session() {
        let Some(cfg) = fake_worker("echo", ECHO) else { return };
        let mut s = WorkerSession::start(&cfg, "echo").unwrap();
        let r = s.call(json!({ "x": 42 })).unwrap();
        assert_eq!(r.get("echo").and_then(|v| v.as_i64()), Some(42));
        // A per-request error (ok:false) is the worker answering,
        // not a dead session: the next call still works.
        let e = s.call(json!({ "fail": true })).unwrap_err();
        assert!(e.to_string().contains("boom"));
        assert!(!s.is_broken());
        let r = s.call(json!({ "x": 7 })).unwrap();
        assert_eq!(r.get("echo").and_then(|v| v.as_i64()), Some(7));
    }

    #[test]
    fn progress_lines_are_forwarded() {
        let Some(cfg) = fake_worker("chatty", CHATTY) else { return };
        let mut s = WorkerSession::start(&cfg, "chatty").unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        let hooks = JobHooks {
            cancel: None,
            progress: Some(Arc::new(move |p: ProgressInfo| {
                seen2.lock().unwrap().push((p.done, p.total));
            })),
        };
        s.call_hooked(json!({}), &hooks).unwrap();
        assert_eq!(*seen.lock().unwrap(), vec![(1, 3), (2, 3), (3, 3)]);
    }

    #[test]
    fn silent_worker_times_out_and_session_breaks() {
        let Some(cfg) = fake_worker("sleepy", SLEEPY) else { return };
        let mut s = WorkerSession::start(&cfg, "sleepy").unwrap();
        s.set_timeout(Duration::from_millis(300));
        let t0 = Instant::now();
        let e = s.call(json!({})).unwrap_err();
        assert!(t0.elapsed() < Duration::from_secs(10), "timeout took {:?}", t0.elapsed());
        assert!(e.to_string().contains("応答しない"), "{e}");
        assert!(s.is_broken());
        // Further calls fail fast instead of hanging again.
        let t0 = Instant::now();
        assert!(s.call(json!({})).is_err());
        assert!(t0.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn cancel_aborts_a_long_call() {
        let Some(cfg) = fake_worker("sleepy2", SLEEPY) else { return };
        let mut s = WorkerSession::start(&cfg, "sleepy2").unwrap();
        s.set_timeout(Duration::from_secs(60));
        let flag = Arc::new(AtomicBool::new(false));
        let hooks = JobHooks {
            cancel: Some(flag.clone()),
            progress: None,
        };
        let t0 = Instant::now();
        let setter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            flag.store(true, Ordering::Relaxed);
        });
        let e = s.call_hooked(json!({}), &hooks).unwrap_err();
        setter.join().unwrap();
        assert!(matches!(e, Error::Cancelled), "{e}");
        assert!(t0.elapsed() < Duration::from_secs(10), "cancel took {:?}", t0.elapsed());
    }
}
