//! Progress / cancel integration tests (W9/W10/W16, W22).
//!
//! Detection runs against a fake face worker (plain python3, no
//! models); clustering runs against the real clustering worker
//! (numpy only) fed with synthetic embeddings written straight
//! into the DB. Both are skipped gracefully when the python they
//! need is unavailable.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use dahlia_core::jobs::{JobHooks, ProgressInfo, WorkerConfig};
use dahlia_core::storage::{Db, NewFace};
use studio_lib::{pipeline, space};

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

fn python_ok(python: &str, need_numpy: bool) -> bool {
    let mut cmd = std::process::Command::new(python);
    cmd.arg("-c");
    if need_numpy {
        cmd.arg("import numpy");
    } else {
        cmd.arg("pass");
    }
    cmd.output().map(|o| o.status.success()).unwrap_or(false)
}

fn python_with_numpy() -> Option<String> {
    let venv = repo_root().join(".venv/bin/python");
    for cand in [venv.to_string_lossy().into_owned(), "python3".to_string()] {
        if python_ok(&cand, true) {
            return Some(cand);
        }
    }
    None
}

type ProgressLog = Arc<Mutex<Vec<ProgressInfo>>>;

fn recording_hooks(log: &ProgressLog, cancel: Option<Arc<AtomicBool>>) -> JobHooks {
    let log = log.clone();
    JobHooks {
        cancel,
        progress: Some(Arc::new(move |p: ProgressInfo| {
            log.lock().unwrap().push(p);
        })),
    }
}

const FAKE_FACE_WORKER: &str = r#"
import json, sys, time
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    time.sleep(0.02)
    resp = {"id": req.get("id"), "ok": True,
            "faces": [{"bbox": [1, 1, 10, 10], "confidence": 0.99}]}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"#;

fn fake_face_config() -> Option<WorkerConfig> {
    if !python_ok("python3", false) {
        return None;
    }
    let dir = std::env::temp_dir().join(format!("dahlia-progress-test-{}", std::process::id()));
    let face_dir = dir.join("face");
    std::fs::create_dir_all(&face_dir).unwrap();
    std::fs::write(face_dir.join("worker.py"), FAKE_FACE_WORKER).unwrap();
    Some(WorkerConfig {
        python: "python3".to_string(),
        workers_dir: dir,
    })
}

fn project_with_assets(db: &Db, n: usize) -> (i64, Vec<i64>) {
    let project = db.create_project("進捗テスト").unwrap();
    let mut ids = Vec::new();
    for i in 0..n {
        let a = db
            .add_asset(project.id, &format!("/nonexistent/photo-{i}.jpg"), None)
            .unwrap();
        ids.push(a.id);
    }
    (project.id, ids)
}

#[test]
fn detect_reports_progress_and_completes() {
    let Some(cfg) = fake_face_config() else {
        eprintln!("skipping: python3 not available");
        return;
    };
    let db = Db::open_in_memory().unwrap();
    let (project_id, _) = project_with_assets(&db, 4);
    let crops = std::env::temp_dir().join("dahlia-progress-crops");
    let proxies = crops.join("proxies");
    let log: ProgressLog = Arc::new(Mutex::new(Vec::new()));
    let hooks = recording_hooks(&log, None);

    let report = pipeline::detect_faces_hooked(&db, &cfg, &crops, &proxies, project_id, &hooks).unwrap();
    assert!(!report.cancelled);
    assert_eq!(report.images_processed, 4);
    assert_eq!(report.faces_found, 4);
    let events = log.lock().unwrap();
    assert_eq!((events.first().unwrap().done, events.first().unwrap().total), (0, 4));
    assert_eq!((events.last().unwrap().done, events.last().unwrap().total), (4, 4));
    assert!(events.windows(2).all(|w| w[0].done <= w[1].done), "progress must not go backwards");
}

#[test]
fn detect_cancel_stops_safely_with_partial_results_saved() {
    let Some(cfg) = fake_face_config() else {
        eprintln!("skipping: python3 not available");
        return;
    };
    let db = Db::open_in_memory().unwrap();
    let (project_id, asset_ids) = project_with_assets(&db, 10);
    let crops = std::env::temp_dir().join("dahlia-progress-crops");
    let proxies = crops.join("proxies");
    // The progress sink itself pulls the brake once 3 photos are
    // done — the same thing the UI's cancel button does.
    let flag = Arc::new(AtomicBool::new(false));
    let flag2 = flag.clone();
    let hooks = JobHooks {
        cancel: Some(flag.clone()),
        progress: Some(Arc::new(move |p: ProgressInfo| {
            if p.done >= 3 {
                flag2.store(true, Ordering::Relaxed);
            }
        })),
    };

    let report = pipeline::detect_faces_hooked(&db, &cfg, &crops, &proxies, project_id, &hooks).unwrap();
    assert!(report.cancelled, "report must say the job was cancelled");
    assert_eq!(report.images_processed, 3);
    // The 3 finished photos kept their faces; the rest are untouched.
    for (i, asset_id) in asset_ids.iter().enumerate() {
        let faces = db.list_faces(*asset_id).unwrap();
        assert_eq!(faces.len(), if i < 3 { 1 } else { 0 }, "asset {i}");
    }
    // And a rerun simply picks up the remaining photos (resume).
    let report = pipeline::detect_faces_hooked(
        &db,
        &cfg,
        &crops,
        &proxies,
        project_id,
        &recording_hooks(&Arc::new(Mutex::new(Vec::new())), None),
    )
    .unwrap();
    assert!(!report.cancelled);
    assert_eq!(report.images_processed, 7);
    assert_eq!(report.skipped, 3);
}

/// Small deterministic RNG for synthetic embeddings.
struct XorShift(u64);
impl XorShift {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
    }
}

fn synthetic_vector(rng: &mut XorShift, base: Option<&[f32]>) -> Vec<f32> {
    let mut v: Vec<f32> = (0..512).map(|_| rng.next_f32()).collect();
    if let Some(b) = base {
        for (x, y) in v.iter_mut().zip(b.iter()) {
            *x = y + 0.08 * *x;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn project_with_embeddings(db: &Db, people: usize, per_person: usize) -> i64 {
    let project = db.create_project("分類テスト").unwrap();
    let asset = db.add_asset(project.id, "/nonexistent/all.jpg", None).unwrap();
    let mut rng = XorShift(0x1234_5678_9abc_def1);
    let mut new_faces = Vec::new();
    let mut vectors = Vec::new();
    for _ in 0..people {
        let base = synthetic_vector(&mut rng, None);
        for _ in 0..per_person {
            new_faces.push(NewFace { bbox: [0.0, 0.0, 10.0, 10.0], confidence: 0.99 });
            vectors.push(synthetic_vector(&mut rng, Some(&base)));
        }
    }
    let saved = db.replace_faces(asset.id, &new_faces).unwrap();
    for (face, v) in saved.iter().zip(vectors.iter()) {
        db.save_embedding(face.id, pipeline::EMBED_MODEL, v).unwrap();
    }
    project.id
}

#[test]
fn cluster_streams_chunks_and_reports_both_phases() {
    let Some(python) = python_with_numpy() else {
        eprintln!("skipping: no python with numpy");
        return;
    };
    let cfg = WorkerConfig {
        python,
        workers_dir: repo_root().join("workers"),
    };
    let db = Db::open_in_memory().unwrap();
    // 600 faces = 150 people x 4: more than one 500-item chunk.
    let project_id = project_with_embeddings(&db, 150, 4);
    let log: ProgressLog = Arc::new(Mutex::new(Vec::new()));
    let hooks = recording_hooks(&log, None);

    let report = pipeline::cluster_identities_hooked(&db, &cfg, project_id, &hooks).unwrap();
    assert!(!report.cancelled);
    assert_eq!(report.identities, 150);
    assert_eq!(report.clustered_faces, 600);

    let events = log.lock().unwrap();
    assert!(
        events.iter().any(|p| p.phase.as_deref() == Some("データを送信中") && p.done == 600 && p.total == 1200),
        "send phase must reach 600/1200: {events:?}"
    );
    let last = events.last().unwrap();
    assert_eq!(last.phase.as_deref(), Some("分類を計算中"));
    assert_eq!((last.done, last.total), (1200, 1200));
    assert!(events.windows(2).all(|w| w[0].done <= w[1].done), "progress must not go backwards");
}

#[test]
fn cluster_cancel_during_send_keeps_previous_identities() {
    let Some(python) = python_with_numpy() else {
        eprintln!("skipping: no python with numpy");
        return;
    };
    let cfg = WorkerConfig {
        python,
        workers_dir: repo_root().join("workers"),
    };
    let db = Db::open_in_memory().unwrap();
    let project_id = project_with_embeddings(&db, 150, 4);
    let flag = Arc::new(AtomicBool::new(false));
    let flag2 = flag.clone();
    let hooks = JobHooks {
        cancel: Some(flag.clone()),
        progress: Some(Arc::new(move |p: ProgressInfo| {
            if p.done >= 500 {
                flag2.store(true, Ordering::Relaxed);
            }
        })),
    };

    let report = pipeline::cluster_identities_hooked(&db, &cfg, project_id, &hooks).unwrap();
    assert!(report.cancelled);
    // Nothing was replaced: the project still has no identities
    // (and a later full run can proceed normally).
    assert!(db.list_identities(project_id).unwrap().is_empty());
    let report = pipeline::cluster_identities_hooked(
        &db,
        &cfg,
        project_id,
        &recording_hooks(&Arc::new(Mutex::new(Vec::new())), None),
    )
    .unwrap();
    assert!(!report.cancelled);
    assert_eq!(report.identities, 150);
}

#[test]
fn space_streams_chunks_and_reports_both_phases() {
    let Some(python) = python_with_numpy() else {
        eprintln!("skipping: no python with numpy");
        return;
    };
    let cfg = WorkerConfig {
        python,
        workers_dir: repo_root().join("workers"),
    };
    let db = Db::open_in_memory().unwrap();
    // 600 faces = 150 people x 4: more than one 500-item chunk (W23).
    let project_id = project_with_embeddings(&db, 150, 4);
    let crops = std::env::temp_dir().join("dahlia-progress-crops");
    let log: ProgressLog = Arc::new(Mutex::new(Vec::new()));
    let hooks = recording_hooks(&log, None);

    let data = space::identity_space_hooked(&db, &cfg, &crops, project_id, &hooks).unwrap();
    assert_eq!(data.points.len(), 600);
    assert_eq!(data.method, "pca");

    let events = log.lock().unwrap();
    assert!(
        events.iter().any(|p| p.phase.as_deref() == Some("データを送信中") && p.done == 600 && p.total == 1200),
        "send phase must reach 600/1200: {events:?}"
    );
    let last = events.last().unwrap();
    assert_eq!(last.phase.as_deref(), Some("空間を計算中"));
    assert_eq!((last.done, last.total), (1200, 1200));
    assert!(events.windows(2).all(|w| w[0].done <= w[1].done), "progress must not go backwards");
}

#[test]
fn space_cancel_during_send_aborts_cleanly() {
    let Some(python) = python_with_numpy() else {
        eprintln!("skipping: no python with numpy");
        return;
    };
    let cfg = WorkerConfig {
        python,
        workers_dir: repo_root().join("workers"),
    };
    let db = Db::open_in_memory().unwrap();
    let project_id = project_with_embeddings(&db, 150, 4);
    let crops = std::env::temp_dir().join("dahlia-progress-crops");
    let flag = Arc::new(AtomicBool::new(false));
    let flag2 = flag.clone();
    let hooks = JobHooks {
        cancel: Some(flag.clone()),
        progress: Some(Arc::new(move |p: ProgressInfo| {
            if p.done >= 500 {
                flag2.store(true, Ordering::Relaxed);
            }
        })),
    };

    let err = match space::identity_space_hooked(&db, &cfg, &crops, project_id, &hooks) {
        Ok(_) => panic!("cancelled space computation must not succeed"),
        Err(e) => e,
    };
    assert!(err.contains("中断"), "unexpected error: {err}");
    // Nothing was persisted by the space computation, and a later
    // full run proceeds normally (no state leaked anywhere).
    let data = space::identity_space_hooked(
        &db,
        &cfg,
        &crops,
        project_id,
        &recording_hooks(&Arc::new(Mutex::new(Vec::new())), None),
    )
    .unwrap();
    assert_eq!(data.points.len(), 600);
}
