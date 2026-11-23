//! End-to-end pipeline test: real Rust core + real Python workers +
//! real ONNX models over the sample photos in tests/samples.
//!
//! Requires the worker venv (DAHLIA_PYTHON) and models; skipped
//! gracefully when they are absent (e.g. on CI without models).
use dahlia_core::jobs::WorkerConfig;
use dahlia_core::storage::{Db, NewFace};
use studio_lib::{import, pipeline};

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

#[test]
fn full_pipeline_on_samples() {
    let root = repo_root();
    let python = root.join(".venv/bin/python");
    let face_model = root.join("workers/face/models/face_detection_yunet_2023mar.onnx");
    if !python.exists() || !face_model.exists() {
        eprintln!("skipping: venv or models not present");
        return;
    }
    let cfg = WorkerConfig {
        python: python.to_string_lossy().into_owned(),
        workers_dir: root.join("workers"),
    };
    let samples = root.join("tests/samples");
    let db = Db::open_in_memory().unwrap();
    let project = db.create_project("E2E").unwrap();
    let base = std::env::temp_dir().join("dahlia-pipeline-e2e");
    let _ = std::fs::remove_dir_all(&base);
    let thumbs = base.join("thumbnails");
    let proxies = base.join("proxies");
    let crops = base.join("faces");
    let imported = import::import_folder(&db, &cfg, &thumbs, &proxies, project.id, &samples).unwrap();
    assert_eq!(imported.added, 4);
    assert_eq!(imported.failed, 0);
    let assets = db.list_assets(project.id).unwrap();
    let landscape = assets.iter().find(|a| a.path.ends_with("landscape.jpg")).unwrap();

    // Phase 6: detection
    let report = pipeline::detect_faces(&db, &cfg, &crops, &proxies, project.id).unwrap();
    assert_eq!(report.failed, 0);
    assert_eq!(report.faces_found, 3, "one face per portrait, none in landscape");
    assert_eq!(db.list_faces(landscape.id).unwrap().len(), 0);

    let old_face = assets.iter().find_map(|a| db.list_faces(a.id).unwrap().first().map(|f| f.id)).unwrap();
    db.save_embedding(old_face, "arcface-mbf", &[1.0; 512]).unwrap();

    // Phase 7: embeddings
    let report = pipeline::generate_embeddings(&db, &cfg, project.id).unwrap();
    assert_eq!(report.embedded, 3);
    let fes = db.faces_with_embeddings(project.id, pipeline::EMBED_MODEL).unwrap();
    assert_eq!(fes.len(), 3);
    assert_eq!(fes[0].vector.len(), 128);
    assert_eq!(db.faces_with_embeddings(project.id, "arcface-mbf").unwrap().len(), 1);

    // Phase 8: clustering -> Einstein pair + Curie single = 2 identities
    let report = pipeline::cluster_identities(&db, &cfg, project.id).unwrap();
    assert_eq!(report.identities, 2);
    assert_eq!(report.clustered_faces, 2);
    assert_eq!(report.single_faces, 1);
    let identities = db.list_identities(project.id).unwrap();
    assert_eq!(identities.len(), 2);
    let sizes: Vec<i64> = identities.iter().map(|i| i.face_count).collect();
    assert!(sizes.contains(&2) && sizes.contains(&1));
}

#[test]
fn storage_new_face_type_is_used() {
    // Compile-time guard that pipeline/storage share the NewFace type.
    let _f = NewFace { bbox: [0.0; 4], confidence: 0.5 };
}
