//! End-to-end pipeline test: real Rust core + real Python workers +
//! real ONNX models over the sample photos in tests/samples.
//!
//! Requires the worker venv (DAHLIA_PYTHON) and models; skipped
//! gracefully when they are absent (e.g. on CI without models).
use dahlia_core::jobs::WorkerConfig;
use dahlia_core::storage::{Db, NewFace};
use studio_lib::pipeline;

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
    let face_model = root.join("workers/face/models/det_500m.onnx");
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
    let crops = std::env::temp_dir().join("dahlia-e2e-crops");
    let _ = std::fs::remove_dir_all(&crops);

    let mut asset_ids = Vec::new();
    for name in ["einstein_a.jpg", "einstein_b.jpg", "curie_a.jpg", "landscape.jpg"] {
        let asset = db
            .add_asset(project.id, samples.join(name).to_str().unwrap(), None)
            .unwrap();
        asset_ids.push(asset.id);
    }

    // Phase 6: detection
    let report = pipeline::detect_faces(&db, &cfg, &crops, &crops.join("proxies"), project.id).unwrap();
    assert_eq!(report.failed, 0);
    assert_eq!(report.faces_found, 3, "one face per portrait, none in landscape");
    assert_eq!(db.list_faces(asset_ids[3]).unwrap().len(), 0);

    // Phase 7: embeddings
    let report = pipeline::generate_embeddings(&db, &cfg, project.id).unwrap();
    assert_eq!(report.embedded, 3);
    let fes = db.faces_with_embeddings(project.id, pipeline::EMBED_MODEL).unwrap();
    assert_eq!(fes.len(), 3);
    assert_eq!(fes[0].vector.len(), 512);

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
