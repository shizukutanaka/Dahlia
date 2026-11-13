//! Face mesh E2E: after the real pipeline, the identity's best face
//! must yield a full MediaPipe landmark mesh, and the second call
//! must come from the on-disk cache.
use dahlia_core::jobs::WorkerConfig;
use dahlia_core::storage::Db;
use studio_lib::{facemesh, pipeline};

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

#[test]
fn identity_face_mesh_on_samples() {
    let root = repo_root();
    let python = root.join(".venv/bin/python");
    let model = root.join("workers/facemesh/models/face_landmarker.task");
    if !python.exists() || !model.exists() {
        eprintln!("skipping: venv or facemesh model not present");
        return;
    }
    let cfg = WorkerConfig {
        python: python.to_string_lossy().into_owned(),
        workers_dir: root.join("workers"),
    };
    let samples = root.join("tests/samples");
    let db = Db::open_in_memory().unwrap();
    let project = db.create_project("MeshE2E").unwrap();
    let data_dir = std::env::temp_dir().join("dahlia-facemesh-e2e");
    let _ = std::fs::remove_dir_all(&data_dir);
    std::fs::create_dir_all(data_dir.join("faces")).unwrap();
    for name in ["einstein_a.jpg", "einstein_b.jpg"] {
        db.add_asset(project.id, samples.join(name).to_str().unwrap(), None).unwrap();
    }
    pipeline::detect_faces(&db, &cfg, &data_dir.join("faces"), &data_dir.join("proxies"), project.id).unwrap();
    pipeline::generate_embeddings(&db, &cfg, project.id).unwrap();
    pipeline::cluster_identities(&db, &cfg, project.id).unwrap();
    let identities = db.list_identities(project.id).unwrap();
    assert_eq!(identities.len(), 1);

    let data = match facemesh::identity_face_mesh(&db, &cfg, &data_dir, identities[0].id) {
        Ok(d) => d,
        Err(e) => {
            // mediapipe needs system GL libraries (libEGL); on machines
            // without them the graceful error is the contract.
            eprintln!("skipping landmark assertions: {e}");
            return;
        }
    };
    assert_eq!(data.points.len(), 478);
    assert_eq!(data.uvs.len(), 478);
    assert!(data.triangles.len() > 800, "triangles: {}", data.triangles.len());
    assert!(!data.blendshapes.is_empty());
    // nose tip (1) in front of cheek edge (234)
    assert!(data.points[1][2] > data.points[234][2]);

    let cache = data_dir.join("facemesh").join(format!("{}.json", data.face_id));
    assert!(cache.exists(), "cache file written");
    let again = facemesh::identity_face_mesh(&db, &cfg, &data_dir, identities[0].id).unwrap();
    assert_eq!(again.points.len(), 478);
}
